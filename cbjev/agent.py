"""`Agent`: load a checkpoint, answer typed questions about a state.

    import cbjev
    agent = cbjev.load()                      # the default cbjev checkpoint
    agent.predict({"body": "billed twice, refund please"}, {
        "dept": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "payments", "tech": "bugs"}},
        "angry": {"type": "noul", "instructions": "Is the customer upset?"},
    })

The response has the same shape Jev returns: ``{"model", "answers", "usage"}``.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import warnings
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .layout import Question, Tokens, as_text, classic_row, packed_row, parse_question
from .hooks import Hooks, CallContext

State = Union[str, dict, list]

TEMP_RANGE = (0.5, 5.0)


def _lse(z: np.ndarray) -> float:
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


def _bucket_name(kind: str, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (kind, size)


def _pick_device(device: Optional[str]):
    import torch
    if device:
        d = torch.device(device)
        if d.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("cbjev: CUDA requested but unavailable, using CPU", RuntimeWarning, stacklevel=3)
            return torch.device("cpu")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint(name_or_path: str, subfolder: Optional[str] = None, token: Optional[str] = None) -> str:
    """A local directory as-is, otherwise a Hugging Face repo id fetched into the HF cache."""
    path = os.path.expanduser(name_or_path)
    if not os.path.isdir(path):
        if name_or_path.startswith((".", "/", "~")):
            raise FileNotFoundError("no checkpoint directory at %r" % name_or_path)
        from huggingface_hub import snapshot_download
        pre = subfolder + "/" if subfolder else ""
        pats = [pre + p for p in ("*.json", "model.safetensors", "tokenizer/*", "encoder/*")]
        path = snapshot_download(name_or_path, allow_patterns=pats, token=token or os.environ.get("HF_TOKEN") or None)
    if subfolder:
        path = os.path.join(path, subfolder)
    if not os.path.isdir(path):
        raise FileNotFoundError("subfolder %r not found in %r" % (subfolder, name_or_path))
    return path


_TOKENIZERS: Dict[str, Tokens] = {}
_TOK_LOCK = threading.Lock()


def _tokens_for(tok_dir: str) -> Tokens:
    """One parsed tokenizer per directory per process (mmBERT's is 34 MB of JSON)."""
    with _TOK_LOCK:
        hit = _TOKENIZERS.get(tok_dir)
        if hit is None:
            from tokenizers import Tokenizer
            from transformers import PreTrainedTokenizerFast
            cfg = {}
            cfg_path = os.path.join(tok_dir, "tokenizer_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
            specials = {k: cfg[k] for k in ("cls_token", "sep_token", "mask_token", "pad_token", "unk_token")
                        if isinstance(cfg.get(k), str)}
            hf = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json")),
                                         **specials)
            hit = _TOKENIZERS[tok_dir] = Tokens(hf)
        return hit


class Agent:
    """A loaded decision model.

    `layout` comes from the checkpoint: cbjev checkpoints are ``packed`` (the state is encoded
    once for all questions), Laya checkpoints are ``classic`` (one sequence per question) and run
    unchanged. `graphs=False` turns off CUDA graph replay.
    """

    def __init__(self, checkpoint: str = "cbjev", device: Optional[str] = None, subfolder: Optional[str] = None,
                 token: Optional[str] = None, graphs: bool = True, hooks=None, name: Optional[str] = None,
                 order_votes: Optional[int] = None):
        import torch
        from safetensors.torch import load_file
        from .engine import Engine
        from .model import DecisionNet, EncoderSpec, load_weights
        from . import checkpoints

        path = resolve_checkpoint(checkpoints.resolve(checkpoint), subfolder, token)
        cfg_file = next((os.path.join(path, n) for n in ("cbjev_config.json", "rl_agent_config.json")
                         if os.path.exists(os.path.join(path, n))), None)
        if cfg_file is None:
            raise FileNotFoundError("%r has no cbjev_config.json / rl_agent_config.json" % path)
        with open(cfg_file) as f:
            self.cfg = cfg = json.load(f)
        with open(os.path.join(path, "encoder", "config.json")) as f:
            spec = EncoderSpec.from_hf(json.load(f))

        self.name = name or cfg.get("model_name") or os.path.basename(path.rstrip("/"))
        self.path = path
        self.layout = cfg.get("layout", "classic")
        self.max_len = int(cfg.get("max_len", 512))
        self.head_max_len = int(cfg.get("head_max_len", 192))
        self.max_state = int(cfg.get("max_state_tokens", 1024))
        self.max_question = int(cfg.get("max_question_tokens", 768))
        self.tokens = _tokens_for(os.path.join(path, "tokenizer"))
        self.hooks = Hooks.coerce(hooks)
        # packed layout only: also ask every choice/score question with its options reversed and
        # average the two answers. Costs one more short segment per question, not another pass.
        votes = cfg.get("order_votes", 1) if order_votes is None else order_votes
        self.order_votes = int(votes) if self.layout == "packed" else 1

        self.temperature = [self._clamp(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {k: self._clamp(v) for k, v in cfg.get("temperature_by_options", {}).items()}

        self.device = _pick_device(device)
        self.dtype = torch.float32
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        net = DecisionNet(spec, head_layers=int(cfg.get("head_layers", 2)))
        load_weights(net, load_file(os.path.join(path, "model.safetensors")))
        net.eval()
        if self.dtype != torch.float32:
            net.cast_linear(self.dtype)
        self.net = net.to(self.device)
        self.engine = Engine(self.net, self.device, self.dtype, packed=self.layout == "packed", graphs=graphs)

    @staticmethod
    def _clamp(t) -> float:
        try:
            t = float(t)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(t):
            return 1.0
        return min(TEMP_RANGE[1], max(TEMP_RANGE[0], t))

    # ------------------------------------------------------------------ building batches

    def _rows(self, states: Sequence[State], qs: List[Question]):
        """Token rows plus, per state, the (row, marker offset) where each question's options sit."""
        texts = [as_text(s) for s in states]
        enc = self.tokens.encode(texts)
        rows, where = [], []
        for st, ids in zip(states, enc):
            left = isinstance(st, list)          # conversations: keep the newest turns
            if self.layout == "packed":
                r = packed_row(self.tokens, ids, qs, self.max_state, self.max_question, left)
                base = len(rows)
                rows.append(r)
                where.append([(base, m) for m in r.markers])
            else:
                spots = []
                for q in qs:
                    row_ids, mk = classic_row(self.tokens, ids, q, self.max_len, self.head_max_len, left)
                    spots.append((len(rows), mk))
                    rows.append((row_ids, mk, q.qtype))
                where.append(spots)
        return rows, where

    def _collate(self, rows):
        from .engine import Batch, _bucket, _rows_bucket
        packed = self.layout == "packed"
        n = len(rows)
        L = max(len(r.ids if packed else r[0]) for r in rows)
        M = max(sum(len(m) for m in r.markers) if packed else len(r[1]) for r in rows)
        b = Batch(_rows_bucket(n), _bucket(L), _bucket(M, (2, 4, 8, 16, 32, 64, 128, 256)), self.tokens.pad)
        for i, r in enumerate(rows):
            if packed:
                k = len(r.ids)
                b.ids[i, :k], b.pos[i, :k], b.seg[i, :k], b.qtype[i, :k] = r.ids, r.pos, r.seg, r.qtype
                flat = [m for mk in r.markers for m in mk]
            else:
                ids, flat, qt = r
                k = len(ids)
                b.ids[i, :k] = ids
                b.pos[i, :k] = np.arange(k)
                b.seg[i, :k] = 0
                b.qtype[i, :] = qt
            b.valid[i, :k] = True
            b.markers[i, :len(flat)] = flat
        # padded rows still need one readable key so attention never sees an empty row
        b.valid[n:, 0] = True
        return b, n

    # ------------------------------------------------------------------ answers

    def _temp(self, q: Question, k: int) -> float:
        return self.temperature_by_options.get(_bucket_name(q.kind, k), self.temperature[q.qtype])

    def _answer(self, q: Question, z: np.ndarray) -> Dict[str, Any]:
        k = len(z)
        z = z / self._temp(q, k)
        p = np.exp(z - z.max())
        p /= p.sum()
        if q.kind == "noul":
            pt = float(p[1])
            return {"type": "noul", "noul": round(pt, 4), "confidence": round(max(pt, 1 - pt), 4)}
        ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
        conf = 1.0 if k < 2 else max(0.0, min(1.0, 1 - ent / math.log(k)))
        probs = {key: round(float(v), 4) for key, v in zip(q.keys, p)}
        if q.kind == "choice":
            return {"type": "choice", "choice": q.keys[int(p.argmax())], "probabilities": probs,
                    "confidence": round(conf, 4)}
        return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): o.split(": ", 1)[1] for i, o in enumerate(q.options)},
                "probabilities": probs, "confidence": round(conf, 4)}

    def _score(self, states: Sequence[State], asked: List[Question], batch_rows: int = 32):
        qs, twin = list(asked), {}
        if self.order_votes > 1:
            for i, q in enumerate(asked):
                if q.kind != "noul" and len(q.options) > 1:
                    twin[i] = len(qs)
                    qs.append(replace(q, qid=q.qid + "\x00rev", keys=q.keys[::-1], options=q.options[::-1]))
        rows, where = self._rows(states, qs)
        logits = [None] * len(rows)
        tokens = [0] * len(rows)
        packed = self.layout == "packed"
        # sort by length so each forward pads as little as possible
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids if packed else rows[i][0]))
        for s in range(0, len(order), batch_rows):
            part = [rows[i] for i in order[s:s + batch_rows]]
            b, n = self._collate(part)
            out = self.engine.run(b)
            for j, i in enumerate(order[s:s + batch_rows]):
                logits[i] = out[j]
                tokens[i] = int(b.valid[j].sum())
        results = []
        offsets = np.cumsum([0] + [len(q.options) for q in qs])
        for spots in where:
            answers, used = {}, set()
            raw = []
            for qi, (q, spot) in enumerate(zip(qs, spots)):
                if packed:
                    ri, _ = spot
                    lo = offsets[qi]
                    z = logits[ri][lo:lo + len(q.options)]
                else:
                    ri, mk = spot
                    z = logits[ri][:len(mk)]
                raw.append(np.asarray(z, dtype=np.float64))
                used.add(ri)
            for qi, q in enumerate(asked):
                z = raw[qi]
                if qi in twin:
                    # average log-probabilities of the two orders (the twin's are reversed back)
                    a, b = z - _lse(z), raw[twin[qi]][::-1]
                    z = (a + b - _lse(b)) / 2
                answers[q.qid] = self._answer(q, z)
            results.append({"model": self.name, "answers": answers,
                            "usage": {"input_tokens": sum(tokens[r] for r in used), "output_tokens": 0}})
        return results

    # ------------------------------------------------------------------ public API

    def predict_batch(self, states: Sequence[State], questions: Dict[str, Any], batch_rows: int = 32,
                      hooks=None) -> List[Dict[str, Any]]:
        """Answer the same questions for many states, sharing forward passes."""
        if isinstance(states, (str, bytes, dict)):
            raise TypeError("predict_batch takes a list of states; use predict() for one")
        ctx = CallContext(model=self.name, states=list(states), questions=questions, agent=self)
        hk = self.hooks.plus(hooks)
        with hk.around(ctx):
            if ctx.results is None:
                qs = [parse_question(k, v) for k, v in (ctx.questions or {}).items()]
                if not ctx.states:
                    ctx.results = []
                elif not qs:
                    ctx.results = [{"model": self.name, "answers": {},
                                    "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in ctx.states]
                else:
                    ctx.results = self._score(ctx.states, qs, batch_rows)
        return ctx.results

    def predict(self, state: State, questions: Dict[str, Any], hooks=None) -> Dict[str, Any]:
        """Answer every question about one state in one forward pass."""
        return self.predict_batch([state], questions, hooks=hooks)[0]

    system_one = predict

    def logits(self, states: Sequence[State], questions: Sequence[Question]) -> List[List[np.ndarray]]:
        """Raw marker logits per state per question (before temperature), for evaluation code."""
        rows, where = self._rows(states, list(questions))
        packed = self.layout == "packed"
        out_rows = {}
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids if packed else rows[i][0]))
        for s in range(0, len(order), 32):
            part = [rows[i] for i in order[s:s + 32]]
            b, _ = self._collate(part)
            o = self.engine.run(b)
            for j, i in enumerate(order[s:s + 32]):
                out_rows[i] = o[j]
        res = []
        for spots in where:
            per = []
            for qi, (q, spot) in enumerate(zip(questions, spots)):
                if packed:
                    ri, _ = spot
                    lo = sum(len(m) for m in rows[ri].markers[:qi])
                    per.append(out_rows[ri][lo:lo + len(q.options)].copy())
                else:
                    ri, mk = spot
                    per.append(out_rows[ri][:len(mk)].copy())
            res.append(per)
        return res

    def close(self):
        self.engine.clear()
        self.net = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load(checkpoint: str = "cbjev", device: Optional[str] = None, **kw) -> Agent:
    return Agent(checkpoint, device=device, **kw)
