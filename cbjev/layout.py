"""Turning (state, questions) into token rows.

Two layouts share the same question text format:

classic   one row per question, ``[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
          <state> [SEP]``. Byte-for-byte what Laya checkpoints were trained on; used to run them.

packed    one row per *state*: ``[CLS] <state> [SEP]`` followed by every question's segment
          ``<type> question: <ins> [SEP] [MASK] opt0 ... [SEP]``. The state is encoded once, each
          segment reads it through the attention mask, and segments never see each other. Every
          segment restarts its positions right after the state, so each question sees the state
          exactly as if it were alone with it. A 10-question call over a 500-token document is
          ~1k tokens instead of ~6k.

Tokenizing is the one part of a call that runs on the CPU for every request, so question text is
tokenized once and cached (the same question set is usually asked again and again), and the state
is tokenized once per call rather than once per question.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

QTYPES = {"choice": 0, "score": 1, "noul": 2}

NOUL_FALSE = "no, the statement does not hold"
NOUL_TRUE = "yes, the statement holds"


def as_text(value: Any) -> str:
    """States and structured criteria are shown to the model as compact JSON; strings as-is."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _crit_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


@dataclass(frozen=True)
class Question:
    """A validated question in the one shape the rest of the package reads."""
    qid: str
    kind: str                      # choice | score | noul
    instructions: str
    keys: Tuple[str, ...]          # answer keys in option order
    options: Tuple[str, ...]       # option text as the model reads it

    @property
    def qtype(self) -> int:
        return QTYPES[self.kind]

    @property
    def cache_key(self) -> Tuple:
        return (self.kind, self.instructions, self.options)


def parse_question(qid: str, spec: Any) -> Question:
    """Validate one question definition (Jev wire format) and render its options.

    Errors name the question and say what to change, because the alternative is an index error
    three frames deep inside the forward pass.
    """
    if not isinstance(spec, dict):
        raise ValueError("question %r must be an object, got %s" % (qid, type(spec).__name__))
    kind = spec.get("type")
    if kind not in QTYPES:
        raise ValueError("question %r: type must be one of choice/score/noul, got %r" % (qid, kind))
    ins = spec.get("instructions")
    if ins is None:
        raise ValueError("question %r: missing 'instructions'" % (qid,))
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    crit = spec.get("criteria")
    if "labels" in spec and kind != "noul":
        raise ValueError("question %r: 'labels' only applies to noul questions" % (qid,))

    if kind == "choice":
        if isinstance(crit, list):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not crit:
            raise ValueError("question %r: choice needs 'criteria' as a non-empty {label: description} "
                             "object or a list of labels" % (qid,))
        keys = tuple(str(k) for k in crit)
        opts = tuple(k if v is None or v == "" else "%s: %s" % (k, _crit_text(v)) for k, v in zip(keys, crit.values()))
    elif kind == "score":
        if not isinstance(crit, list) or not crit:
            raise ValueError("question %r: score needs 'criteria' as a non-empty list of levels, lowest first"
                             % (qid,))
        keys = tuple(str(i) for i in range(len(crit)))
        opts = tuple("level %d: %s" % (i, _crit_text(c)) for i, c in enumerate(crit))
    else:
        if crit is not None and not isinstance(crit, dict):
            raise ValueError("question %r: noul 'criteria' is an optional {\"true\": ..., \"false\": ...} object"
                             % (qid,))
        crit = {str(k).lower(): v for k, v in (crit or {}).items()}
        extra = set(crit) - {"true", "false"}
        if extra:
            raise ValueError("question %r: noul 'criteria' may only use the keys true/false, got %s; to change "
                             "the wording shown to the model use 'labels'" % (qid, sorted(extra)))
        lf, lt = _noul_labels(qid, spec.get("labels"))
        f, t = crit.get("false"), crit.get("true")
        keys = ("false", "true")
        opts = ("%s: %s" % (lf, NOUL_FALSE if f in (None, "") else _crit_text(f)),
                "%s: %s" % (lt, NOUL_TRUE if t in (None, "") else _crit_text(t)))
    return Question(str(qid), kind, ins, keys, opts)


def _noul_labels(qid, labels) -> Tuple[str, str]:
    if labels is None:
        return "false", "true"
    ok = isinstance(labels, dict) and set(labels) == {"false", "true"} \
        and all(isinstance(v, str) and v.strip() for v in labels.values())
    if not ok or labels["false"].strip() == labels["true"].strip():
        raise ValueError("question %r: 'labels' must map exactly false and true to two different "
                         "non-empty strings" % (qid,))
    return labels["false"].strip(), labels["true"].strip()


class Tokens:
    """Thin, thread-safe wrapper over a HF fast tokenizer with a question-part cache."""

    def __init__(self, hf_tokenizer, cache_size: int = 4096):
        self.hf = hf_tokenizer
        self.fast = hf_tokenizer.backend_tokenizer
        self.cls, self.sep = hf_tokenizer.cls_token_id, hf_tokenizer.sep_token_id
        self.mask, self.pad = hf_tokenizer.mask_token_id, hf_tokenizer.pad_token_id
        self.mask_text = hf_tokenizer.mask_token
        self._cache: "OrderedDict[Tuple, Tuple[List[int], List[List[int]]]]" = OrderedDict()
        self._size = cache_size
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> List[List[int]]:
        clean = [t.replace(self.mask_text, " ") for t in texts]
        return [e.ids for e in self.fast.encode_batch(clean, add_special_tokens=False)]

    def question(self, q: Question) -> Tuple[List[int], List[List[int]]]:
        """(header ids, per-option ids) for a question; options are capped at 48 tokens each."""
        key = q.cache_key
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        enc = self.encode(["%s question: %s" % (q.kind, q.instructions)] + [" " + o for o in q.options])
        val = (enc[0], [o[:48] for o in enc[1:]])
        with self._lock:
            self._cache[key] = val
            if len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return val


def _fit_options(opts: List[List[int]], budget: int) -> List[List[int]]:
    """Laya's rule: if the options leave < 16 tokens of the head budget, trim each evenly."""
    total = sum(len(o) + 1 for o in opts)
    if budget - total < 16:
        per = max(4, (budget - 16) // max(1, len(opts)))
        opts = [o[: per - 1] for o in opts]
    return opts


def classic_row(tk: Tokens, state_ids: List[int], q: Question, max_len: int, head_max_len: int,
                truncate_left: bool) -> Tuple[List[int], List[int]]:
    """One Laya-format row. Mirrors laya.common.build_sequence token for token."""
    head, opts = tk.question(q)
    opts = [[tk.mask] + o for o in opts]
    left = head_max_len - sum(len(o) for o in opts)
    if left < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opts)))
        opts = [o[:per] for o in opts]
        left = head_max_len - sum(len(o) for o in opts)
    ids = [tk.cls] + head[: max(8, left)] + [tk.sep]
    markers = []
    for o in opts:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tk.sep)
    room = max(0, max_len - len(ids) - 1)
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = (ids + st + [tk.sep])[:max_len]
    return ids, [m for m in markers if m < max_len]


@dataclass
class PackedRow:
    ids: List[int]
    pos: List[int]
    seg: List[int]
    qtype: List[int]
    markers: List[List[int]]       # per question, row-local marker indices


def state_type_code(qs: Sequence[Question]) -> int:
    """How many questions of each type a row holds, packed into one int for the state tokens.

    In the shared layout the state is read together with every question, so it gets the average
    of their type embeddings (exactly the question's own when there is one, as in Laya).
    """
    n = [0, 0, 0]
    for q in qs:
        n[q.qtype] += 1
    return 3 + min(n[0], 63) + 64 * min(n[1], 63) + 4096 * min(n[2], 63)


def _segment(tk: Tokens, q: Question, max_question: int):
    head, opts = tk.question(q)
    opts = _fit_options(opts, max_question - 2)
    room = max_question - 2 - sum(len(o) + 1 for o in opts)
    part = head[: max(8, room)] + [tk.sep]
    mk = []
    for o in opts:
        mk.append(len(part))
        part += [tk.mask] + o
    part.append(tk.sep)
    return part, mk


def packed_row(tk: Tokens, state_ids: List[int], qs: Sequence[Question], max_state: int,
               max_question: int, truncate_left: bool, shared: bool = False) -> PackedRow:
    """The state once plus one segment per question (see module docstring).

    shared=False  ``[CLS] state [SEP]`` first, segments after it at positions that continue from
                  the state; the state is encoded without looking at any question.
    shared=True   ``[CLS]`` + segments, each at positions 1..len (they overlap: every question
                  believes it sits right after [CLS]), then ``state [SEP]`` after the longest
                  segment. The state reads all questions. With one question the row is token
                  for token, position for position, what a Laya checkpoint was trained on.
    """
    st = state_ids[max(0, len(state_ids) - max_state):] if truncate_left else state_ids[:max_state]
    parts = [_segment(tk, q, max_question) for q in qs]
    markers: List[List[int]] = []
    if shared:
        ids, pos, seg = [tk.cls], [0], [0]
        code = state_type_code(qs)
        qtype = [code]
        for si, (q, (part, mk)) in enumerate(zip(qs, parts), 1):
            markers.append([len(ids) + m for m in mk])
            ids += part
            pos += range(1, 1 + len(part))
            seg += [si] * len(part)
            qtype += [q.qtype] * len(part)
        start = 1 + max(len(p) for p, _ in parts)
        tail = st + [tk.sep]
        ids += tail
        pos += range(start, start + len(tail))
        seg += [0] * len(tail)
        qtype += [code] * len(tail)
        return PackedRow(ids, pos, seg, qtype, markers)
    ids = [tk.cls] + st + [tk.sep]
    n0 = len(ids)
    pos = list(range(n0))
    seg = [0] * n0
    qtype = [0] * n0
    for si, (q, (part, mk)) in enumerate(zip(qs, parts), 1):
        markers.append([len(ids) + m for m in mk])
        ids += part
        pos += range(n0, n0 + len(part))
        seg += [si] * len(part)
        qtype += [q.qtype] * len(part)
    return PackedRow(ids, pos, seg, qtype, markers)
