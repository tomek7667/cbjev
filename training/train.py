"""Fine-tune a decision checkpoint into cbjev's packed layout.

    python training/train.py --data .work/data --init laya --out ~/.cache/cbjev/cbjev

Starts from a Laya checkpoint (its encoder, head and scorer are exactly the modules cbjev runs)
and trains on packed rows: one state, then every question of the case as its own segment.
The loss per question is cross-entropy against a soft target,

    target = (1 - a) * gold + a * teacher        if the question has a gold label
    target = teacher                             otherwise (auxiliary questions)

with `a` = --teacher-mix. Option order is reshuffled every epoch for choice questions so the
model cannot lean on position.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cbjev.agent import _tokens_for, resolve_checkpoint  # noqa: E402
from cbjev.checkpoints import resolve  # noqa: E402
from cbjev.layout import as_text, packed_row, parse_question  # noqa: E402
from cbjev.model import DecisionNet, EncoderSpec, load_weights  # noqa: E402

SRC_WEIGHT = {"typed_decisions": 1.5}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def permute_choice(spec, dists, rng):
    """Shuffle a choice question's options; permute its target vectors the same way."""
    if spec["type"] != "choice":
        return spec, dists
    crit = spec["criteria"]
    items = list(crit.items()) if isinstance(crit, dict) else [(c, None) for c in crit]
    order = list(range(len(items)))
    rng.shuffle(order)
    spec = dict(spec, criteria={items[i][0]: items[i][1] for i in order})
    return spec, [None if d is None else [d[i] for i in order] for d in dists]


class Rows:
    """Cases -> packed rows with per-question targets, rebuilt each epoch (fresh option order)."""

    def __init__(self, cases, tokens, max_state, max_question, teacher_mix, seed, shared=False):
        self.cases, self.tk, self.shared = cases, tokens, shared
        self.max_state, self.max_question, self.mix = max_state, max_question, teacher_mix
        self.seed = seed
        texts = [as_text(c["state"]) for c in cases]
        self.state_ids = []
        for i in range(0, len(texts), 4096):
            self.state_ids += tokens.encode(texts[i:i + 4096])

    def build(self, epoch):
        rng = random.Random("%d-%d" % (self.seed, epoch))
        rows = []
        for c, sid in zip(self.cases, self.state_ids):
            qs, tg, wt = [], [], []
            items = list(c["questions"].items())
            rng.shuffle(items)
            for qid, spec in items:
                gold, teach = c["gold"].get(qid), c["teacher"].get(qid)
                spec, (gold, teach) = permute_choice(spec, [gold, teach], rng)
                q = parse_question(qid, spec)
                if gold is not None:
                    t = [(1 - self.mix) * g + self.mix * h for g, h in zip(gold, teach)] if teach else gold
                    w = SRC_WEIGHT.get(c["src"], 1.0)
                else:
                    t, w = teach, 0.5
                qs.append(q)
                tg.append(t)
                wt.append(w)
            r = packed_row(self.tk, sid, qs, self.max_state, self.max_question, isinstance(c["state"], list),
                           self.shared)
            keep = [i for i, mk in enumerate(r.markers) if len(mk) == len(tg[i])]
            if not keep:
                continue
            rows.append((r, tg, [wt[i] if i in keep else 0.0 for i in range(len(wt))], c["src"]))
        return rows


def batches(rows, budget, rng):
    """Length-bucketed batches of at most `budget` padded tokens."""
    idx = sorted(range(len(rows)), key=lambda i: len(rows[i][0].ids) + rng.random() * 24)
    out, cur, Lmax = [], [], 0
    for i in idx:
        L = len(rows[i][0].ids)
        if cur and (max(Lmax, L) * (len(cur) + 1) > budget or len(cur) >= 64):
            out.append(cur)
            cur, Lmax = [], 0
        cur.append(i)
        Lmax = max(Lmax, L)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(rows, sel, pad, device):
    B = len(sel)
    L = max(len(rows[i][0].ids) for i in sel)
    M = max(sum(len(m) for m in rows[i][0].markers) for i in sel)
    ids = np.full((B, L), pad, np.int64)
    pos = np.zeros((B, L), np.int64)
    seg = np.full((B, L), -1, np.int64)
    qt = np.zeros((B, L), np.int64)
    valid = np.zeros((B, L), bool)
    mk = np.full((B, M), -1, np.int64)
    qrow, qstart, qk, tgt, wts = [], [], [], [], []
    for b, i in enumerate(sel):
        r, tg, wt = rows[i][:3]
        n = len(r.ids)
        ids[b, :n], pos[b, :n], seg[b, :n], qt[b, :n] = r.ids, r.pos, r.seg, r.qtype
        valid[b, :n] = True
        off = 0
        for m, t, w in zip(r.markers, tg, wt):
            mk[b, off:off + len(m)] = m
            if w > 0:
                qrow.append(b)
                qstart.append(off)
                qk.append(len(m))
                tgt.append(t)
                wts.append(w)
            off += len(m)
    K = max(qk) if qk else 1
    T = np.zeros((len(qk), K), np.float32)
    for j, t in enumerate(tgt):
        T[j, :len(t)] = t
    dev = lambda a: torch.from_numpy(a).to(device, non_blocking=True)  # noqa: E731
    return (dev(ids), dev(pos), dev(seg), dev(qt), dev(valid), dev(mk)), (
        dev(np.array(qrow)), dev(np.array(qstart)), dev(np.array(qk)), dev(T), dev(np.array(wts, np.float32)))


def question_logprobs(z, qrow, qstart, qk):
    K = int(qk.max())
    ar = torch.arange(K, device=z.device)
    col = (qstart[:, None] + ar[None, :]).clamp(max=z.size(1) - 1)
    zz = z[qrow[:, None], col]
    zz = zz.masked_fill(ar[None, :] >= qk[:, None], -1e4)
    return F.log_softmax(zz.float(), -1), ar[None, :] < qk[:, None]


def evaluate(net, rows, pad, device, budget):
    net.eval()
    hit = tot = 0
    by = {}
    nll = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for sel in batches(rows, budget, random.Random(0)):
            x, (qrow, qstart, qk, T, W) = collate(rows, sel, pad, device)
            if len(qk) == 0:
                continue
            lp, m = question_logprobs(net(*x, packed=True), qrow, qstart, qk)
            pred = lp.argmax(-1)
            gold = T.argmax(-1)
            ok = (pred == gold).float()
            hit += ok.sum().item()
            tot += len(ok)
            nll += -(T * lp.masked_fill(~m, 0)).sum(-1).sum().item()
    net.train()
    return {"acc": hit / max(1, tot), "nll": nll / max(1, tot), "n": tot}


def save(net, cfg, init_path, out):
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in net.state_dict().items()
          if not k.startswith("encoder.rope_")}
    save_file(sd, os.path.join(out, "model.safetensors"))
    with open(os.path.join(out, "cbjev_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for d in ("encoder", "tokenizer"):
        dst = os.path.join(out, d)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(init_path, d), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, ".work", "data"))
    ap.add_argument("--init", default="laya")
    ap.add_argument("--init-sub", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/cbjev/cbjev"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--budget", type=int, default=12288, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--teacher-mix", type=float, default=0.2)
    ap.add_argument("--max-state", type=int, default=1024)
    ap.add_argument("--max-question", type=int, default=768)
    ap.add_argument("--checkpointing", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--name", default="cbjev")
    ap.add_argument("--layout", default="shared", choices=["shared", "packed"])
    ap.add_argument("--vram-frac", type=float, default=0.5, help="cap on this process's share of GPU memory")
    ap.add_argument("--repeat", default="typed_decisions=8",
                    help="oversample sources, e.g. 'typed_decisions=8,go_emotions=2'")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    if a.vram_frac < 1:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    init_path = resolve_checkpoint(resolve(a.init), a.init_sub)
    cfg_name = "cbjev_config.json" if os.path.exists(os.path.join(init_path, "cbjev_config.json")) else "rl_agent_config.json"
    with open(os.path.join(init_path, cfg_name)) as f:
        base_cfg = json.load(f)
    with open(os.path.join(init_path, "encoder", "config.json")) as f:
        spec = EncoderSpec.from_hf(json.load(f))
    tokens = _tokens_for(os.path.join(init_path, "tokenizer"))
    net = DecisionNet(spec, head_layers=int(base_cfg.get("head_layers", 2)))
    from safetensors.torch import load_file
    load_weights(net, load_file(os.path.join(init_path, "model.safetensors")))
    net.float().to(device).train()
    net.encoder.checkpointing = bool(a.checkpointing)

    train_cases = read_jsonl(os.path.join(a.data, "train.jsonl"))
    rep = {k: int(v) for k, v in (x.split("=") for x in a.repeat.split(",") if x)}
    train_cases = [c for c in train_cases for _ in range(rep.get(c["src"], 1))]
    dev_cases = read_jsonl(os.path.join(a.data, "dev.jsonl"))
    shared = a.layout == "shared"
    net.shared = shared
    tr = Rows(train_cases, tokens, a.max_state, a.max_question, a.teacher_mix, a.seed, shared)
    dv = Rows(dev_cases, tokens, a.max_state, a.max_question, 0.0, a.seed + 99, shared)
    dev_rows = dv.build(0)
    dev_td = [r for r in dev_rows if r[3] == "typed_decisions"]
    print("train cases %d, dev rows %d (typed-decisions %d)" % (len(train_cases), len(dev_rows), len(dev_td)),
          flush=True)

    enc_params = [p for n, p in net.named_parameters() if n.startswith("encoder.")]
    other = [p for n, p in net.named_parameters() if not n.startswith("encoder.")]
    decay = lambda ps: [p for p in ps if p.ndim >= 2]  # noqa: E731
    nodecay = lambda ps: [p for p in ps if p.ndim < 2]  # noqa: E731
    opt = torch.optim.AdamW([
        {"params": decay(enc_params), "lr": a.lr, "weight_decay": 0.01},
        {"params": nodecay(enc_params), "lr": a.lr, "weight_decay": 0.0},
        {"params": decay(other), "lr": a.head_lr, "weight_decay": 0.01},
        {"params": nodecay(other), "lr": a.head_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.98), eps=1e-6, fused=True)
    base_lrs = [g["lr"] for g in opt.param_groups]

    rng = random.Random(a.seed)
    ep_rows = tr.build(0)
    steps_per_epoch = math.ceil(len(batches(ep_rows, a.budget, random.Random(0))) / a.accum)
    total = int(steps_per_epoch * a.epochs)
    warm = min(400, total // 15)
    print("steps/epoch %d, total optimizer steps %d" % (steps_per_epoch, total), flush=True)

    cfg = dict(base_cfg)
    for k in ("temperature", "temperature_by_options", "training", "fine_tuned", "gradient_checkpointing",
              "max_tokens_per_batch", "act_costs", "cost_wrong_act", "max_prefixes"):
        cfg.pop(k, None)
    cfg.update({"layout": a.layout, "model_name": a.name, "max_state_tokens": a.max_state,
                "max_question_tokens": a.max_question, "temperature": [1.0, 1.0, 1.0],
                "trained_from": a.init + ("/" + a.init_sub if a.init_sub else "")})

    step, micro, t0, epoch, done = 0, 0, time.time(), 0, False
    run_loss, run_n = 0.0, 0
    best = None
    while not done:
        if epoch > 0:
            ep_rows = tr.build(epoch)
        for sel in batches(ep_rows, a.budget, rng):
            x, (qrow, qstart, qk, T, W) = collate(ep_rows, sel, tokens.pad, device)
            if len(qk) == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = net(*x, packed=True)
            lp, m = question_logprobs(z, qrow, qstart, qk)
            loss_q = -(T * lp.masked_fill(~m, 0)).sum(-1)
            loss = (loss_q * W).sum() / W.sum().clamp(min=1e-6)
            (loss / a.accum).backward()
            run_loss += loss.item()
            run_n += 1
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            f = step / max(1, warm) if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print("step %5d/%d  loss %.4f  %.1f min elapsed, eta %.1f min" % (
                    step, total, run_loss / run_n, el / 60, el / step * (total - step) / 60), flush=True)
                run_loss, run_n = 0.0, 0
            if step % a.eval_every == 0 or step == total:
                ev, evt = evaluate(net, dev_rows, tokens.pad, device, a.budget), evaluate(net, dev_td, tokens.pad, device, a.budget)
                print("  dev acc %.4f nll %.4f | typed-decisions dev acc %.4f nll %.4f" % (
                    ev["acc"], ev["nll"], evt["acc"], evt["nll"]), flush=True)
                save(net, cfg, init_path, a.out)
            if step >= total:
                done = True
                break
        epoch += 1
    save(net, cfg, init_path, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
