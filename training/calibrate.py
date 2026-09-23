"""Fit softmax temperatures on the dev split and write them into the checkpoint config.

    python training/calibrate.py --ckpt ~/.cache/cbjev/cbjev --data .work/data

One temperature per question type, plus one per (type, option-count bucket) where the dev split
has enough questions. Fitting minimises the negative log-likelihood of the gold option (soft gold
for typed-decisions), which is a proper scoring rule, so it cannot "improve" calibration by
simply hedging. Benchmark data is never touched here: dev.jsonl comes from the training sources'
held-back rows only.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cbjev  # noqa: E402
from cbjev.agent import _bucket_name  # noqa: E402
from cbjev.layout import parse_question  # noqa: E402


def fit(rows, lo=0.3, hi=5.0):
    """Temperature minimising NLL of targets under softmax(z / T); rows = [(z, target)]."""
    K = max(len(z) for z, _ in rows)
    Z = torch.full((len(rows), K), -1e4, dtype=torch.float64)
    T = torch.zeros((len(rows), K), dtype=torch.float64)
    for i, (z, t) in enumerate(rows):
        Z[i, :len(z)] = torch.as_tensor(z, dtype=torch.float64)
        T[i, :len(t)] = torch.as_tensor(t, dtype=torch.float64)
    grid = torch.exp(torch.linspace(np.log(lo), np.log(hi), 400, dtype=torch.float64))
    nll = torch.stack([-(T * torch.log_softmax(Z / g, -1)).sum(-1).mean() for g in grid])
    return float(grid[int(nll.argmin())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, ".work", "data"))
    ap.add_argument("--min-bucket", type=int, default=150)
    a = ap.parse_args()
    ag = cbjev.load(a.ckpt, device="cuda", graphs=False)
    cases = [json.loads(l) for l in open(os.path.join(a.data, "dev.jsonl"))]
    by_type, by_bucket = {}, {}
    for c in cases:
        qids = [q for q, g in c["gold"].items() if g is not None]
        if not qids:
            continue
        qs = [parse_question(q, c["questions"][q]) for q in qids]
        zs = ag.logits([c["state"]], qs)[0]
        for q, z in zip(qs, zs):
            if len(z) != len(q.options):
                continue
            row = (np.asarray(z, float), c["gold"][q.qid])
            by_type.setdefault(q.kind, []).append(row)
            by_bucket.setdefault(_bucket_name(q.kind, len(z)), []).append(row)
    types = ["choice", "score", "noul"]
    temps = [round(fit(by_type[t]), 4) if len(by_type.get(t, [])) >= 30 else 1.0 for t in types]
    buckets = {b: round(fit(r), 4) for b, r in sorted(by_bucket.items()) if len(r) >= a.min_bucket}
    print("per type:", dict(zip(types, temps)), {t: len(by_type.get(t, [])) for t in types})
    print("per bucket:", buckets, {b: len(r) for b, r in by_bucket.items()})
    path = os.path.join(a.ckpt, "cbjev_config.json")
    cfg = json.load(open(path))
    cfg["temperature"] = temps
    cfg["temperature_by_options"] = buckets
    cfg["calibration"] = {"data": "dev split of training sources", "objective": "NLL",
                          "n_by_type": {t: len(by_type.get(t, [])) for t in types}}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
