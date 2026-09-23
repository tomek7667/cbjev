"""Latency of one `predict` call, cbjev vs Laya, measured round-robin in one process.

Engines take turns call by call, so GPU clock changes, thermal drift and anything else that
shares the card hit every engine alike. Reported numbers are medians over `--reps` calls
after warm-up, end to end (validation, tokenization, forward, decoding).

    python benchmarks/speed.py --engines laya,laya-fast,cbjev --out benchmarks/results/speed.json
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import ENGINES, make_engine  # noqa: E402

TICKET = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
          "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will "
                  "cancel our plan."}
BASE = [
    {"type": "choice", "instructions": "Which department should handle this request?",
     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                  "sales": "pricing, new contracts", "other": "everything else"}},
    {"type": "score", "instructions": "How urgent is this request?",
     "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
    {"type": "choice", "instructions": "Which language is the message written in?",
     "criteria": ["english", "german", "french", "other"]},
]


def questions(n):
    return {"q%d" % i: dict(BASE[i % len(BASE)], instructions=BASE[i % len(BASE)]["instructions"] + " " * (i // 5))
            for i in range(n)}


def long_state(tokens):
    words = ("We were billed twice for March and the support line keeps dropping the call. "
             "Our finance team needs the duplicate refunded before the quarter closes. ").split()
    return {"subject": "Billing problem", "body": " ".join(words[i % len(words)] for i in range(int(tokens * 0.8)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="laya,laya-fast,cbjev")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    names = a.engines.split(",")
    engines = {n: make_engine(n) for n in names}
    cases = [("short ticket", TICKET, n) for n in (1, 3, 5, 10, 30, 50)]
    cases += [("~500-token doc", long_state(500), n) for n in (1, 5, 10, 30)]
    res = []
    for label, state, n in cases:
        qs = questions(n)
        for e in engines.values():
            for _ in range(5):
                e(state, qs)
        times = {k: [] for k in engines}
        for _ in range(a.reps):
            for k, e in engines.items():
                t = time.perf_counter()
                e(state, qs)
                times[k].append((time.perf_counter() - t) * 1000)
        row = {"case": label, "questions": n}
        row.update({k: round(statistics.median(v), 2) for k, v in times.items()})
        res.append(row)
        print("%-16s %3d q  " % (label, n) + "  ".join("%s %7.2f ms" % (k, row[k]) for k in engines), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        import torch
        meta = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__, "reps": a.reps, "engines": names}
        with open(a.out, "w") as f:
            json.dump({"meta": meta, "rows": res}, f, indent=2)


if __name__ == "__main__":
    main()
