"""Option-order robustness: how often does reversing a choice question's options change the answer?

    python benchmarks/robustness.py --engines laya,laya-td,cbjev --out benchmarks/results/robustness.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from accuracy import load_suite  # noqa: E402
from common import load_agent  # noqa: E402

SUITES = ["ag_news", "emotion", "support_triage", "model_routing", "massive_en", "xnli_en"]


def flip_rate(agent, cases):
    flips = n = 0
    for state, qs, gold in cases:
        rev = {}
        for qid, q in qs.items():
            if q["type"] == "choice" and isinstance(q["criteria"], dict):
                rev[qid] = dict(q, criteria=dict(reversed(list(q["criteria"].items()))))
        if not rev:
            continue
        a = agent.predict(state, {k: qs[k] for k in rev})["answers"]
        b = agent.predict(state, rev)["answers"]
        for k in rev:
            flips += a[k]["choice"] != b[k]["choice"]
            n += 1
    return flips / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="laya,laya-td,cbjev")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    engines = a.engines.split(",")
    rows = {s: {} for s in SUITES}
    for e in engines:
        ag = load_agent(e)
        for s in SUITES:
            rows[s][e] = round(flip_rate(ag, load_suite(s, a.n)), 4)
            print(e, s, rows[s][e], flush=True)
        del ag
    rows["mean"] = {e: round(sum(rows[s][e] for s in SUITES) / len(SUITES), 4) for e in engines}
    if a.out:
        json.dump({"engines": engines, "rows": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
