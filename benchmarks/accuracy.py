"""Accuracy and calibration of every engine on every suite, through the public `predict` API.

    python benchmarks/accuracy.py --engines laya,cbjev --n 400 --out benchmarks/results/accuracy.json

Answers are read from what `predict` returns (shipped temperatures included), so this measures
what a caller actually gets. Suites are built once and cached under `.work/suites/`.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from common import load_agent  # noqa: E402
from suites import DEFAULT, MULTILINGUAL, SUITES  # noqa: E402

CACHE = os.path.join(os.path.dirname(HERE), ".work", "suites")

JEV_PUBLISHED = {   # third-party numbers, never measured here (no TypeSafe API access)
    "ag_news": {"accuracy": 0.910, "source": "AbdelStark/jev-benchmarks"},
    "emotion": {"accuracy": 0.480, "source": "AbdelStark/jev-benchmarks"},
    "banking77": {"accuracy": 0.870, "note": "72 labels", "source": "AbdelStark/jev-benchmarks"},
    "typed_decisions": {"accuracy": 0.727, "soft_accuracy": 0.580, "ece": 0.144, "score_mae": 0.391,
                        "source": "LocalLLaMA/typed-decisions leaderboard via Laya README"},
    "latency_p50_ms": "236-276 (hosted API; AbdelStark, nibzard)",
    "ece": {"value": 0.246, "source": "nibzard/decision-model-benchmark"},
    "option_order_flip_rate": {"value": 0.13, "source": "nibzard/decision-model-benchmark"},
}


def load_suite(name, n):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "%s_%d.json" % (name.replace("/", "_"), n))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    cases = SUITES[name]["build"](n)
    with open(path, "w") as f:
        json.dump(cases, f, ensure_ascii=False)
    return json.loads(json.dumps(cases))


def probs_of(ans, q):
    if q["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    if q["type"] == "score":
        return [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]
    keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else [str(c) for c in q["criteria"]]
    return [ans["probabilities"][k] for k in keys]


def ece(conf, corr, bins=15):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def macro_f1(g, p):
    f = []
    for c in sorted(set(g) | set(p)):
        tp = sum(1 for a, b in zip(g, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(g, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(g, p) if a == c and b != c)
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def summarise(rows):
    """rows: (gold index, prob vector, extra) -> metric dict."""
    g = [r[0] for r in rows]
    P = [np.asarray(r[1], float) / max(1e-12, sum(r[1])) for r in rows]
    pred = [int(np.argmax(p)) for p in P]
    conf = [float(p.max()) for p in P]
    corr = [float(a == b) for a, b in zip(g, pred)]
    m = {"n": len(rows), "accuracy": round(float(np.mean(corr)), 4), "macro_f1": round(macro_f1(g, pred), 4),
         "ece": round(ece(conf, corr), 4),
         "brier": round(float(np.mean([((p - np.eye(len(p))[gi]) ** 2).sum() for p, gi in zip(P, g)])), 4),
         "nll": round(float(np.mean([-math.log(max(p[gi], 1e-12)) for p, gi in zip(P, g)])), 4),
         "mean_confidence": round(float(np.mean(conf)), 4)}
    order = np.argsort(-np.asarray(conf))
    m["acc_at_50_coverage"] = round(float(np.asarray(corr)[order[:max(1, len(conf) // 2)]].mean()), 4)
    dists = [r[2].get("dist") for r in rows if isinstance(r[2], dict) and r[2].get("dist")]
    if dists:
        m["soft_accuracy"] = round(float(np.mean([float(np.dot(p, d)) for p, (_, _, x) in zip(P, rows)
                                                 for d in [x["dist"]]])), 4)
        sc = [(p, x) for p, (_, _, x) in zip(P, rows) if x.get("kind") == "score"]
        if sc:
            m["score_mae"] = round(float(np.mean([abs(float(np.dot(p, np.arange(len(p))))
                                                      - float(np.dot(x["dist"], np.arange(len(p)))))
                                                  for p, x in sc])), 4)
    return m


def run_suite(agent, cases, bs=16):
    rows = []
    t0 = time.perf_counter()
    for i in range(0, len(cases), 1):
        state, qs, gold = cases[i]
        ans = agent.predict(state, qs)["answers"]
        for qid, gi in gold.items():
            extra = gi if isinstance(gi, dict) else {}
            idx = gi["index"] if isinstance(gi, dict) else gi
            rows.append((idx, probs_of(ans[qid], qs[qid]), extra))
    secs = time.perf_counter() - t0
    m = summarise(rows)
    m["ms_per_case"] = round(1000 * secs / max(1, len(cases)), 2)
    if rows and rows[0][2].get("workflow"):
        by = {}
        for r in rows:
            by.setdefault(r[2]["workflow"], []).append(r)
        m["by_workflow"] = {k: summarise(v)["accuracy"] for k, v in sorted(by.items())}
        byk = {}
        for r in rows:
            byk.setdefault(r[2]["kind"], []).append(r)
        m["by_type"] = {k: summarise(v)["accuracy"] for k, v in sorted(byk.items())}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="laya,cbjev")
    ap.add_argument("--suites", default=",".join(DEFAULT))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    names = MULTILINGUAL if a.suites == "multilingual" else [s for s in a.suites.split(",") if s]
    out = {"meta": {"n": a.n, "time": time.strftime("%Y-%m-%d %H:%M"), "suites": {
        s: {"relation": SUITES[s]["relation"], "note": SUITES[s]["note"]} for s in names}},
        "jev_published": JEV_PUBLISHED, "results": {}}
    if a.out and os.path.exists(a.out):
        with open(a.out) as f:
            prev = json.load(f)
        out["results"] = prev.get("results", {})
    data = {s: load_suite(s, a.n) for s in names}
    for e in a.engines.split(","):
        ag = load_agent(e)
        res = out["results"].setdefault(e, {})
        for s in names:
            res[s] = run_suite(ag, data[s])
            r = res[s]
            print("%-14s %-17s acc %.3f  f1 %.3f  ece %.3f  brier %.3f  %6.1f ms/case%s" % (
                e, s, r["accuracy"], r["macro_f1"], r["ece"], r["brier"], r["ms_per_case"],
                ("  soft %.3f" % r["soft_accuracy"]) if "soft_accuracy" in r else ""), flush=True)
            if a.out:
                with open(a.out, "w") as f:
                    json.dump(out, f, indent=2)
        del ag
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
