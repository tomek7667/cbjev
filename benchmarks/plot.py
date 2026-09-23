"""Render assets/cbjev_vs_laya_vs_jev.png from the benchmark result files.

    python benchmarks/plot.py --acc benchmarks/results/accuracy.json --speed benchmarks/results/speed.json

cbjev and Laya numbers are measured here on the same machine and the same cases. Jev numbers
are third-party published figures (see accuracy.JEV_PUBLISHED) and are drawn hatched.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

C_CB, C_LA, C_JEV, C_LF = "#2563eb", "#f59e0b", "#9ca3af", "#b45309"
LABELS = {
    "ag_news": "AG News", "email_spam": "E-mail spam", "phishing": "Phishing", "rag_relevance": "RAG relevance",
    "support_triage": "Support triage", "typed_decisions": "typed-decisions", "emotion": "DAIR emotion*",
    "banking77": "Banking77 (77 labels)*", "jailbreak": "Jailbreak*", "toxicity": "Toxicity*",
    "prompt_injection": "Prompt injection*", "model_routing": "Model routing*", "sst5": "SST-5*",
    "massive_en": "MASSIVE intent*", "xnli_en": "XNLI*",
}
JEV_ACC = {"ag_news": 0.910, "emotion": 0.480, "banking77": 0.870, "typed_decisions": 0.727}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="benchmarks/results/accuracy.json")
    ap.add_argument("--speed", default="benchmarks/results/speed.json")
    ap.add_argument("--cbjev", default="cbjev")
    ap.add_argument("--laya", default="laya,laya-td")
    ap.add_argument("--out", default="assets/cbjev_vs_laya_vs_jev.png")
    a = ap.parse_args()
    acc = json.load(open(a.acc))["results"]
    cb = acc[a.cbjev]
    lays = [acc[x] for x in a.laya.split(",") if x in acc]
    suites = [s for s in LABELS if s in cb]
    best = {s: max(l[s]["accuracy"] for l in lays if s in l) for s in suites}
    best_ece = {s: min(l[s]["ece"] for l in lays if s in l) for s in suites}

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16, 10.5), dpi=130)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.42, wspace=0.28)

    # accuracy per suite
    ax = fig.add_subplot(gs[0, :])
    x = range(len(suites))
    w = 0.27
    ax.bar([i - w for i in x], [cb[s]["accuracy"] for s in suites], w, color=C_CB, label="cbjev")
    ax.bar(list(x), [best[s] for s in suites], w, color=C_LA, label="Laya (best of laya / laya-typed-decisions)")
    for i, s in enumerate(suites):
        if s in JEV_ACC:
            ax.bar(i + w, JEV_ACC[s], w, color="white", edgecolor=C_JEV, hatch="///", lw=1.2)
    for i, s in enumerate(suites):
        d = cb[s]["accuracy"] - best[s]
        ax.text(i - w, cb[s]["accuracy"] + 0.012, "%+.1f" % (100 * d), ha="center", fontsize=8,
                color="#166534" if d >= 0 else "#991b1b", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[s] for s in suites], rotation=28, ha="right")
    ax.set_ylim(0.3, 1.06)
    ax.set_ylabel("accuracy")
    wins = sum(cb[s]["accuracy"] >= best[s] for s in suites)
    ax.set_title("Accuracy on %d suites: cbjev >= Laya's best checkpoint on %d/%d   (* = dataset held out of training)"
                 % (len(suites), wins, len(suites)), loc="left", fontweight="bold")
    handles = ax.get_legend_handles_labels()[0] + [Patch(facecolor="white", edgecolor=C_JEV, hatch="///",
                                                         label="TypeSafe Jev (published, not measured here)")]
    ax.legend(handles=handles, loc="lower left", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # latency
    if os.path.exists(a.speed):
        sp = json.load(open(a.speed))
        for col, case, title in ((0, "short ticket", "Latency, short ticket"), (1, "~500-token doc",
                                                                                 "Latency, ~500-token document")):
            ax = fig.add_subplot(gs[1, col])
            rows = [r for r in sp["rows"] if r["case"] == case]
            n = [r["questions"] for r in rows]
            for key, color, lab, ls in (("cbjev", C_CB, "cbjev", "-"), ("laya-fast", C_LF, "Laya fast path (TileLang)", "--"),
                                        ("laya", C_LA, "Laya", "-")):
                if rows and key in rows[0]:
                    ax.plot(n, [r[key] for r in rows], ls, marker="o", color=color, label=lab, lw=2)
            ax.axhspan(236, 276, color=C_JEV, alpha=0.25, label="Jev hosted API p50 (published)")
            ax.set_yscale("log")
            ax.set_xlabel("questions per call")
            ax.set_ylabel("ms per call (median, %s)" % sp["meta"]["gpu"].replace("NVIDIA GeForce ", ""))
            ax.set_title(title, loc="left", fontweight="bold")
            ax.grid(alpha=0.25, which="both")
            if col == 0:
                ax.legend(frameon=False, fontsize=8)

    # calibration
    ax = fig.add_subplot(gs[1, 2])
    mean_cb = sum(cb[s]["ece"] for s in suites) / len(suites)
    mean_la = sum(best_ece[s] for s in suites) / len(suites)
    bars = ax.bar(["cbjev", "Laya\n(best per suite)", "Jev\n(published)"], [mean_cb, mean_la, 0.246],
                  color=[C_CB, C_LA, "white"], edgecolor=["none", "none", C_JEV], hatch=["", "", "///"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, "%.3f" % b.get_height(), ha="center")
    ax.set_title("Calibration: mean ECE (lower is better)", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("cbjev vs Laya vs TypeSafe Jev", fontsize=17, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.945, "cbjev and Laya measured on the same GPU and the same cases; Jev figures are third-party "
             "published results (different samples).", fontsize=9, color="#4b5563")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
