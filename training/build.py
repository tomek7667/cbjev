"""Build the training set: sources -> cases -> teacher answers -> jsonl.

    python training/build.py --out .work/data --mix english

Teacher: the two English Laya checkpoints (root and typed-decisions) run in their own classic
layout through cbjev's engine; their logits are averaged before the softmax. Every question gets
a teacher distribution, including the ones that also carry a gold label, so the trainer can blend
the two. A few extra open-ended questions per state (below) carry the teacher's answer only and
keep the general question-following behaviour from drifting while the gold tasks are learned.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sources import ENGLISH_MIX, SOURCES  # noqa: E402

AUX = [
    {"type": "noul", "instructions": "Is the author upset or angry?"},
    {"type": "noul", "instructions": "Does the text ask a question?"},
    {"type": "noul", "instructions": "Does the text mention money, prices or payments?"},
    {"type": "noul", "instructions": "Is this written in a formal register?"},
    {"type": "noul", "instructions": "Does the text contain personal contact information such as an email address or phone number?"},
    {"type": "noul", "instructions": "Is the writer requesting that someone take an action?"},
    {"type": "noul", "instructions": "Is this a complaint?"},
    {"type": "noul", "instructions": "Does the text talk about a specific date or deadline?"},
    {"type": "noul", "instructions": "Is the content safe to show to a general audience?"},
    {"type": "noul", "instructions": "Does the text describe a technical problem?"},
    {"type": "noul", "instructions": "Is the author satisfied?"},
    {"type": "noul", "instructions": "Does this message need a reply?"},
    {"type": "noul", "instructions": "Is the text about sports?"},
    {"type": "noul", "instructions": "Does the text mention a company or brand?"},
    {"type": "choice", "instructions": "What is the overall sentiment?",
     "criteria": {"positive": None, "neutral": None, "negative": None}},
    {"type": "choice", "instructions": "Who is most likely the author?",
     "criteria": {"customer": "a customer or user", "employee": "someone writing for a company",
                  "journalist": "a reporter or news outlet", "other": "anyone else"}},
    {"type": "choice", "instructions": "What is the main purpose of the text?",
     "criteria": {"inform": "sharing information", "request": "asking for something",
                  "complain": "expressing dissatisfaction", "persuade": "trying to convince or sell",
                  "entertain": "fun or casual chat"}},
    {"type": "choice", "instructions": "Which department should see this first?",
     "criteria": {"support": "help with a product", "sales": "buying, pricing", "legal": "contracts, compliance",
                  "security": "threats, fraud, abuse", "none": "no department needed"}},
    {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    {"type": "score", "instructions": "How negative is the tone?", "criteria": ["not negative", "slightly negative", "negative", "very negative"]},
    {"type": "score", "instructions": "How long and detailed is the text?", "criteria": ["a few words", "a sentence or two", "a paragraph", "a long document"]},
    {"type": "score", "instructions": "How risky would it be to act on this automatically?",
     "criteria": ["no risk", "low risk", "moderate risk", "high risk"]},
    {"type": "score", "instructions": "How confident does the author sound?", "criteria": ["unsure", "neutral", "confident"]},
]


def make_cases(mix, seed, dev_frac=0.03):
    train, dev = [], []
    for name, n in mix.items():
        t = time.time()
        rng = random.Random("%s-%d" % (name, seed))
        cases = SOURCES[name](n, rng)
        rng.shuffle(cases)
        k = max(1, int(len(cases) * dev_frac)) if name != "typed_decisions" else 0
        dev += cases[:k]
        train += cases[k:]
        print("  %-22s %6d cases %6d questions  %.0fs" % (
            name, len(cases), sum(len(c["questions"]) for c in cases), time.time() - t), flush=True)
    if "typed_decisions" in mix:
        rng = random.Random("td-dev-%d" % seed)
        dev += SOURCES["typed_decisions"](100, rng, split="dev")
    return train, dev


def add_aux(cases, seed, frac=0.2):
    # Only choice/score extras: Laya's noul answers to open-ended questions lean hard towards
    # "false" whatever the state says (its issue #156), and that is not something to distil.
    bank = [q for q in AUX if q["type"] != "noul"]
    rng = random.Random("aux-%d" % seed)
    for c in cases:
        if c["src"] == "typed_decisions" or rng.random() > frac:
            continue
        for j, q in enumerate(rng.sample(bank, rng.randint(1, 2))):
            q = json.loads(json.dumps(q))
            if q["type"] == "choice":
                items = list(q["criteria"].items())
                rng.shuffle(items)
                q["criteria"] = dict(items)
            c["questions"]["aux_%d" % j] = q
            c["gold"]["aux_%d" % j] = None


def label_with_teachers(cases, teachers, rows_per_batch=96):
    """Average the teachers' raw logits per question and store the softmax as c["teacher"]."""
    from cbjev.engine import Batch
    from cbjev.layout import as_text, classic_row, parse_question
    for c in cases:
        c["teacher"] = {}
    for ag in teachers:
        items = []
        texts = [as_text(c["state"]) for c in cases]
        t0 = time.time()
        enc = []
        for i in range(0, len(texts), 2048):
            enc += ag.tokens.encode(texts[i:i + 2048])
        for ci, c in enumerate(cases):
            left = isinstance(c["state"], list)
            for qid, spec in c["questions"].items():
                q = parse_question(qid, spec)
                ids, mk = classic_row(ag.tokens, enc[ci], q, ag.max_len, ag.head_max_len, left)
                items.append((len(ids), ci, qid, ids, mk, q.qtype, len(q.options)))
        items.sort(key=lambda x: x[0])
        i = 0
        while i < len(items):
            L = items[min(len(items), i + rows_per_batch) - 1][0]
            j = min(len(items), i + max(1, min(rows_per_batch, 24576 // max(L, 1))))
            part = items[i:j]
            M = max(len(p[4]) for p in part)
            b = Batch(len(part), max(p[0] for p in part), M, ag.tokens.pad)
            for r, (n, ci, qid, ids, mk, qt, k) in enumerate(part):
                b.ids[r, :n] = ids
                b.pos[r, :n] = np.arange(n)
                b.seg[r, :n] = 0
                b.qtype[r, :] = qt
                b.valid[r, :n] = True
                b.markers[r, :len(mk)] = mk
            out = ag.engine.run(b)
            for r, (n, ci, qid, ids, mk, qt, k) in enumerate(part):
                z = out[r, :len(mk)].astype(np.float64)
                prev = cases[ci]["teacher"].get(qid)
                cases[ci]["teacher"][qid] = z if prev is None else prev + z
            i = j
        print("  teacher %s: %d questions in %.0fs" % (ag.name, len(items), time.time() - t0), flush=True)
    for c in cases:
        for qid, z in c["teacher"].items():
            z = z / len(teachers)
            p = np.exp(z - z.max())
            c["teacher"][qid] = [round(float(x), 5) for x in p / p.sum()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, ".work", "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--teachers", default="laya,laya-td")
    ap.add_argument("--mix", default="english", choices=["english", "multilingual"])
    ap.add_argument("--only", default="", help="comma list: build just these sources (for patching a set)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = ENGLISH_MIX
    if a.mix == "multilingual":
        from sources_ml import MULTILINGUAL_MIX as base
    mix = {k: max(50, int(v * a.scale)) if k != "typed_decisions" else v for k, v in base.items()}
    if a.only:
        mix = {k: v for k, v in mix.items() if k in a.only.split(",")}
    train, dev = make_cases(mix, a.seed)
    add_aux(train, a.seed)
    add_aux(dev, a.seed + 1)
    print("train %d cases / dev %d cases" % (len(train), len(dev)), flush=True)
    import cbjev
    from cbjev.checkpoints import resolve
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    teachers = []
    for t in a.teachers.split(","):
        sub = {"laya": None, "laya-td": "typed-decisions", "laya-ml": "multilingual"}[t]
        teachers.append(cbjev.load(resolve("laya"), device="cuda", subfolder=sub, graphs=False, name=t))
    label_with_teachers(train + dev, teachers)
    for name, cs in (("train", train), ("dev", dev)):
        with open(os.path.join(a.out, name + ".jsonl"), "w") as f:
            for c in cs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
