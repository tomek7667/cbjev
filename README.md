# cbjev

**Typed decisions — `choice`, `score`, `noul` — about any text or JSON state, from one encoder pass.**
No generation, nothing to parse. A drop-in, self-hosted answer to TypeSafe's Jev wire format, and a
faster, more accurate successor to [Laya](https://github.com/NandhaKishorM/laya).

<p align="center">
  <img src="assets/cbjev_vs_laya_vs_jev.png" alt="cbjev vs Laya vs TypeSafe Jev: accuracy on 15 suites, latency, calibration" width="100%" />
</p>

<!-- RESULTS -->

---

## What is different from Laya

Laya encodes the state once **per question**: ten questions about a 500-token document means ten
500-token sequences. cbjev packs a call into one row — the state once, then every question as its
own segment:

```
Laya   [CLS] q1 [SEP] opts [SEP] state [SEP]      (x N questions)
cbjev  [CLS] q1 | q2 | ... | qN | state [SEP]     (once)
```

The attention mask keeps questions from seeing each other, while the state reads all of them. Each
question segment restarts its positions right after `[CLS]`, so with a single question the row is
token-for-token and position-for-position exactly what a Laya checkpoint was trained on — which is why
fine-tuning from Laya starts from Laya's full ability instead of relearning it.

On top of that:

* **Own encoder forward.** A from-scratch ModernBERT/mmBERT implementation (`cbjev/model.py`) with an
  explicit attention mask, bf16 matmuls over an fp32 residual stream, and **CUDA-graph replay per
  shape bucket** (`cbjev/engine.py`) — a call is one graph launch instead of ~300 kernel launches.
* **Cheaper CPU side.** The state is tokenized once per call, question text is tokenized once and
  cached, and everything is shipped to the GPU in one pinned copy.
* **Retrained checkpoints.** Fine-tuned from Laya on 35 public datasets in many question framings,
  distilled against both English Laya checkpoints, with temperatures fitted on a held-back dev split.
* **Optional order voting.** `cbjev.load(order_votes=2)` also asks every choice/score question with its
  options reversed and averages the answers — one extra short segment, not another pass.
* **Laya checkpoints still run.** `cbjev.load("laya")` runs the original weights in their own layout
  (verified to match Laya's probabilities within bf16 rounding).

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .            # core
.venv/bin/pip install -e ".[serve]"   # + HTTP server
```

Weights: point `CBJEV_HOME` (default `~/.cache/cbjev`) at a directory holding `cbjev/` and
`cbjev-multilingual/`, or pass a path to `cbjev.load(...)`. They are produced by `training/`
(see [Training](#training)).

## Quickstart

```python
import cbjev

agent = cbjev.load()                                   # English checkpoint, GPU if available
res = agent.predict(
    {"subject": "Duplicate charge on invoice #4411",
     "body": "We were billed twice for March. Refund it today or we cancel."},
    {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                                    "sales": "pricing, contracts", "other": "everything else"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
                    "criteria": ["can wait", "this week", "today", "blocking right now"]},
        "churn": {"type": "noul", "instructions": "Does the customer threaten to leave?"},
    },
)
res["answers"]["department"]["choice"]      # 'billing'
res["answers"]["churn"]["noul"]             # P(true)
```

Many states at once: `agent.predict_batch(states, questions)`.

### Routing between languages

```python
from cbjev import Router
router = Router()                                      # english + multilingual, loaded lazily
router.predict({"body": "Mir wurde zweimal abgebucht"}, questions)["routing"]
# {'model': 'multilingual', 'reason': "Latin script, language looks like 'de'"}
```

### HTTP server (Jev wire format)

```bash
cbjev-serve                              # 127.0.0.1:8000, POST /v1/systemone
CBJEV_API_KEY=secret CBJEV_DEVICE=cuda CBJEV_PRELOAD=1 cbjev-serve
```

### Command line

```bash
cbjev "I was charged twice, please refund"            # routing decision only, no model load
cbjev "I was charged twice, please refund" --predict  # triage preset answers as JSON
```

### Presets and e-mail

`cbjev.triage_questions()`, `guard_questions()`, `moderation_questions()`, `router_questions()`,
`email_questions()`; `cbjev.email_state(subject, body)` strips quoted history, signatures and
disclaimers before the model sees an e-mail.

## Benchmarks

See **[BENCHMARKS.md](BENCHMARKS.md)** for every number, how it was measured, and how to reproduce it.

```bash
python benchmarks/accuracy.py --engines laya,laya-td,cbjev --out benchmarks/results/accuracy.json
python benchmarks/speed.py    --engines laya,laya-fast,cbjev --out benchmarks/results/speed.json
python benchmarks/plot.py
```

## Training

```bash
python training/build.py --out .work/data                  # 35 public datasets -> cases + teacher answers
python training/train.py --data .work/data --out ~/.cache/cbjev/cbjev --repeat typed_decisions=8
python training/calibrate.py --ckpt ~/.cache/cbjev/cbjev --data .work/data
```

`training/duty.py --duty 0.5 -- <command>` runs any of these on part of the GPU if you share the machine.

## Honest limits

<!-- LIMITS -->

## License

GPL-3.0-or-later (see `LICENSE`). cbjev's checkpoints are fine-tuned from the Apache-2.0 Laya
checkpoints by Convai Innovations; see `NOTICE` for attribution and the datasets used.
