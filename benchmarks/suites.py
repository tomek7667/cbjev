"""Evaluation suites: real labelled data, fixed seed, identical questions for every engine.

Each suite is a list of cases ``(state, questions, gold)`` where ``gold`` maps a question id to
the index of the correct option (``typed-decisions`` also carries the teacher's distribution).

Suites are tagged by how they relate to cbjev's training mix (see training/sources.py):

  train-split  the dataset's *train* split was trained on; these cases come from its test split
               (or, for single-split datasets, from rows the training loader skips)
  held-out     the dataset was never trained on, by cbjev or (per its docs) by Laya

The case selection copies Laya's research/scripts/bench_apps.py where a suite exists there, so
numbers line up with the ones Laya publishes.
"""
import json
import random

SEED = 13

SUITES = {}


def suite(name, relation, note=""):
    def wrap(fn):
        SUITES[name] = {"build": fn, "relation": relation, "note": note}
        return fn
    return wrap


def _ds(*a, **k):
    from datasets import load_dataset
    return load_dataset(*a, **k)


def _choice(qid, ins, crit):
    return {qid: {"type": "choice", "instructions": ins, "criteria": crit}}


# ----------------------------------------------------------------------------- train-split

@suite("ag_news", "train-split", "Jev 0.910 (published)")
def ag_news(n):
    d = _ds("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    return [({"article": r["text"]}, _choice("topic", "What is the topic of `article`?", crit),
             {"topic": int(r["label"])}) for r in list(d)[:n]]


@suite("email_spam", "train-split")
def email_spam(n):
    from cbjev.email import email_state
    d = _ds("SetFit/enron_spam", split="test")
    q = {"is_spam": {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"}}
    return [(email_state(r.get("subject") or "", (r.get("message") or "")[:3000]), q, {"is_spam": int(r["label"])})
            for r in list(d)[:n]]


@suite("phishing", "train-split", "rows 0-5999 are never trained on")
def phishing(n):
    d = _ds("zefang-liu/phishing-email-dataset", split="train")
    rows = [r for r in list(d)[:6000]
            if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
    random.Random(SEED).shuffle(rows)
    q = {"is_phishing": {"type": "noul",
                         "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                         "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email (even if promotional)"}}}
    return [({"email": r["Email Text"][:3000]}, q, {"is_phishing": int(r["Email Type"] == "Phishing Email")})
            for r in rows[:n]]


@suite("rag_relevance", "train-split", "MS MARCO validation")
def rag_relevance(n):
    rng = random.Random(SEED)
    d = _ds("microsoft/ms_marco", "v1.1", split="validation")
    q = {"relevant": {"type": "noul", "instructions": "Does `passage` help answer `query`?"}}
    out = []
    for r in d:
        texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        neg = [t for t, s in zip(texts, sel) if s == 0]
        if not pos or not neg:
            continue
        take = len(out) % 2 == 0
        out.append(({"query": r["query"], "passage": rng.choice(pos if take else neg)}, q, {"relevant": int(take)}))
        if len(out) >= n:
            break
    return out


SUPPORT_QUEUES = {
    "Technical Support": "technical problems, bugs, outages, integrations",
    "Product Support": "help using a product or feature",
    "Customer Service": "general account or service questions",
    "IT Support": "internal IT, devices, access, networks",
    "Billing and Payments": "invoices, charges, refunds, payment methods",
    "Returns and Exchanges": "returning or exchanging an item",
    "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
    "Sales and Pre-Sales": "pricing, quotes, buying",
    "Human Resources": "employment, payroll, leave, hiring",
    "General Inquiry": "anything else",
}


@suite("support_triage", "train-split", "first 2,000 English tickets are never trained on")
def support_triage(n):
    d = _ds("Tobi-Bueck/customer-support-tickets", split="train")
    keys = list(SUPPORT_QUEUES)
    q = _choice("queue", "Which support queue should handle this ticket?", dict(SUPPORT_QUEUES))
    out = []
    for r in d:
        if r.get("language") != "en" or r.get("queue") not in SUPPORT_QUEUES or not r.get("body"):
            continue
        out.append(({"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]}, q,
                    {"queue": keys.index(r["queue"])}))
        if len(out) >= n:
            break
    return out


@suite("typed_decisions", "train-split", "Jev 0.727 (published); 400 cases x 5 decisions")
def typed_decisions(n):
    d = _ds("LocalLLaMA/typed-decisions", "all", split="test")
    out = []
    for r in d:
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        g = {}
        for qid, q in qs.items():
            if qid not in gold:
                continue
            keys = list(q["criteria"]) if q["type"] == "choice" else \
                [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else ["false", "true"]
            probs = [float(gold[qid]["probabilities"].get(k, 0.0)) for k in keys]
            s = sum(probs) or 1.0
            g[qid] = {"index": keys.index(str(gold[qid]["label"]).lower() if q["type"] == "noul"
                                          else str(gold[qid]["label"])),
                      "dist": [p / s for p in probs], "kind": q["type"], "workflow": r["workflow"]}
        out.append((json.loads(r["state"]), qs, g))
    return out[:max(n, 400)]


# ----------------------------------------------------------------------------- held-out

@suite("emotion", "held-out", "DAIR emotion; Jev 0.480 (published)")
def emotion(n):
    d = _ds("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    q = _choice("emotion", "Which emotion is most strongly expressed in `text`?", {k: None for k in names})
    return [({"text": r["text"]}, q, {"emotion": int(r["label"])}) for r in list(d)[:n]]


@suite("banking77", "held-out", "all 77 labels in one question; Jev 0.870 (published, 72 labels)")
def banking77(n):
    d = _ds("mteb/banking77", split="test")
    labels = sorted(set(d["label_text"]))
    keys = [x.replace("_", " ") for x in labels]
    q = _choice("intent", "Which banking intent does `message` express?", {k: None for k in keys})
    return [({"message": r["text"]}, q, {"intent": keys.index(r["label_text"].replace("_", " "))})
            for r in list(d)[:n]]


def _toxic_chat():
    d = _ds("lmsys/toxic-chat", "toxicchat0124", split="test")
    return [r for r in d if (r.get("user_input") or "").strip()]


@suite("jailbreak", "held-out", "lmsys/toxic-chat jailbreaking flag, balanced")
def jailbreak(n):
    rows = _toxic_chat()
    jb = [r for r in rows if int(r.get("jailbreaking", 0)) == 1][:n // 2]
    nj = [r for r in rows if int(r.get("jailbreaking", 0)) == 0][:n - len(jb)]
    mix = jb + nj
    random.Random(SEED).shuffle(mix)
    q = {"jailbreak": {"type": "noul", "instructions":
                       "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"}}
    return [({"prompt": r["user_input"][:3000]}, q, {"jailbreak": int(r["jailbreaking"])}) for r in mix]


@suite("toxicity", "held-out", "lmsys/toxic-chat toxicity flag, balanced")
def toxicity(n):
    rows = _toxic_chat()
    tox = [r for r in rows if int(r.get("toxicity", 0)) == 1][:n // 2]
    ntox = [r for r in rows if int(r.get("toxicity", 0)) == 0][:n - len(tox)]
    mix = tox + ntox
    random.Random(SEED).shuffle(mix)
    q = {"toxic": {"type": "noul", "instructions":
                   "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?"}}
    return [({"post": r["user_input"][:3000]}, q, {"toxic": int(r["toxicity"])}) for r in mix]


@suite("prompt_injection", "held-out", "deepset/prompt-injections test split")
def prompt_injection(n):
    d = _ds("deepset/prompt-injections", split="test")
    q = {"injection": {"type": "noul", "instructions":
                       "Is `text` a prompt injection: an attempt to override or hijack an AI system's instructions?"}}
    return [({"text": r["text"]}, q, {"injection": int(r["label"])}) for r in list(d)[:n]]


ROUTING_DOMAINS = {
    "code": "software engineering, programming, refactoring, architecture, debugging",
    "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
    "writing": "creative writing, essays, emails, blog posts, copywriting",
    "factual_lookup": "facts, definitions, trivia, history",
    "data_analysis": "statistics, SQL, data manipulation, metrics",
    "chitchat": "casual conversation, greetings, small talk",
}


@suite("model_routing", "held-out", "gsm8k / mbpp / ag_news requests -> domain")
def model_routing(n):
    rng = random.Random(SEED)
    keys = list(ROUTING_DOMAINS)
    pool = [(r["question"], "math_or_logic") for r in list(_ds("openai/gsm8k", "main", split="test"))[:n // 3]]
    pool += [(r["text"], "code") for r in list(_ds("google-research-datasets/mbpp", "full", split="test"))[:n // 3]]
    pool += [(r["text"][:400], "factual_lookup") for r in list(_ds("fancyzhx/ag_news", split="test"))[:n // 3]]
    rng.shuffle(pool)
    q = _choice("domain", "What domain does `request` belong to?", dict(ROUTING_DOMAINS))
    return [({"request": t}, q, {"domain": keys.index(dom)}) for t, dom in pool[:n]]


@suite("sst5", "held-out", "SST-5 as a 5-level score question")
def sst5(n):
    d = _ds("SetFit/sst5", split="test")
    q = {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of `review`?",
                       "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}}
    return [({"review": r["text"]}, q, {"sentiment": int(r["label"])}) for r in list(d)[:n]]


@suite("massive_en", "held-out", "MASSIVE intent (English), 20 options per question")
def massive_en(n):
    return massive("en", n)


def massive(lang, n, n_opts=20):
    d = _ds("mteb/amazon_massive_intent", lang, split="test")
    labels = sorted(set(d["label_text"]))
    rng = random.Random(SEED)
    out = []
    for r in list(d)[:n]:
        pool = [x for x in labels if x != r["label_text"]]
        keys = [r["label_text"]] + rng.sample(pool, min(n_opts - 1, len(pool)))
        rng.shuffle(keys)
        q = _choice("intent", "What is the user asking for in `utterance`?",
                    {k: k.replace("_", " ").replace(".", ": ") for k in keys})
        out.append(({"utterance": r["text"]}, q, {"intent": keys.index(r["label_text"])}))
    return out


@suite("xnli_en", "held-out", "XNLI English test (MNLI-style; MNLI train is in cbjev's mix)")
def xnli_en(n):
    d = list(_ds("facebook/xnli", "en", split="test"))
    random.Random(SEED).shuffle(d)
    q = _choice("relation", "How does `hypothesis` relate to `premise`?",
                {"entailment": "the premise implies the hypothesis is true",
                 "neutral": "the hypothesis may or may not be true",
                 "contradiction": "the premise implies the hypothesis is false"})
    return [({"premise": r["premise"], "hypothesis": r["hypothesis"]}, q, {"relation": int(r["label"])})
            for r in d[:n]]


DEFAULT = ["ag_news", "email_spam", "phishing", "rag_relevance", "support_triage", "typed_decisions",
           "emotion", "banking77", "jailbreak", "toxicity", "prompt_injection", "model_routing", "sst5",
           "massive_en", "xnli_en"]


MASSIVE_LANGS = ["af", "am", "ar", "az", "bn", "cy", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hu",
                 "hy", "id", "is", "it", "ja", "jv", "ka", "km", "kn", "ko", "lv", "ml", "mn", "ms", "my", "nb", "nl",
                 "pl", "pt", "ro", "ru", "sl", "sq", "sv", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "zh-CN",
                 "zh-TW"]
for _lg in MASSIVE_LANGS:
    SUITES["massive." + _lg] = {"build": (lambda lg: lambda n: massive(lg, n))(_lg), "relation": "held-out",
                                "note": "MASSIVE intent (%s), 20 options" % _lg}
MULTILINGUAL = ["massive." + lg for lg in MASSIVE_LANGS]
