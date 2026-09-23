"""Training data: public labelled datasets rendered as typed-decision cases.

A case is ``{"state", "questions": {qid: Jev question}, "gold": {qid: distribution or None},
"src"}``. ``gold`` is None for questions that only carry the teacher's answer (see build.py).

Every task is rendered through several framings (a choice over labels, a noul per label, a score
for ordinal labels, opaque letter keys with the label text as description, with and without
descriptions) and several instruction wordings, so the model learns to read the question rather
than memorise one prompt per dataset.

What is *not* here matters as much. These never enter training, because benchmarks/suites.py
evaluates on them: dair-ai/emotion, banking77, lmsys/toxic-chat, deepset/prompt-injections,
SST-5 (and SST-2, which shares its sentences), MASSIVE, XNLI, gsm8k and mbpp. For datasets whose
test cases the benchmarks take from the train split, the rows they use are skipped below.
"""
import json
import random
from typing import Callable, Dict, List, Optional

from datasets import load_dataset

SOURCES: Dict[str, Callable] = {}


def source(name):
    def wrap(fn):
        SOURCES[name] = fn
        return fn
    return wrap


def _take(ds, n, rng, filt=None):
    rows = list(ds) if filt is None else [r for r in ds if filt(r)]
    rng.shuffle(rows)
    return rows[:n]


def _onehot(k, i):
    return [1.0 if j == i else 0.0 for j in range(k)]


def _letters(k):
    import string
    return list(string.ascii_uppercase[:k]) if k <= 26 else ["L%d" % i for i in range(1, k + 1)]


def classify(rng, state, labels: List[str], gold: int, instructions: List[str], descs: Optional[List[str]] = None,
             noul_prompts: Optional[List[str]] = None, ordinal: bool = False, max_extra: int = 2):
    """Questions for one labelled example, in a few different framings.

    labels are human-readable class names; descs optional descriptions; noul_prompts a list of
    templates with ``{label}``; ordinal adds a score framing.
    """
    qs, gold_d = {}, {}
    k = len(labels)
    order = list(range(k))
    rng.shuffle(order)
    style = rng.random()
    ins = rng.choice(instructions)
    if style < 0.18 and k <= 26:
        keys = _letters(k)
        crit = {keys[j]: (labels[order[j]] + (" - " + descs[order[j]] if descs and rng.random() < 0.5 else ""))
                for j in range(k)}
    elif style < 0.55 and descs:
        crit = {labels[o]: descs[o] for o in order}
    else:
        crit = {labels[o]: None for o in order}
    qs["label"] = {"type": "choice", "instructions": ins, "criteria": crit}
    gold_d["label"] = _onehot(k, order.index(gold))
    if ordinal and rng.random() < 0.6:
        qs["level"] = {"type": "score", "instructions": ins, "criteria": [
            labels[i] + (": " + descs[i] if descs and rng.random() < 0.3 else "") for i in range(k)]}
        gold_d["level"] = _onehot(k, gold)
    if noul_prompts:
        picks = [gold] if rng.random() < 0.5 else []
        others = [i for i in range(k) if i != gold]
        rng.shuffle(others)
        picks += others[:rng.randint(0, max_extra)]
        for j, i in enumerate(picks):
            t = rng.choice(noul_prompts)
            name = descs[i] if descs and rng.random() < 0.3 else labels[i]
            q = {"type": "noul", "instructions": t.format(label=name)}
            if rng.random() < 0.15:
                q["criteria"] = {"true": "yes, it is " + name, "false": "no, it is something else"}
            qs["is_%d" % j] = q
            gold_d["is_%d" % j] = [0.0, 1.0] if i == gold else [1.0, 0.0]
    if rng.random() < 0.35 and len(qs) > 1:
        # sometimes only one framing, so single-question rows are common too
        keep = rng.choice(list(qs))
        qs, gold_d = {keep: qs[keep]}, {keep: gold_d[keep]}
    return qs, gold_d


def binary(rng, instructions, positive: bool, true_desc=None, false_desc=None, as_choice=None):
    """A yes/no fact as a noul, or occasionally as a two-option choice."""
    ins = rng.choice(instructions)
    if as_choice and rng.random() < 0.25:
        yes, no = as_choice
        if rng.random() < 0.3:
            keys = ["A", "B"] if rng.random() < 0.5 else ["B", "A"]
            crit = {keys[0]: yes, keys[1]: no}
            first_yes = True
        else:
            first_yes = rng.random() < 0.5
            crit = {yes: None, no: None} if first_yes else {no: None, yes: None}
        q = {"type": "choice", "instructions": ins, "criteria": crit}
        idx = (0 if positive else 1) if first_yes else (1 if positive else 0)
        return q, _onehot(2, idx)
    q = {"type": "noul", "instructions": ins}
    if (true_desc or false_desc) and rng.random() < 0.5:
        q["criteria"] = {k: v for k, v in (("true", true_desc), ("false", false_desc)) if v}
    return q, ([0.0, 1.0] if positive else [1.0, 0.0])


def case(src, state, qs, gold):
    return {"src": src, "state": state, "questions": qs, "gold": gold}


# ============================================================================ tasks
# Each source returns a list of cases. `n` is the number of source rows to use.

@source("typed_decisions")
def typed_decisions(n, rng, split="train"):
    d = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    rows = list(d)
    random.Random(7).shuffle(rows)
    rows = rows[100:] if split == "train" else rows[:100]      # 100 train cases kept for dev/calibration
    out = []
    for r in rows[:n]:
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        g = {}
        for qid, q in qs.items():
            if qid not in gold:
                continue
            if q["type"] == "choice":
                keys = list(q["criteria"])
            elif q["type"] == "score":
                keys = [str(i) for i in range(len(q["criteria"]))]
            else:
                keys = ["false", "true"]
            p = [float(gold[qid]["probabilities"].get(k, 0.0)) for k in keys]
            s = sum(p) or 1.0
            g[qid] = [x / s for x in p]
        out.append(case("typed_decisions", json.loads(r["state"]), {k: qs[k] for k in g}, g))
    return out


TOPICS_AG = ["world", "sports", "business", "science and technology"]
TOPICS_AG_D = ["world news and international politics", "sports", "business and economy", "science and technology"]


@source("ag_news")
def ag_news(n, rng):
    d = load_dataset("fancyzhx/ag_news", split="train")
    ins = ["What is the topic of `article`?", "Which news section does this article belong in?",
           "Classify the news story by topic.", "What is this article mainly about?"]
    out = []
    for r in _take(d, n, rng):
        st = {rng.choice(["article", "article", "text", "story"]): r["text"]}
        qs, g = classify(rng, st, TOPICS_AG, int(r["label"]), ins, TOPICS_AG_D,
                         ["Is this article about {label}?", "Does the story belong to the {label} section?"])
        out.append(case("ag_news", st, qs, g))
    return out


@source("enron_spam")
def enron_spam(n, rng):
    from cbjev.email import email_state
    d = load_dataset("SetFit/enron_spam", split="train")
    ins = ["Is this email unsolicited spam or bulk marketing?", "Is this message spam?",
           "Should this email go to the spam folder?", "Is this an unwanted mass-mailing?"]
    out = []
    for r in _take(d, n, rng):
        st = email_state(r.get("subject") or "", (r.get("message") or "")[:3000])
        q, g = binary(rng, ins, int(r["label"]) == 1, "spam or bulk marketing", "a normal personal or work email",
                      as_choice=("spam", "not spam"))
        out.append(case("enron_spam", st, {"spam": q}, {"spam": g}))
    return out


@source("phishing")
def phishing(n, rng):
    d = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    rows = [r for r in list(d)[6000:] if (r.get("Email Text") or "").strip()
            and r.get("Email Type") in ("Safe Email", "Phishing Email")]    # 0-5999 are benchmark rows
    rng.shuffle(rows)
    ins = ["Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
           "Is this a phishing email?", "Does this message try to trick the reader into giving away data or money?",
           "Is this email fraudulent?"]
    out = []
    for r in rows[:n]:
        q, g = binary(rng, ins, r["Email Type"] == "Phishing Email", "phishing, scam, or fraud",
                      "a legitimate email (even if promotional)", as_choice=("phishing", "legitimate"))
        out.append(case("phishing", {"email": r["Email Text"][:3000]}, {"phish": q}, {"phish": g}))
    return out


@source("ms_marco")
def ms_marco(n, rng):
    d = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    ins = ["Does `passage` help answer `query`?", "Is the passage relevant to the query?",
           "Would this passage be a useful search result for the query?",
           "Does the passage contain the answer to the question?"]
    out = []
    for r in d:
        texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        neg = [t for t, s in zip(texts, sel) if s == 0]
        if not pos or not neg:
            continue
        take = rng.random() < 0.5
        q, g = binary(rng, ins, take, "the passage is relevant", "the passage does not help")
        out.append(case("ms_marco", {"query": r["query"], "passage": rng.choice(pos if take else neg)},
                        {"relevant": q}, {"relevant": g}))
        if len(out) >= n:
            break
    return out


QUEUES = ["Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments",
          "Returns and Exchanges", "Service Outages and Maintenance", "Sales and Pre-Sales", "Human Resources",
          "General Inquiry"]
QUEUES_D = ["technical problems, bugs, outages, integrations", "help using a product or feature",
            "general account or service questions", "internal IT, devices, access, networks",
            "invoices, charges, refunds, payment methods", "returning or exchanging an item",
            "downtime, outages, scheduled maintenance", "pricing, quotes, buying", "employment, payroll, leave, hiring",
            "anything else"]


@source("support_tickets")
def support_tickets(n, rng, lang="en"):
    d = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    rows, seen_en = [], 0
    for r in d:
        if r.get("language") == "en":
            seen_en += 1
            if seen_en <= 2000:           # the benchmark reads the first English tickets
                continue
        if (lang == "any" or r.get("language") == lang) and r.get("queue") in QUEUES and r.get("body"):
            rows.append(r)
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        st = {"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]}
        qs, g = classify(rng, st, QUEUES, QUEUES.index(r["queue"]),
                         ["Which support queue should handle this ticket?", "Route this ticket to a team.",
                          "Which department owns this request?"], QUEUES_D,
                         ["Should the {label} team handle this ticket?"], max_extra=1)
        pr = {"low": 0, "medium": 1, "high": 2}.get(str(r.get("priority")).lower())
        if pr is not None and rng.random() < 0.5:
            qs["priority"] = {"type": "score", "instructions": rng.choice(
                ["How urgent is this ticket?", "What priority should this ticket get?"]),
                "criteria": ["low", "medium", "high"]}
            g["priority"] = _onehot(3, pr)
        tp = r.get("type")
        types = ["Incident", "Request", "Problem", "Change"]
        if tp in types and rng.random() < 0.4:
            order = types[:]
            rng.shuffle(order)
            qs["kind"] = {"type": "choice", "instructions": "What kind of ticket is this?",
                          "criteria": {t: None for t in order}}
            g["kind"] = _onehot(4, order.index(tp))
        out.append(case("support_tickets", st, qs, g))
    return out


NLI = ["entailment", "neutral", "contradiction"]
NLI_D = ["the premise implies the hypothesis is true", "the hypothesis may or may not be true",
         "the premise implies the hypothesis is false"]


def _nli_case(rng, src, premise, hypothesis, label):
    keys = rng.choice([("premise", "hypothesis"), ("text", "claim"), ("context", "statement")])
    st = {keys[0]: premise, keys[1]: hypothesis}
    qs, g = classify(rng, st, NLI, label, [
        "How does `%s` relate to `%s`?" % (keys[1], keys[0]),
        "Is the %s supported, contradicted, or neither by the %s?" % (keys[1], keys[0]),
        "What is the logical relationship between the two texts?"], NLI_D)
    if rng.random() < 0.6:
        q, gg = binary(rng, ["Does the %s follow from the %s?" % (keys[1], keys[0]),
                             "Is the %s true given the %s?" % (keys[1], keys[0]),
                             "Can the %s be concluded from the %s?" % (keys[1], keys[0])], label == 0)
        qs["follows"], g["follows"] = q, gg
    if rng.random() < 0.3:
        q, gg = binary(rng, ["Does the %s contradict the %s?" % (keys[1], keys[0])], label == 2)
        qs["contradicts"], g["contradicts"] = q, gg
    return case(src, st, qs, g)


@source("mnli")
def mnli(n, rng):
    d = load_dataset("nyu-mll/multi_nli", split="train")
    return [_nli_case(rng, "mnli", r["premise"], r["hypothesis"], int(r["label"]))
            for r in _take(d.select(range(120000)), n, rng) if int(r["label"]) in (0, 1, 2)]


@source("snli")
def snli(n, rng):
    d = load_dataset("stanfordnlp/snli", split="train").select(range(100000))
    return [_nli_case(rng, "snli", r["premise"], r["hypothesis"], int(r["label"]))
            for r in _take(d, n, rng) if int(r["label"]) in (0, 1, 2)]


@source("scitail")
def scitail(n, rng):
    d = load_dataset("allenai/scitail", "snli_format", split="train")
    out = []
    for r in _take(d, n, rng):
        pos = r["gold_label"] == "entailment"
        q, g = binary(rng, ["Does the premise support the hypothesis?", "Is the claim entailed by the text?",
                            "Does `premise` provide evidence that `hypothesis` is true?"], pos)
        out.append(case("scitail", {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, {"q": q}, {"q": g}))
    return out


@source("rte")
def rte(n, rng):
    d = load_dataset("aps/super_glue", "rte", split="train")
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Does the premise entail the hypothesis?", "Is `hypothesis` true according to `premise`?"],
                      int(r["label"]) == 0)
        out.append(case("rte", {"premise": r["premise"], "hypothesis": r["hypothesis"]}, {"q": q}, {"q": g}))
    return out


@source("boolq")
def boolq(n, rng):
    d = load_dataset("google/boolq", split="train")
    out = []
    for r in _take(d, n, rng):
        qtext = r["question"].strip().rstrip("?") + "?"
        qtext = qtext[0].upper() + qtext[1:]
        q, g = binary(rng, [qtext, "According to the passage: " + qtext], bool(r["answer"]),
                      as_choice=("yes", "no"))
        out.append(case("boolq", {"passage": r["passage"]}, {"answer": q}, {"answer": g}))
    return out


SENT_INS = ["What is the sentiment of `review`?", "Is this review positive or negative?",
            "How does the author feel about the product?", "What is the overall tone of the text?"]


def _polarity(rng, src, text, pos, field="review"):
    st = {field: text}
    if rng.random() < 0.55:
        qs, g = classify(rng, st, ["negative", "positive"], int(pos), SENT_INS, None,
                         ["Is the {label} sentiment expressed here?", "Is this a {label} review?"], max_extra=1)
    else:
        q, gg = binary(rng, ["Is this review positive?", "Did the customer like it?",
                             "Would the author recommend it?", "Is the sentiment positive?"], pos,
                       "positive sentiment", "negative sentiment", as_choice=("positive", "negative"))
        qs, g = {"positive": q}, {"positive": gg}
    return case(src, st, qs, g)


@source("imdb")
def imdb(n, rng):
    d = load_dataset("stanfordnlp/imdb", "plain_text", split="train")
    return [_polarity(rng, "imdb", r["text"][:2500], int(r["label"]) == 1) for r in _take(d, n, rng)]


@source("yelp_polarity")
def yelp_polarity(n, rng):
    d = load_dataset("fancyzhx/yelp_polarity", split="train").select(range(60000))
    return [_polarity(rng, "yelp_polarity", r["text"][:2500], int(r["label"]) == 1) for r in _take(d, n, rng)]


@source("amazon_polarity")
def amazon_polarity(n, rng):
    d = load_dataset("fancyzhx/amazon_polarity", split="train").select(range(60000))
    return [_polarity(rng, "amazon_polarity", (r["title"] + ". " + r["content"])[:2500], int(r["label"]) == 1)
            for r in _take(d, n, rng)]


@source("tweet_sentiment")
def tweet_sentiment(n, rng):
    d = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
    labels = ["negative", "neutral", "positive"]
    out = []
    for r in _take(d, n, rng):
        st = {rng.choice(["tweet", "post", "text"]): r["text"]}
        qs, g = classify(rng, st, labels, int(r["label"]), ["What is the sentiment of this tweet?",
                                                           "Is the post positive, negative or neutral?",
                                                           "How does the author feel?"],
                         None, ["Is the tone {label}?"], ordinal=True, max_extra=1)
        out.append(case("tweet_sentiment", st, qs, g))
    return out


EKMAN = {"anger": ["anger", "annoyance", "disapproval"], "disgust": ["disgust"], "fear": ["fear", "nervousness"],
         "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
                 "admiration", "desire", "caring"],
         "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
         "surprise": ["surprise", "realization", "confusion", "curiosity"], "neutral": ["neutral"]}


# a six-way scheme (with "love" split from joy) is common in emotion datasets, so train it too
SIX_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SIX = {"sadness": "sadness", "grief": "sadness", "disappointment": "sadness", "remorse": "sadness",
       "joy": "joy", "amusement": "joy", "excitement": "joy", "gratitude": "joy", "optimism": "joy",
       "relief": "joy", "pride": "joy", "love": "love", "caring": "love", "admiration": "love", "desire": "love",
       "anger": "anger", "annoyance": "anger", "disgust": "anger", "fear": "fear", "nervousness": "fear",
       "surprise": "surprise"}


@source("tweet_emotion")
def tweet_emotion(n, rng):
    d = load_dataset("cardiffnlp/tweet_eval", "emotion", split="train")
    labels = ["anger", "joy", "optimism", "sadness"]
    out = []
    for r in _take(d, n, rng):
        st = {"text": r["text"]}
        qs, g = classify(rng, st, labels, int(r["label"]), ["Which emotion is most strongly expressed in `text`?",
                                                           "What does the author feel?"], None,
                         ["Is the author feeling {label}?"], max_extra=1)
        out.append(case("tweet_emotion", st, qs, g))
    return out


@source("go_emotions")
def go_emotions(n, rng):
    d = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    names = d.features["labels"].feature.names
    fine2coarse = {f: c for c, fs in EKMAN.items() for f in fs}
    out = []
    for r in _take(d, n * 2, rng):
        if len(r["labels"]) != 1:
            continue
        fine = names[r["labels"][0]]
        coarse = list(EKMAN)
        st = {"text": r["text"]}
        roll = rng.random()
        if roll < 0.35 and fine in SIX:
            labels = list(SIX_NAMES)
            qs, g = classify(rng, st, labels, labels.index(SIX[fine]),
                             ["Which emotion is most strongly expressed in `text`?", "What is the author feeling?",
                              "Which emotion best describes this message?"], None,
                             ["Does the author express {label}?"], max_extra=1)
        elif roll < 0.7:
            labels = coarse if rng.random() < 0.5 else [c for c in coarse if c != "neutral"]
            if fine2coarse[fine] not in labels:
                continue
            qs, g = classify(rng, st, labels, labels.index(fine2coarse[fine]),
                             ["Which emotion is most strongly expressed in `text`?", "What is the author feeling?",
                              "Which emotion best describes this message?"], None,
                             ["Does the author express {label}?", "Is the writer feeling {label}?"], max_extra=1)
        else:
            pool = [x for x in names if fine2coarse.get(x) != fine2coarse[fine]]
            rng.shuffle(pool)
            labels = [fine] + pool[:rng.randint(3, 9)]
            rng.shuffle(labels)
            qs, g = classify(rng, st, labels, labels.index(fine), ["Which emotion does the text express?",
                                                                  "Pick the emotion that fits the message."])
        out.append(case("go_emotions", st, qs, g))
        if len(out) >= n:
            break
    return out


@source("clinc")
def clinc(n, rng):
    d = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = d.features["intent"].names
    pretty = [x.replace("_", " ") for x in names]
    out = []
    for r in _take(d, n, rng):
        gi = int(r["intent"])
        k = rng.choice([5, 10, 20, 20, 30, 50, 80, 150])
        pool = [i for i in range(len(names)) if i != gi]
        rng.shuffle(pool)
        pick = [gi] + pool[:k - 1]
        rng.shuffle(pick)
        labels = [pretty[i] for i in pick]
        st = {rng.choice(["utterance", "message", "request"]): r["text"]}
        ins = rng.choice(["What is the user asking for?", "Which intent does the message express?",
                          "Classify the user's request.", "What does the user want to do?"])
        crit = {lab: None for lab in labels}
        out.append(case("clinc", st, {"intent": {"type": "choice", "instructions": ins, "criteria": crit}},
                        {"intent": _onehot(len(labels), pick.index(gi))}))
    return out


@source("dbpedia")
def dbpedia(n, rng):
    d = load_dataset("fancyzhx/dbpedia_14", split="train")
    labels = ["company", "educational institution", "artist", "athlete", "office holder", "means of transportation",
              "building", "natural place", "village", "animal", "plant", "album", "film", "written work"]
    out = []
    for r in _take(d, n, rng):
        st = {"title": r["title"], "abstract": r["content"].strip()}
        qs, g = classify(rng, st, labels, int(r["label"]), ["What kind of entity is described?",
                                                           "Which category does this article's subject belong to?",
                                                           "What is this encyclopedia entry about?"], None,
                         ["Is the subject a {label}?", "Does this describe a {label}?"], max_extra=1)
        out.append(case("dbpedia", st, qs, g))
    return out


@source("yahoo")
def yahoo(n, rng):
    d = load_dataset("community-datasets/yahoo_answers_topics", split="train").select(range(200000))
    labels = ["society and culture", "science and mathematics", "health", "education and reference",
              "computers and internet", "sports", "business and finance", "entertainment and music",
              "family and relationships", "politics and government"]
    out = []
    for r in _take(d, n, rng):
        st = {"question": (r["question_title"] + " " + (r["question_content"] or "")).strip()[:1500]}
        if rng.random() < 0.5:
            st["best_answer"] = (r["best_answer"] or "")[:1500]
        qs, g = classify(rng, st, labels, int(r["topic"]), ["Which topic does this question belong to?",
                                                           "Categorise the question.", "What is this about?"],
                         None, ["Is this a question about {label}?"], max_extra=1)
        out.append(case("yahoo", st, qs, g))
    return out


@source("civil_comments")
def civil_comments(n, rng):
    d = load_dataset("google/civil_comments", split="train").select(range(400000))
    tox = [r for r in d if r["toxicity"] >= 0.6]
    ok = [r for r in d.select(range(60000)) if r["toxicity"] < 0.1]
    rng.shuffle(tox)
    rng.shuffle(ok)
    rows = tox[:n // 2] + ok[:n - n // 2]
    rng.shuffle(rows)
    out = []
    for r in rows:
        st = {rng.choice(["comment", "post", "text"]): r["text"][:2000]}
        qs, g = {}, {}
        q, gg = binary(rng, ["Is this comment toxic?", "Is `comment` rude, disrespectful or hateful?",
                             "Should a moderator remove this comment for toxicity?",
                             "Is the text abusive?"], r["toxicity"] >= 0.5,
                       "toxic, insulting or abusive", "civil", as_choice=("toxic", "not toxic"))
        qs["toxic"], g["toxic"] = q, gg
        for attr, ins in (("insult", "Does the comment insult someone?"), ("threat", "Does the comment contain a threat?"),
                          ("obscene", "Is the comment obscene or vulgar?"),
                          ("identity_attack", "Does the comment attack a group based on identity?")):
            if rng.random() < 0.25 and (r[attr] >= 0.5 or r[attr] < 0.1):
                q, gg = binary(rng, [ins], r[attr] >= 0.5)
                qs[attr], g[attr] = q, gg
        if rng.random() < 0.25:
            lvl = 0 if r["toxicity"] < 0.1 else 1 if r["toxicity"] < 0.6 else 2 if r["severe_toxicity"] < 0.1 else 3
            qs["severity"] = {"type": "score", "instructions": "How toxic is this comment?",
                              "criteria": ["not toxic", "mildly toxic", "toxic", "severely toxic"]}
            g["severity"] = _onehot(4, lvl)
        out.append(case("civil_comments", st, qs, g))
    return out


@source("toxic_conversations")
def toxic_conversations(n, rng):
    d = load_dataset("mteb/toxic_conversations_50k", split="train")
    rows = [r for r in d if int(r["label"]) == 1]
    rows = rows[:n // 2] + _take([r for r in d if int(r["label"]) == 0], n - min(len(rows), n // 2), rng)
    rng.shuffle(rows)
    out = []
    for r in rows:
        q, g = binary(rng, ["Is this message toxic?", "Is the text offensive or harassing?",
                            "Would this message violate a civility policy?"], int(r["label"]) == 1)
        out.append(case("toxic_conversations", {"message": r["text"][:2000]}, {"toxic": q}, {"toxic": g}))
    return out


@source("hate_offensive")
def hate_offensive(n, rng):
    d = load_dataset("tdavidson/hate_speech_offensive", split="train")
    labels = ["hate speech", "offensive but not hate speech", "neither"]
    out = []
    for r in _take(d, n, rng):
        st = {"tweet": r["tweet"]}
        qs, g = classify(rng, st, labels, int(r["class"]), ["Classify this tweet.",
                                                           "Is the tweet hate speech, offensive, or neither?"],
                         None, ["Is this tweet {label}?"], max_extra=1)
        out.append(case("hate_offensive", st, qs, g))
    return out


@source("tweet_offensive")
def tweet_offensive(n, rng):
    d = load_dataset("cardiffnlp/tweet_eval", "offensive", split="train")
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Is this tweet offensive?", "Does the post contain offensive language?"],
                      int(r["label"]) == 1)
        out.append(case("tweet_offensive", {"tweet": r["text"]}, {"offensive": q}, {"offensive": g}))
    return out


@source("tweet_hate")
def tweet_hate(n, rng):
    d = load_dataset("cardiffnlp/tweet_eval", "hate", split="train")
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Is this tweet hateful towards immigrants or women?", "Does the tweet contain hate speech?"],
                      int(r["label"]) == 1)
        out.append(case("tweet_hate", {"tweet": r["text"]}, {"hate": q}, {"hate": g}))
    return out


@source("tweet_irony")
def tweet_irony(n, rng):
    d = load_dataset("cardiffnlp/tweet_eval", "irony", split="train")
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Is this tweet ironic?", "Is the author being sarcastic or ironic?"], int(r["label"]) == 1)
        out.append(case("tweet_irony", {"tweet": r["text"]}, {"irony": q}, {"irony": g}))
    return out


@source("jailbreak_cls")
def jailbreak_cls(n, rng):
    d = load_dataset("jackhhao/jailbreak-classification", split="train")
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Is this prompt a jailbreak attempt?",
                            "Does `prompt` try to make an AI assistant ignore its rules or safety guidelines?",
                            "Is the user trying to bypass the model's restrictions?"],
                      r["type"] == "jailbreak", "an attempt to bypass the assistant's rules",
                      "an ordinary request", as_choice=("jailbreak", "benign"))
        out.append(case("jailbreak_cls", {"prompt": r["prompt"][:3000]}, {"jb": q}, {"jb": g}))
    return out


@source("safeguard_injection")
def safeguard_injection(n, rng):
    """Prompt-injection examples from a collection unrelated to the held-out deepset benchmark;
    any text that also appears in deepset/prompt-injections is dropped."""
    held = set()
    for sp in ("train", "test"):
        held |= {r["text"].strip().lower() for r in load_dataset("deepset/prompt-injections", split=sp)}
    d = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
    rows = [r for r in d if r["text"].strip().lower() not in held]
    out = []
    for r in _take(rows, n, rng):
        q, g = binary(rng, ["Is `text` a prompt injection: an attempt to override or hijack an AI system's instructions?",
                            "Does this input try to make the AI ignore its previous instructions?",
                            "Is this a prompt injection attack?",
                            "Does the text try to manipulate an AI assistant into a different task or role?"],
                      int(r["label"]) == 1, "an attempt to override the system's instructions",
                      "an ordinary input", as_choice=("injection", "benign"))
        out.append(case("safeguard_injection", {rng.choice(["text", "prompt", "input"]): r["text"][:3000]},
                        {"inj": q}, {"inj": g}))
    return out


@source("squad_v2")
def squad_v2(n, rng):
    d = load_dataset("rajpurkar/squad_v2", split="train")
    out = []
    for r in _take(d.select(range(80000)), n, rng):
        ans = bool(r["answers"]["text"])
        q, g = binary(rng, ["Does `context` contain the answer to `question`?",
                            "Can the question be answered from the context?",
                            "Is the answer to the question stated in the text?"], ans)
        out.append(case("squad_v2", {"context": r["context"], "question": r["question"]}, {"answerable": q},
                        {"answerable": g}))
    return out


@source("paws")
def paws(n, rng):
    d = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Do the two sentences mean the same thing?", "Is `b` a paraphrase of `a`?"],
                      int(r["label"]) == 1)
        out.append(case("paws", {"a": r["sentence1"], "b": r["sentence2"]}, {"same": q}, {"same": g}))
    return out


@source("qqp")
def qqp(n, rng):
    d = load_dataset("nyu-mll/glue", "qqp", split="train").select(range(100000))
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Are these two questions asking the same thing?", "Are the questions duplicates?"],
                      int(r["label"]) == 1)
        out.append(case("qqp", {"question_1": r["question1"], "question_2": r["question2"]}, {"dup": q}, {"dup": g}))
    return out


def _mcq(rng, src, state, question, options, gold):
    order = list(range(len(options)))
    rng.shuffle(order)
    if rng.random() < 0.5:
        keys = _letters(len(options))
        crit = {keys[j]: options[order[j]] for j in range(len(options))}
    else:
        crit = {options[o]: None for o in order}
    return case(src, state, {"answer": {"type": "choice", "instructions": question, "criteria": crit}},
                {"answer": _onehot(len(options), order.index(gold))})


@source("race")
def race(n, rng):
    d = load_dataset("ehovy/race", "all", split="train")
    out = []
    for r in _take(d, n, rng):
        opts = list(r["options"])
        if len(set(opts)) != len(opts):
            continue
        out.append(_mcq(rng, "race", {"article": r["article"][:3500]}, r["question"], opts, "ABCD".index(r["answer"])))
    return out


@source("openbookqa")
def openbookqa(n, rng):
    d = load_dataset("allenai/openbookqa", "main", split="train")
    out = []
    for r in _take(d, n, rng):
        opts, labs = r["choices"]["text"], r["choices"]["label"]
        if len(set(opts)) != len(opts):
            continue
        st = {"question": r["question_stem"]} if rng.random() < 0.5 else r["question_stem"]
        out.append(_mcq(rng, "openbookqa", st, rng.choice(["Which answer is correct?", "Answer the question.",
                                                           r["question_stem"]]), opts, labs.index(r["answerKey"])))
    return out


@source("commonsense_qa")
def commonsense_qa(n, rng):
    d = load_dataset("tau/commonsense_qa", split="train")
    out = []
    for r in _take(d, n, rng):
        opts, labs = r["choices"]["text"], r["choices"]["label"]
        if len(set(opts)) != len(opts) or r["answerKey"] not in labs:
            continue
        out.append(_mcq(rng, "commonsense_qa", {"question": r["question"]},
                        rng.choice(["Which answer is most sensible?", "Pick the best answer to `question`.",
                                    r["question"]]), opts, labs.index(r["answerKey"])))
    return out


@source("arc")
def arc(n, rng):
    out = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        d = load_dataset("allenai/ai2_arc", cfg, split="train")
        for r in _take(d, n // 2, rng):
            opts, labs = r["choices"]["text"], r["choices"]["label"]
            if len(set(opts)) != len(opts) or r["answerKey"] not in labs:
                continue
            out.append(_mcq(rng, "arc", {"question": r["question"]}, "Which answer is correct?", opts,
                            labs.index(r["answerKey"])))
    return out


@source("subjectivity")
def subjectivity(n, rng):
    d = load_dataset("SetFit/subj", split="train")
    out = []
    for r in _take(d, n, rng):
        q, g = binary(rng, ["Is this sentence an opinion rather than a statement of fact?",
                            "Is the text subjective?"], int(r["label"]) == 1,
                      as_choice=("subjective", "objective"))
        out.append(case("subjectivity", {"sentence": r["text"]}, {"subj": q}, {"subj": g}))
    return out


@source("hh_rlhf")
def hh_rlhf(n, rng):
    d = load_dataset("Anthropic/hh-rlhf", split="train").select(range(80000))
    out = []
    for r in _take(d, n, rng):
        a, b = r["chosen"], r["rejected"]
        cut = 0
        while cut < min(len(a), len(b)) and a[cut] == b[cut]:
            cut += 1
        cut = a.rfind("Assistant:", 0, cut + 1)
        if cut < 0 or len(a) > 4000:
            continue
        convo, ra, rb = a[:cut].strip()[-2000:], a[cut + 10:].strip(), b[cut + 10:].strip()
        if not ra or not rb or ra == rb:
            continue
        first_good = rng.random() < 0.5
        st = {"conversation": convo, "response_a": ra if first_good else rb, "response_b": rb if first_good else ra}
        crit = {"response_a": None, "response_b": None}
        out.append(case("hh_rlhf", st, {"better": {"type": "choice", "instructions": rng.choice(
            ["Which response is more helpful and harmless?", "Which assistant reply is better?",
             "Which response would a careful reviewer prefer?"]), "criteria": crit}},
                        {"better": [1.0, 0.0] if first_good else [0.0, 1.0]}))
    return out


# ============================================================================ mixtures

ENGLISH_MIX = {
    "typed_decisions": 1100, "ag_news": 7000, "enron_spam": 5000, "phishing": 5000, "ms_marco": 9000,
    "support_tickets": 6000, "mnli": 9000, "snli": 4000, "scitail": 3000, "rte": 2400, "boolq": 7000,
    "imdb": 3000, "yelp_polarity": 4000, "amazon_polarity": 4000, "tweet_sentiment": 5000, "go_emotions": 12000, "tweet_emotion": 3200,
    "clinc": 12000, "dbpedia": 4000, "yahoo": 6000, "civil_comments": 9000, "toxic_conversations": 4000,
    "hate_offensive": 4000, "tweet_offensive": 4000, "tweet_hate": 3000, "tweet_irony": 2500,
    "jailbreak_cls": 1100, "safeguard_injection": 6000, "squad_v2": 6000, "paws": 4000, "qqp": 4000, "race": 6000, "openbookqa": 4000,
    "commonsense_qa": 6000, "arc": 3000, "subjectivity": 3000, "hh_rlhf": 5000,
}
