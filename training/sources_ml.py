"""Multilingual training sources for cbjev-multilingual (mmBERT encoder).

Held out for evaluation and never read here: MASSIVE (all 51 languages), and the first 200 rows
per language of textdetox/multilingual_toxicity_dataset.
"""
import random

from datasets import load_dataset

from sources import SOURCES, _onehot, binary, case, classify, source

NLI_LANGS = ["ar", "bn", "de", "es", "fa", "fr", "he", "hi", "id", "it", "ja", "ko", "mr", "nl", "pl", "ps", "pt",
             "ru", "sv", "sw", "ta", "tr", "uk", "ur", "vi", "zh"]


@source("ml_nli")
def ml_nli(n, rng):
    from sources import _nli_case
    out = []
    per = max(1, n // len(NLI_LANGS))
    for lg in NLI_LANGS:
        for sub in ("mnli", "fever"):
            try:
                d = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", split="%s_%s" % (lg, sub))
            except Exception:
                continue
            rows = list(d.select(range(min(len(d), per * 4))))
            rng.shuffle(rows)
            for r in rows[:per // 2]:
                if int(r["label"]) in (0, 1, 2):
                    out.append(_nli_case(rng, "ml_nli", r["premise"], r["hypothesis"], int(r["label"])))
    return out


STARS = ["1 star: very bad", "2 stars: bad", "3 stars: okay", "4 stars: good", "5 stars: excellent"]


@source("ml_amazon")
def ml_amazon(n, rng):
    out = []
    langs = ["de", "es", "fr", "ja", "zh", "en"]
    for lg in langs:
        d = load_dataset("SetFit/amazon_reviews_multi_" + lg, split="train")
        rows = list(d.select(range(min(len(d), n))))
        rng.shuffle(rows)
        for r in rows[:n // len(langs)]:
            st = {"review": r["text"]}
            lab = int(r["label"])
            if rng.random() < 0.5:
                qs = {"stars": {"type": "score", "instructions": rng.choice(
                    ["How many stars did the customer give?", "How satisfied is the reviewer?"]),
                    "criteria": STARS}}
                g = {"stars": _onehot(5, lab)}
            elif lab != 2:
                q, gg = binary(rng, ["Is this review positive?", "Did the customer like the product?"], lab >= 3)
                qs, g = {"pos": q}, {"pos": gg}
            else:
                continue
            out.append(case("ml_amazon", st, qs, g))
    return out


@source("ml_tweets")
def ml_tweets(n, rng):
    out = []
    langs = ["arabic", "english", "french", "german", "hindi", "italian", "portuguese", "spanish"]
    for lg in langs:
        d = load_dataset("mteb/tweet_sentiment_multilingual", lg, split="train")
        for r in random.Random(lg).sample(list(d), min(len(d), n // len(langs))):
            st = {"tweet": r["text"]}
            qs, g = classify(rng, st, ["negative", "neutral", "positive"], int(r["label"]),
                             ["What is the sentiment of this tweet?", "How does the author feel?"], None,
                             ["Is the tone {label}?"], ordinal=True, max_extra=1)
            out.append(case("ml_tweets", st, qs, g))
    return out


SIB = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]


@source("ml_sib200")
def ml_sib200(n, rng):
    from huggingface_hub import HfApi
    files = [s.rfilename for s in HfApi().dataset_info("Davlan/sib200").siblings or []]
    langs = sorted({f.split("/")[1] for f in files if f.startswith("data/") and f.count("/") >= 2})
    rng2 = random.Random(3)
    langs = rng2.sample(langs, min(80, len(langs)))
    out = []
    for lg in langs:
        try:
            d = load_dataset("Davlan/sib200", lg, split="train")
        except Exception:
            continue
        for r in random.Random(lg).sample(list(d), min(len(d), max(1, n // len(langs)))):
            if r["category"] not in SIB:
                continue
            st = {"text": r["text"]}
            qs, g = classify(rng, st, SIB, SIB.index(r["category"]), ["What is the topic of the text?",
                                                                     "Which category fits this sentence?"],
                             None, ["Is the text about {label}?"], max_extra=1)
            out.append(case("ml_sib200", st, qs, g))
    return out


@source("ml_toxicity")
def ml_toxicity(n, rng):
    out = []
    langs = ["en", "ru", "uk", "de", "es", "am", "zh", "ar", "hi", "it", "fr", "he", "ja", "tt"]
    for lg in langs:
        try:
            d = load_dataset("textdetox/multilingual_toxicity_dataset", split=lg)
        except Exception:
            continue
        rows = list(d)[200:]
        rng.shuffle(rows)
        for r in rows[:n // len(langs)]:
            q, g = binary(rng, ["Is this text toxic?", "Is the message rude, insulting or hateful?",
                                "Would a moderator remove this?"], int(r["toxic"]) == 1)
            out.append(case("ml_toxicity", {"text": r["text"][:2000]}, {"toxic": q}, {"toxic": g}))
    return out


LANG_NAMES = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
              "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
              "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu",
              "vi": "Vietnamese", "zh": "Chinese"}


@source("ml_langid")
def ml_langid(n, rng):
    d = load_dataset("papluca/language-identification", split="train")
    names = list(LANG_NAMES)
    out = []
    for r in random.Random(5).sample(list(d), n):
        if r["labels"] not in LANG_NAMES:
            continue
        pool = [x for x in names if x != r["labels"]]
        pick = [r["labels"]] + rng.sample(pool, rng.randint(3, 8))
        labels = [LANG_NAMES[x] for x in pick]
        qs, g = classify(rng, {"text": r["text"]}, labels, 0, ["Which language is the text written in?",
                                                              "What language is this?"], None,
                         ["Is the text written in {label}?"], max_extra=1)
        out.append(case("ml_langid", {"text": r["text"]}, qs, g))
    return out


@source("ml_support")
def ml_support(n, rng):
    return [dict(c, src="ml_support") for c in SOURCES["support_tickets"](n, rng, lang="any")]


MULTILINGUAL_MIX = {
    "ml_nli": 26000, "ml_amazon": 18000, "ml_tweets": 12000, "ml_sib200": 9000, "ml_toxicity": 9000,
    "ml_langid": 6000, "ml_support": 8000,
    # English tasks keep the question-following skills the multilingual checkpoint shares
    "typed_decisions": 1100, "ag_news": 3000, "enron_spam": 2000, "phishing": 2000, "ms_marco": 4000,
    "boolq": 3000, "go_emotions": 4000, "clinc": 6000, "civil_comments": 4000, "jailbreak_cls": 1100,
    "yahoo": 3000, "race": 3000, "commonsense_qa": 3000, "hh_rlhf": 2000, "imdb": 1500, "squad_v2": 3000,
    "hate_offensive": 2000, "tweet_offensive": 2000,
}
