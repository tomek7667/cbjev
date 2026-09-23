"""Which script, and for Latin text which language: just enough to pick a checkpoint.

The English checkpoint reads English well, other Latin-script languages poorly, and other scripts
not at all. So the decision is script first (exact, from Unicode ranges), then, for Latin text, a
function-word vote. The vote is deliberately timid: a language is only named with a clear margin
and at least one word that belongs to it alone; otherwise the answer is ``None`` (undecided), and
undecided is *not* English. Pure Python, no dependencies, well under a millisecond for a message.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------- scripts

_RANGES: List[Tuple[int, int, str]] = sorted([
    (0x0370, 0x03FF, "greek"), (0x1F00, 0x1FFF, "greek"),
    (0x0400, 0x052F, "cyrillic"), (0x1C80, 0x1C8F, "cyrillic"), (0x2DE0, 0x2DFF, "cyrillic"),
    (0xA640, 0xA69F, "cyrillic"),
    (0x0530, 0x058F, "armenian"),
    (0x0590, 0x05FF, "hebrew"), (0xFB1D, 0xFB4F, "hebrew"),
    (0x0600, 0x06FF, "arabic"), (0x0750, 0x077F, "arabic"), (0x0870, 0x08FF, "arabic"),
    (0xFB50, 0xFDFF, "arabic"), (0xFE70, 0xFEFF, "arabic"),
    (0x0700, 0x074F, "syriac"), (0x0780, 0x07BF, "thaana"),
    (0x0900, 0x097F, "devanagari"), (0xA8E0, 0xA8FF, "devanagari"),
    (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"), (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"), (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"), (0x0D80, 0x0DFF, "sinhala"),
    (0x0E00, 0x0E7F, "thai"), (0x0E80, 0x0EFF, "lao"), (0x0F00, 0x0FFF, "tibetan"),
    (0x1000, 0x109F, "myanmar"), (0x10A0, 0x10FF, "georgian"), (0x1C90, 0x1CBF, "georgian"),
    (0x1100, 0x11FF, "hangul"), (0x3130, 0x318F, "hangul"), (0xA960, 0xA97F, "hangul"),
    (0xAC00, 0xD7AF, "hangul"), (0xFFA0, 0xFFDC, "hangul"),
    (0x1200, 0x139F, "ethiopic"), (0x2D80, 0x2DDF, "ethiopic"),
    (0x13A0, 0x13FF, "cherokee"), (0x1780, 0x17FF, "khmer"), (0x1800, 0x18AF, "mongolian"),
    (0x3040, 0x309F, "kana"), (0x30A0, 0x30FF, "kana"), (0x31F0, 0x31FF, "kana"),
    (0xFF66, 0xFF9F, "kana"), (0x1B000, 0x1B16F, "kana"),
    (0x3100, 0x312F, "han"), (0x31A0, 0x31BF, "han"),       # bopomofo reads with han
    (0x3400, 0x4DBF, "han"), (0x4E00, 0x9FFF, "han"), (0xF900, 0xFAFF, "han"),
    (0x20000, 0x323AF, "han"),
])
_STARTS = [r[0] for r in _RANGES]

# scripts that name a single language; the rest (cyrillic, arabic, devanagari, han...) do not
_SCRIPT_LANG = {"greek": "el", "armenian": "hy", "hebrew": "he", "georgian": "ka", "thai": "th",
                "lao": "lo", "khmer": "km", "hangul": "ko", "kana": "ja", "bengali": "bn",
                "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml", "sinhala": "si",
                "gujarati": "gu", "gurmukhi": "pa", "oriya": "or", "myanmar": "my", "tibetan": "bo"}


def _is_latin(cp: int) -> bool:
    # Basic Latin .. Latin Extended-B + IPA, Latin Extended Additional, fullwidth Latin
    return cp < 0x0300 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF5A


def script_of(ch: str) -> str:
    """Script name of one letter: 'latin', a named script, or 'other' for an unlisted one."""
    cp = ord(ch)
    if _is_latin(cp):
        return "latin"
    i = bisect_right(_STARTS, cp) - 1
    if i >= 0 and cp <= _RANGES[i][1]:
        return _RANGES[i][2]
    return "other"          # an unlisted script is still not something the English model reads


def script_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            s = "latin" if ord(ch) < 0x0300 else script_of(ch)
            counts[s] = counts.get(s, 0) + 1
    return counts


def _foreign_word_letters(text: str) -> int:
    """Letters in non-Latin *words*: runs of 2+ letters of one script that do not start upper-case.

    Skips what English prose carries without being foreign text: a lone symbol (``α = 0.05``) and
    a proper name (``Дмитрий``). Caseless scripts (CJK, Arabic, ...) are never skipped.
    """
    n, run, cur = 0, 0, None
    first_upper = False
    for ch in text:
        if unicodedata.combining(ch):
            continue
        s = None if (not ch.isalpha() or ord(ch) < 0x0300) else script_of(ch)
        if s == "latin":
            s = None
        if s is not None and s == cur:
            run += 1
            continue
        if cur is not None and run >= 2 and not first_upper:
            n += run
        cur, run, first_upper = s, (1 if s else 0), (ch.isupper() if s else False)
    if cur is not None and run >= 2 and not first_upper:
        n += run
    return n


# ---------------------------------------------------------------------------- latin languages

def _w(s: str) -> frozenset:
    return frozenset(s.split())


_WORDS: Dict[str, frozenset] = {
    "en": _w("the and is are was were to of for with that this it you have has not but on at be as "
             "from will can would there their what which please we i my your our they been do does "
             "did should could about if an or"),
    "de": _w("der die das und ist ein eine einen einem einer den dem nicht mit für auf von zu sich auch "
             "werden wurde haben sind oder aber ich wir mir mich dir dich uns mein meine meinen diese "
             "dieser dieses wie wann welche im zum zur aus bei nach noch bitte heute jetzt kann habe "
             "gibt wird keine kein schon sehr dass warum"),
    "fr": _w("le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais sont "
             "ont aux ce et du au ou je il elle ils mon ma mes ces très bien tout tous fait veux peux "
             "peut dois doit merci bonjour jour mois fois quand comment pourquoi alors donc j'ai "
             "n'est c'est depuis toujours encore"),
    "es": _w("el los las que por con para una es se del como pero son está este esta todo más muy hay "
             "sus la un y al lo le su mi tu nos dos fue ser tiene tengo puede quiero necesito hemos han "
             "cuando donde porque también ya eso esto nada algo aquí hoy gracias hola usted"),
    "pt": _w("os as que em um uma para com não é se do da dos das mas são está este esta muito pelo pela "
             "o e na nas nos ao aos por foi era ser tem tenho pode quero preciso eu meu minha seu sua "
             "isso aqui como quando onde porque mais já ainda agora hoje dois tudo nada obrigado "
             "obrigada olá você voce vc nao pra gostaria também tambem estou depois deu ficou"),
    "it": _w("il lo gli che di per con non è si del della sono questo questa anche come più nella alla "
             "la le un uno una e ed o da su tra mi ci ne ho hai ha abbiamo hanno era stato devo voglio "
             "vorrei mio mia quando dove perché molto sempre mai già ancora oggi grazie ciao nel sul "
             "sulla dal dalla dei delle degli alle"),
    "nl": _w("het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt ik "
             "je jij wij we mijn zijn heb hebben kan kunnen wil moet nog geen bedankt alstublieft "
             "graag waarom wanneer hoe"),
    "pl": _w("i w na z że się nie to jest do jak ale co tak od po za czy dla już jestem mam mój moja "
             "moje przez jego jej są było był była bardzo może można proszę dziękuję dzień jeszcze "
             "tylko kiedy gdzie dlaczego który która które oraz lub"),
    "sv": _w("och att det som är på för med av till den har inte jag om ett så eller "
             "kan vi du min mitt mina hur när varför tack hej också bara måste skulle"),
    "da": _w("og at det som er på for med af til den har ikke jeg om et så eller kan vi "
             "du min mit mine hvordan hvornår hvorfor tak hej også kun skal skulle jeres gerne meget hvad nogen "
             "fordi"),
    "no": _w("og at det som er på for med av til den har ikke jeg om et så eller kan vi "
             "du min mitt mine hvordan når hvorfor takk hei også bare må skulle dere ikkje eg gjerne veldig hva "
             "noen fordi"),
    "fi": _w("ja on ei se että oli ovat mutta kun jos niin tai kuin myös vain olen olet minä sinä hän "
             "me te he tämä tuo mitä miksi missä milloin kiitos hei voi pitää haluan minun sinun"),
    "tr": _w("ve bir bu için ile da ne çok daha ama gibi olan olarak yok ben sen biz siz onlar "
             "şu mı mi mu mü değil kadar sonra önce neden nasıl nerede teşekkürler merhaba lütfen "
             "benim sizin"),
    "ro": _w("și să este sunt care pentru din dar după până fără ale lui în fost acum vreau trebuie "
             "foarte acest această acesta mulțumesc bună vă nu"),
    "cs": _w("a je se na v že to s z do jsem jsou ale jak už nebo pro jeho její mám jste byl byla "
             "bylo když kde proč děkuji prosím dobrý ještě také jen který která které"),
    "hu": _w("a az és egy hogy nem is van meg ez azt csak már mint vagy volt lesz kell nagyon "
             "köszönöm kérem szia jó miért hol mikor én te mi ti ők"),
    "id": _w("dan yang di ini itu dengan untuk tidak dari ke pada ada saya kami kita anda mereka akan "
             "sudah belum bisa juga atau karena jika tapi tetapi terima kasih tolong apa bagaimana "
             "kenapa mengapa kapan dimana"),
    "vi": _w("của và là không có được những một cho với này các tôi bạn chúng người trong khi đã sẽ "
             "rất cũng như nhưng vì nếu thì làm ạ cảm ơn xin chào"),
}
_LANGS = tuple(_WORDS)
# a word two lists share (la, de, og, und...) counts toward a total but names no language itself
_SHARED = frozenset(w for w in set().union(*_WORDS.values()) if sum(w in s for s in _WORDS.values()) > 1)

# letters ordinary English never uses; a high rate means "not English" even with no word match
_FOREIGN_LETTERS = frozenset(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ" "ăâîșțşţ" "ąćęłńśźż" "čďěňřšťůž" "őű" "ğı" "āēģīķļņū" "đ" "ə"
    "ơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
# letters that point at one language when words cannot decide
_TELL = {"ł": "pl", "ą": "pl", "ę": "pl", "ś": "pl", "ź": "pl", "ż": "pl", "ń": "pl",
         "ß": "de", "ő": "hu", "ű": "hu", "ğ": "tr", "ı": "tr", "ș": "ro", "ț": "ro", "ă": "ro",
         "ř": "cs", "ě": "cs", "ů": "cs", "ã": "pt", "õ": "pt", "ñ": "es",
         "ơ": "vi", "ư": "vi", "đ": "vi"}

FOREIGN_RATE = 0.02         # share of letters that are _FOREIGN_LETTERS
MIN_WORDS = 4               # below this the language is left undecided
FOREIGN_SHARE = 0.2         # non-Latin share of letters that overrides a Latin plurality...
FOREIGN_SHARE_LOW = 0.1     # ...or this share when it is also at least FOREIGN_MIN_LETTERS letters
FOREIGN_MIN_LETTERS = 10

_NOISE = re.compile(r"\b(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\w[\w-]*(?:\.[\w-]+)+", re.U)
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.U)


def latin_language(text: str) -> Dict[str, Any]:
    """Function-word vote for Latin text: ``language`` (or None), ``english_hits``, ``foreign_rate``."""
    clean = _NOISE.sub(" ", text).replace("İ", "i").lower()
    letters = foreign = 0
    tells: Dict[str, int] = {}
    for ch in clean:
        if ch.isalpha():
            letters += 1
            if ch in _FOREIGN_LETTERS:
                foreign += 1
                t = _TELL.get(ch)
                if t:
                    tells[t] = tells.get(t, 0) + 1
    rate = foreign / letters if letters else 0.0
    looks_foreign = rate >= FOREIGN_RATE
    words = _WORD_RE.findall(clean)
    out = {"language": None, "english_hits": 0, "foreign_rate": round(rate, 4), "looks_foreign": looks_foreign}
    if len(words) < MIN_WORDS:
        return out
    hits = dict.fromkeys(_LANGS, 0)
    own = dict.fromkeys(_LANGS, 0)
    for w in words:
        for lg in _LANGS:
            if w in _WORDS[lg]:
                hits[lg] += 1
                if w not in _SHARED:
                    own[lg] += 1
    en = out["english_hits"] = hits["en"]
    # only a language with a word of its own may win; ties broken by letters only it uses
    cands = [lg for lg in _LANGS if lg != "en" and own[lg]]
    best = max(cands, key=lambda lg: (hits[lg], tells.get(lg, 0), own[lg]), default=None)
    b = hits[best] if best else 0
    top = max(hits[lg] for lg in _LANGS if lg != "en")
    if best and b >= max(2, en + 2):
        out["language"] = best
    elif best and looks_foreign and b >= max(2, en):
        out["language"] = best
    elif top >= max(3, en + 2):
        out["looks_foreign"] = True     # shared words only: plainly not English, but no single language
    elif en and not looks_foreign and en >= b:
        out["language"] = "en"
    return out


# ------------------------------------------------------------------------------------ analysis

def state_text(state: Any, max_chars: int = 4000) -> str:
    """The prose of a state: string leaves of dicts/lists (keys skipped, they are usually English)."""
    parts: List[str] = []
    budget = max_chars
    stack = [(state, 0)]
    while stack and budget > 0:
        v, d = stack.pop()
        if isinstance(v, str):
            parts.append(v[:budget])
            budget -= len(v) + 1
        elif d < 8 and isinstance(v, dict):
            stack.extend((x, d + 1) for x in reversed(list(v.values())))
        elif d < 8 and isinstance(v, (list, tuple)):
            stack.extend((x, d + 1) for x in reversed(v))
    return " ".join(parts)[:max_chars]


def analyse(text: Any) -> Dict[str, Any]:
    """Detection for a string (or a state, via `state_text`).

    Keys: ``script`` ('latin', 'cyrillic', 'han', ..., 'unknown' for no letters), ``language``
    (ISO 639-1 code or None when undecided), ``is_english`` (safe for the English checkpoint),
    ``undecided``, ``foreign_rate`` and ``non_latin`` (share of letters outside Latin).
    """
    if not isinstance(text, str):
        text = state_text(text)
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return {"script": "unknown", "language": None, "is_english": True, "undecided": True,
                "foreign_rate": 0.0, "non_latin": 0.0}
    latin = counts.get("latin", 0)
    non_latin = (total - latin) / total
    script = max(counts, key=counts.get)
    if script == "latin" and non_latin >= FOREIGN_SHARE_LOW:
        # a Latin plurality (brand names, order codes, English field text) around a foreign request
        n = _foreign_word_letters(text)
        if n and (n / total >= FOREIGN_SHARE or (n / total >= FOREIGN_SHARE_LOW and n >= FOREIGN_MIN_LETTERS)):
            script = max((s for s in counts if s != "latin"), key=counts.get)
    if script != "latin":
        lang = _SCRIPT_LANG.get(script)
        if script == "han":
            lang = "ja" if counts.get("kana") else "zh"
        if total < 2:
            lang = None
        return {"script": script, "language": lang, "is_english": False, "undecided": lang is None,
                "foreign_rate": 0.0, "non_latin": round(non_latin, 4)}
    lat = latin_language(text)
    lang = lat["language"]
    return {"script": "latin", "language": lang,
            "is_english": lang == "en" or (lang is None and not lat["looks_foreign"]),
            "undecided": lang is None, "foreign_rate": lat["foreign_rate"], "non_latin": round(non_latin, 4)}


def detect_language(text: Any) -> Optional[str]:
    return analyse(text)["language"]


def detect_script(text: Any) -> str:
    return analyse(text)["script"]


def is_english(text: Any) -> bool:
    return bool(analyse(text)["is_english"])


_ENGLISH_CODES = frozenset(("en", "eng", "english"))


def english_code(code: Any) -> Optional[bool]:
    """True/False for a language code (``en``, ``en_US.UTF-8``, ``eng``, ``English``, ``de``...);
    None when there is nothing to go on, so a caller's detector can abstain."""
    if code is None:
        return None
    c = str(code).strip().lower().split(".", 1)[0].replace("_", "-").split("-", 1)[0]
    return (c in _ENGLISH_CODES) if c else None
