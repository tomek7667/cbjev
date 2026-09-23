"""E-mail to state: keep what the sender wrote this time, drop what they did not.

Quoted history (often a different request), signatures, device footers and legal disclaimers
are all text the model would otherwise weigh as heavily as the new message. The rules are
deliberately conservative: a line is only removed when it looks like boilerplate *on its own*,
so a request that merely mentions "confidential" or starts with "Thanks for" survives.
Covers English, Portuguese, Spanish, German and Polish clients.
"""
import re
from typing import Dict, Optional

_QUOTE_START = [re.compile(p, re.I) for p in (
    r"^\s*On .{0,300}wrote:\s*$",
    r"^\s*Em (?=.*\d).{0,300}escreveu:\s*$",
    r"^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$",
    r"^\s*Am (?=.*\d).{0,300}schrieb .{0,120}:\s*$",
    r"^\s*(W dniu|Dnia) (?=.*\d).{0,300}(napisał|napisała)\(?a?\)?:\s*$",
    r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}",
    r"^\s*-{2,}\s*(Mensagem original|Mensaje original|Ursprüngliche Nachricht|Wiadomość oryginalna)\s*-{2,}",
    r"^\s*_{8,}\s*$",
    r"^\s*(From|Von|Od):\s.*[@<]",
    r"^\s*De:\s.*[@<]",
)]
_SIGN_OFF = re.compile(
    r"^\s*(?i:best regards|best wishes|best|kind regards|regards|warm regards|warmest regards|cheers|thanks|"
    r"thank you|many thanks|sincerely|yours sincerely|yours truly|"
    r"atenciosamente|abraços?|cordialmente|saludos( cordiales)?|atentamente|mit freundlichen grüßen|"
    r"viele grüße|lg|pozdrawiam|z poważaniem)"
    r"[\s,;:!.]*(?:[^\W\d_a-zß-öø-ÿ][\w'-]*[\s,.]*){0,3}$")
_FOOTER = re.compile(
    r"^\s*(sent from my \w+( \w+)?|enviado d[oe] (meu|mi) \w+( \w+)?|von meinem \w+ gesendet|"
    r"wysłane z (mojego )?\w+|get outlook for (ios|android))[\s.!]*$", re.I)
_DISCLAIMER = re.compile(
    r"(\b(e-?mail|message|information|communication|transmission)\b[^.]{0,60}\bconfidential\b[^.]{0,60}"
    r"\b(intended|solely|addressee|recipient|privileged|disclos|unauthori[sz]ed)"
    r"|if you (have )?received this (e-?mail|message) in error"
    r"|\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)"
    r"|diese (e-?mail|nachricht)\b[^.]{0,80}vertraulich)", re.I)


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")[: max_chars * 4]
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\n", "\n")        # exports that stored newlines as a literal backslash-n
    kept = []
    for line in text.split("\n"):
        if kept and any(p.match(line) for p in _QUOTE_START):
            break                                   # everything below is history
        if line.lstrip().startswith(">"):
            continue
        kept.append(line.rstrip())
    # a sign-off only counts near the end: what follows it must look like a signature block
    # (a few short lines), otherwise "Thanks!" on line two would take the request with it
    for i in range(1, len(kept)):
        s = kept[i].strip()
        tail = [t for t in kept[i + 1:] if t.strip()]
        if len(tail) > 6 or any(len(t.strip()) > 60 for t in tail):
            continue
        if (len(s) <= 40 and _SIGN_OFF.match(s)) or (len(s) <= 60 and _FOOTER.match(s)) or s == "--":
            kept = kept[:i]
            break
    paras = []
    for p in re.split(r"\n\s*\n", "\n".join(kept)):
        if _DISCLAIMER.search(p):
            p = " ".join(s for s in re.split(r"(?<=[.!?])\s+", p) if not _DISCLAIMER.search(s))
        if p.strip():
            paras.append(re.sub(r"[ \t]+", " ", p.strip()))
    return "\n\n".join(paras)[:max_chars]


def email_state(subject: str, body: str, sender: Optional[str] = None, clean: bool = True, **extra) -> Dict:
    """A state dict for an e-mail: subject, cleaned body, optional sender and extra fields."""
    st = {"subject": (subject or "").strip(), "body": clean_email_body(body) if clean else (body or "")}
    if sender:
        st["from"] = sender
    st.update({k: v for k, v in extra.items() if v is not None})
    return st
