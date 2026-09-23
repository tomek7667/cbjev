import pytest

from cbjev import email_state
from cbjev.email import clean_email_body as clean

ASK = "Please refund order 1234, it was charged twice.\nIt has been a week now."


@pytest.mark.parametrize("header", [
    "On Mon, 3 Mar 2025 at 10:00, Support <support@acme.com> wrote:",
    "-----Original Message-----",
    "From: Support <support@acme.com>",
    "Am 03.03.2025 um 10:00 schrieb Support <s@acme.com>:",
    "Em seg., 3 de mar. de 2025 às 10:00, Suporte <s@a.com> escreveu:",
    "W dniu 3.03.2025 o 10:00 Support napisał:",
])
def test_quoted_history_dropped(header):
    out = clean(ASK + "\n\n" + header + "\nWe issued a partial credit last month.\n> Old stuff")
    assert "refund order 1234" in out
    assert "partial credit" not in out and "Old stuff" not in out


def test_inline_quote_lines_dropped():
    out = clean("Can you reset my password?\n> earlier quoted line\nstill here")
    assert out == "Can you reset my password?\nstill here"


def test_header_on_first_line_is_kept():
    # a message cannot *start* with history; keep it rather than returning nothing
    assert "Support" in clean("From: Support <support@acme.com>\nplease help")


@pytest.mark.parametrize("signoff", ["Kind regards,", "Regards", "Thanks,", "Thank you!", "Cheers", "Best,",
                                     "Sincerely,", "Many thanks,", "Pozdrawiam", "Mit freundlichen Grüßen",
                                     "Atenciosamente,"])
def test_sign_off_and_signature_dropped(signoff):
    out = clean(ASK + "\n\n" + signoff + "\nAnna Kowalska\nHead of Ops")
    assert "refund order 1234" in out and "Anna" not in out and "Head of Ops" not in out


def test_best_regards_sign_off():
    assert "Anna" not in clean(ASK + "\n\nBest regards,\nAnna Kowalska")


def test_thanks_sentence_is_not_a_sign_off():
    out = clean(ASK + "\nThanks for the quick help last time.\n\nKind regards,\nAnna")
    assert "Thanks for the quick help last time." in out and "Anna" not in out


@pytest.mark.parametrize("footer", ["Sent from my iPhone", "Get Outlook for Android", "Enviado do meu iPhone"])
def test_device_footer_dropped(footer):
    assert clean(ASK + "\n\n" + footer) == ASK


def test_disclaimer_dropped_but_confidential_request_survives():
    body = ("Please send me the confidential pricing sheet before the board meeting.\n\n"
            "This email and any attachments are confidential and intended solely for the addressee. "
            "If you have received this email in error please notify the sender.")
    out = clean(body)
    assert out == "Please send me the confidential pricing sheet before the board meeting."
    # the same word inside the request paragraph is untouched
    assert "confidential" in clean("Is the Q3 report confidential, or can I share it with the auditors?")


def test_disclaimer_sentence_removed_inside_paragraph():
    out = clean("I need the March invoice. This message is confidential and intended only for the recipient.")
    assert out == "I need the March invoice."


def test_whitespace_newlines_and_limit():
    assert clean("a\r\nb\rc") == "a\nb\nc"
    assert clean("line1\\nline2") == "line1\nline2"          # literal \n from JSON-escaped sources
    assert clean("a   b\t\tc") == "a b c"
    assert len(clean("word " * 5000, max_chars=100)) <= 100
    assert clean("") == "" and clean(None) == ""


def test_email_state():
    st = email_state("  Refund  ", "Can you refund me?\n\nSent from my iPhone", sender="a@b.c", ticket=None,
                     lang="en")
    assert st == {"subject": "Refund", "body": "Can you refund me?", "from": "a@b.c", "lang": "en"}
    raw = email_state("s", "x\n> q", clean=False)
    assert raw == {"subject": "s", "body": "x\n> q"}


def test_early_thanks_keeps_request():
    out = clean("Hello,\nThanks!\nMy account is locked and I cannot log in since Monday, please help.")
    assert "account is locked" in out


def test_literal_backslash_n_in_paths_survives():
    assert "C:\\new_folder" in clean("Please restore C:\\new_folder from backup.\nIt was deleted today.")
