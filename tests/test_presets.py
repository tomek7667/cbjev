import pytest

from cbjev import presets
from cbjev.layout import parse_question

FNS = [presets.router_questions, presets.guard_questions, presets.moderation_questions,
       presets.triage_questions, presets.email_questions]


@pytest.mark.parametrize("fn", FNS, ids=lambda f: f.__name__)
def test_every_preset_validates(fn):
    qs = fn()
    assert qs
    for qid, spec in qs.items():
        q = parse_question(qid, spec)
        assert q.qid == qid and len(q.options) >= 2


@pytest.mark.parametrize("fn", FNS, ids=lambda f: f.__name__)
def test_fresh_copies(fn):
    a, b = fn(), fn()
    assert a == b and a is not b
    first = next(iter(a))
    a[first]["instructions"] = "changed"
    assert fn()[first]["instructions"] != "changed"


def test_expected_keys():
    assert {"complexity", "domain", "needs_tools"} <= set(presets.router_questions())
    assert set(presets.guard_questions()) == {"jailbreak", "prompt_injection", "data_exfiltration",
                                              "system_prompt_leak"}
    mod = presets.moderation_questions()
    assert {"toxic", "harassment", "hate", "threat", "sexual", "self_harm", "severity"} == set(mod)
    assert mod["severity"]["type"] == "score"
    tri = presets.triage_questions()
    assert (tri["department"]["type"], tri["urgency"]["type"], tri["frustration"]["type"],
            tri["churn_risk"]["type"], tri["refund_requested"]["type"]) == ("choice", "score", "score", "noul", "noul")
    assert set(presets.email_questions()) == {"spam", "phishing", "needs_reply", "urgency"}


def test_custom_choices():
    qs = presets.triage_questions({"a": "first", "b": "second"})
    assert parse_question("department", qs["department"]).keys == ("a", "b")
    assert parse_question("domain", presets.router_questions({"x": "", "y": None})["domain"]).keys == ("x", "y")


def test_exported_from_package():
    import cbjev
    assert cbjev.triage_questions is presets.triage_questions
    assert cbjev.presets.ALL["guard"] is presets.guard_questions
