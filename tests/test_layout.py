import os

import pytest

from cbjev.layout import NOUL_FALSE, NOUL_TRUE, as_text, classic_row, packed_row, parse_question


# ------------------------------------------------------------------ parse_question

@pytest.mark.parametrize("spec,msg", [
    ("nope", "must be an object"),
    ({"type": "maybe", "instructions": "?"}, "type must be one of"),
    ({"type": "noul"}, "missing 'instructions'"),
    ({"type": "choice", "instructions": "?"}, "choice needs 'criteria'"),
    ({"type": "choice", "instructions": "?", "criteria": {}}, "choice needs 'criteria'"),
    ({"type": "choice", "instructions": "?", "criteria": "a,b"}, "choice needs 'criteria'"),
    ({"type": "score", "instructions": "?", "criteria": {"a": 1}}, "score needs 'criteria'"),
    ({"type": "score", "instructions": "?", "criteria": []}, "score needs 'criteria'"),
    ({"type": "noul", "instructions": "?", "criteria": ["x"]}, "noul 'criteria' is an optional"),
    ({"type": "noul", "instructions": "?", "criteria": {"yes": "x"}}, "only use the keys true/false"),
    ({"type": "choice", "instructions": "?", "criteria": ["a"], "labels": {}}, "'labels' only applies"),
    ({"type": "noul", "instructions": "?", "labels": {"false": "no"}}, "'labels' must map"),
    ({"type": "noul", "instructions": "?", "labels": {"false": "x", "true": "x"}}, "'labels' must map"),
    ({"type": "noul", "instructions": "?", "labels": {"false": " ", "true": "y"}}, "'labels' must map"),
])
def test_parse_question_errors(spec, msg):
    with pytest.raises(ValueError) as e:
        parse_question("qx", spec)
    assert msg in str(e.value) and "qx" in str(e.value)


def test_parse_question_shapes():
    q = parse_question("d", {"type": "choice", "instructions": "Team?",
                             "criteria": {"billing": "money", "tech": "", "misc": None, "n": 0, "j": {"a": 1}}})
    assert q.keys == ("billing", "tech", "misc", "n", "j")
    assert q.options == ("billing: money", "tech", "misc", "n: 0", 'j: {"a": 1}')
    assert parse_question("d", {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}).options == ("a", "b")
    s = parse_question("s", {"type": "score", "instructions": "?", "criteria": ["low", "high"]})
    assert s.keys == ("0", "1") and s.options == ("level 0: low", "level 1: high") and s.qtype == 1
    n = parse_question("n", {"type": "noul", "instructions": "?"})
    assert n.keys == ("false", "true") and n.options == ("false: " + NOUL_FALSE, "true: " + NOUL_TRUE)
    n = parse_question("n", {"type": "noul", "instructions": "?", "criteria": {"TRUE": "spam"},
                             "labels": {"false": " ham ", "true": "spam"}})
    assert n.options == ("ham: " + NOUL_FALSE, "spam: spam")
    assert parse_question("x", {"type": "noul", "instructions": {"ask": 1}}).instructions == '{"ask": 1}'


# ------------------------------------------------------------------ token rows (real tokenizer)

@pytest.fixture(scope="module")
def snapshot():
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download("convaiinnovations/laya", allow_patterns=["tokenizer/*"])
    except Exception as e:  # offline and not cached
        pytest.skip("Laya tokenizer unavailable: %s" % e)
    return os.path.join(path, "tokenizer")


@pytest.fixture(scope="module")
def tk(snapshot):
    from cbjev.agent import _tokens_for
    return _tokens_for(snapshot)


def _ids(tk, state):
    return tk.encode([as_text(state)])[0]


QS = {
    "dept": {"type": "choice", "instructions": "Which team should handle `message`?",
             "criteria": {"billing": "charges and refunds", "tech": "bugs", "other": None}},
    "mood": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "annoyed", {"desc": "furious", "hint": "caps"}]},
    "spam": {"type": "noul", "instructions": "Is it spam?"},
}


def test_packed_row_structure(tk):
    state = {"message": "I was charged twice for my plan, please refund the second charge."}
    qs = [parse_question(k, v) for k, v in QS.items()]
    sid = _ids(tk, state)
    row = packed_row(tk, sid, qs, max_state=1024, max_question=768, truncate_left=False)
    n0 = len(sid) + 2
    assert row.ids[:n0] == [tk.cls] + sid + [tk.sep]
    assert row.pos[:n0] == list(range(n0)) and row.seg[:n0] == [0] * n0
    assert len(row.ids) == len(row.pos) == len(row.seg) == len(row.qtype)
    assert len(row.markers) == len(qs)
    start = n0
    for si, (q, mk) in enumerate(zip(qs, row.markers), 1):
        end = start
        while end < len(row.ids) and row.seg[end] == si:
            end += 1
        # each segment restarts its positions right after the state
        assert row.pos[start:end] == list(range(n0, n0 + end - start))
        assert set(row.qtype[start:end]) == {q.qtype}
        assert len(mk) == len(q.options)
        assert all(start <= m < end and row.ids[m] == tk.mask for m in mk)
        assert row.ids[end - 1] == tk.sep
        start = end
    assert start == len(row.ids)
    assert sum(i == tk.mask for i in row.ids) == sum(len(q.options) for q in qs)


def test_packed_row_state_truncation(tk):
    state = " ".join("word%d" % i for i in range(400))
    q = [parse_question("spam", QS["spam"])]
    sid = _ids(tk, state)
    right = packed_row(tk, sid, q, max_state=16, max_question=64, truncate_left=False)
    left = packed_row(tk, sid, q, max_state=16, max_question=64, truncate_left=True)
    assert right.ids[1:17] == sid[:16] and left.ids[1:17] == sid[-16:]
    assert right.seg[:18] == [0] * 18 and right.seg[18] == 1


def test_packed_row_tight_question_budget(tk):
    q = parse_question("c", {"type": "choice", "instructions": "Pick one of these many options " * 10,
                             "criteria": {"opt%d" % i: "a long description of option %d " % i * 5 for i in range(12)}})
    row = packed_row(tk, _ids(tk, "hello"), [q], max_state=1024, max_question=96, truncate_left=False)
    assert len(row.markers[0]) == 12
    assert all(row.ids[m] == tk.mask for m in row.markers[0])


def _laya_q(spec):
    q = {"t": spec["type"], "ins": spec["instructions"]}
    crit = spec.get("criteria")
    if spec["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    q["crit"] = crit
    if "labels" in spec:
        q["labels"] = spec["labels"]
    return q


CASES = [
    ({"message": "I was charged twice, please refund me."}, QS["dept"], 512, 192, False),
    ({"message": "Mein Konto wurde zweimal belastet."}, QS["mood"], 512, 192, False),
    ("plain string state with a [MASK] token inside", QS["spam"], 512, 192, False),
    ([{"role": "user", "content": "turn %d " % i * 20} for i in range(30)], QS["dept"], 256, 192, True),
    ("x " * 2000, QS["mood"], 128, 64, False),
    ({"body": "short"}, {"type": "noul", "instructions": "Is it urgent? [MASK]",
                         "criteria": {"true": "yes, due today"}, "labels": {"false": "calm", "true": "rush"}},
     512, 192, False),
    ({"body": "many options"}, {"type": "choice", "instructions": "Which of the many categories applies here?",
                                "criteria": {"cat%d" % i: "category number %d with a long explanation " % i * 4
                                             for i in range(15)}}, 512, 96, False),
    ({"body": "list crit"}, {"type": "choice", "instructions": "?", "criteria": ["a", "b", "c"]}, 64, 32, False),
]


@pytest.mark.parametrize("state,spec,max_len,head_max_len,left", CASES)
def test_classic_row_matches_laya(tk, snapshot, state, spec, max_len, head_max_len, left):
    common = pytest.importorskip("laya.common")
    from transformers import AutoTokenizer
    ltok = AutoTokenizer.from_pretrained(snapshot)
    want_ids, want_mk = common.build_sequence(ltok, state, _laya_q(spec), max_len=max_len,
                                              head_max_len=head_max_len, truncate_left=left)
    ids, mk = classic_row(tk, _ids(tk, state), parse_question("q", spec), max_len, head_max_len, left)
    assert ids == want_ids
    assert mk == want_mk
