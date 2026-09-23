import pytest

from cbjev.router import Route, Router

Q = {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}
Q2 = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]}}
EN = {"message": "I was charged twice for my subscription, please refund me."}
DE = {"message": "Mein Konto wurde zweimal belastet, bitte erstatten Sie mir das Geld."}
ZH = {"message": "我想取消我的订阅"}


class FakeAgent:
    def __init__(self, ckpt):
        self.name = ckpt
        self.calls = []
        self.closed = False

    def _one(self, state, questions):
        return {"model": self.name, "answers": {k: {"state": state} for k in questions},
                "usage": {"input_tokens": 1, "output_tokens": 0}}

    def predict(self, state, questions, hooks=None):
        self.calls.append(("one", [state], questions))
        return self._one(state, questions)

    def predict_batch(self, states, questions, batch_rows=32):
        self.calls.append(("batch", list(states), questions))
        return [self._one(s, questions) for s in states]

    def close(self):
        self.closed = True


class Loader:
    def __init__(self):
        self.built = []

    def __call__(self, ckpt):
        a = FakeAgent(ckpt)
        self.built.append(a)
        return a


@pytest.fixture
def router():
    return Router(loader=Loader())


def test_route_detection(router):
    assert router.route(EN).model == "english"
    assert router.route(DE).model == "multilingual"
    assert router.route(ZH).model == "multilingual"
    assert router.route("Hi").model == "english"            # undecided -> default
    assert router.route("12345").model == "english"
    assert Router(default="multilingual", loader=Loader()).route("Hi").model == "multilingual"
    r = router.route(DE)
    assert isinstance(r, Route) and r.analysis["language"] == "de" and "de" in r.reason
    assert router._loader.built == []                       # routing loads nothing


def test_precedence(router):
    assert router.route(DE, model="english").model == "english"
    assert router.route(DE, model="cbjev").model == "english"          # checkpoint name
    assert router.route(EN, model="en", lang="de").model == "english"  # model beats lang
    assert router.route(EN, lang="de").model == "multilingual"
    assert router.route(DE, lang="en_US").model == "english"
    assert router.route(DE, lang="eng", lang_guess="de").model == "english"   # lang beats guess
    assert router.route(EN, lang_guess="pt").model == "multilingual"
    assert router.route(DE, lang_guess=lambda s: "english").model == "english"
    # an abstaining guess falls through to detection
    assert router.route(DE, lang_guess=lambda s: None).model == "multilingual"
    assert router.route(DE, lang_guess="").model == "multilingual"
    assert router.route(DE, lang="").model == "multilingual"
    with pytest.raises(ValueError):
        router.route(EN, model="nope")


def test_router_level_guess():
    r = Router(loader=Loader(), lang_guess=lambda s: "en")
    assert r.route(DE).model == "english"
    assert r.route(EN, lang_guess="de").model == "multilingual"      # per call first


def test_custom_models():
    r = Router(models={"english": "a", "multilingual": "b", "fast": "c"}, loader=Loader())
    assert r.route(EN, model="fast").model == "fast"
    assert r.predict(EN, Q, model="fast")["model"] == "c"
    solo = Router(models={"english": "a"}, loader=Loader())
    assert solo.route(DE).model == "english"                # no multilingual: fall back to default


def test_predict_adds_routing(router):
    res = router.predict(DE, Q)
    assert res["model"] == "cbjev-multilingual"
    assert res["routing"]["model"] == "multilingual" and res["routing"]["reason"]
    assert set(res) >= {"model", "answers", "usage", "routing"}


def test_lru(router):
    router.max_loaded = 1
    router.predict(EN, Q)
    first = router._loader.built[0]
    router.predict(DE, Q)
    assert router.loaded == ["multilingual"] and first.closed
    router.max_loaded = 2
    router.predict(EN, Q)
    router.predict(DE, Q)
    assert router.loaded == ["english", "multilingual"]
    assert len(router._loader.built) == 3


def test_preload_attach_unload(router):
    assert router.preload() == ["english", "multilingual"]
    with pytest.raises(ValueError):
        Router(models={"a": 1, "b": 2, "c": 3}, default="a", loader=Loader()).preload()
    mine = FakeAgent("mine")
    router.attach("english", mine)
    assert router.get("english") is mine
    router.unload("english")
    assert mine.closed and router.loaded == ["multilingual"]
    with Router(loader=Loader()) as r:
        a = r.get("english")
    assert a.closed and r.loaded == []


def test_predict_batch_groups_and_order(router):
    reqs = [
        {"state": EN, "questions": Q},
        {"state": DE, "questions": Q},
        {"state": {"message": "Please refund the double charge on my card."}, "questions": Q},
        {"state": EN, "questions": Q2},
        {"state": ZH, "questions": Q},
        {"state": DE, "questions": Q, "model": "english"},
        {"state": EN, "questions": dict(reversed(list({**Q, **Q2}.items())))},
        {"state": DE, "questions": {**Q, **Q2}, "lang": "en"},
    ]
    out = router.predict_batch(reqs, batch_size=4)
    assert len(out) == len(reqs)
    for req, res in zip(reqs, out):
        assert all(a["state"] == req["state"] for a in res["answers"].values())
        assert set(res["answers"]) == set(req["questions"])
    assert [r["routing"]["model"] for r in out] == [
        "english", "multilingual", "english", "english", "multilingual", "english", "english", "english"]
    en, ml = {a.name: a for a in router._loader.built}["cbjev"], {a.name: a for a in router._loader.built}[
        "cbjev-multilingual"]
    # english: Q x4 (0, 2, 5), Q2 x1, Q+Q2 in either key order -> 3 calls; multilingual: one call
    assert len(en.calls) == 3 and len(ml.calls) == 1
    assert all(kind == "batch" for kind, _, _ in en.calls + ml.calls)
    assert sorted(len(states) for _, states, _ in en.calls) == [1, 2, 3]
    assert len(ml.calls[0][1]) == 2


def test_predict_batch_empty_and_invalid(router):
    assert router.predict_batch([]) == []
    with pytest.raises(ValueError):
        router.predict_batch([{"state": "x"}])
