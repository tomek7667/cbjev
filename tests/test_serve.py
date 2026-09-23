import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cbjev import serve  # noqa: E402
from cbjev.router import Router  # noqa: E402

Q = {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}


class FakeAgent:
    def __init__(self, name):
        self.name = name

    def predict(self, state, questions):
        return self.predict_batch([state], questions)[0]

    def predict_batch(self, states, questions, **kw):
        return [{"model": self.name, "answers": {k: {"type": "noul", "noul": 0.5, "confidence": 0.5}
                                                 for k in questions},
                 "usage": {"input_tokens": 3, "output_tokens": 0}} for _ in states]


class Boom(FakeAgent):
    def predict_batch(self, states, questions, **kw):
        raise RuntimeError("CUDA OOM at /home/secret/weights/model.safetensors")


def make(env=None, monkeypatch=None, agent=FakeAgent, **cfg):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    router = Router(loader=agent)
    return TestClient(serve.create_app(router, config=cfg or None)), router


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("CBJEV_API_KEY", "CBJEV_MAX_BODY", "CBJEV_MAX_QUESTIONS", "CBJEV_PORT", "CBJEV_PRELOAD"):
        monkeypatch.delenv(k, raising=False)


def test_happy_path():
    c, router = make()
    r = c.post("/v1/systemone", json={"state": {"message": "Mein Konto wurde zweimal belastet, bitte helfen Sie"},
                                      "questions": Q, "model": "jev-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"model", "answers", "usage", "routing"}
    assert body["model"] == "cbjev-multilingual" and body["routing"]["model"] == "multilingual"
    assert body["answers"]["angry"]["type"] == "noul"
    r = c.post("/v1/systemone", json={"state": "hallo", "questions": Q, "model": "english"})
    assert r.json()["routing"]["model"] == "english"
    h = c.get("/health").json()
    assert h["status"] == "ok" and set(h["loaded"]) == {"english", "multilingual"}


def test_batch():
    c, _ = make()
    reqs = [{"state": "I was charged twice, please refund me now", "questions": Q},
            {"state": "我想退款", "questions": Q},
            {"state": "x", "questions": {}}]
    r = c.post("/v1/systemone/batch", json={"requests": reqs})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert [x["routing"]["model"] for x in res] == ["english", "multilingual", "english"]
    assert c.post("/v1/systemone/batch", json={"nope": 1}).status_code == 400
    assert c.post("/v1/systemone/batch", json={"requests": [{"state": "x"}]}).status_code == 400


def test_auth(monkeypatch):
    c, _ = make({"CBJEV_API_KEY": "s3cret"}, monkeypatch)
    body = {"state": "hi", "questions": Q}
    assert c.post("/v1/systemone", json=body).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "s3cret"}).status_code == 401
    assert c.post("/v1/systemone/batch", json={"requests": []}).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_bad_json():
    c, _ = make()
    assert c.post("/v1/systemone", content=b"{not json").status_code == 400
    assert c.post("/v1/systemone", content=b"\xff\xfe").status_code == 400
    assert c.post("/v1/systemone", content=b"").status_code == 400
    assert c.post("/v1/systemone", json=[1, 2]).status_code == 400
    assert c.post("/v1/systemone", json={"state": "x"}).status_code == 400
    assert c.post("/v1/systemone", json={"state": "x", "questions": [1]}).status_code == 400


def test_too_large(monkeypatch):
    c, _ = make({"CBJEV_MAX_BODY": "200", "CBJEV_MAX_QUESTIONS": "2"}, monkeypatch)
    r = c.post("/v1/systemone", json={"state": "x" * 500, "questions": Q})
    assert r.status_code == 413
    # no Content-Length (streamed body): the limit still holds
    r = c.post("/v1/systemone", content=iter([json.dumps({"state": "x" * 500, "questions": Q}).encode()]))
    assert r.status_code == 413
    many = {"q%d" % i: dict(Q["angry"]) for i in range(3)}
    c2, _ = make(max_questions=2)
    assert c2.post("/v1/systemone", json={"state": "x", "questions": many}).status_code == 413
    assert c2.post("/v1/systemone/batch", json={"requests": [{"state": "x", "questions": many}]}).status_code == 413


def test_invalid_question_is_422():
    c, _ = make()
    r = c.post("/v1/systemone", json={"state": "x", "questions": {"dept": {"type": "choice", "instructions": "?"}}})
    assert r.status_code == 422
    assert "dept" in r.json()["detail"]
    r = c.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}})
    assert r.status_code == 422


def test_errors_do_not_leak():
    c, _ = make(agent=Boom)
    r = c.post("/v1/systemone", json={"state": "hello there", "questions": Q})
    assert r.status_code == 500
    assert "/home" not in r.text and "safetensors" not in r.text


def test_settings(monkeypatch):
    monkeypatch.setenv("CBJEV_PORT", "70000")
    with pytest.raises(ValueError):
        serve.settings()
    monkeypatch.setenv("CBJEV_PORT", "abc")
    with pytest.raises(ValueError):
        serve.settings()
    monkeypatch.setenv("CBJEV_PORT", "9001")
    s = serve.settings()
    assert s["port"] == 9001 and s["host"] == "127.0.0.1" and s["max_body"] == 2 * 1024 * 1024
    assert s["max_questions"] == 64 and s["preload"] is False
