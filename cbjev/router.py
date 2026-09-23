"""`Router`: pick the English or the multilingual checkpoint per request, load it on demand.

    r = cbjev.Router()
    r.predict({"message": "I was charged twice"}, qs)                # -> english
    r.predict({"message": "Mein Konto wurde zweimal belastet"}, qs)  # -> multilingual

Precedence: explicit ``model=`` > ``lang=`` > ``lang_guess=`` (per call, then the Router's) >
detection (`cbjev.lang`) > ``default``. Detection never loads anything; at most ``max_loaded``
agents are kept, least recently used first out.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import lang as _lang

DEFAULT_MODELS = {"english": "cbjev", "multilingual": "cbjev-multilingual"}
_ALIASES = {"en": "english", "eng": "english", "default": "english",
            "multi": "multilingual", "ml": "multilingual", "xx": "multilingual"}


@dataclass
class Route:
    model: str                                  # a key of Router.models
    reason: str
    analysis: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def as_dict(self) -> Dict[str, str]:
        return {"model": self.model, "reason": self.reason}


class Router:
    """Lazily built agents keyed by model name, and the rule that picks one per request.

    `models` maps a name to anything `loader` accepts (a checkpoint name, path or hub id).
    `loader(checkpoint)` builds an Agent; the default is ``cbjev.load(checkpoint, device=device)``.
    `lang_guess` is an optional default hint: a language code or a callable ``state -> code``.
    """

    def __init__(self, models: Optional[Dict[str, Any]] = None, default: str = "english", max_loaded: int = 2,
                 device: Optional[str] = None, loader: Optional[Callable[[Any], Any]] = None,
                 lang_guess: Any = None):
        self.models = dict(DEFAULT_MODELS if models is None else models)
        if not self.models:
            raise ValueError("Router needs at least one model")
        if int(max_loaded) < 1:
            raise ValueError("max_loaded must be >= 1")
        self.max_loaded = int(max_loaded)
        self.device = device
        self.lang_guess = lang_guess
        self._loader = loader
        self._agents: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()
        self.default = self.resolve(default)

    # ------------------------------------------------------------------ names and agents

    def resolve(self, model: str) -> str:
        """Normalise a model name, alias or checkpoint to a key of `models` (ValueError if unknown)."""
        key = str(model).strip()
        if key in self.models:
            return key
        low = key.lower()
        if low in self.models:
            return low
        alias = _ALIASES.get(low)
        if alias in self.models:
            return alias
        for k, ckpt in self.models.items():         # a checkpoint name picks its model
            if isinstance(ckpt, str) and ckpt == key:
                return k
        raise ValueError("unknown model %r; known: %s" % (model, ", ".join(sorted(self.models))))

    def _build(self, key: str):
        ckpt = self.models[key]
        if self._loader is not None:
            return self._loader(ckpt)
        from .agent import load
        return load(ckpt, device=self.device)

    def get(self, model: str):
        """The agent for `model`, loading it (and evicting the least recently used) if needed."""
        key = self.resolve(model)
        with self._lock:
            agent = self._agents.get(key)
            if agent is not None:
                self._agents.move_to_end(key)
                return agent
            # free room first, so two big checkpoints are never resident at once past the cap
            self._evict(self.max_loaded - 1)
            agent = self._build(key)
            self._agents[key] = agent
            return agent

    load = get

    def attach(self, name: str, agent: Any) -> None:
        """Serve an already built agent under `name` (added to `models` if new)."""
        with self._lock:
            if name not in self.models:
                self.models[name] = getattr(agent, "path", name)
            old = self._agents.pop(name, None)
            if old is not None and old is not agent:
                _close(old)
            self._evict(self.max_loaded - 1)
            self._agents[name] = agent

    def preload(self, names: Optional[Sequence[str]] = None) -> List[str]:
        keys = [self.resolve(n) for n in (names if names is not None else self.models)]
        if len(set(keys)) > self.max_loaded:
            raise ValueError("cannot preload %d models with max_loaded=%d" % (len(set(keys)), self.max_loaded))
        for k in keys:
            self.get(k)
        return self.loaded

    def unload(self, name: Optional[str] = None) -> None:
        with self._lock:
            keys = list(self._agents) if name is None else [self.resolve(name)]
            for k in keys:
                agent = self._agents.pop(k, None)
                if agent is not None:
                    _close(agent)

    close = unload

    def _evict(self, keep: int) -> None:
        while len(self._agents) > keep:
            _, agent = self._agents.popitem(last=False)
            _close(agent)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._agents)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.unload()
        return False

    def __repr__(self):
        return "Router(models=%r, loaded=%r, default=%r)" % (list(self.models), self.loaded, self.default)

    # ------------------------------------------------------------------ routing

    def _pick(self, english: bool) -> str:
        want = "english" if english else "multilingual"
        return want if want in self.models else self.default

    def route(self, state: Any, questions: Optional[Dict[str, Any]] = None, model: Optional[str] = None,
              lang: Optional[str] = None, lang_guess: Any = None) -> Route:
        """Decide which model answers; loads nothing. `questions` is accepted for symmetry."""
        if model is not None and str(model).strip():
            return Route(self.resolve(model), "explicit model=%r" % (model,))
        hint = _lang.english_code(lang)
        if hint is not None:
            return Route(self._pick(hint), "explicit lang=%r" % (lang,))
        for src, guess in (("lang_guess", lang_guess), ("Router.lang_guess", self.lang_guess)):
            if guess is None:
                continue
            code = guess(state) if callable(guess) else guess
            hint = _lang.english_code(code)
            if hint is not None:
                return Route(self._pick(hint), "%s said %r" % (src, code))
        a = _lang.analyse(state)
        if a["script"] == "unknown":
            return Route(self.default, "no letters in state; default", a)
        if a["script"] != "latin":
            return Route(self._pick(False), "%s script" % a["script"], a)
        if a["language"] == "en":
            return Route(self._pick(True), "English text", a)
        if a["language"]:
            return Route(self._pick(False), "Latin script, looks like %r" % a["language"], a)
        if not a["is_english"]:
            return Route(self._pick(False), "Latin script, language unclear but not English", a)
        return Route(self.default, "language undecided; default", a)

    # ------------------------------------------------------------------ predicting

    def predict(self, state: Any, questions: Dict[str, Any], model: Optional[str] = None,
                lang: Optional[str] = None, lang_guess: Any = None, hooks=None) -> Dict[str, Any]:
        r = self.route(state, questions, model=model, lang=lang, lang_guess=lang_guess)
        agent = self.get(r.model)
        res = agent.predict(state, questions) if hooks is None else agent.predict(state, questions, hooks=hooks)
        res = dict(res)
        res["routing"] = r.as_dict()
        return res

    system_one = predict

    def predict_batch(self, requests: Sequence[Dict[str, Any]], batch_size: Optional[int] = None
                      ) -> List[Dict[str, Any]]:
        """Many requests, each ``{"state", "questions"[, "model", "lang", "lang_guess"]}``.

        Requests are grouped by model and then by question set, so each group is one
        `Agent.predict_batch`; results come back in input order, each with its ``routing``.
        """
        routes: List[Route] = []
        for i, req in enumerate(requests):
            if not isinstance(req, dict) or not isinstance(req.get("questions"), dict):
                raise ValueError("request %d must be an object with 'state' and a 'questions' object" % i)
            routes.append(self.route(req.get("state"), req["questions"], model=req.get("model"),
                                     lang=req.get("lang"), lang_guess=req.get("lang_guess")))
        groups: "OrderedDict[tuple, List[int]]" = OrderedDict()
        for i, (req, r) in enumerate(zip(requests, routes)):
            sig = json.dumps(req["questions"], sort_keys=True, ensure_ascii=False, default=str)
            groups.setdefault((r.model, sig), []).append(i)
        out: List[Optional[Dict[str, Any]]] = [None] * len(requests)
        kw = {} if batch_size is None else {"batch_rows": int(batch_size)}
        for (key, _), idx in groups.items():
            agent = self.get(key)
            res = agent.predict_batch([requests[i].get("state") for i in idx], requests[idx[0]]["questions"], **kw)
            if len(res) != len(idx):
                raise RuntimeError("agent %r returned %d results for %d states" % (key, len(res), len(idx)))
            for i, r in zip(idx, res):
                r = dict(r)
                r["routing"] = routes[i].as_dict()
                out[i] = r
        return out  # type: ignore[return-value]


def _close(agent: Any) -> None:
    fn = getattr(agent, "close", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


_default: Optional[Router] = None


def route(state: Any, questions: Optional[Dict[str, Any]] = None, **kw) -> Route:
    """Routing decision with the default Router (loads nothing)."""
    global _default
    if _default is None:
        _default = Router()
    return _default.route(state, questions, **kw)
