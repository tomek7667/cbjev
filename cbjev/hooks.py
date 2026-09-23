"""Call hooks: observe, rewrite or short-circuit a prediction without subclassing.

A hook is any object with some of ``on_start(ctx)``, ``on_end(ctx)``, ``on_error(ctx)``, or a
bare callable (treated as ``on_end``). ``on_start`` may edit ``ctx.states`` / ``ctx.questions``
or set ``ctx.results`` to skip the model (a cache); ``on_end`` may replace ``ctx.results``.
With no hooks installed nothing extra runs.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CallContext:
    model: str
    states: List[Any]
    questions: Dict[str, Any]
    agent: Any = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[BaseException] = None
    route: Optional[Dict[str, Any]] = None
    started: float = field(default_factory=time.perf_counter)
    elapsed_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class _Fn:
    def __init__(self, fn):
        self.on_end = fn


class Hooks:
    def __init__(self, items=()):
        self.items = list(items)

    @classmethod
    def coerce(cls, hooks) -> "Hooks":
        if hooks is None:
            return cls()
        if isinstance(hooks, Hooks):
            return cls(hooks.items)
        if not isinstance(hooks, (list, tuple)):
            hooks = [hooks]
        return cls(h if any(hasattr(h, n) for n in ("on_start", "on_end", "on_error")) else _Fn(h)
                   for h in hooks)

    def plus(self, more) -> "Hooks":
        return self if more is None else Hooks(self.items + Hooks.coerce(more).items)

    def _fire(self, name: str, ctx: CallContext):
        for h in self.items:
            fn = getattr(h, name, None)
            if fn is not None:
                fn(ctx)

    @contextmanager
    def around(self, ctx: CallContext):
        if not self.items:
            yield ctx
            return
        try:
            self._fire("on_start", ctx)
            yield ctx
        except BaseException as e:
            ctx.error = e
            self._fire("on_error", ctx)
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started) * 1000
            if ctx.error is None:
                self._fire("on_end", ctx)
