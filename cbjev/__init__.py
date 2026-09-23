"""cbjev: typed decisions (choice / score / noul) from one encoder forward pass."""
__version__ = "0.1.0"

from .agent import Agent, load  # noqa: F401
from .layout import parse_question  # noqa: F401
from .email import email_state, clean_email_body  # noqa: F401
from .presets import (  # noqa: F401
    email_questions, guard_questions, moderation_questions, router_questions, triage_questions,
)
from . import presets  # noqa: F401

# light, but kept lazy so `import cbjev` stays cheap; none of these import torch
_LAZY = {"Router": "router", "Route": "router", "route": "router", "analyse": "lang"}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError("module 'cbjev' has no attribute %r" % name)
    import importlib
    value = getattr(importlib.import_module("." + mod, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = ["Agent", "load", "parse_question", "email_state", "clean_email_body", "presets",
           "router_questions", "guard_questions", "moderation_questions", "triage_questions",
           "email_questions", *_LAZY]
