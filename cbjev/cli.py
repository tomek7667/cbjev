"""Command line: see where a text would be routed, or run the triage preset on it.

    cbjev "Mein Konto wurde zweimal belastet"          # routing decision only, no model load
    cbjev --predict "I was charged twice, refund me"   # load, answer the triage questions, print JSON
    cbjev                                              # interactive: one text per line
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cbjev", description="Typed decisions from one encoder pass.")
    p.add_argument("text", nargs="*", help="text to route (omit for an interactive prompt)")
    p.add_argument("--predict", action="store_true", help="load the routed model and answer the triage preset")
    p.add_argument("--model", help="force a model (english, multilingual, or a checkpoint name)")
    p.add_argument("--lang", help="language code of the text (en, de, pt_BR, ...)")
    p.add_argument("--device", help="torch device for --predict (cpu, cuda, cuda:1, mps)")
    return p


def _state(text: str):
    return {"message": text}


def run(text: str, args, router, out=sys.stdout) -> dict:
    state = _state(text)
    r = router.route(state, model=args.model, lang=args.lang)
    if not args.predict:
        a = r.analysis or {}
        info = {"model": r.model, "reason": r.reason}
        if a:
            info.update(script=a.get("script"), language=a.get("language"))
        print(json.dumps(info, ensure_ascii=False), file=out)
        return info
    from .presets import triage_questions
    res = router.predict(state, triage_questions(), model=r.model)
    print(json.dumps(res, ensure_ascii=False, indent=2), file=out)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from .router import Router
    router = Router(device=args.device)
    try:
        if args.model:
            router.resolve(args.model)          # fail early on a typo
    except ValueError as e:
        print("cbjev: %s" % e, file=sys.stderr)
        return 2
    try:
        if args.text:
            run(" ".join(args.text), args, router)
            return 0
        interactive = sys.stdin.isatty()
        while True:
            if interactive:
                print("> ", end="", flush=True)
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line in ("exit", "quit") and interactive:
                break
            if line:
                run(line, args, router)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        router.unload()


if __name__ == "__main__":
    sys.exit(main())
