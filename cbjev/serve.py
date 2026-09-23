"""HTTP server speaking the Jev ``/v1/systemone`` wire format, backed by a `Router`.

    pip install "cbjev[serve]"
    CBJEV_API_KEY=secret cbjev-serve

Environment: CBJEV_HOST (127.0.0.1), CBJEV_PORT (8000), CBJEV_DEVICE, CBJEV_PRELOAD=1,
CBJEV_API_KEY (require ``Authorization: Bearer <key>``), CBJEV_MAX_BODY (bytes, 2 MB),
CBJEV_MAX_QUESTIONS (64 per request), CBJEV_MAX_BATCH (256 requests per batch call).
"""

import asyncio
import hmac
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from .layout import parse_question

_MISSING = 'cbjev.serve needs FastAPI and uvicorn: pip install "cbjev[serve]"'


def _int_env(name: str, default: int, lo: int = 1, hi: Optional[int] = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        v = int(raw.strip())
    except ValueError:
        v = lo - 1
    if v < lo or (hi is not None and v > hi):
        raise ValueError("%s must be an integer in %d..%s, got %r" % (name, lo, hi if hi else "inf", raw))
    return v


def settings() -> Dict[str, Any]:
    """Server settings from the environment (ValueError on a bad number)."""
    return {
        "host": os.environ.get("CBJEV_HOST") or "127.0.0.1",
        "port": _int_env("CBJEV_PORT", 8000, 1, 65535),
        "device": os.environ.get("CBJEV_DEVICE") or None,
        "preload": (os.environ.get("CBJEV_PRELOAD") or "").strip() == "1",
        "api_key": os.environ.get("CBJEV_API_KEY") or None,
        "max_body": _int_env("CBJEV_MAX_BODY", 2 * 1024 * 1024),
        "max_questions": _int_env("CBJEV_MAX_QUESTIONS", 64),
        "max_batch": _int_env("CBJEV_MAX_BATCH", 256),
    }


def build_router(cfg: Optional[Dict[str, Any]] = None):
    from .router import Router
    cfg = cfg or settings()
    router = Router(device=cfg["device"])
    if cfg["preload"]:
        router.preload()
    return router


def create_app(router: Any = None, config: Optional[Dict[str, Any]] = None):
    """The FastAPI app. Pass `router` (anything with route/predict/predict_batch) to inject one."""
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(_MISSING) from e

    cfg = dict(settings(), **(config or {}))
    if router is None:
        router = build_router(cfg)
    key = cfg["api_key"].encode() if cfg["api_key"] else None
    # inference is blocking; one worker keeps it off the event loop and runs calls one at a time
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cbjev-infer")

    app = FastAPI(title="cbjev", summary="Typed System-1 decisions over the Jev /v1/systemone protocol")

    def check_auth(request: Request) -> None:
        if key is None:
            return
        got = (request.headers.get("authorization") or "").encode("utf-8", "replace")
        if not hmac.compare_digest(got, b"Bearer " + key):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token",
                                headers={"WWW-Authenticate": "Bearer"})

    async def read_json(request: Request) -> Any:
        cl = request.headers.get("content-length")
        if cl and cl.strip().isdigit() and int(cl) > cfg["max_body"]:
            raise HTTPException(status_code=413, detail="request body too large")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > cfg["max_body"]:
                raise HTTPException(status_code=413, detail="request body too large")
        try:
            return json.loads(raw)
        except ValueError:              # JSONDecodeError and UnicodeDecodeError
            raise HTTPException(status_code=400, detail="request body must be valid JSON")

    def check_request(req: Any, where: str = "request") -> Dict[str, Any]:
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="%s must be a JSON object" % where)
        qs = req.get("questions")
        if not isinstance(qs, dict):
            raise HTTPException(status_code=400, detail="%s needs a 'questions' object" % where)
        if len(qs) > cfg["max_questions"]:
            raise HTTPException(status_code=413, detail="%s has %d questions, the limit is %d"
                                % (where, len(qs), cfg["max_questions"]))
        try:
            for qid, spec in qs.items():
                parse_question(qid, spec)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        out = {"state": req.get("state"), "questions": qs}
        model = _model_for(router, req.get("model"))
        if model:
            out["model"] = model
        for k in ("lang", "lang_guess"):
            if isinstance(req.get(k), str) and req[k].strip():
                out[k] = req[k]
        return out

    async def run(fn):
        try:
            return await asyncio.get_running_loop().run_in_executor(pool, fn)
        except HTTPException:
            raise
        except Exception:                # never echo internals (paths, CUDA errors) to clients
            raise HTTPException(status_code=500, detail="inference failed")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "loaded": list(getattr(router, "loaded", []) or []),
                "models": sorted(getattr(router, "models", {}) or {})}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        check_auth(request)
        req = check_request(await read_json(request))
        state, qs = req.pop("state"), req.pop("questions")
        return await run(lambda: router.predict(state, qs, **req))

    @app.post("/v1/systemone/batch")
    async def systemone_batch(request: Request):
        check_auth(request)
        body = await read_json(request)
        reqs = body.get("requests") if isinstance(body, dict) else None
        if not isinstance(reqs, list):
            raise HTTPException(status_code=400, detail="request body must be an object with a 'requests' list")
        if len(reqs) > cfg["max_batch"]:
            raise HTTPException(status_code=413, detail="batch has %d requests, the limit is %d"
                                % (len(reqs), cfg["max_batch"]))
        clean = [check_request(r, "requests[%d]" % i) for i, r in enumerate(reqs)]
        results = await run(lambda: router.predict_batch(clean))
        return {"results": results}

    app.state.router = router
    app.state.config = cfg
    return app


def _model_for(router: Any, model: Any) -> Optional[str]:
    """A client's `model` as a Router model name, or None to auto-route (Jev clients send e.g. 'jev-1')."""
    if not isinstance(model, str) or not model.strip():
        return None
    resolve = getattr(router, "resolve", None)
    if resolve is None:
        return model if model in (getattr(router, "models", None) or {}) else None
    try:
        return resolve(model)
    except ValueError:
        return None


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(_MISSING)
    try:
        cfg = settings()
    except ValueError as e:
        raise SystemExit("cbjev-serve: %s" % e)
    uvicorn.run(create_app(config=cfg), host=cfg["host"], port=cfg["port"],
                log_level=os.environ.get("CBJEV_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
