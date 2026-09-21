"""HTTP surface for the Laya typed-decision model.

The upstream package ships no server, so this is a thin wrapper: load one
checkpoint at startup, expose a single forward pass over HTTP, and report
readiness honestly so callers never race the (slow) first load.
"""
from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .model import ModelState, state_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("laya.api")

# Request/response tracing. Off by default and gated on a plain boolean, so the
# hot path costs one attribute lookup when it is not in use.
#   LAYA_LOG_IO=compact   one line in, one line out (good for watching a game)
#   LAYA_LOG_IO=full      the whole request and response as indented JSON
IO_MODE = os.environ.get("LAYA_LOG_IO", "").strip().lower()
IO_LOG = IO_MODE in ("1", "true", "compact", "full")
IO_FULL = IO_MODE == "full"
IO_STATE_CHARS = int(os.environ.get("LAYA_LOG_IO_CHARS", "160"))
iolog = logging.getLogger("laya.io")


def _brief(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    text = " ".join(text.split())
    return text if len(text) <= IO_STATE_CHARS else text[: IO_STATE_CHARS - 1] + "\u2026"


def _answer_brief(key: str, ans: dict[str, Any]) -> str:
    kind = ans.get("type")
    if kind == "choice":
        probs = ans.get("probabilities") or {}
        top = ans.get("choice")
        return f"{key}={top} p={probs.get(top, 0):.3f} conf={ans.get('confidence', 0):.3f}"
    if kind == "score":
        return f"{key}={ans.get('score', 0):.3f}"
    if kind == "noul":
        return f"{key}={ans.get('noul', 0):.3f}"
    return f"{key}={ans.get('choice', ans)}"


def trace_io(state: Any, questions: dict[str, Any], result: dict[str, Any],
             latency_ms: float) -> None:
    if IO_FULL:
        iolog.info("--> %s", json.dumps({"state": state, "questions": questions}, indent=2))
        iolog.info("<-- %s", json.dumps(result, indent=2))
        return
    asked = ", ".join(
        f"{k}:{v.get('type')}" + (f"({len(v.get('criteria') or [])})" if v.get("criteria") else "")
        for k, v in questions.items())
    iolog.info("--> %s   [%s]", _brief(state), asked)
    answers = result.get("answers") or {}
    iolog.info("<-- %s   %.1fms",
               " | ".join(_answer_brief(k, v) for k, v in answers.items()), latency_ms)

QTYPES = ("choice", "score", "noul")

model: ModelState = state_from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load off-thread so /healthz answers immediately and orchestrators can tell
    # "starting" apart from "dead" during the ~3min first-run weight download.
    threading.Thread(target=model.load, name="laya-load", daemon=True).start()
    yield


app = FastAPI(title="laya-service", version="0.1.0", lifespan=lifespan)

# The game autopilot runs inside the browser page and calls /predict directly,
# so the service has to be reachable cross-origin. This is a local dev service
# with no auth and no side effects; if that ever changes, narrow this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("LAYA_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Question(BaseModel):
    type: Literal["choice", "score", "noul"]
    instructions: str = Field(min_length=1)
    # choice -> {label: description}; score -> [level, ...]; noul -> absent
    criteria: dict[str, str] | list[str] | None = None

    @field_validator("criteria")
    @classmethod
    def _check(cls, v, info):
        qtype = info.data.get("type")
        if qtype == "choice":
            if not isinstance(v, dict) or len(v) < 2:
                raise ValueError("choice requires a criteria object with at least 2 labels")
        elif qtype == "score":
            if not isinstance(v, list) or len(v) < 2:
                raise ValueError("score requires a criteria list with at least 2 levels")
        return v


class PredictRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)


class PredictResponse(BaseModel):
    model_config = {"extra": "allow"}
    answers: dict[str, Any]
    latency_ms: float


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness only: the process is up. Says nothing about the checkpoint."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    if model.ready:
        return JSONResponse({"status": "ready", "load_seconds": model.load_seconds})
    if model.error:
        return JSONResponse({"status": "failed", "error": model.error}, status_code=503)
    return JSONResponse({"status": "loading"}, status_code=503)


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "checkpoint": model.checkpoint,
        "subfolder": model.subfolder,
        # backend matters for reading results: the MLX path is an independent
        # FP16 port, so its numbers are not interchangeable with torch fp32.
        "backend": model.backend,
        "backend_requested": model.backend_requested,
        "runtime": model.runtime_version,
        "dtype": model.dtype if model.backend == "mlx" else "float32",
        "device": model.device,
        "device_requested": model.device_requested,
        "torch": model.torch_version,
        "threads": model.threads,
        "ready": model.ready,
        "load_seconds": model.load_seconds,
        "question_types": list(QTYPES),
    }


# Both runtimes ship the same preset helpers, under different package names.
PRESET_PACKAGES = ("laya", "laya_mlx")
KNOWN_PRESETS = ("router", "guard", "moderation", "triage", "email")


@functools.lru_cache(maxsize=1)
def _preset_module():
    for name in PRESET_PACKAGES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


@app.get("/presets/{name}")
def presets(name: str) -> dict[str, Any]:
    """Expose the built-in question sets so tests and demos share one source.

    Resolved against whichever runtime is installed: the macOS venv has
    laya_mlx and no laya, so importing `laya` unconditionally 500s there.
    """
    mod = _preset_module()
    if mod is not None:
        fn = getattr(mod, f"{name}_questions", None)
        if fn is None:
            raise HTTPException(404, f"unknown preset '{name}'")
        return {"name": name, "questions": fn()}
    if name in KNOWN_PRESETS:
        raise HTTPException(503, f"no laya runtime installed to supply preset '{name}'")
    raise HTTPException(404, f"unknown preset '{name}'")


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> dict[str, Any]:
    if not model.ready:
        raise HTTPException(503, model.error or "model still loading")
    questions = {k: v.model_dump(exclude_none=True) for k, v in req.questions.items()}
    try:
        result, latency = await anyio.to_thread.run_sync(model.predict, req.state, questions)
    except Exception as exc:
        log.exception("inference failed")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    if IO_LOG:
        # after the forward pass, so formatting never sits inside the lock
        trace_io(req.state, questions, result, latency)
    return {**result, "latency_ms": round(latency, 1)}
