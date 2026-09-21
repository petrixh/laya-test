"""HTTP surface for the Laya typed-decision model.

The upstream package ships no server, so this is a thin wrapper: load one
checkpoint at startup, expose a single forward pass over HTTP, and report
readiness honestly so callers never race the (slow) first load.
"""
from __future__ import annotations

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
        "device": model.device,
        "device_requested": model.device_requested,
        "torch": model.torch_version,
        "threads": model.threads,
        "ready": model.ready,
        "load_seconds": model.load_seconds,
        "question_types": list(QTYPES),
    }


@app.get("/presets/{name}")
def presets(name: str) -> dict[str, Any]:
    """Expose laya's built-in question sets so tests and demos share one source."""
    import laya

    fn = getattr(laya, f"{name}_questions", None)
    if fn is None:
        raise HTTPException(404, f"unknown preset '{name}'")
    return {"name": name, "questions": fn()}


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
    return {**result, "latency_ms": round(latency, 1)}
