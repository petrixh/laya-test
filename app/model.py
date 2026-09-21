"""Checkpoint loading and thread-safe inference."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("laya.model")


def _resolve_device(requested: str) -> str:
    """'auto' picks the best available accelerator. cuda covers ROCm builds too."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class ModelState:
    checkpoint: str
    subfolder: str | None
    device_requested: str
    threads: int

    agent: Any = None
    device: str | None = None
    torch_version: str | None = None
    load_seconds: float | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def ready(self) -> bool:
        return self.agent is not None

    def load(self) -> None:
        """Blocking load. Called once from a background thread at startup."""
        try:
            import torch

            self.torch_version = torch.__version__
            if self.threads > 0:
                torch.set_num_threads(self.threads)
            self.device = _resolve_device(self.device_requested)

            import laya

            log.info("loading %s (subfolder=%s) on %s", self.checkpoint, self.subfolder, self.device)
            t0 = time.monotonic()
            agent = laya.load(self.checkpoint, device=self.device, subfolder=self.subfolder)
            self.load_seconds = time.monotonic() - t0
            self.agent = agent
            log.info("ready in %.1fs", self.load_seconds)
        except Exception as exc:  # surfaced via /readyz rather than crashing the process
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("checkpoint load failed")

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], float]:
        """Serialised inference. The agent holds one set of module buffers, so we
        do not let two forward passes touch it at once."""
        if self.agent is None:
            raise RuntimeError("model not loaded")
        with self._lock:
            t0 = time.monotonic()
            result = self.agent.predict(state, questions)
            return result, (time.monotonic() - t0) * 1000.0


def state_from_env() -> ModelState:
    return ModelState(
        checkpoint=os.environ.get("LAYA_CHECKPOINT", "convaiinnovations/laya"),
        subfolder=os.environ.get("LAYA_SUBFOLDER") or None,
        device_requested=os.environ.get("LAYA_DEVICE", "auto"),
        threads=int(os.environ.get("TORCH_NUM_THREADS", "0")),
    )
