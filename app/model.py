"""Checkpoint loading and thread-safe inference."""
from __future__ import annotations

import importlib.util
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("laya.model")


# The MLX port ships each checkpoint as its own repo rather than as subfolders,
# so LAYA_SUBFOLDER keeps working across both backends.
MLX_REPOS = {
    None: "aac6fef/laya-mlx",
    "multilingual": "aac6fef/laya-multilingual-mlx",
    "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
}


def _resolve_backend(requested: str) -> str:
    """'auto' prefers MLX on Apple silicon, where it is roughly 30x faster than
    the torch CPU path on the game prompt, and falls back to torch elsewhere."""
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("laya_mlx") is not None:
            return "mlx"
    return "torch"


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
    backend_requested: str = "auto"
    dtype: str = "float16"

    agent: Any = None
    backend: str | None = None
    device: str | None = None
    torch_version: str | None = None
    runtime_version: str | None = None
    load_seconds: float | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def ready(self) -> bool:
        return self.agent is not None

    def load(self) -> None:
        """Blocking load. Called once from a background thread at startup."""
        try:
            self.backend = _resolve_backend(self.backend_requested)
            t0 = time.monotonic()
            if self.backend == "mlx":
                self.agent = self._load_mlx()
            else:
                self.agent = self._load_torch()
            self.load_seconds = time.monotonic() - t0
            log.info("ready in %.1fs (%s on %s)", self.load_seconds, self.backend, self.device)
        except Exception as exc:  # surfaced via /readyz rather than crashing the process
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("checkpoint load failed")

    def _load_torch(self):
        import torch

        self.torch_version = torch.__version__
        self.runtime_version = f"torch {torch.__version__}"
        if self.threads > 0:
            torch.set_num_threads(self.threads)
        self.device = _resolve_device(self.device_requested)

        import laya

        log.info("loading %s (subfolder=%s) on %s", self.checkpoint, self.subfolder, self.device)
        return laya.load(self.checkpoint, device=self.device, subfolder=self.subfolder)

    def _load_mlx(self):
        """Apple silicon path. laya_mlx is an independent FP16 port with the same
        predict() contract, so nothing downstream changes -- but the weights are
        not bit-identical to the fp32 originals, which is why /info reports the
        backend and every result file records it."""
        import laya_mlx
        import mlx.core  # noqa: F401  -- fail loudly here rather than at predict time

        repo = self.checkpoint
        if repo == DEFAULT_CHECKPOINT:
            if self.subfolder not in MLX_REPOS:
                raise ValueError(
                    f"no MLX repo for subfolder {self.subfolder!r}; "
                    f"known: {sorted(k for k in MLX_REPOS if k)}"
                )
            repo = MLX_REPOS[self.subfolder]
        self.device = "mlx"
        self.runtime_version = f"laya-mlx {getattr(laya_mlx, '__version__', 'unknown')}"

        log.info("loading %s via MLX (dtype=%s)", repo, self.dtype)
        return laya_mlx.load(repo, dtype=self.dtype)

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], float]:
        """Serialised inference. The agent holds one set of module buffers, so we
        do not let two forward passes touch it at once."""
        if self.agent is None:
            raise RuntimeError("model not loaded")
        with self._lock:
            t0 = time.monotonic()
            result = self.agent.predict(state, questions)
            return result, (time.monotonic() - t0) * 1000.0


DEFAULT_CHECKPOINT = "convaiinnovations/laya"


def state_from_env() -> ModelState:
    return ModelState(
        checkpoint=os.environ.get("LAYA_CHECKPOINT", DEFAULT_CHECKPOINT),
        subfolder=os.environ.get("LAYA_SUBFOLDER") or None,
        device_requested=os.environ.get("LAYA_DEVICE", "auto"),
        threads=int(os.environ.get("TORCH_NUM_THREADS", "0")),
        backend_requested=os.environ.get("LAYA_BACKEND", "auto"),
        dtype=os.environ.get("LAYA_DTYPE", "float16"),
    )
