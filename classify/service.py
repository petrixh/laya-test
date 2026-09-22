"""HTTP client for the Laya service.

Deliberately depends on nothing but httpx: the suite talks to the service only
through its public contract, so it cannot accidentally test a second, local
implementation of the model instead of the real one.

The endpoint comes from LAYA_URL and defaults to the local container. No host
is hard-coded anywhere in this repository -- to run against a faster machine,
set LAYA_URL in your environment.
"""
from __future__ import annotations

import os
import time

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"


def url_from_env() -> str:
    return os.environ.get("LAYA_URL", DEFAULT_URL)


class Service:
    def __init__(self, url: str | None = None, timeout: float = 300.0):
        self.url = url or url_from_env()
        self.client = httpx.Client(base_url=self.url, timeout=timeout)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.client.close()

    def wait_ready(self, timeout: float | None = None) -> dict:
        timeout = timeout if timeout is not None else float(
            os.environ.get("LAYA_READY_TIMEOUT", "900"))
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                r = self.client.get("/readyz")
                if r.status_code == 200:
                    return self.info()
                last = r.text
            except httpx.HTTPError as exc:
                last = str(exc)
            time.sleep(3)
        raise RuntimeError(f"service at {self.url} never became ready; last={last}")

    def info(self) -> dict:
        return self.client.get("/info").json()

    def predict(self, state, questions) -> dict:
        r = self.client.post("/predict", json={"state": state, "questions": questions})
        r.raise_for_status()
        self.calls += 1
        return r.json()


def run_identity(info: dict, **extra) -> str:
    """Filename stem encoding everything that makes two runs incomparable.

    Carried over from the abandoned branch together with its lesson: an n=120
    run once silently overwrote an n=150 baseline that differed only in that.
    Backend belongs in the stem too, because the MLX port is an independent
    FP16 conversion rather than the same numbers on faster hardware.
    """
    sub = info.get("subfolder") or "base"
    stem = f"{sub}-{info.get('backend') or 'torch'}-{info.get('device', 'cpu')}"
    for key, value in extra.items():
        if value is not None:
            stem += f"-{key}{value}"
    return stem
