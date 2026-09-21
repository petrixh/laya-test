"""HTTP client for the Laya service.

Everything else that lived here -- the Banking77 task runner, the ladder
statistics -- went with the scripts that used it."""
from __future__ import annotations

import time

import httpx


class Service:
    def __init__(self, url: str, timeout: float = 300.0):
        self.client = httpx.Client(base_url=url, timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.client.close()

    def wait_ready(self, timeout: float = 900.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.client.get("/readyz").status_code == 200:
                    return self.info()
            except httpx.HTTPError:
                pass
            time.sleep(3)
        raise RuntimeError("service never became ready")

    def info(self) -> dict:
        return self.client.get("/info").json()

    def predict(self, state, questions) -> dict:
        r = self.client.post("/predict", json={"state": state, "questions": questions})
        r.raise_for_status()
        return r.json()
