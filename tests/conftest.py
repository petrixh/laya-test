import os
import time

import httpx
import pytest

BASE_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
READY_TIMEOUT = float(os.environ.get("LAYA_READY_TIMEOUT", "900"))


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    """Block until the checkpoint is loaded so no test races the first download."""
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as c:
        deadline = time.monotonic() + READY_TIMEOUT
        last = None
        while time.monotonic() < deadline:
            try:
                r = c.get("/readyz")
                if r.status_code == 200:
                    yield c
                    return
                last = r.text
                if r.json().get("status") == "failed":
                    pytest.fail(f"checkpoint load failed: {last}")
            except httpx.HTTPError as exc:
                last = str(exc)
            time.sleep(3)
        pytest.fail(f"service not ready within {READY_TIMEOUT}s; last={last}")


@pytest.fixture(scope="session")
def predict(client):
    def _predict(state, questions, expect=200):
        r = client.post("/predict", json={"state": state, "questions": questions})
        assert r.status_code == expect, f"{r.status_code}: {r.text[:400]}"
        return r.json()

    return _predict
