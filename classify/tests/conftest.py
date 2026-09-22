import os

import pytest

from classify.service import Service


@pytest.fixture(scope="session")
def svc():
    """A ready service. Skips rather than fails when none is reachable, so the
    offline tests still run on a machine with no model."""
    s = Service()
    try:
        info = s.wait_ready(timeout=float(os.environ.get("LAYA_READY_TIMEOUT", "900")))
    except Exception as exc:
        s.client.close()
        pytest.skip(f"no laya service at {s.url}: {exc}")
    s.info_cached = info
    yield s
    s.client.close()
