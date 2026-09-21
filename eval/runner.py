"""Drives the Laya service over HTTP and collects per-example records."""
from __future__ import annotations

import math
import sys
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


def run_choice_task(svc: Service, examples: list[dict], labels: list[str],
                    instructions: str, key: str = "intent", progress: bool = True) -> list[dict]:
    """examples: [{text, label_text}]. One choice question over `labels`."""
    from .data import humanise

    questions = {key: {
        "type": "choice",
        "instructions": instructions,
        "criteria": {lab: humanise(lab) for lab in labels},
    }}

    records = []
    for i, ex in enumerate(examples, 1):
        res = svc.predict(ex["text"], questions)
        ans = res["answers"][key]
        probs = ans["probabilities"]
        records.append({
            "text": ex["text"],
            "gold": ex["label_text"],
            "pred": ans["choice"],
            "probabilities": probs,
            "confidence": ans["confidence"],
            "top_prob": probs[ans["choice"]],
            "latency_ms": res["latency_ms"],
            "input_tokens": res.get("usage", {}).get("input_tokens"),
        })
        if progress and (i % 25 == 0 or i == len(examples)):
            acc = sum(r["gold"] == r["pred"] for r in records) / len(records)
            print(f"    {i}/{len(examples)}  running acc={acc:.3f}", file=sys.stderr, flush=True)
    return records


def kendall_tau(ranks: list[int], values: list[float]) -> float:
    """Concordance between the intended ordering and the model's scores.
    +1 = perfectly ordered, 0 = unrelated, -1 = exactly reversed."""
    n = len(ranks)
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (ranks[i] - ranks[j]) * (values[i] - values[j])
            if a > 0:
                con += 1
            elif a < 0:
                dis += 1
    total = con + dis
    return (con - dis) / total if total else 0.0


def balanced_for_k(rows: list[dict], labels: list[str], n: int, seed: int = 0) -> list[dict]:
    """Roughly balanced sample of total size ~n across `labels`."""
    from .data import balanced_sample

    per = max(1, math.ceil(n / len(labels)))
    sample = balanced_sample(rows, labels, per, seed=seed)
    return sample[:n]
