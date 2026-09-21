"""Latency sanity check over HTTP. Not a pass/fail test -- CPU numbers vary
wildly by host, so this reports and leaves judgement to the reader.

  docker compose --profile test run --rm --entrypoint python tests bench.py
"""
import os
import statistics
import time

import httpx

from fixtures import ROUTING_CASES, TRIAGE_QUESTIONS

BASE_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
N = int(os.environ.get("BENCH_N", "20"))
STATE = ROUTING_CASES[0][1]
ONE_Q = {"department": TRIAGE_QUESTIONS["department"]}


def run(client, questions, n, label):
    server = []
    wall = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/predict", json={"state": STATE, "questions": questions})
        wall.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        server.append(r.json()["latency_ms"])
    q = statistics.quantiles(wall, n=20)
    print(
        f"{label:<26} n={n:<3} "
        f"p50={statistics.median(wall):7.1f}ms  p95={q[18]:7.1f}ms  "
        f"min={min(wall):7.1f}ms  model_p50={statistics.median(server):7.1f}ms"
    )
    return statistics.median(server)


def main():
    with httpx.Client(base_url=BASE_URL, timeout=300.0) as c:
        info = c.get("/info").json()
        print(f"device={info['device']} torch={info['torch']} threads={info['threads']} "
              f"checkpoint={info['checkpoint']}\n")

        for _ in range(3):  # warm
            c.post("/predict", json={"state": STATE, "questions": ONE_Q})

        one = run(c, ONE_Q, N, "1 question")
        three = run(c, TRIAGE_QUESTIONS, N, "3 questions (1 pass)")

        big = {f"q{i}": TRIAGE_QUESTIONS["department"] for i in range(10)}
        ten = run(c, big, max(5, N // 2), "10 questions (1 pass)")

        print(f"\nper-question cost: 1q={one:.1f}ms  3q={three/3:.1f}ms  10q={ten/10:.1f}ms")
        print("batching amortises correctly" if ten / 10 < one else "WARNING: batching gives no benefit")


if __name__ == "__main__":
    main()
