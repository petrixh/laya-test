# Recorded run

One run of the autopilot against Laya on the MLX backend, so the result can be seen
without standing the service up.

- `reindeer.gif` — nine seconds of it, the version embedded in the README
- `run.webm` — the whole 90-second run
- `summary.json` — headline figures
- `trace.json` — every decision: the obstacle, what the model answered, the
  probabilities, server and wall latency. Rows tagged `stage: "lane"` are lane choices.
- `deaths.json` — one row per crash, with the verdict that preceded it, so a death after
  a *correct* reading (an execution bug) is distinguishable from one after a wrong
  reading (a model error)

| | |
|---|---|
| obstacles classified | 150, accuracy **1.00** (jump 59/59, duck 52/52, block 39/39) |
| lane questions | 12 — 9 decided by the model, 3 exact ties broken by the harness |
| of those, chose a barrier | **0** |
| crashes | **0** |
| furthest run | **1117m** |
| obstacle question, p50 | 239ms wall |
| lane question, p50 | 193ms wall |

Rows tagged `stage: "lane"` in `trace.json` carry `blocked_scores` — the model's
per-lane score for each candidate — so every lane change can be checked against the
numbers it was made from.
