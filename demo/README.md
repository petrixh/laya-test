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
| obstacles classified | 130, accuracy **1.00** (jump 44/44, duck 43/43, block 43/43) |
| lane choices by the model | 10 |
| chose the barrier it was standing in | **5** |
| took the first option offered | 90% |
| crashes | 4 |
| furthest run | 322m |
| latency p50 | 249ms wall, ~30ms of it inference |

Every crash traces to the lane question. Reading obstacles did not fail once.
