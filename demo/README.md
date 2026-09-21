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
| obstacles classified | 131, accuracy **1.00** (jump 44/44, duck 39/39, block 48/48) |
| lane choices by the model | 12 |
| took the first of the two lanes offered | **12 / 12** |
| of those, the first lane was a barrier | 3 |
| crashes | **3** — the same three |
| furthest run | 488m |
| latency p50 | 168ms wall, ~30ms of it inference |

Every crash traces to the lane question; reading obstacles did not fail once. The
offered pair alternates between `[left, right]` and `[middle, right]` depending on where
the reindeer is standing, so always taking the first is a position preference rather
than a lane preference. `trace.json` rows tagged `stage: "lane"` show each one.
