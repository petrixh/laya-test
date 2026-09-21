# laya-test

Containerised [Laya](https://github.com/NandhaKishorM/laya) — Convai's Apache-2.0,
421M-param non-autoregressive "System 1" decision model, the open counterpart to
TypeSafe AI's closed Jev.

You give it a state plus typed questions; one forward pass returns a calibrated
probability distribution per question. Nothing is generated token by token, so
there is no JSON to repair and no parse step.

Upstream ships a Python library only — no server, no Dockerfile, no CLI. This repo
adds the container and the proof that it works.

## Quick start

```bash
make up      # first run downloads ~1.7GB of weights into the laya-models volume
make ready   # blocks until the checkpoint is loaded
make test    # 22 HTTP tests
make bench   # latency at 1 / 3 / 10 questions per pass
```

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "state": {"subject": "Duplicate charge on invoice #4411",
            "body": "We were billed twice for March. Please refund."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages",
                                "sales": "pricing, contracts",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["not urgent", "soon", "critical"]},
    "churn_risk": {"type": "noul", "instructions": "Threatening to cancel?"}
  }}' | jq
```

## Endpoints

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness. Answers immediately, says nothing about the model. |
| `GET /readyz` | 200 only once the checkpoint is loaded; 503 while `loading`, 503 + error if `failed`. |
| `GET /info` | Checkpoint, resolved device, torch version, thread count, load time. |
| `GET /presets/{name}` | Built-in question sets: `router`, `guard`, `moderation`, `triage`, `email`. |
| `POST /predict` | `{state, questions}` → laya's result plus `latency_ms`. |

Question types: `choice` (label + per-option probabilities + confidence),
`score` (expected level on an ordinal rubric), `noul` (calibrated P(true)).

## Design notes

- **Weights are not baked into the image.** They live in the external `laya-models`
  volume mounted at `HF_HOME=/models`, so the image stays ~1.5GB and rebuilds cost
  seconds. First load ~3min (download), subsequent loads ~20s.
- **`transformers>=4.48` is pinned.** laya declares `>=4.45`, but ModernBERT support
  only landed in 4.48.
- **`USE_TF=0 USE_JAX=0`** — the model card warns `laya.load()` can deadlock while
  transformers probes for TensorFlow.
- **One checkpoint, loaded eagerly, in a background thread.** `/healthz` answers
  during the load so orchestrators can distinguish starting from dead. `Router(preload=True)`
  is avoided deliberately: it pulls several checkpoints and will OOM a small box.
- **Inference is serialised** behind a lock and run off the event loop, since one
  agent instance holds a single set of module buffers.
- **Docker `HEALTHCHECK` targets `/readyz`**, so `--wait` means "model ready", not
  "port open".

## Measured on this host (8-core aarch64 CPU, 4 torch threads)

| Questions per pass | p50 total | per question |
|---|---|---|
| 1 | 278ms | 278ms |
| 3 | 713ms | 237ms |
| 10 | 2287ms | 229ms |

Single-question latency sits in the published CPU band (193–464ms).

**Batching barely helps on CPU** — 278ms → 229ms per question, about 1.2x. On a T4
upstream reports 33ms → 6.8ms, roughly 5x, because there the cost is kernel-launch
overhead rather than compute. Anything that needs many questions per decision is
GPU-shaped work.

## Behaviour observed

- Routing is confident and correct on clear cases (`billing` at p=0.97).
- `score` and `noul` order correctly: urgent > casual, explicit cancellation > happy
  customer. Tests assert the *ordering*, not absolute thresholds.
- Output is bit-exact deterministic across repeats, as expected with no sampling.
- Answers do not change when other questions are added to the same pass.
- **`noul` keys on vocabulary, not intent strength.** The churn ladder scores
  tau=0.72 with five inversions. "We are no longer sure we will renew" scores **0.923**,
  higher than "We are cancelling our subscription effective immediately" (0.769), while
  "we have started evaluating other vendors" -- a clear churn signal with no
  cancel/renew vocabulary -- scores 0.299. This is lexical triggering, not graded
  intent. Do not put a threshold on `noul` for subtle intent without probing your own
  phrasings first.
- **`score` is well behaved.** The urgency ladder is perfectly ordered (tau=1.0, no
  inversions, 0.46 -> 1.85). Ordinal scoring looks trustworthy where boolean does not.
- `score.confidence` is not comparable to `choice.confidence` — a 0.11 there reflects a
  spread-out ordinal distribution, not a broken answer.

## Evaluation

```bash
make eval          # probes + label-budget curve against the running checkpoint
make eval-typed    # same, against the laya-typed-decisions checkpoint
make report        # render everything in results/ as comparison tables
```

Two tasks, both aimed at the failure modes the smoke tests cannot see.

**Label-budget curve** (`eval/run.py labels`) runs Banking77 as a single `choice`
question at k = 4, 8, 16, 32, 77 labels. Label subsets are nested and seeded, so
the curve isolates *label count* rather than which labels were drawn. Reports
accuracy, lift over chance, macro-F1, ECE, Brier and mean input tokens -- the last
matters because the English checkpoint budgets only ~192 tokens for options, so
large label sets get truncated.

**Graded ladders** (`eval/run.py probes`) score hand-written cases ordered by
intended intensity and measure Kendall tau against that order. A single
assert-cancel-beats-happy test passes even when the middle of the range is
scrambled; the ladder catches it.

Macro-F1 sits alongside accuracy deliberately: with balanced sampling they track
each other, but collapse onto a majority label shows up in F1 first.

### Results: label-budget curve (base checkpoint, 150 examples per point)

| k labels | chance | accuracy | lift | macro-F1 | ECE | mean conf | input tokens | p50 |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.250 | 0.913 | 3.7x | 0.911 | 0.187 | 0.726 | 82 | 269ms |
| 8 | 0.125 | 0.920 | 7.4x | 0.920 | **0.057** | 0.929 | 128 | 386ms |
| 16 | 0.063 | 0.787 | 12.6x | 0.777 | 0.208 | 0.995 | 186 | 506ms |
| 32 | 0.031 | 0.540 | 17.3x | 0.518 | 0.444 | 0.984 | 189 | 510ms |
| 77 | 0.013 | 0.380 | 29.2x | 0.288 | **0.609** | 0.989 | 333 | 880ms |

Our k=77 result (0.380) lands near the published Banking77 figure (0.425), which is a
useful sanity check on the harness.

The `laya-typed-decisions` checkpoint is indistinguishable on this task
(0.908 / 0.925 / 0.783 at k = 4 / 8 / 16, against 0.913 / 0.920 / 0.787 for base,
n=120), and it behaves identically on the game probes. Nothing here separates the
two checkpoints.

**Two things to take from this.**

*Accuracy has a cliff between 8 and 16 labels.* Up to k=8 it is ~0.92; by k=32 it is
0.54. Mean input tokens cross the English checkpoint's ~192-token option budget right
where the cliff starts, so this reads as option truncation, not a reasoning limit.

*Calibration fails worse than accuracy, and in the dangerous direction.* ECE goes
0.057 at k=8 to 0.609 at k=77 while mean confidence stays pinned near 0.99 and accuracy
falls to 0.38. The model is not merely wrong at high label counts, it is wrong while
reporting ~99% confidence. Since calibrated probability is the whole pitch, this is the
single most important caveat here: **`confidence` is only usable as a gate at small
label counts.** Above ~16 labels it carries close to no information.

Practical consequence: keep `choice` questions at 8 labels or fewer and route
hierarchically (coarse choice, then a fine choice within the winning group) rather than
presenting one flat label set. Two cheap forward passes beat one truncated one.

## Playing Reindeer Jump

`index.html` already exposes `window.__rj` for automation, so nothing in the game
needed changing. The autopilot is injected alongside it.

```bash
make agent-deps          # once: link playwright
make play-mock           # prove the rig with a fake decision service
make play                # autopilot plays against the live model, records video
make play-watch          # open a real browser and watch (needs a display)
```

Artifacts land in `runs/<name>/`: `run.webm`, `summary.json`, `trace.json`,
`deaths.json`.

**Where this runs.** The agent lives *in the page* (`agent/autopilot.js`), so the
only thing it needs is an HTTP route to the service -- which is why the service
sends CORS headers. To watch it yourself, open the page in a normal browser and
point it at the container; no driver process is involved. `agent/play.mjs` is
for the headless case: it serves the page, injects the agent, records video and
writes the trace, which is what makes it runnable in CI. One agent, two ways in.

**The model decides what, the harness decides when.** Inference costs ~280ms on
CPU while the collision box is only 65ms wide at top speed, so acting on the
response the moment it lands cannot work. Instead each obstacle is classified
once, early -- obstacles are visible 2.8-5.9s out -- and the verdict is cached and
fired on a timer as the obstacle arrives.

**CPU is fast enough, but only just, and only with lookahead.** The game needs
**0.6-1.5 decisions per second**. The real game prompt costs **940ms p50**, not
the 278ms the Banking77 bench showed -- the scene state and three long option
descriptions are several times more tokens. That leaves about 1.06 decisions per
second, inside the requirement but not comfortably.

At that latency a perfect oracle still crashed **11 times in 30**, because the
autopilot only classified the *next* obstacle and its verdict landed after
impact. Classifying every obstacle inside a 2.6s horizon as it enters, up to
three in flight, took that to **0 crashes over 30 decisions, 556m**. Obstacles
are independent questions; pipelining them is what makes CPU viable.

**Every run is self-labelling.** Ground truth comes from the game's own collision
geometry (`def.kind === 'air'` means duck, otherwise jump), so accuracy, latency
and per-class breakdown are computed with no hand labelling. `deaths.json`
separates deaths that followed a *correct* verdict -- an execution bug -- from
deaths that followed a wrong one, which is a model error. That distinction found
a real bug: the autopilot dropped an obstacle from consideration at `z >= 0`,
but the game's collision box extends to `z = +1.1`, so it released its duck
while still inside a garland it had correctly classified.

The state encoding is switchable (`--encoding json|nl`) so the JSON-dict and
natural-language framings can be compared on the same task.

### Result: Laya cannot play this game

40 decisions on the base checkpoint, CPU:

| | |
|---|---|
| accuracy | **0.35** (chance 0.33) |
| ground obstacles | **0 / 26** |
| air obstacles | **14 / 14** |
| mean confidence | 0.13 |

**It answers `duck` to everything**, so 0.35 is simply the base rate of air
obstacles. All ten crashes are attributed to wrong verdicts and none to latency,
so this is the model, not the rig. Recorded run and stills in `demo/`.

`python -m eval.game_probes` runs the diagnostic ladder that locates the failure:

| rung | accuracy | mean confidence |
|---|---|---|
| 1 control (sentiment) | **1.00** | 0.795 |
| 2 action -- the question the autopilot asks | **0.20** | 0.084 |
| 3 position -- "where is the obstacle?" | **1.00** | 0.580 |
| 4 knowledge -- object name only | 0.60 | 0.103 |

Rung 1 clears the pipeline of suspicion. Rung 2 is *below* chance at near-zero
confidence. Rung 3 shows it can classify the scene when the position is stated in
the text -- so it can read, it just cannot turn "the obstacle is on the ground"
into "jump". Rung 4 gives only the object's name and it answers `on_the_ground`
for all five, including the garland and the baubles: no world knowledge that a
garland hangs, which means rung 3 was paraphrase matching rather than inference.
The typed-decisions checkpoint behaves the same way.

**The useful part is the confidence.** Mean 0.13 here against 0.73 on Banking77
at four labels, and 0.795 on the sentiment control. The model is not confidently
wrong -- it is abstaining, correctly. That is the exact complement of the
label-budget finding above, where at k=77 it was 0.99 confident and 38% accurate.
Taken together: confidence collapses honestly when the model cannot do a task,
and lies only when the option list is truncated. A confidence gate would have
caught this failure before it reached the game.

### The telemetry panel

`agent/hud.js` draws the live side panel: the action being executed, per-action
probability bars, an inference-latency sparkline with p50/p95, running accuracy,
distance and crash count, and a rolling decision log. Colours are the validated
dark-mode categorical palette, checked against the panel's own surface rather
than a generic dark background:

```
node scripts/validate_palette.js "#3987e5,#d95926,#199e70" \
     --mode dark --surface "#0d1630" --pairs all     # all checks pass
```

Correct/incorrect in the log carries a glyph as well as colour, and every action
bar is directly labelled, so nothing depends on colour alone. The panel is a live
readout rather than an explorable chart, so there is no hover layer; `trace.json`
is the table view.

## Running on Apple silicon (MLX)

There is an independent MLX port, [`laya-mlx`](https://github.com/mizorewww/laya-mlx),
with the same `predict()` contract. It reports **13.4ms p50** on an M3 Max for the
English checkpoint and 7.4ms multilingual, against 940ms on the CPU container
here. `app/model.py` will use it automatically on Apple silicon.

Docker is not involved: MLX needs Metal, which containers on macOS cannot reach.
So this runs natively in a venv.

```bash
./scripts/serve-macos.sh                         # creates .venv, installs, serves on 0.0.0.0:8000
LAYA_SUBFOLDER=multilingual ./scripts/serve-macos.sh
```

or by hand:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mlx.txt              # laya-mlx + fastapi, no torch
LAYA_BACKEND=mlx python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Requires macOS 14+ and Python 3.11+. `LAYA_BACKEND` is `auto` by default, which
picks MLX on Apple silicon when `laya_mlx` imports and torch everywhere else;
force it with `mlx` or `torch`. `LAYA_SUBFOLDER` works on both backends -- the MLX
port publishes each checkpoint as its own repo, and `MLX_REPOS` in `app/model.py`
maps them.

**Check which backend answered.** `GET /info` reports `backend`, `runtime` and
`dtype`. This matters: the MLX port is an independent **FP16** conversion, so its
answers are not guaranteed bit-identical to the fp32 originals and its numbers
should not be pooled with the CPU results in this README without re-running.
The eval harness is the way to check -- point it at the Mac and compare:

```bash
LAYA_URL=http://<mac>:8000 python -m eval.run labels --k 4,8,16,32,77
LAYA_URL=http://<mac>:8000 python -m eval.game_probes
```

At 13ms a question the full Banking77 curve takes about a minute rather than
the twenty it takes on CPU.

### Pointing the game at any service

The page boots the autopilot itself when `?autopilot=1` is present, so no driver
process is needed and you can watch it in a real browser:

```bash
python3 -m http.server 8080          # serve the repo (not file://, the agent scripts need an origin)
```

then open:

```
http://localhost:8080/index.html?autopilot=1&laya=http://<mac>:8000
```

Query parameters: `laya` (service URL, default `http://127.0.0.1:8000`),
`encoding` (`json` or `nl`), `decisions` (stop after N, `0` for unlimited).
The service sends permissive CORS headers; narrow them with `LAYA_CORS_ORIGINS`
if it is ever exposed beyond a LAN.

`node agent/check-page.mjs --endpoint http://<mac>:8000` verifies that path
end to end -- loader, endpoint wiring, HUD, and a live decision -- without
recording anything.

**A faster server will not make it play better.** The failure documented above
is accuracy, not latency: the oracle control already scores 30/30 with zero
crashes at 975ms. MLX buys a watchable frame rate and much quicker eval sweeps,
not a working player.

## Running on a GPU

`laya.load()` takes `device` directly and `LAYA_DEVICE=auto` already resolves to
`cuda` when it is available, so it is a build arg plus a device reservation:

```bash
# NVIDIA (needs nvidia-container-toolkit)
TORCH_INDEX=https://download.pytorch.org/whl/cu124 LAYA_TAG=cu124 docker compose build
# AMD
TORCH_INDEX=https://download.pytorch.org/whl/rocm6.2 LAYA_TAG=rocm docker compose build
```

Then uncomment the `deploy.resources.reservations.devices` block in `compose.yml`.
`GET /info` reports the device actually in use — check it rather than assuming.

## Layout

```
Dockerfile            base (deps) + runtime stages; TORCH_INDEX build arg picks CPU/CUDA/ROCm
Dockerfile.tests      test/eval runner; no torch, no laya -- HTTP contract only
compose.yml           service + weights volume + `test` and `eval` profiles
app/model.py          checkpoint loading, device resolution, serialised inference
app/main.py           FastAPI surface and request validation
tests/test_api.py     service / correctness / contract / determinism (22 tests)
tests/test_metrics.py unit tests for the eval metrics; needs no service (8 tests)
tests/bench.py        latency report
eval/data.py          dataset fetch + caching + nested label subsets
eval/metrics.py       accuracy, macro-F1, ECE, Brier, reliability buckets
eval/probes.py        graded intensity ladders
eval/runner.py        HTTP driver, Kendall tau
eval/run.py           eval CLI
eval/report.py        renders results/ as comparison tables
scripts/smoke.py      in-process load + predict, bypasses HTTP
scripts/introspect.py prints laya's real signatures and return shape
scripts/serve-macos.sh  native Apple-silicon service (MLX backend, no Docker)
requirements-mlx.txt  macOS dependency set: laya-mlx + fastapi, no torch
agent/check-page.mjs  verifies the ?autopilot=1 loader against a live service
index.html            the game (upstream); exposes window.__rj for automation
agent/autopilot.js    in-page agent: reads game state, asks Laya, acts, self-grades
agent/hud.js          live telemetry panel
agent/play.mjs        Playwright driver: serves, injects, records video, writes trace
```

`scripts/introspect.py` (`make introspect`) is the diagnostic to re-run after a laya
version bump. It is how we found that `laya.load()` accepts `device=` and that
responses carry `action.act_probability` and `usage` fields -- none of which the
README or model card mention. Cheap insurance against coding to stale docs.
