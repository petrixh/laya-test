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
```

`scripts/introspect.py` (`make introspect`) is the diagnostic to re-run after a laya
version bump. It is how we found that `laya.load()` accepts `device=` and that
responses carry `action.act_probability` and `usage` fields -- none of which the
README or model card mention. Cheap insurance against coding to stale docs.
