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
- **Soft churn signals are underweighted.** "Please refund or we will have to look at
  other vendors" scored `churn_risk = 0.13`. Explicit cancellation language scores high,
  hints do not. Worth probing before trusting it on subtle intent.
- `score.confidence` is not comparable to `choice.confidence` — a 0.11 there reflects a
  spread-out ordinal distribution, not a broken answer.

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
Dockerfile          base (deps) + runtime stages; TORCH_INDEX build arg picks CPU/CUDA/ROCm
Dockerfile.tests    test runner; no torch, no laya -- HTTP contract only
compose.yml         service + weights volume + `test` profile
app/model.py        checkpoint loading, device resolution, serialised inference
app/main.py         FastAPI surface and request validation
tests/test_api.py   service / correctness / contract / determinism
tests/bench.py      latency report
scripts/smoke.py    in-process load + predict, bypasses HTTP
```
