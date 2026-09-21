"""Minimal load-and-predict check. Run inside the container before trusting anything else."""
import json, os, time, sys

t0 = time.time()
import torch
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
import laya
print(f"[smoke] imports ok in {time.time()-t0:.1f}s  torch={torch.__version__}", flush=True)

ckpt = os.environ.get("LAYA_CHECKPOINT", "convaiinnovations/laya")
sub = os.environ.get("LAYA_SUBFOLDER") or None
t0 = time.time()
agent = laya.load(ckpt, subfolder=sub) if sub else laya.load(ckpt)
print(f"[smoke] loaded {ckpt} subfolder={sub} in {time.time()-t0:.1f}s", flush=True)

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the extra charge or we will have to look at other vendors.",
}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does user threaten to cancel?"},
}

t0 = time.time()
res = agent.predict(state, questions)
first = time.time() - t0
lat = []
for _ in range(5):
    t = time.time(); agent.predict(state, questions); lat.append((time.time() - t) * 1000)

print(f"[smoke] first predict {first*1000:.0f}ms; warm {min(lat):.0f}-{max(lat):.0f}ms", flush=True)
print(json.dumps(res, indent=2, default=str))
