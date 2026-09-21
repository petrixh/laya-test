"""Eval CLI.

  python -m eval.run labels   --k 4,8,16,32,77 --n 150   # label-budget curve
  python -m eval.run probes                              # graded intensity ladders
  python -m eval.run all
Results land in /results/<checkpoint>__<task>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys

from .data import fetch_split, pick_labels
from .metrics import summarise
from .probes import run_all as run_probes
from .runner import Service, balanced_for_k, run_choice_task

RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results"))
URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")

INSTRUCTIONS = "Which banking intent does this customer message express?"


def tag(info: dict) -> str:
    sub = info.get("subfolder") or "base"
    return f"{sub}-{info.get('device', 'cpu')}"


def write(name: str, payload: dict) -> pathlib.Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"  -> {path}", file=sys.stderr)
    return path


def task_labels(svc: Service, info: dict, ks: list[int], n: int, seed: int) -> dict:
    rows = fetch_split("mteb/banking77")
    out = {"task": "label_budget", "dataset": "mteb/banking77", "n_per_k": n,
           "seed": seed, "service": info, "generated": dt.datetime.now(dt.UTC).isoformat(),
           "results": {}}
    for k in ks:
        labels = pick_labels(rows, k, seed=seed)
        examples = balanced_for_k(rows, labels, n, seed=seed)
        print(f"  k={k:<3} labels={len(labels)} examples={len(examples)}", file=sys.stderr, flush=True)
        records = run_choice_task(svc, examples, labels, INSTRUCTIONS)
        summary = summarise(records)
        summary["k"] = k
        summary["random_baseline"] = round(1.0 / k, 4)
        out["results"][str(k)] = summary
        print(f"  k={k:<3} acc={summary['accuracy']:.3f} "
              f"(chance {summary['random_baseline']:.3f}) ece={summary['ece_confidence']:.3f} "
              f"tokens={summary['mean_input_tokens']}", file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["labels", "probes", "all"])
    ap.add_argument("--k", default="4,8,16,32,77")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with Service(URL) as svc:
        info = svc.wait_ready()
        name = tag(info)
        print(f"service: checkpoint={info['checkpoint']} subfolder={info.get('subfolder')} "
              f"device={info['device']}", file=sys.stderr)

        if args.task in ("labels", "all"):
            ks = [int(x) for x in args.k.split(",")]
            write(f"{name}__labels", task_labels(svc, info, ks, args.n, args.seed))

        if args.task in ("probes", "all"):
            print("  probes: churn + urgency ladders", file=sys.stderr)
            write(f"{name}__probes", {"task": "probes", "service": info,
                                      "generated": dt.datetime.now(dt.UTC).isoformat(),
                                      "results": run_probes(svc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
