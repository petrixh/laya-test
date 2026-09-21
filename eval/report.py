"""Render every JSON in results/ as comparison tables.

  python -m eval.report
"""
from __future__ import annotations

import json
import os
import pathlib

RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results"))


def load() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}


def label_budget_table(runs: dict[str, dict]) -> None:
    runs = {k: v for k, v in runs.items() if v.get("task") == "label_budget"}
    if not runs:
        return
    print("\n=== Label-budget curve (Banking77, accuracy / macro-F1 / ECE) ===\n")
    ks = sorted({int(k) for r in runs.values() for k in r["results"]})
    name_w = max(len(n) for n in runs) + 2
    print(f"{'run':<{name_w}}" + "".join(f"k={k:<14}" for k in ks))
    for name, run in runs.items():
        cells = ""
        for k in ks:
            s = run["results"].get(str(k))
            cells += f"{s['accuracy']:.3f}/{s['macro_f1']:.3f}    " if s else " " * 16
        print(f"{name:<{name_w}}{cells}")
    print()
    for name, run in runs.items():
        print(f"  {name}")
        print(f"    {'k':>4} {'chance':>7} {'acc':>7} {'lift':>7} {'macroF1':>8} "
              f"{'ECE':>6} {'Brier':>6} {'meanConf':>9} {'tokens':>7} {'p50ms':>7}")
        for k in ks:
            s = run["results"].get(str(k))
            if not s:
                continue
            lift = s["accuracy"] / s["random_baseline"]
            print(f"    {k:>4} {s['random_baseline']:>7.3f} {s['accuracy']:>7.3f} "
                  f"{lift:>6.1f}x {s['macro_f1']:>8.3f} {s['ece_confidence']:>6.3f} "
                  f"{s['brier']:>6.3f} {s['mean_confidence']:>9.3f} "
                  f"{str(s['mean_input_tokens']):>7} {s['latency_p50_ms']:>7.0f}")
        print()


def probe_table(runs: dict[str, dict]) -> None:
    runs = {k: v for k, v in runs.items() if v.get("task") == "probes"}
    if not runs:
        return
    print("\n=== Graded intensity ladders (tau=1.0 means perfectly ordered) ===\n")
    for name, run in runs.items():
        print(f"  {name}")
        for probe, r in run["results"].items():
            print(f"    {probe:<10} tau={r['kendall_tau']:>6.3f}  spread={r['spread']:.3f}  "
                  f"inversions={len(r['inversions'])}")
            for row in r["rows"]:
                print(f"       {row['rank']}  {row['case']:<18} {row['value']:.4f}")
        print()


def main() -> int:
    runs = load()
    if not runs:
        print(f"no results in {RESULTS}")
        return 1
    label_budget_table(runs)
    probe_table(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
