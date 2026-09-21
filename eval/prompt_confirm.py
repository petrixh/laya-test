"""Confirm the prompt sweep's winners on a larger, harder scene set.

The sweep scores each variant on five scenes, so accuracy is quantised to 0.2
and a 1.00 means very little -- with 192 variants, some top over by luck alone.
This re-tests the leaders on many more scenes, including paraphrases and
objects the criteria never mention, which is where a variant that merely
pattern-matches the criteria text will come apart.

  python -m eval.prompt_confirm [--top 8]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from .runner import Service
from .prompt_sweep import CRITERIA, INSTRUCTIONS, LABELS, LEAK, OPTIONS, STATES, THIRD

URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results"))

# Objects the criteria name explicitly.
SEEN = [("snowman", False), ("pile of gifts", False), ("log", False),
        ("garland", True), ("string of baubles", True)]
# Objects they never mention: the real test of whether knowledge transfers
# rather than the criteria text being matched literally.
HELD_OUT = [("sledge", False), ("stack of firewood", False), ("snow drift", False),
            ("wooden crate", False), ("hanging lantern", True), ("string of fairy lights", True),
            ("low banner strung between two poles", True), ("bunting", True)]

# Four phrasings per object, to stop a single sentence template carrying the result.
TEMPLATES = [
    "There is a {o} on the track ahead of the running reindeer.",
    "The reindeer is galloping down the track. A {o} lies in its path.",
    "Ahead: a {o}, right in the middle of the snowy track.",
    "Coming up fast, the reindeer sees a {o} blocking the way.",
]


def scenes(include_held_out: bool) -> list[tuple[str, bool, bool]]:
    out = []
    for obj, air in SEEN:
        for t in TEMPLATES:
            out.append((t.format(o=obj), air, False))
    if include_held_out:
        for obj, air in HELD_OUT:
            for t in TEMPLATES[:2]:
                out.append((t.format(o=obj), air, True))
    return out


def build_question(v: dict) -> dict:
    ground, air = LABELS[v["labels"]]
    criteria = CRITERIA[v["criteria"]](ground, air)
    if OPTIONS[v["options"]]:
        lab, text = THIRD[v["labels"]]
        criteria[lab] = text
    return {"type": "choice", "instructions": INSTRUCTIONS[v["instructions"]],
            "criteria": criteria}, ground, air


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    files = sorted(RESULTS.glob("prompt_sweep__*.json"))
    if not files:
        print("run `python -m eval.prompt_sweep` first", file=sys.stderr)
        return 1
    rows = json.loads(files[-1].read_text())["rows"]

    # Only variants that give nothing away are worth confirming: a variant fed
    # "sitting on the ground" is being asked to paraphrase, not to decide.
    clean = [r for r in rows if r["leak"] == "none" and r["distinct_answers"] > 1]
    picks = clean[: args.top]
    cases = scenes(include_held_out=True)
    print(f"confirming {len(picks)} clean variants on {len(cases)} scenes "
          f"({sum(1 for c in cases if c[2])} using objects the criteria never name)\n",
          file=sys.stderr)

    out = []
    with Service(URL) as svc:
        info = svc.wait_ready()
        for i, v in enumerate(picks, 1):
            question, ground, air = build_question(v)
            tally = {"seen": [0, 0], "held": [0, 0]}
            confs, chosen = [], []
            for text, is_air, held in cases:
                ans = svc.predict(text, {"m": question})["answers"]["m"]
                want = air if is_air else ground
                bucket = tally["held" if held else "seen"]
                bucket[1] += 1
                bucket[0] += ans["choice"] == want
                confs.append(ans["confidence"])
                chosen.append(ans["choice"])
            seen_acc = tally["seen"][0] / tally["seen"][1]
            held_acc = tally["held"][0] / tally["held"][1]
            total = (tally["seen"][0] + tally["held"][0]) / len(cases)
            out.append({**v, "sweep_accuracy": v["accuracy"],
                        "accuracy": round(total, 3),
                        "named_objects": round(seen_acc, 3),
                        "held_out_objects": round(held_acc, 3),
                        "mean_confidence": round(sum(confs) / len(confs), 3),
                        "distinct_answers": len(set(chosen))})
            print(f"  {i}/{len(picks)} done", file=sys.stderr, flush=True)

    out.sort(key=lambda r: -r["accuracy"])
    dest = RESULTS / f"prompt_confirm__{info.get('subfolder') or 'base'}.json"
    dest.write_text(json.dumps({"service": info, "scenes": len(cases), "rows": out}, indent=2))

    print(f"{'sweep':>6} {'all':>6} {'named':>6} {'held':>6} {'conf':>6}  "
          f"{'labels':<8} {'criteria':<10} {'instr':<9} {'opts':<6}")
    for r in out:
        print(f"{r['sweep_accuracy']:>6.2f} {r['accuracy']:>6.2f} {r['named_objects']:>6.2f} "
              f"{r['held_out_objects']:>6.2f} {r['mean_confidence']:>6.3f}  "
              f"{r['labels']:<8} {r['criteria']:<10} {r['instructions']:<9} {r['options']:<6}")
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
