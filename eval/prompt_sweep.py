"""Prompt triage: which framing, if any, gets Laya to choose the right manoeuvre?

The first autopilot attempt answered `duck` to everything. That prompt already
carried the domain knowledge in its criteria ("a snowman, a pile of gifts or a
log"), so "it does not know where a garland hangs" cannot be the whole story.
This crosses the framing axes systematically instead of guessing.

The line we do not cross: guidance may be **static and instance-independent** --
the same sentence on every call, describing the world. It may not be
**per-instance** -- pre-ranking the options for the scene at hand, the way the
laya-mlx Snake demo labels directions "Best" / "Blocked. Collision." That hands
the model the answer and measures nothing.

The `state` axis is graded by how much it gives away, and the report prints it,
so no variant's score can be read without seeing what it was told:

  name_only     just the object. Needs world knowledge + mapping. Gives nothing.
  geometry      numeric extents and the reindeer's two heights. States facts,
                never the category -- the model must compare numbers.
  says_position "sitting on the ground" / "hanging in the air". States the
                category in words; choosing the action is then paraphrase.

  python -m eval.prompt_sweep [--limit N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import sys

from .runner import Service

URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results"))

# (object, is_air). The five the game actually spawns.
OBJECTS = [("snowman", False), ("pile of gifts", False), ("log", False),
           ("garland", True), ("string of baubles", True)]

GEOM = {False: ("0.00", "0.95"), True: ("1.20", "2.90")}

# ---------------------------------------------------------------- axes

def state_name_only(obj: str, air: bool) -> str:
    return f"There is a {obj} on the track ahead of the running reindeer."


def state_geometry(obj: str, air: bool) -> str:
    lo, hi = GEOM[air]
    return (f"There is a {obj} on the track ahead of the running reindeer. "
            f"It occupies the space from {lo} to {hi} metres above the snow. "
            f"The reindeer is 1.75 metres tall standing and 0.95 metres tall crouching.")


def state_says_position(obj: str, air: bool) -> str:
    where = "hanging in the air" if air else "sitting on the ground"
    return f"A {obj} is {where} directly ahead of the running reindeer."


def state_json_facts(obj: str, air: bool) -> dict:
    """Same information as `geometry`, as a dict instead of prose. Isolates
    format: this is what the first autopilot actually sent."""
    lo, hi = GEOM[air]
    return {"obstacle": obj,
            "obstacle_bottom_m": float(lo), "obstacle_top_m": float(hi),
            "reindeer_height_standing_m": 1.75,
            "reindeer_height_crouching_m": 0.95}


STATES = {"name_only": state_name_only, "geometry": state_geometry,
          "json_facts": state_json_facts, "says_position": state_says_position}
LEAK = {"name_only": "none", "geometry": "facts", "json_facts": "facts",
        "says_position": "category"}

# The first autopilot offered a third "nothing to do yet" option. Whether that
# alone poisons the decision is worth an axis of its own.
THIRD = {"action": ("run", "keep running normally, nothing is close enough to act on yet"),
         "motion": ("hold_course", "keep running normally, nothing is close enough to act on yet"),
         "neutral": ("option_c", "neither manoeuvre is needed")}
OPTIONS = {"two": False, "three": True}

LABELS = {
    "action": ("jump", "duck"),
    "motion": ("go_over", "go_under"),
    "neutral": ("option_a", "option_b"),
}

# criteria[style](label_for_ground, label_for_air) -> dict
def crit_bare(g: str, a: str) -> dict:
    return {g: "jump over the obstacle", a: "duck under the obstacle"}


def crit_knowledge(g: str, a: str) -> dict:
    return {g: "for obstacles that sit on the ground such as a snowman, a pile of "
               "gifts or a log, which must be jumped over",
            a: "for obstacles that hang overhead such as a garland or a string of "
               "baubles, which must be ducked under"}


def crit_mechanism(g: str, a: str) -> dict:
    return {g: "the obstacle occupies the space near the snow, so the reindeer must "
               "leave the ground to clear it",
            a: "the obstacle occupies the space above head height, so the reindeer "
               "must lower itself to pass beneath it"}


def crit_both(g: str, a: str) -> dict:
    return {g: "obstacles resting on the snow, such as a snowman, a pile of gifts or "
               "a log: they block the space near the ground, so leave the ground to clear them",
            a: "obstacles suspended overhead, such as a garland or a string of baubles: "
               "they block the space above head height, so lower yourself to pass beneath"}


CRITERIA = {"bare": crit_bare, "knowledge": crit_knowledge,
            "mechanism": crit_mechanism, "both": crit_both}

INSTRUCTIONS = {
    "act": "A running reindeer must get past the obstacle ahead without touching it. What should it do?",
    "classify": "Which manoeuvre clears the obstacle described?",
}


def variants():
    for lab, crit, st, ins, opt in itertools.product(
            LABELS, CRITERIA, STATES, INSTRUCTIONS, OPTIONS):
        yield {"labels": lab, "criteria": crit, "state": st,
               "instructions": ins, "options": opt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N variants")
    args = ap.parse_args()

    all_variants = list(variants())
    if args.limit:
        all_variants = all_variants[: args.limit]

    with Service(URL) as svc:
        info = svc.wait_ready()
        print(f"checkpoint={info['checkpoint']} subfolder={info.get('subfolder')} "
              f"backend={info.get('backend')}  variants={len(all_variants)} "
              f"scenes={len(OBJECTS)}\n", file=sys.stderr)

        rows = []
        for i, v in enumerate(all_variants, 1):
            ground, air = LABELS[v["labels"]]
            criteria = CRITERIA[v["criteria"]](ground, air)
            if OPTIONS[v["options"]]:
                third_label, third_text = THIRD[v["labels"]]
                criteria[third_label] = third_text
            question = {
                "type": "choice",
                "instructions": INSTRUCTIONS[v["instructions"]],
                "criteria": criteria,
            }
            build = STATES[v["state"]]
            ok, confs, picks = 0, [], []
            for obj, is_air in OBJECTS:
                ans = svc.predict(build(obj, is_air), {"m": question})["answers"]["m"]
                want = air if is_air else ground
                ok += ans["choice"] == want
                confs.append(ans["confidence"])
                picks.append(ans["choice"])
            rows.append({**v,
                         "leak": LEAK[v["state"]],
                         "accuracy": round(ok / len(OBJECTS), 3),
                         "mean_confidence": round(sum(confs) / len(confs), 3),
                         # a variant that answers the same thing every time is
                         # not deciding, whatever its score
                         "distinct_answers": len(set(picks)),
                         "picks": picks})
            if i % 12 == 0:
                print(f"  {i}/{len(all_variants)}", file=sys.stderr, flush=True)

    rows.sort(key=lambda r: (-r["accuracy"], -r["distinct_answers"], -r["mean_confidence"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"prompt_sweep__{info.get('subfolder') or 'base'}.json"
    out.write_text(json.dumps({"service": info, "rows": rows}, indent=2))

    hdr = (f"{'acc':>5} {'dist':>4} {'conf':>6}  {'labels':<8} {'criteria':<10} "
           f"{'state':<14} {'leak':<9} {'instr':<9} {'opts':<6}")
    line = lambda r: (f"{r['accuracy']:>5.2f} {r['distinct_answers']:>4} "
                      f"{r['mean_confidence']:>6.3f}  {r['labels']:<8} {r['criteria']:<10} "
                      f"{r['state']:<14} {r['leak']:<9} {r['instructions']:<9} {r['options']:<6}")
    print(hdr)
    for r in rows[:15]:
        print(line(r))

    print("\n--- marginal effect of each axis on accuracy ---")
    for axis in ("labels", "criteria", "state", "instructions", "options"):
        levels = sorted({r[axis] for r in rows})
        parts = []
        for lv in levels:
            sub = [r["accuracy"] for r in rows if r[axis] == lv]
            parts.append(f"{lv}={sum(sub)/len(sub):.2f}")
        print(f"  {axis:<13} " + "  ".join(parts))

    print("\n--- best variants that give nothing away (leak=none) ---")
    for r in [r for r in rows if r["leak"] == "none"][:5]:
        print(line(r) + f"  picks={r['picks']}")

    collapsed = sum(1 for r in rows if r["distinct_answers"] == 1)
    print(f"\n{collapsed}/{len(rows)} variants answered identically for all five objects.")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
