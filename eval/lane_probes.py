"""Which framing works for the three-lane game?

The single-lane agent asked one question about one obstacle. Three lanes adds a
binding problem: the model must work out *which* obstacle is in *which* lane
before it can say what that lane needs. Three framings are plausible and they
differ in how much of that work the harness does:

  scene_3q   one call. The state lists all three lanes; three questions, one per
             lane, differing only in their instruction. The model does the
             binding. Hardest, and the most interesting if it works.
  scene_1q   one call, one question: which lane should the reindeer take?
             The model does the binding and the choosing.
  per_lane   three calls, each with a state naming only that lane's object.
             The harness does the binding by reading obstacle.lane from the
             game. Reuses the validated single-lane prompt verbatim.

per_lane is the safe option but it makes the task no harder than before, so it
is worth knowing what the others cost.

  python -m eval.lane_probes
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import random
import sys

from .runner import Service

URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results"))

GROUND = ["snowman", "pile of gifts", "log"]
AIR = ["garland", "string of baubles"]
WALL = "tall ice wall"
LANE_NAMES = ["left", "middle", "right"]

# requirement -> (descriptive label, neutral label)
REQS = ["clear", "jump", "duck", "blocked"]
LABELS = {
    "descriptive": {"clear": "clear", "jump": "jump_over", "duck": "duck_under",
                    "blocked": "blocked"},
    "neutral": {"clear": "option_a", "jump": "option_b", "duck": "option_c",
                "blocked": "option_d"},
}
CRIT_TEXT = {
    "clear": "nothing is in this lane, the reindeer can run straight through it",
    "jump": "something rests on the snow in this lane, such as a snowman, a pile of "
            "gifts or a log: it blocks the space near the ground, so leave the ground "
            "to clear it",
    "duck": "something hangs overhead in this lane, such as a garland or a string of "
            "baubles: it blocks the space above head height, so lower yourself to pass "
            "beneath it",
    "blocked": "a solid barrier fills this lane from the snow to above head height, "
               "such as a tall ice wall: it cannot be jumped over or ducked under at all",
}


def criteria_for(style: str) -> dict:
    lab = LABELS[style]
    return {lab[r]: CRIT_TEXT[r] for r in REQS}


def make_scene(rng: random.Random) -> dict:
    """One wave. At least one lane is passable, as the game guarantees."""
    while True:
        lanes = []
        for _ in range(3):
            r = rng.random()
            if r < 0.28:
                lanes.append(("nothing", "clear"))
            elif r < 0.55:
                lanes.append((rng.choice(GROUND), "jump"))
            elif r < 0.80:
                lanes.append((rng.choice(AIR), "duck"))
            else:
                lanes.append((WALL, "blocked"))
        if any(r != "blocked" for _, r in lanes):
            return {"lanes": lanes}


def scene_text(scene: dict) -> str:
    bits = []
    for name, (obj, _) in zip(LANE_NAMES, scene["lanes"]):
        bits.append(f"{name.capitalize()} lane: " +
                    ("nothing." if obj == "nothing" else f"a {obj}."))
    return ("The reindeer is running down a track with three lanes. " + " ".join(bits))


def passable(scene: dict) -> list[int]:
    return [i for i, (_, r) in enumerate(scene["lanes"]) if r != "blocked"]


def run(svc: Service, framing: str, style: str, scenes: list[dict]) -> dict:
    lab = LABELS[style]
    inv = {v: k for k, v in lab.items()}
    crit = criteria_for(style)
    per_lane_hits = per_lane_n = 0
    chose_passable = chose_n = survives = 0
    all_blocked = 0
    confs = []

    for sc in scenes:
        got: list[str | None] = [None, None, None]

        if framing == "per_lane":
            for i, (obj, _) in enumerate(sc["lanes"]):
                state = ("There is nothing on the track ahead of the running reindeer."
                         if obj == "nothing"
                         else f"There is a {obj} on the track ahead of the running reindeer.")
                q = {"type": "choice", "instructions": "Which manoeuvre clears the obstacle described?",
                     "criteria": crit}
                a = svc.predict(state, {"m": q})["answers"]["m"]
                got[i] = inv.get(a["choice"])
                confs.append(a["confidence"])

        elif framing == "scene_3q":
            qs = {}
            for i, name in enumerate(LANE_NAMES):
                qs[f"lane_{i}"] = {
                    "type": "choice",
                    "instructions": f"What does the {name} lane require?",
                    "criteria": crit,
                }
            ans = svc.predict(scene_text(sc), qs)["answers"]
            for i in range(3):
                a = ans[f"lane_{i}"]
                got[i] = inv.get(a["choice"])
                confs.append(a["confidence"])

        elif framing == "scene_1q":
            q = {"type": "choice",
                 "instructions": "Which lane can the reindeer get through?",
                 "criteria": {"left": "the left lane is passable",
                              "middle": "the middle lane is passable",
                              "right": "the right lane is passable"}}
            a = svc.predict(scene_text(sc), {"m": q})["answers"]["m"]
            confs.append(a["confidence"])
            pick = LANE_NAMES.index(a["choice"])
            chose_n += 1
            ok_lane = pick in passable(sc)
            chose_passable += ok_lane
            # this framing never says what to do once there, so surviving the
            # manoeuvre is out of scope for it
            survives += ok_lane
            continue

        for i, (_, want) in enumerate(sc["lanes"]):
            per_lane_n += 1
            per_lane_hits += got[i] == want
        # The decision that actually matters. Surviving needs BOTH a passable
        # lane and the right manoeuvre once in it -- reading a garland as clear
        # kills you just as dead as picking a walled lane.
        open_lanes = [i for i in range(3) if got[i] != "blocked"]
        chose_n += 1
        if not open_lanes:
            all_blocked += 1        # no lane offered: the harness has nothing to pick
            continue
        clear = [i for i in open_lanes if got[i] == "clear"]
        pick = (clear or open_lanes)[0]
        ok_lane = pick in passable(sc)
        chose_passable += ok_lane
        survives += ok_lane and got[pick] == sc["lanes"][pick][1]

    return {
        "framing": framing, "labels": style,
        "per_lane_accuracy": round(per_lane_hits / per_lane_n, 3) if per_lane_n else None,
        "picks_passable_lane": round(chose_passable / chose_n, 3) if chose_n else 0.0,
        "survives_wave": round(survives / chose_n, 3) if chose_n else 0.0,
        "no_lane_offered": round(all_blocked / chose_n, 3) if chose_n else 0.0,
        "mean_confidence": round(sum(confs) / len(confs), 3),
    }


def main() -> int:
    rng = random.Random(7)
    scenes = [make_scene(rng) for _ in range(30)]
    rows = []
    with Service(URL) as svc:
        info = svc.wait_ready()
        print(f"backend={info.get('backend')} scenes={len(scenes)}\n", file=sys.stderr)
        for framing, style in itertools.product(("per_lane", "scene_3q", "scene_1q"),
                                                ("descriptive", "neutral")):
            rows.append(run(svc, framing, style, scenes))
            print(f"  done {framing}/{style}", file=sys.stderr, flush=True)

    rows.sort(key=lambda r: -(r["survives_wave"] or 0))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lane_probes.json").write_text(json.dumps({"service": info, "rows": rows}, indent=2))

    print(f"{'framing':<10} {'labels':<12} {'lane acc':>9} {'passable':>9} "
          f"{'SURVIVES':>9} {'no lane':>8} {'conf':>7}")
    for r in rows:
        pl = "     -   " if r["per_lane_accuracy"] is None else f"{r['per_lane_accuracy']:>9.3f}"
        print(f"{r['framing']:<10} {r['labels']:<12} {pl} {r['picks_passable_lane']:>9.3f} "
              f"{r['survives_wave']:>9.3f} {r['no_lane_offered']:>8.3f} {r['mean_confidence']:>7.3f}")
    print("\nSURVIVES = picked a passable lane AND read its manoeuvre correctly.\n"
          "no lane   = called all three lanes blocked, leaving the harness nothing to pick.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
