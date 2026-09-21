"""Stage two: can the model pick the lane, given its own stage-one readings?

Stage one classifies each obstacle (jump / duck / block). Stage two is handed a
description of the three lanes built from those classifications -- the model's
own output, not ground truth -- and asked which lane to take.

The preference itself (prefer a clear lane, never a barrier) is static guidance
and may live in the instructions, the same way "logs sit on the ground" lives in
the criteria. What may not happen is the scene pre-ranking the lanes for it.

  python -m eval.lane_choice
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

LANE_WORDS = ["left", "middle", "right"]
DESC = {
    None: "clear, with nothing in it",
    "jump": "something resting on the snow that has to be jumped over",
    "duck": "something hanging overhead that has to be ducked under",
    "block": "a solid barrier that cannot be passed at all",
}

LABELS = {
    "named": ("left", "middle", "right"),
    "neutral": ("option_a", "option_b", "option_c"),
}
CRITERIA = {
    "bare": lambda L: {L[i]: f"the {LANE_WORDS[i]} lane" for i in range(3)},
    "verbose": lambda L: {L[i]: f"send the reindeer into the {LANE_WORDS[i]} lane"
                          for i in range(3)},
}
INSTRUCTIONS = {
    "plain": "Which lane should the reindeer take?",
    "preference": ("Which lane should the reindeer take? Prefer a lane that is clear. "
                   "A lane needing a jump or a duck is acceptable. Never choose a lane "
                   "with a solid barrier."),
}


def make_scene(rng: random.Random) -> dict:
    while True:
        lanes = [rng.choice([None, "jump", "duck", "block"]) for _ in range(3)]
        if any(l != "block" for l in lanes):
            return {"lanes": lanes, "here": rng.randrange(3)}


def scene_text(sc: dict, with_here: bool) -> str:
    bits = [f"The {LANE_WORDS[i]} lane is {DESC[sc['lanes'][i]]}." for i in range(3)]
    if with_here:
        bits.append(f"The reindeer is currently in the {LANE_WORDS[sc['here']]} lane.")
    return " ".join(bits)


def run_baseline(kind, scenes, rng):
    """Constant and random strategies.

    Without these a score is uninterpretable: on scenes where each lane is a
    barrier a quarter of the time, "always answer option_a" already picks a
    passable lane about 75% of the time. Any framing has to beat that before it
    can be said to be choosing anything.
    """
    passable = best = n = 0
    for sc in scenes:
        pick = rng.randrange(3) if kind == "random" else int(kind[-1])
        n += 1
        passable += sc["lanes"][pick] != "block"
        clear = [i for i in range(3) if sc["lanes"][i] is None]
        best += (pick in clear) if clear else (sc["lanes"][pick] != "block")
    return {"labels": f"BASELINE {kind}", "criteria": "-", "instructions": "-",
            "current_lane_stated": False,
            "picks_passable": round(passable / n, 3),
            "picks_best": round(best / n, 3), "mean_confidence": 0.0}


def run_shuffled(svc, scenes, rng):
    """Best framing, but the label->lane mapping is randomised per call.

    If the model is keyed on option position rather than on the scene, this
    destroys the score. If it is actually reading, it should be unaffected.
    """
    L = LABELS["neutral"]
    passable = best = n = 0
    confs = []
    for sc in scenes:
        order = [0, 1, 2]
        rng.shuffle(order)                       # order[k] = lane shown as label k
        crit = {L[k]: f"the {LANE_WORDS[order[k]]} lane" for k in range(3)}
        q = {"type": "choice", "instructions": INSTRUCTIONS["plain"], "criteria": crit}
        a = svc.predict(scene_text(sc, False), {"m": q})["answers"]["m"]
        pick = order[L.index(a["choice"])]
        confs.append(a["confidence"])
        n += 1
        passable += sc["lanes"][pick] != "block"
        clear = [i for i in range(3) if sc["lanes"][i] is None]
        best += (pick in clear) if clear else (sc["lanes"][pick] != "block")
    return {"labels": "neutral+shuffled", "criteria": "bare", "instructions": "plain",
            "current_lane_stated": False,
            "picks_passable": round(passable / n, 3),
            "picks_best": round(best / n, 3),
            "mean_confidence": round(sum(confs) / len(confs), 3)}


NOUL_Q = {
    "good": "Is the {w} lane a good lane for the reindeer to take?",
    "avoid": "Must the reindeer avoid the {w} lane?",
}


def run_noul(svc, phrasing, with_here, scenes):
    """One binary question per lane in a single pass, then argmax.

    The three-option choice tops out at 0.875, and every earlier framing
    problem on this model has yielded to splitting rather than widening.
    """
    qs = {f"lane_{i}": {"type": "noul",
                        "instructions": NOUL_Q[phrasing].format(w=LANE_WORDS[i])}
          for i in range(3)}
    passable = best = n = 0
    confs = []
    for sc in scenes:
        ans = svc.predict(scene_text(sc, with_here), qs)["answers"]
        score = [ans[f"lane_{i}"]["noul"] for i in range(3)]
        pick = max(range(3), key=lambda i: score[i] if phrasing == "good" else -score[i])
        confs.extend(ans[f"lane_{i}"]["confidence"] for i in range(3))
        n += 1
        passable += sc["lanes"][pick] != "block"
        clear = [i for i in range(3) if sc["lanes"][i] is None]
        best += (pick in clear) if clear else (sc["lanes"][pick] != "block")
    return {"labels": f"noul:{phrasing}", "criteria": "-", "instructions": "-",
            "current_lane_stated": with_here,
            "picks_passable": round(passable / n, 3),
            "picks_best": round(best / n, 3),
            "mean_confidence": round(sum(confs) / len(confs), 3)}


def run(svc, style, crit, instr, with_here, scenes):
    L = LABELS[style]
    q = {"type": "choice", "instructions": INSTRUCTIONS[instr], "criteria": CRITERIA[crit](L)}
    passable = best = n = 0
    confs = []
    for sc in scenes:
        a = svc.predict(scene_text(sc, with_here), {"m": q})["answers"]["m"]
        pick = L.index(a["choice"])
        confs.append(a["confidence"])
        n += 1
        passable += sc["lanes"][pick] != "block"
        clear = [i for i in range(3) if sc["lanes"][i] is None]
        best += (pick in clear) if clear else (sc["lanes"][pick] != "block")
    return {"labels": style, "criteria": crit, "instructions": instr,
            "current_lane_stated": with_here,
            "picks_passable": round(passable / n, 3),
            "picks_best": round(best / n, 3),
            "mean_confidence": round(sum(confs) / len(confs), 3)}


def main() -> int:
    rng = random.Random(11)
    scenes = [make_scene(rng) for _ in range(40)]
    rows = []
    with Service(URL) as svc:
        info = svc.wait_ready()
        print(f"backend={info.get('backend')} scenes={len(scenes)}\n", file=sys.stderr)
        for style, crit, instr, here in itertools.product(
                LABELS, CRITERIA, INSTRUCTIONS, (False, True)):
            rows.append(run(svc, style, crit, instr, here, scenes))
            print(f"  {len(rows)}/20", file=sys.stderr, flush=True)
        for phrasing, here in itertools.product(NOUL_Q, (False, True)):
            rows.append(run_noul(svc, phrasing, here, scenes))
            print(f"  {len(rows)}/25", file=sys.stderr, flush=True)
        rows.append(run_shuffled(svc, scenes, random.Random(3)))
        for kind in ("always_0", "always_1", "always_2", "random"):
            rows.append(run_baseline(kind, scenes, random.Random(5)))

    rows.sort(key=lambda r: (-r["picks_passable"], -r["picks_best"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lane_choice.json").write_text(json.dumps({"service": info, "rows": rows}, indent=2))
    print(f"{'labels':<11} {'criteria':<9} {'instr':<11} {'here':<6} "
          f"{'passable':>9} {'best':>7} {'conf':>7}")
    for r in rows:
        print(f"{r['labels']:<11} {r['criteria']:<9} {r['instructions']:<11} "
              f"{str(r['current_lane_stated']):<6} {r['picks_passable']:>9.3f} "
              f"{r['picks_best']:>7.3f} {r['mean_confidence']:>7.3f}")
    print("\npassable = never walks into a barrier (survival).\n"
          "best     = also takes a clear lane when one exists (quality).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
