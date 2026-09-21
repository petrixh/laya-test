"""How few options can we get away with when there are three obstacle classes?

lane_probes showed a 4-option question collapsing (0.078 per-lane accuracy,
every lane called blocked). The sweep on the single-lane game had already found
option count to be a strong axis -- two 0.65 against three 0.51 -- so the fix is
probably to decompose rather than to reword.

The harness already knows which lanes hold an object and what each object is
called; it read exactly that in the single-lane version. What the model has to
supply is what the object *requires*. These are the ways to ask it:

  choice3     one 3-option question: jump / duck / impassable
  split       the validated 2-option ground-vs-air question, plus a separate
              noul for "is this a solid barrier?"
  two_noul    two independent noul questions: is it a barrier, does it hang

  python -m eval.obstacle_class
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys

from .runner import Service

URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results"))

# (object, truth). Objects the criteria name.
NAMED = [("snowman", "jump"), ("pile of gifts", "jump"), ("log", "jump"),
         ("garland", "duck"), ("string of baubles", "duck"),
         ("tall ice wall", "block")]
# Objects they never name: the transfer test.
HELD = [("sledge", "jump"), ("stack of firewood", "jump"), ("wooden crate", "jump"),
        ("hanging lantern", "duck"), ("string of fairy lights", "duck"), ("bunting", "duck"),
        ("solid stone barricade", "block"), ("high wooden fence", "block"),
        ("wall of packed snow", "block")]

TEMPLATES = [
    "There is a {o} on the track ahead of the running reindeer.",
    "Ahead: a {o}, right in the middle of the snowy track.",
]

GROUND_TXT = ("obstacles resting on the snow, such as a snowman, a pile of gifts or a log: "
              "they block the space near the ground, so leave the ground to clear them")
AIR_TXT = ("obstacles suspended overhead, such as a garland or a string of baubles: "
           "they block the space above head height, so lower yourself to pass beneath")
BLOCK_TXT = ("a solid barrier filling the lane from the snow to above head height, such as "
             "a tall ice wall: it cannot be cleared by leaving the ground or by lowering "
             "yourself, so it must be gone around")

LABELS = {"neutral": ("option_a", "option_b", "option_c"),
          "descriptive": ("jump_over", "duck_under", "impassable")}

INSTR = "Which manoeuvre clears the obstacle described?"
BARRIER_Q = {"type": "noul",
             "instructions": "Is this a solid barrier that fills the whole lane, too tall to "
                             "leave the ground over and too low to pass beneath?"}
HANGS_Q = {"type": "noul",
           "instructions": "Does this obstacle hang overhead, above head height, rather than "
                           "resting on the ground?"}


def classify(svc: Service, framing: str, style: str, obj: str) -> str:
    g, a, b = LABELS[style]
    state = TEMPLATES[0].format(o=obj)

    if framing == "choice3":
        q = {"type": "choice", "instructions": INSTR,
             "criteria": {g: GROUND_TXT, a: AIR_TXT, b: BLOCK_TXT}}
        pick = svc.predict(state, {"m": q})["answers"]["m"]["choice"]
        return {g: "jump", a: "duck", b: "block"}[pick]

    if framing == "split":
        q = {"type": "choice", "instructions": INSTR, "criteria": {g: GROUND_TXT, a: AIR_TXT}}
        res = svc.predict(state, {"m": q, "barrier": BARRIER_Q})["answers"]
        if res["barrier"]["noul"] >= 0.5:
            return "block"
        return "jump" if res["m"]["choice"] == g else "duck"

    res = svc.predict(state, {"barrier": BARRIER_Q, "hangs": HANGS_Q})["answers"]
    if res["barrier"]["noul"] >= 0.5:
        return "block"
    return "duck" if res["hangs"]["noul"] >= 0.5 else "jump"


def main() -> int:
    rows = []
    with Service(URL) as svc:
        info = svc.wait_ready()
        print(f"backend={info.get('backend')}\n", file=sys.stderr)
        for framing, style in itertools.product(("choice3", "split", "two_noul"),
                                                ("neutral", "descriptive")):
            if framing == "two_noul" and style == "descriptive":
                continue                      # no labels involved, one run is enough
            hits = {"named": [0, 0], "held": [0, 0]}
            confusion: dict[str, int] = {}
            for group, cases in (("named", NAMED), ("held", HELD)):
                for obj, want in cases:
                    got = classify(svc, framing, style, obj)
                    hits[group][1] += 1
                    hits[group][0] += got == want
                    if got != want:
                        confusion[f"{want}->{got}"] = confusion.get(f"{want}->{got}", 0) + 1
            n = hits["named"][1] + hits["held"][1]
            rows.append({
                "framing": framing, "labels": style,
                "accuracy": round((hits["named"][0] + hits["held"][0]) / n, 3),
                "named": round(hits["named"][0] / hits["named"][1], 3),
                "held_out": round(hits["held"][0] / hits["held"][1], 3),
                "errors": confusion,
            })
            print(f"  done {framing}/{style}", file=sys.stderr, flush=True)

    rows.sort(key=lambda r: -r["accuracy"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "obstacle_class.json").write_text(json.dumps({"service": info, "rows": rows}, indent=2))
    print(f"{'framing':<10} {'labels':<12} {'all':>6} {'named':>7} {'held-out':>9}  errors")
    for r in rows:
        print(f"{r['framing']:<10} {r['labels']:<12} {r['accuracy']:>6.3f} {r['named']:>7.3f} "
              f"{r['held_out']:>9.3f}  {r['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
