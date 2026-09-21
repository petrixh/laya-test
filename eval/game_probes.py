"""Why Laya cannot play Reindeer Jump -- a diagnostic ladder.

The game autopilot scored 0.45 against a 0.33 chance baseline. This isolates
where that breaks, rung by rung, so the result is a diagnosis rather than a
number:

  1 control      a task the model is built for (sentiment). If this fails the
                 pipeline is broken, not the model.
  2 action       the question the autopilot actually asks: given the scene,
                 jump or duck?
  3 position     the same scene, but asked "where is the obstacle?" instead of
                 "what should we do?". Separates reading the scene from
                 choosing an action.
  4 knowledge    only the object's name, no positional words. Tests whether the
                 model knows a garland hangs and a log does not -- i.e. whether
                 rung 3 was real inference or just paraphrase matching.

  python -m eval.game_probes
"""
from __future__ import annotations

import os
import sys

from .runner import Service

URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")

GROUND = ["snowman", "gifts", "log"]
AIR = ["garland", "baubles"]

ACTION_Q = {
    "type": "choice",
    "instructions": ("A running reindeer must get past the obstacle ahead without "
                     "touching it. What should it do?"),
    "criteria": {
        "jump": "leap over a ground-level obstacle such as a snowman, a pile of gifts or a log",
        "duck": "crouch under a hanging obstacle such as a garland or a string of baubles",
        "run": "keep running normally, nothing is close enough to need action yet",
    },
}
POSITION_Q = {
    "type": "choice",
    "instructions": "Where is the obstacle?",
    "criteria": {
        "on_the_ground": "the obstacle rests on the ground",
        "in_the_air": "the obstacle hangs in the air above the ground",
    },
}
KNOWLEDGE_Q = {
    "type": "choice",
    "instructions": "Where would this festive object be found on a snowy track?",
    "criteria": {
        "on_the_ground": "it rests on the ground, you would trip over it",
        "in_the_air": "it hangs overhead, you would walk underneath it",
    },
}
SENTIMENT_Q = {
    "type": "choice",
    "instructions": "What is the sentiment of this message?",
    "criteria": {"positive": "the writer is happy", "negative": "the writer is unhappy"},
}

SENTIMENT_CASES = [
    ("This product is fantastic, I love it.", "positive"),
    ("Terrible, broke on day one, furious.", "negative"),
    ("Exactly what I needed, arrived early.", "positive"),
    ("Useless. Waste of money. Never again.", "negative"),
]


def scene(obj: str, air: bool) -> str:
    where = "hanging in the air" if air else "sitting on the ground"
    return f"A {obj} is {where} directly ahead of the running reindeer."


def run_rung(svc: Service, name: str, cases: list[tuple[str, str]], question: dict) -> dict:
    ok, confs, rows = 0, [], []
    for state, want in cases:
        ans = svc.predict(state, {"a": question})["answers"]["a"]
        hit = ans["choice"] == want
        ok += hit
        confs.append(ans["confidence"])
        rows.append({"state": state, "want": want, "got": ans["choice"],
                     "correct": hit, "confidence": round(ans["confidence"], 4)})
    return {"rung": name, "n": len(cases), "correct": ok,
            "accuracy": round(ok / len(cases), 3),
            "mean_confidence": round(sum(confs) / len(confs), 3), "rows": rows}


def main() -> int:
    with Service(URL) as svc:
        info = svc.wait_ready()
        print(f"checkpoint={info['checkpoint']} subfolder={info.get('subfolder')} "
              f"device={info['device']}\n", file=sys.stderr)

        rungs = [
            run_rung(svc, "1 control (sentiment)", SENTIMENT_CASES, SENTIMENT_Q),
            run_rung(svc, "2 action (the real ask)",
                     [(scene(o, False), "jump") for o in GROUND] +
                     [(scene(o, True), "duck") for o in AIR], ACTION_Q),
            run_rung(svc, "3 position (fact stated)",
                     [(scene(o, False), "on_the_ground") for o in GROUND] +
                     [(scene(o, True), "in_the_air") for o in AIR], POSITION_Q),
            run_rung(svc, "4 knowledge (name only)",
                     [(o, "on_the_ground") for o in GROUND] +
                     [(o, "in_the_air") for o in AIR], KNOWLEDGE_Q),
        ]

        print(f"{'rung':<26} {'acc':>6} {'mean conf':>10}")
        for r in rungs:
            print(f"{r['rung']:<26} {r['accuracy']:>6.2f} {r['mean_confidence']:>10.3f}")
        print("\nRead: rung 1 high means the pipeline is fine. Rung 2 at chance with low\n"
              "confidence means the model cannot pick the action. Rung 3 high but rung 4\n"
              "at chance means rung 3 was paraphrase matching, not world knowledge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
