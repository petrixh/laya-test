"""Two ways to ask which lane to move to, and why the autopilot uses the second.

**As a choice between the candidate lanes.** This is what the agent used to do
and it does not work: the model answers by option position. The control needs
no baseline -- present the same pair with the options swapped, and a model
reading the state names the same lane twice. Order-consistency comes out at
0.00-0.20, and in a recorded run the agent took the first-listed option 12
times out of 12.

**As one yes/no per lane**, each about that lane alone, lower score wins. No
option list, so nothing to prefer the front of. This is what the autopilot
does now, and `run_per_lane` below measures it as a decision rule over pairs
across three wordings of each lane content.

The first section below also reports a quality metric for the choice framing.
Survival there is inherited from stage one, so the metric has to be quality:
when one candidate is clear and the other needs a manoeuvre, does it take the
clear one? Chance is 0.50, and "always take the first option offered" is the
other baseline that matters.

  python -m eval.lane_forced
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys

from .runner import Service

URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results"))
LANE_WORDS = ["left", "middle", "right"]
DESC = {None: "clear, with nothing in it",
        "jump": "something resting on the snow that has to be jumped over",
        "duck": "something hanging overhead that has to be ducked under",
        "block": "a solid barrier that cannot be passed at all"}
LABELS = ["option_a", "option_b"]


def make_scene(rng):
    """Current lane is a barrier; the other two differ in quality, one clear
    and one needing a manoeuvre. That is the case where the choice matters."""
    here = rng.randrange(3)
    others = [i for i in range(3) if i != here]
    rng.shuffle(others)
    lanes = [None, None, None]
    lanes[here] = "block"
    lanes[others[0]] = None                       # the clear one
    lanes[others[1]] = rng.choice(["jump", "duck"])
    return {"lanes": lanes, "here": here, "good": others[0], "meh": others[1]}


def state_of(sc, candidates):
    return " ".join(f"The {LANE_WORDS[i]} lane is {DESC[sc['lanes'][i]]}."
                    for i in candidates)


def run(svc, scenes, shuffle_options, rng):
    hits = first = n = 0
    confs = []
    for sc in scenes:
        cands = sorted([sc["good"], sc["meh"]])
        if shuffle_options:
            cands = cands[:]
            rng.shuffle(cands)
        crit = {LABELS[k]: f"the {LANE_WORDS[cands[k]]} lane" for k in range(2)}
        q = {"type": "choice", "instructions": "Which lane should the reindeer take?",
             "criteria": crit}
        a = svc.predict(state_of(sc, cands), {"m": q})["answers"]["m"]
        pick = cands[LABELS.index(a["choice"])]
        confs.append(a["confidence"])
        n += 1
        hits += pick == sc["good"]
        first += pick == cands[0]
    return {"picks_clear": round(hits / n, 3),
            "picks_first_option": round(first / n, 3),
            "mean_confidence": round(sum(confs) / len(confs), 3)}


PAIRS = [("block", None), ("block", "jump"), ("block", "duck"),
         (None, "jump"), (None, "duck"), ("jump", "duck")]
BETTER = {"block": 0, None: 3, "jump": 2, "duck": 2}   # higher is better to be in


def run_pairs(svc, rng, reps=10):
    """Every pair type, each presented in both orders.

    Order-consistency is the bias test that needs no baseline: ask the same
    question with the options swapped, and a model that is reading the scene
    names the same lane twice. A model keyed on position names whichever lane
    is listed first, and scores 0.
    """
    out = []
    for a, b in PAIRS:
        agree = correct = n = 0
        confs = []
        for _ in range(reps):
            lanes = rng.sample(range(3), 2)
            content = {lanes[0]: a, lanes[1]: b}
            picks = []
            for order in ([lanes[0], lanes[1]], [lanes[1], lanes[0]]):
                crit = {LABELS[k]: f"the {LANE_WORDS[order[k]]} lane" for k in range(2)}
                st = " ".join(f"The {LANE_WORDS[i]} lane is {DESC[content[i]]}." for i in order)
                ans = svc.predict(st, {"m": {"type": "choice",
                                             "instructions": "Which lane should the reindeer take?",
                                             "criteria": crit}})["answers"]["m"]
                picks.append(order[LABELS.index(ans["choice"])])
                confs.append(ans["confidence"])
            n += 1
            agree += picks[0] == picks[1]
            if BETTER[a] != BETTER[b]:
                want = lanes[0] if BETTER[a] > BETTER[b] else lanes[1]
                correct += sum(1 for p in picks if p == want) / 2
        label = f"{a or 'clear'} vs {b or 'clear'}"
        # with two equally good options there is no better one to pick, so the
        # column would otherwise report "picks whichever was listed as a"
        comparable = BETTER[a] != BETTER[b]
        out.append({"pair": label, "order_consistent": round(agree / n, 3),
                    "picks_better": round(correct / n, 3) if comparable else None,
                    "mean_confidence": round(sum(confs) / len(confs), 3)})
    return out


BLOCKED_Q = {"blocked": {"type": "noul", "instructions": "Is this lane blocked?"}}

# three wordings per class, so the ranking cannot be an artefact of one string
WORDINGS = {
    None: ["clear, with nothing in it", "empty", "open, with a clear path through"],
    "jump": ["something resting on the snow that has to be jumped over",
             "a log lying across it", "a snowman standing in it"],
    "duck": ["something hanging overhead that has to be ducked under",
             "a garland strung across it above head height",
             "a string of baubles hanging over it"],
    "block": ["a solid barrier that cannot be passed at all",
              "a tall ice wall filling it from the snow to above head height",
              "completely walled off"],
}


def run_per_lane(svc):
    """The framing the autopilot uses: one yes/no per lane, lower score wins.

    No option list, so nothing to prefer the front of. Reported as a decision
    rule over pairs, which is how it is actually used.
    """
    import itertools

    score = {}
    for cls, words in WORDINGS.items():
        for w in words:
            a = svc.predict(f"This lane is {w}.", BLOCKED_Q)["answers"]["blocked"]
            score[(cls, w)] = a["noul"]

    survive = sn = quality = qn = ties = tn = 0
    for (c1, w1), (c2, w2) in itertools.combinations(score, 2):
        if c1 == c2:
            # Two lanes holding the same thing. The model scores each lane on
            # its own and is deterministic, so these tie exactly and it has no
            # preference to express -- the harness breaks them. Excluded from
            # the rates below, and reported separately, because they are about
            # a fifth of live lane questions.
            tn += 1
            ties += score[(c1, w1)] == score[(c2, w2)]
            continue
        pick = (c1, w1) if score[(c1, w1)] < score[(c2, w2)] else (c2, w2)
        if (c1 == "block") != (c2 == "block"):
            sn += 1
            survive += pick[0] != "block"
        if None in (c1, c2) and "block" not in (c1, c2):
            qn += 1
            quality += pick[0] is None
    by_class = {str(c or "clear"): round(sum(score[(c, w)] for w in ws) / len(ws), 3)
                for c, ws in WORDINGS.items()}
    return {"mean_score_by_class": by_class,
            "avoids_the_barrier": f"{survive}/{sn}",
            "prefers_a_clear_lane": f"{quality}/{qn}",
            "same_content_pairs_that_tie_exactly": f"{ties}/{tn}"}


def main() -> int:
    rng = random.Random(23)
    scenes = [make_scene(rng) for _ in range(60)]
    with Service(URL) as svc:
        info = svc.wait_ready()
        print(f"backend={info.get('backend')} scenes={len(scenes)}\n", file=sys.stderr)
        fixed = run(svc, scenes, False, random.Random(1))
        shuf = run(svc, scenes, True, random.Random(2))
        pairs = run_pairs(svc, random.Random(31))
        per_lane = run_per_lane(svc)

    # baselines on the same scenes
    r = random.Random(4)
    coin = sum(1 for sc in scenes if r.random() < 0.5) / len(scenes)
    const = sum(1 for sc in scenes if sorted([sc["good"], sc["meh"]])[0] == sc["good"]) / len(scenes)

    rows = {"fixed order": fixed, "shuffled order": shuf,
            "BASELINE always first option": {"picks_clear": round(const, 3),
                                             "picks_first_option": 1.0, "mean_confidence": 0.0},
            "BASELINE coin flip": {"picks_clear": round(coin, 3),
                                   "picks_first_option": None, "mean_confidence": 0.0}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lane_forced.json").write_text(json.dumps({"service": info, "rows": rows}, indent=2))

    print(f"{'variant':<30} {'picks clear':>12} {'picks 1st':>10} {'conf':>7}")
    for k, v in rows.items():
        pf = "  -" if v["picks_first_option"] is None else f"{v['picks_first_option']:>10.3f}"
        print(f"{k:<30} {v['picks_clear']:>12.3f} {pf} {v['mean_confidence']:>7.3f}")
    print("\npicks clear = took the empty lane over the one needing a manoeuvre (chance 0.50).\n"
          "picks 1st   = how often it just took whichever option was listed first.")

    print(f"\n{'pair (both orders)':<22} {'consistent':>11} {'picks better':>13} {'conf':>7}")
    for r in pairs:
        better = "          n/a" if r["picks_better"] is None else f"{r['picks_better']:>13.2f}"
        print(f"{r['pair']:<22} {r['order_consistent']:>11.2f} {better} "
              f"{r['mean_confidence']:>7.3f}")
    print("\nconsistent = named the same lane with the options swapped. 0 means it is\n"
          "answering by position; 1 means it is reading the scene.")

    print("\n=== the framing the autopilot uses instead: one yes/no per lane ===")
    print("  mean 'is this lane blocked?' by content:")
    for k, v in per_lane["mean_score_by_class"].items():
        print(f"    {k:<6} {v:.3f}")
    print(f"  as a rule, lower score wins:")
    print(f"    avoids the barrier    {per_lane['avoids_the_barrier']}")
    print(f"    prefers a clear lane  {per_lane['prefers_a_clear_lane']}")
    print(f"  same-content pairs that tie exactly, leaving the harness to break them:")
    print(f"    {per_lane['same_content_pairs_that_tie_exactly']}")
    (RESULTS / "lane_forced.json").write_text(json.dumps(
        {"service": info, "rows": rows, "pairs": pairs, "per_lane": per_lane}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
