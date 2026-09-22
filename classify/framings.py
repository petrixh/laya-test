"""Ways of asking "which category is this document?", and how each is scored.

Part one of this project found that option count is the sharpest edge on this
model: a two-option question was reliable where a three-option one collapsed,
and the fix was to decompose the question rather than to reword it. These
framings carry that finding onto a classification task with a real label space.

  choice          one k-option question, each label with a written description
  choice_bare     the same question with bare label names and no descriptions
  noul_per_label  k independent yes/no questions, argmax of the yes score
  two_stage       a group question, then a label question inside that group

Every framing returns a distribution over the on-menu labels, so they are scored
identically and the comparison is like for like. `noul_per_label` and `choice`
both cost one forward pass; `two_stage` costs two.
"""
from __future__ import annotations

import math

INSTRUCTION = "Which category does this message belong to?"
GROUP_INSTRUCTION = "Which broad area does this message belong to?"

# Undocumented but supported: a noul question accepts a criteria object giving
# the false and true readings. laya.render_options falls back to "no, the
# statement does not hold" / "yes, the statement holds" when it is absent, so
# supplying it is a real knob and `noul_described` below measures whether it pays.
NOUL_BARE = "Is this message about {desc}?"


def _norm(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


def _confidence(probs: dict[str, float]) -> float:
    """Normalised Shannon entropy confidence, 1 - H(p)/log(k).

    This mirrors laya.confidence_from_probs so that a framing which builds its
    own distribution reports confidence on the same scale as one that gets it
    from the service. Note the log(k) denominator: it is why mean confidence
    drifts upward as k grows even when the distribution is no sharper.
    """
    k = len(probs)
    if k < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, min(1.0, 1 - h / math.log(k)))


# --- the framings ---------------------------------------------------------


def ask_choice(svc, state, labels, descriptions, *, described=True, order=None):
    """One k-option choice. `order` presents the options in a given sequence."""
    seq = order or labels
    criteria = {l: (descriptions[l] if described else "") for l in seq}
    q = {"cat": {"type": "choice", "instructions": INSTRUCTION, "criteria": criteria}}
    res = svc.predict(state, q)
    ans = res["answers"]["cat"]
    probs = {l: ans["probabilities"].get(l, 0.0) for l in labels}
    return {"probs": probs, "pred": ans["choice"], "confidence": ans["confidence"],
            "latency_ms": res["latency_ms"], "passes": 1,
            "chosen_index": seq.index(ans["choice"]) if ans["choice"] in seq else -1}


def ask_noul_per_label(svc, state, labels, descriptions, *, described=False, order=None):
    """One yes/no per label, all in a single forward pass.

    There is no option list, so there is no front of a list to prefer. The
    scores are independent, which is also what makes this the only framing that
    can name more than one label -- see `multilabel_from`.
    """
    seq = order or labels
    questions = {}
    for l in seq:
        q = {"type": "noul", "instructions": NOUL_BARE.format(desc=descriptions[l])}
        if described:
            q["criteria"] = {"true": f"yes, this message is about {descriptions[l]}",
                             "false": f"no, this message is not about {descriptions[l]}"}
        questions[f"is_{l}"] = q
    res = svc.predict(state, questions)
    raw = {l: res["answers"][f"is_{l}"]["noul"] for l in labels}
    probs = _norm(raw)
    pred = max(raw, key=lambda l: raw[l])
    return {"probs": probs, "raw": raw, "pred": pred, "confidence": _confidence(probs),
            "latency_ms": res["latency_ms"], "passes": 1,
            "chosen_index": seq.index(pred) if pred in seq else -1}


def ask_two_stage(svc, state, labels, descriptions, groups, group_descriptions, *,
                  described=True, order=None):
    """Pick a group, then a label inside it. Two passes, both small questions.

    Built to exploit the part-one finding directly: at k=16 this asks a
    4-option question and then another 4-option question, where the flat
    framing asks one 16-option question.
    """
    on_menu = set(labels)
    live = {g: [l for l in ls if l in on_menu] for g, ls in groups.items()}
    live = {g: ls for g, ls in live.items() if ls}

    if len(live) == 1:
        group = next(iter(live))
        stage1_latency, passes = 0.0, 0
    else:
        crit = {g: (group_descriptions[g] if described else "") for g in live}
        r1 = svc.predict(state, {"grp": {"type": "choice", "instructions": GROUP_INSTRUCTION,
                                         "criteria": crit}})
        group = r1["answers"]["grp"]["choice"]
        stage1_latency, passes = r1["latency_ms"], 1
        if group not in live:
            group = next(iter(live))

    members = live[group]
    if len(members) == 1:
        pred, stage2_latency = members[0], 0.0
    else:
        crit = {l: (descriptions[l] if described else "") for l in members}
        r2 = svc.predict(state, {"cat": {"type": "choice", "instructions": INSTRUCTION,
                                         "criteria": crit}})
        pred, stage2_latency = r2["answers"]["cat"]["choice"], r2["latency_ms"]
        passes += 1

    # A distribution over the whole menu is not available from a staged
    # decision, so calibration is not reported for this framing rather than
    # being invented. The hard prediction is what it is scored on.
    probs = {l: (1.0 if l == pred else 0.0) for l in labels}
    return {"probs": probs, "pred": pred, "confidence": None,
            "latency_ms": stage1_latency + stage2_latency, "passes": passes,
            "chosen_index": -1, "group": group}


# --- multi-label ----------------------------------------------------------


def multilabel_from(result: dict, threshold: float) -> set[str]:
    """Labels a framing asserts, at a given threshold.

    Only `noul_per_label` produces independent per-label scores, so only it can
    name more than one. For the choice framings the raw scores are a simplex and
    thresholding them is not the same operation; they are handled by taking the
    argmax alone, which is exactly the ceiling this comparison exists to show.
    """
    raw = result.get("raw")
    if raw is None:
        return {result["pred"]}
    hits = {l for l, v in raw.items() if v >= threshold}
    return hits or {result["pred"]}      # never return nothing


# --- post-hoc calibration -------------------------------------------------


def temperature_scale(probs: dict[str, float], t: float) -> dict[str, float]:
    """Raise a distribution to the power 1/t and renormalise.

    The one sense in which this model has a "temperature". It has no sampling
    temperature -- predict() takes no such parameter and one forward pass is
    deterministic -- but the returned distribution can still be softened after
    the fact. t > 1 flattens, t < 1 sharpens. Fitted on a split of the corpus
    and applied to the rest, this is the standard remedy for the overconfidence
    the sweep measures, and it changes no prediction: argmax is invariant.
    """
    if t <= 0:
        raise ValueError("temperature must be positive")
    powered = {k: (v ** (1 / t) if v > 0 else 0.0) for k, v in probs.items()}
    return _norm(powered)
