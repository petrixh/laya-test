"""Accuracy, calibration and multi-label metrics.

The single-label half of this file (accuracy, macro_f1, ece, brier,
reliability_table, summarise) is recovered from the abandoned `three-lanes`
branch, where it was written for the Banking77 label-budget sweep and then
deleted along with that sweep. It is unchanged apart from this note: it was
already exercised by unit tests, which come back with it.

Implemented here rather than pulled from sklearn so the suite needs nothing but
httpx and runs anywhere the service is reachable.
"""
from __future__ import annotations

from collections import defaultdict


def accuracy(pairs: list[tuple[str, str]]) -> float:
    """pairs: (gold, pred)"""
    return sum(g == p for g, p in pairs) / len(pairs) if pairs else 0.0


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = {g for g, _ in pairs} | {p for _, p in pairs}
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy| averaged over bins,
    weighted by bin population. 0 is perfect."""
    if not confidences:
        return 0.0
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, ok in zip(confidences, correct):
        idx = min(int(c * bins), bins - 1)
        buckets[idx].append((c, ok))
    n = len(confidences)
    total = 0.0
    for items in buckets.values():
        avg_conf = sum(c for c, _ in items) / len(items)
        avg_acc = sum(ok for _, ok in items) / len(items)
        total += (len(items) / n) * abs(avg_conf - avg_acc)
    return total


def brier(prob_dists: list[dict[str, float]], golds: list[str]) -> float:
    """Multiclass Brier score: mean squared error of the whole distribution.
    Lower is better; 0 is perfect."""
    if not prob_dists:
        return 0.0
    total = 0.0
    for probs, gold in zip(prob_dists, golds):
        total += sum((p - (1.0 if lab == gold else 0.0)) ** 2 for lab, p in probs.items())
    return total / len(prob_dists)


def reliability_table(confidences: list[float], correct: list[bool], bins: int = 10) -> list[dict]:
    """Per-bucket confidence vs realised accuracy. Shows *where* miscalibration lives."""
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, ok in zip(confidences, correct):
        buckets[min(int(c * bins), bins - 1)].append((c, ok))
    rows = []
    for idx in sorted(buckets):
        items = buckets[idx]
        rows.append({
            "bucket": f"{idx / bins:.1f}-{(idx + 1) / bins:.1f}",
            "n": len(items),
            "mean_confidence": round(sum(c for c, _ in items) / len(items), 4),
            "accuracy": round(sum(ok for _, ok in items) / len(items), 4),
        })
    return rows


def confusion(pairs: list[tuple[str, str]], top: int = 12) -> list[dict]:
    """The most frequent gold->pred mistakes. Diagonal excluded."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for g, p in pairs:
        if g != p:
            counts[(g, p)] += 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return [{"gold": g, "pred": p, "n": n} for (g, p), n in ranked]


def summarise(records: list[dict]) -> dict:
    """records: {gold, pred, probabilities, confidence, top_prob, latency_ms}"""
    if not records:
        return {"n": 0}
    pairs = [(r["gold"], r["pred"]) for r in records]
    correct = [r["gold"] == r["pred"] for r in records]
    lat = sorted(r["latency_ms"] for r in records)
    return {
        "n": len(records),
        "accuracy": round(accuracy(pairs), 4),
        "macro_f1": round(macro_f1(pairs), 4),
        # Two calibration views: the raw top-label probability, and laya's own
        # derived `confidence`. Callers threshold on the latter.
        "ece_top_prob": round(ece([r["top_prob"] for r in records], correct), 4),
        "ece_confidence": round(ece([r["confidence"] for r in records], correct), 4),
        "brier": round(brier([r["probabilities"] for r in records], [r["gold"] for r in records]), 4),
        "mean_top_prob": round(sum(r["top_prob"] for r in records) / len(records), 4),
        "mean_confidence": round(sum(r["confidence"] for r in records) / len(records), 4),
        "latency_p50_ms": round(lat[len(lat) // 2], 1),
        "reliability": reliability_table([r["confidence"] for r in records], correct),
        "confusion": confusion(pairs),
    }


# --- multi-label ----------------------------------------------------------
# A document may carry more than one gold label. `choice` can only ever name
# one, so it is scored here too in order to show that ceiling explicitly rather
# than leaving the comparison unstated.


def multilabel_scores(gold_sets: list[set[str]], pred_sets: list[set[str]]) -> dict:
    """Micro and macro P/R/F1, plus exact-set match.

    Micro pools every (document, label) decision, so frequent labels dominate.
    Macro averages per-label F1, so a label the model never predicts costs as
    much as a common one. They separate "mostly right on the easy labels" from
    "right across the board", which is the distinction that matters here.
    """
    assert len(gold_sets) == len(pred_sets)
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    labels = sorted({l for s in gold_sets for l in s} | {l for s in pred_sets for l in s})
    per_label, f1s = {}, []
    for lab in labels:
        ltp = sum(lab in g and lab in p for g, p in zip(gold_sets, pred_sets))
        lfp = sum(lab not in g and lab in p for g, p in zip(gold_sets, pred_sets))
        lfn = sum(lab in g and lab not in p for g, p in zip(gold_sets, pred_sets))
        p_ = ltp / (ltp + lfp) if ltp + lfp else 0.0
        r_ = ltp / (ltp + lfn) if ltp + lfn else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        f1s.append(f_)
        per_label[lab] = {"support": sum(lab in g for g in gold_sets),
                          "precision": round(p_, 4), "recall": round(r_, 4), "f1": round(f_, 4)}

    exact = sum(g == p for g, p in zip(gold_sets, pred_sets)) / len(gold_sets)
    # Did it find at least one correct label? The lenient reading, and the one
    # that matters if the output is a routing hint rather than a final answer.
    any_hit = sum(bool(g & p) for g, p in zip(gold_sets, pred_sets)) / len(gold_sets)
    return {
        "n": len(gold_sets),
        "micro_precision": round(micro_p, 4), "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "exact_set_match": round(exact, 4),
        "at_least_one_correct": round(any_hit, 4),
        "mean_predicted_labels": round(sum(len(p) for p in pred_sets) / len(pred_sets), 3),
        "mean_gold_labels": round(sum(len(g) for g in gold_sets) / len(gold_sets), 3),
        "per_label": per_label,
    }


def order_consistency(pred_lists: list[list[str]]) -> float:
    """Each inner list is the prediction for one document under several
    presentation orders of the same options. 1.0 means the answer never moved.

    This is the control that needs no baseline: a model reading the document
    names the same label however the options are arranged. One keyed on option
    position scores near 0 whenever the orders disagree.
    """
    if not pred_lists:
        return 0.0
    return round(sum(len(set(p)) == 1 for p in pred_lists) / len(pred_lists), 4)


def position_bias(chosen_index: list[int], k: int) -> dict:
    """How often each option slot won, against the 1/k a position-blind model gives.

    `max_excess` is the largest amount by which any one slot beat its share --
    the single number that says "it likes slot j".
    """
    if not chosen_index:
        return {"k": k, "n": 0}
    n = len(chosen_index)
    share = [round(sum(i == j for i in chosen_index) / n, 4) for j in range(k)]
    return {"k": k, "n": n, "uniform": round(1 / k, 4), "share_by_position": share,
            "max_excess": round(max(share) - 1 / k, 4)}
