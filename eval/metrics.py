"""Accuracy and calibration metrics.

Implemented here rather than pulled from sklearn/laya so the eval runner needs
nothing but httpx and can run anywhere the service is reachable.
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


def summarise(records: list[dict]) -> dict:
    """records: {gold, pred, probabilities, confidence, top_prob, latency_ms, input_tokens}"""
    if not records:
        return {"n": 0}
    pairs = [(r["gold"], r["pred"]) for r in records]
    correct = [r["gold"] == r["pred"] for r in records]
    lat = sorted(r["latency_ms"] for r in records)
    toks = [r["input_tokens"] for r in records if r.get("input_tokens")]
    return {
        "n": len(records),
        "accuracy": round(accuracy(pairs), 4),
        "macro_f1": round(macro_f1(pairs), 4),
        # Two calibration views: the raw top-label probability, and laya's own
        # derived `confidence` field. Callers threshold on the latter.
        "ece_top_prob": round(ece([r["top_prob"] for r in records], correct), 4),
        "ece_confidence": round(ece([r["confidence"] for r in records], correct), 4),
        "brier": round(brier([r["probabilities"] for r in records], [r["gold"] for r in records]), 4),
        "mean_top_prob": round(sum(r["top_prob"] for r in records) / len(records), 4),
        "mean_confidence": round(sum(r["confidence"] for r in records) / len(records), 4),
        "latency_p50_ms": round(lat[len(lat) // 2], 1),
        "mean_input_tokens": round(sum(toks) / len(toks), 1) if toks else None,
        "reliability": reliability_table([r["confidence"] for r in records], correct),
    }
