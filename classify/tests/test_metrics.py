"""Unit tests for the metrics. No service required.

The single-label half is recovered from the abandoned `three-lanes` branch
alongside the code it tests. The multi-label, order-consistency, position-bias
and temperature-scaling tests are new.
"""
from __future__ import annotations

import pytest

from classify.framings import temperature_scale
from classify.metrics import (
    accuracy,
    brier,
    confusion,
    ece,
    macro_f1,
    multilabel_scores,
    order_consistency,
    position_bias,
    reliability_table,
)


# --- recovered -------------------------------------------------------------

def test_accuracy():
    assert accuracy([("a", "a"), ("a", "b")]) == 0.5
    assert accuracy([]) == 0.0


def test_macro_f1_perfect_and_worst():
    assert macro_f1([("a", "a"), ("b", "b")]) == 1.0
    assert macro_f1([("a", "b"), ("b", "a")]) == 0.0


def test_macro_f1_penalises_majority_collapse():
    """Always predicting the majority class must score far below accuracy."""
    pairs = [("a", "a")] * 9 + [("b", "a")]
    assert accuracy(pairs) == 0.9
    assert macro_f1(pairs) < 0.5


def test_ece_bounds():
    assert ece([1.0] * 10, [True] * 10) == 0.0            # confident and right
    assert ece([1.0] * 10, [False] * 10) == 1.0           # confident and wrong
    assert ece([0.5] * 10, [True] * 5 + [False] * 5) == 0.0   # honestly uncertain


def test_brier_perfect_and_worst():
    assert brier([{"a": 1.0, "b": 0.0}], ["a"]) == 0.0
    assert brier([{"a": 0.0, "b": 1.0}], ["a"]) == 2.0


def test_reliability_table_buckets():
    rows = reliability_table([0.05, 0.95, 0.96], [False, True, True])
    assert [r["n"] for r in rows] == [1, 2]
    assert rows[-1]["accuracy"] == 1.0


def test_confusion_excludes_the_diagonal():
    rows = confusion([("a", "a"), ("a", "b"), ("a", "b"), ("c", "b")])
    assert rows[0] == {"gold": "a", "pred": "b", "n": 2}
    assert all(r["gold"] != r["pred"] for r in rows)


# --- multi-label -----------------------------------------------------------

def test_multilabel_perfect():
    s = multilabel_scores([{"a", "b"}, {"c"}], [{"a", "b"}, {"c"}])
    assert s["micro_f1"] == 1.0
    assert s["exact_set_match"] == 1.0
    assert s["at_least_one_correct"] == 1.0


def test_multilabel_partial_credit():
    """Naming one of two gold labels is half the recall but full precision."""
    s = multilabel_scores([{"a", "b"}], [{"a"}])
    assert s["micro_precision"] == 1.0
    assert s["micro_recall"] == 0.5
    assert s["exact_set_match"] == 0.0
    assert s["at_least_one_correct"] == 1.0


def test_multilabel_overprediction_costs_precision():
    s = multilabel_scores([{"a"}], [{"a", "b", "c"}])
    assert s["micro_recall"] == 1.0
    assert round(s["micro_precision"], 4) == round(1 / 3, 4)


def test_multilabel_macro_punishes_an_ignored_label():
    """A label the predictor never names scores 0 and drags the macro down,
    which micro-averaging would hide."""
    gold = [{"a"}] * 9 + [{"b"}]
    pred = [{"a"}] * 10
    s = multilabel_scores(gold, pred)
    assert s["micro_f1"] == 0.9          # 9 TP against one FP and one FN
    assert s["macro_f1"] < 0.5
    assert s["per_label"]["b"]["f1"] == 0.0


# --- presentation controls -------------------------------------------------

def test_order_consistency():
    assert order_consistency([["a", "a", "a"], ["b", "b", "b"]]) == 1.0
    assert order_consistency([["a", "b"], ["c", "c"]]) == 0.5
    assert order_consistency([]) == 0.0


def test_position_bias_detects_a_favoured_slot():
    flat = position_bias([0, 1, 2, 3], k=4)
    assert flat["max_excess"] == 0.0
    always_first = position_bias([0, 0, 0, 0], k=4)
    assert always_first["max_excess"] == 0.75
    assert always_first["share_by_position"][0] == 1.0


# --- post-hoc calibration --------------------------------------------------

def test_temperature_scaling_preserves_the_argmax():
    """The whole point: it changes confidence, never the decision."""
    p = {"a": 0.6, "b": 0.3, "c": 0.1}
    for t in (0.5, 1.0, 2.0, 5.0):
        scaled = temperature_scale(p, t)
        assert max(scaled, key=scaled.get) == "a"
        assert round(sum(scaled.values()), 9) == 1.0


def test_temperature_above_one_flattens():
    p = {"a": 0.9, "b": 0.1}
    assert temperature_scale(p, 5.0)["a"] < p["a"]
    assert temperature_scale(p, 0.2)["a"] > p["a"]


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        temperature_scale({"a": 1.0}, 0)
