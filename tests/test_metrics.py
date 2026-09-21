"""Unit tests for the eval metrics. No service required."""
import pytest

from eval.metrics import accuracy, brier, ece, macro_f1, reliability_table
from eval.runner import kendall_tau


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
    assert ece([1.0] * 10, [True] * 10) == 0.0          # confident and right
    assert ece([1.0] * 10, [False] * 10) == 1.0         # confident and wrong
    assert ece([0.5] * 10, [True] * 5 + [False] * 5) == 0.0  # honestly uncertain


def test_ece_detects_overconfidence():
    # claims 90%, delivers 50%
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)


def test_brier():
    assert brier([{"a": 1.0, "b": 0.0}], ["a"]) == 0.0
    assert brier([{"a": 0.0, "b": 1.0}], ["a"]) == 2.0   # max for one-hot wrong


def test_reliability_table_buckets():
    rows = reliability_table([0.15, 0.95, 0.92], [False, True, True])
    assert [r["bucket"] for r in rows] == ["0.1-0.2", "0.9-1.0"]
    assert rows[1]["n"] == 2
    assert rows[1]["accuracy"] == 1.0


def test_kendall_tau_penalises_ties():
    """A ladder that is flat except for one step is not perfectly ordered.

    Dropping ties from the denominator as well as the numerator scores this
    1.0, which would hide exactly the collapse the ladders exist to catch.
    """
    flat = kendall_tau(list(range(9, 0, -1)), [0.5] * 8 + [0.4])
    assert 0.2 < flat < 0.7, flat
    assert kendall_tau([3, 2, 1], [0.5, 0.5, 0.5]) == 0.0


def test_kendall_tau():
    assert kendall_tau([3, 2, 1], [3.0, 2.0, 1.0]) == 1.0
    assert kendall_tau([3, 2, 1], [1.0, 2.0, 3.0]) == -1.0
    # one adjacent swap out of three pairs -> 1 discordant, 2 concordant
    assert kendall_tau([3, 2, 1], [3.0, 1.0, 2.0]) == pytest.approx(1 / 3)
