"""A live slice: proves the whole path works against a real service.

Deliberately small. On the CPU container a forward pass is most of a second, so
this is the check you can afford there; the full sweep wants faster hardware.
Every test here is about wiring and contract, not accuracy -- the sweep measures
accuracy, and a test that asserted a number would start failing the moment the
model or the corpus changed, which is not what a smoke test is for.
"""
from __future__ import annotations

import pytest

from classify import corpus as C
from classify import framings as F

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def tax():
    return C.load_taxonomy()


@pytest.fixture(scope="module")
def sample():
    """Two documents per tier, explicit tier first."""
    docs = C.load_documents()
    out = []
    for tier in C.TIERS:
        out += [d for d in docs if d["tier"] == tier][:2]
    return out


def test_service_reports_its_backend(svc):
    info = svc.info()
    assert info["ready"] is True
    assert info["backend"] in {"torch", "mlx"}
    assert info["checkpoint"]


def test_choice_returns_a_label_on_the_menu(svc, tax, sample):
    labels = C.labels_at(4, tax)
    for doc in sample[:4]:
        res = F.ask_choice(svc, C.as_state(doc), labels, tax["descriptions"])
        assert res["pred"] in labels
        assert 0.0 <= res["confidence"] <= 1.0
        assert round(sum(res["probs"].values()), 3) == 1.0
        assert res["passes"] == 1


def test_noul_per_label_returns_one_score_per_label(svc, tax, sample):
    """All k questions ride in a single forward pass -- that is what makes the
    decomposed framing cost the same as the flat one."""
    labels = C.labels_at(4, tax)
    res = F.ask_noul_per_label(svc, C.as_state(sample[0]), labels, tax["descriptions"])
    assert set(res["raw"]) == set(labels)
    assert all(0.0 <= v <= 1.0 for v in res["raw"].values())
    assert res["pred"] in labels
    assert res["passes"] == 1


def test_two_stage_costs_two_passes_and_lands_on_the_menu(svc, tax, sample):
    labels = C.labels_at(8, tax)
    res = F.ask_two_stage(svc, C.as_state(sample[0]), labels, tax["descriptions"],
                          tax["groups"], tax["group_descriptions"])
    assert res["pred"] in labels
    assert res["passes"] == 2
    assert res["group"] in tax["groups"]


def test_the_same_prompt_scores_identically(svc, tax, sample):
    """Determinism. Every comparison in the sweep assumes it."""
    labels = C.labels_at(4, tax)
    a = F.ask_choice(svc, C.as_state(sample[0]), labels, tax["descriptions"])
    b = F.ask_choice(svc, C.as_state(sample[0]), labels, tax["descriptions"])
    assert a["pred"] == b["pred"]
    assert a["probs"] == b["probs"]


def test_option_order_does_not_change_the_menu(svc, tax, sample):
    """Whether the answer moves is a finding for the sweep to report. What must
    hold regardless is that a reordered menu is still the same menu."""
    labels = C.labels_at(4, tax)
    rev = list(reversed(labels))
    res = F.ask_choice(svc, C.as_state(sample[0]), labels, tax["descriptions"], order=rev)
    assert res["pred"] in labels
    assert set(res["probs"]) == set(labels)


def test_the_easy_tier_is_not_hopeless(svc, tax, sample):
    """One weak sanity assertion, so a totally miswired prompt fails loudly:
    on a 2-option menu with explicit documents, chance is 0.5 and we should be
    clear of it. Anything sharper belongs in the sweep, not in a smoke test."""
    labels = C.labels_at(2, tax)
    docs = [d for d in C.load_documents() if d["tier"] == "explicit" and d["label"] in set(labels)]
    hits = sum(F.ask_choice(svc, C.as_state(d), labels, tax["descriptions"])["pred"] == d["label"]
               for d in docs)
    assert hits > len(docs) * 0.5, f"only {hits}/{len(docs)} on the easiest possible menu"
