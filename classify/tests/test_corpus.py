"""Corpus integrity. No service required.

These guard the properties every number in the report depends on. If the
allocation drifts, a per-tier or per-label comparison stops being like for like
and nothing downstream will notice on its own.
"""
from __future__ import annotations

import pytest

from classify import corpus as C

TIERS = C.TIERS


@pytest.fixture(scope="module")
def docs():
    return C.load_documents()


@pytest.fixture(scope="module")
def tax():
    return C.load_taxonomy()


def test_corpus_size(docs):
    assert len(docs) == 160
    assert len({d["id"] for d in docs}) == 160


def test_every_document_has_the_same_fields(docs):
    want = {"id", "tier", "type", "subject", "body", "label", "labels"}
    assert {frozenset(d) for d in docs} == {frozenset(want)}


def test_labels_are_balanced(docs, tax):
    """Ten documents per label, so chance and the majority-class baseline
    coincide at exactly 1/k and the report can quote one number for both."""
    for label in tax["order"]:
        assert sum(1 for d in docs if d["label"] == label) == 10, label


def test_tiers_are_balanced(docs):
    for tier in TIERS:
        assert sum(1 for d in docs if d["tier"] == tier) == 40, tier


def test_allocation_grid(docs, tax):
    """The grid the corpus was written against: 3 documents where (i+t)%4 < 2,
    otherwise 2. That is what makes both margins come out exactly balanced."""
    for i, label in enumerate(tax["order"]):
        for t, tier in enumerate(TIERS):
            want = 3 if (i + t) % 4 < 2 else 2
            got = sum(1 for d in docs if d["label"] == label and d["tier"] == tier)
            assert got == want, f"{label}/{tier}: want {want}, got {got}"


def test_gold_labels_are_in_the_taxonomy(docs, tax):
    known = set(tax["order"])
    for d in docs:
        assert d["label"] in known, d["id"]
        assert set(d["labels"]) <= known, d["id"]
        assert d["labels"][0] == d["label"], d["id"]
        assert len(set(d["labels"])) == len(d["labels"]), f"{d['id']} repeats a label"


def test_multi_label_documents_exist(docs):
    multi = [d for d in docs if len(d["labels"]) > 1]
    assert len(multi) >= 30, "the multi-label half of the sweep needs material"


def test_taxonomy_nests(tax):
    """Every rung is a prefix of the next, which is what lets the curve over k
    be read as an effect of option count rather than of label choice."""
    rungs = tax["rungs"]
    for a, b in zip(rungs, rungs[1:]):
        assert C.labels_at(a, tax) == C.labels_at(b, tax)[:a]


def test_groups_partition_the_label_space(tax):
    flat = [l for ls in tax["groups"].values() for l in ls]
    assert sorted(flat) == sorted(tax["order"])
    assert len(flat) == len(set(flat))
    assert set(tax["groups"]) == set(tax["group_descriptions"])


def test_every_label_has_a_description(tax):
    for label in tax["order"]:
        assert tax["descriptions"].get(label, "").strip(), label


def test_docs_at_restricts_to_the_menu(docs, tax):
    labels = C.labels_at(4, tax)
    here = C.docs_at(docs, labels)
    assert here and all(d["label"] in set(labels) for d in here)
    assert len(here) == 40


def test_gold_set_drops_off_menu_labels(docs, tax):
    """A secondary label that is not offerable at this k must not count as a
    miss -- it was never on the menu to be chosen."""
    labels = C.labels_at(2, tax)
    for d in docs:
        assert C.gold_set_at(d, labels) <= set(labels)


def test_no_contact_details_in_the_corpus(docs):
    """The corpus is invented and must stay that way: it ships in a public
    repository, so nothing in it may look like a real person's contact detail."""
    import re
    blob = "\n".join(d["subject"] + "\n" + d["body"] for d in docs)
    assert "@" not in blob
    assert not re.search(r"https?://", blob)
    assert not re.search(r"([0-9]{1,3}\.){3}[0-9]{1,3}", blob)
    assert not re.search(r"[0-9]{11,}", blob), "no card- or account-length digit runs"
    assert not re.search(r"(?i)\b(sk-|pk_|AKIA|ghp_|glpat-|BEGIN [A-Z]+ PRIVATE KEY)", blob)
