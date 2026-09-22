"""Loading the document corpus and slicing the nested taxonomy.

The label list nests: `labels_at(k)` is always a prefix of `labels_at(k')` for
k < k'. A document is only scored at k when its primary gold label is on the
menu at k, so the curve over k measures option count rather than which labels
happened to be drawn.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent
CORPUS_DIR = HERE / "corpus"
TIERS = ("explicit", "realistic", "implicit", "adversarial")


def load_taxonomy() -> dict:
    return json.loads((CORPUS_DIR / "taxonomy.json").read_text())


def load_documents() -> list[dict]:
    path = CORPUS_DIR / "documents.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def labels_at(k: int, taxonomy: dict | None = None) -> list[str]:
    tax = taxonomy or load_taxonomy()
    order = tax["order"]
    if not 2 <= k <= len(order):
        raise ValueError(f"k must be between 2 and {len(order)}, got {k}")
    return order[:k]


def docs_at(docs: list[dict], labels: list[str]) -> list[dict]:
    """Documents whose primary gold label is on this menu."""
    on_menu = set(labels)
    return [d for d in docs if d["label"] in on_menu]


def gold_set_at(doc: dict, labels: list[str]) -> set[str]:
    """Gold labels restricted to the menu. A secondary label off the menu is
    dropped rather than counted as a miss -- it was never offerable."""
    return {l for l in doc["labels"] if l in set(labels)}


def as_state(doc: dict, clean: bool = False) -> dict:
    """The document as a state dict for /predict.

    `clean` routes the body through laya's own `clean_email_body`, which strips
    quoted history, signatures and disclaimers. That helper lives in the model
    package, not here, so it is applied service-side by the caller when
    available; this function only marks the intent.
    """
    state = {"subject": doc["subject"], "body": doc["body"]}
    if doc.get("type") and doc["type"] != "email":
        state["kind"] = doc["type"]
    return state


def stats(docs: list[dict]) -> dict:
    """Shape of the corpus, for the report header."""
    per_label: dict[str, int] = {}
    per_tier: dict[str, int] = {}
    per_type: dict[str, int] = {}
    for d in docs:
        per_label[d["label"]] = per_label.get(d["label"], 0) + 1
        per_tier[d["tier"]] = per_tier.get(d["tier"], 0) + 1
        per_type[d["type"]] = per_type.get(d["type"], 0) + 1
    multi = [d for d in docs if len(d["labels"]) > 1]
    bodies = [len(d["body"]) for d in docs]
    return {
        "n": len(docs),
        "multi_label": len(multi),
        "per_label": dict(sorted(per_label.items())),
        "per_tier": {t: per_tier.get(t, 0) for t in TIERS},
        "per_type": dict(sorted(per_type.items(), key=lambda kv: -kv[1])),
        "body_chars": {"min": min(bodies), "mean": round(sum(bodies) / len(bodies)),
                       "max": max(bodies)},
    }
