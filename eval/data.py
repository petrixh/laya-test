"""Dataset fetch + caching.

Uses the HF datasets-server REST API rather than the `datasets` package so the
eval runner keeps its dependency footprint to httpx.
"""
from __future__ import annotations

import json
import pathlib
import time
from collections import defaultdict

import httpx

CACHE = pathlib.Path(__file__).parent / "data"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def fetch_split(dataset: str, config: str = "default", split: str = "test", limit: int = 4000) -> list[dict]:
    """Download a split page by page and cache it as JSON."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{dataset.replace('/', '__')}__{config}__{split}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    rows: list[dict] = []
    with httpx.Client(timeout=60.0) as c:
        offset = 0
        while offset < limit:
            r = c.get(ROWS_URL, params={
                "dataset": dataset, "config": config, "split": split,
                "offset": offset, "length": PAGE,
            })
            r.raise_for_status()
            batch = r.json()["rows"]
            if not batch:
                break
            rows.extend(item["row"] for item in batch)
            offset += PAGE
            time.sleep(0.05)

    cache_file.write_text(json.dumps(rows))
    return rows


def humanise(label: str) -> str:
    """`card_arrival` -> `card arrival`. Deliberately plain: this is the honest
    zero-effort baseline for label descriptions."""
    return label.replace("_", " ").replace("-", " ").strip()


def balanced_sample(rows: list[dict], labels: list[str], per_label: int, seed: int = 0) -> list[dict]:
    """Take up to `per_label` examples of each label, deterministically."""
    import random

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["label_text"] in labels:
            by_label[r["label_text"]].append(r)
    out = []
    for lab in labels:
        pool = by_label.get(lab, [])
        rng.shuffle(pool)
        out.extend(pool[:per_label])
    rng.shuffle(out)
    return out


def pick_labels(rows: list[dict], k: int, seed: int = 0) -> list[str]:
    """A deterministic k-subset of the label space. Same seed nests: the k=4 set
    is a subset of the k=8 set, so the degradation curve isolates label count
    rather than which labels were drawn."""
    import random

    all_labels = sorted({r["label_text"] for r in rows})
    rng = random.Random(seed)
    shuffled = all_labels[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:k])
