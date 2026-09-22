"""The sweep: how document classification degrades as the menu grows.

Axes, and why each is here:

  option count   part one found option count to be the sharpest edge on this
                 model. The taxonomy nests, so the curve over k isolates it.
  framing        whether decomposing the question recovers what a flat k-option
                 question loses. This is the actionable half.
  difficulty     four tiers, from a one-line message that names its own category
                 to one carrying the vocabulary of a different one.
  presentation   the same question with the options in a different order. A
                 model reading the document answers the same; one keyed on
                 position does not.

  python -m classify.sweep              # full run
  python -m classify.sweep --smoke      # a few dozen calls, for the slow path

LAYA_URL selects the service and defaults to the local container.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import sys
import time

from . import corpus as C
from . import framings as F
from . import metrics as M
from .service import Service, run_identity

RESULTS = pathlib.Path(__file__).parent / "results"

FRAMINGS = ("choice", "choice_bare", "noul_per_label", "noul_described", "two_stage")
# Only the framings where option order could plausibly matter are permuted.
PERMUTED = ("choice", "noul_per_label")
THRESHOLDS = (0.3, 0.5, 0.7)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def classify(svc, framing, state, labels, tax, order):
    desc, groups, gdesc = tax["descriptions"], tax["groups"], tax["group_descriptions"]
    if framing == "choice":
        return F.ask_choice(svc, state, labels, desc, described=True, order=order)
    if framing == "choice_bare":
        return F.ask_choice(svc, state, labels, desc, described=False, order=order)
    if framing == "noul_per_label":
        return F.ask_noul_per_label(svc, state, labels, desc, described=False, order=order)
    if framing == "noul_described":
        return F.ask_noul_per_label(svc, state, labels, desc, described=True, order=order)
    if framing == "two_stage":
        return F.ask_two_stage(svc, state, labels, desc, groups, gdesc, described=True)
    raise ValueError(framing)


def sweep(svc, docs, tax, rungs, framings, orders, want_multilabel=True):
    single: dict = {}
    by_tier: dict = {}
    presentation: dict = {}
    multilabel: dict = {}
    raw_records: list = []

    for framing in framings:
        single[framing], by_tier[framing], presentation[framing] = {}, {}, {}
        n_orders = orders if framing in PERMUTED else 1

        for k in rungs:
            labels = C.labels_at(k, tax)
            here = C.docs_at(docs, labels)
            if not here:
                continue
            rng = random.Random(1000 + k)
            orderings = [labels] + [random.Random(2000 + k + i).sample(labels, len(labels))
                                    for i in range(1, n_orders)]

            per_order_preds: list[list[str]] = [[] for _ in here]
            chosen_idx: list[int] = []
            records = []
            t0 = time.monotonic()

            for oi, ordering in enumerate(orderings):
                for di, doc in enumerate(here):
                    res = classify(svc, framing, C.as_state(doc), labels, tax, ordering)
                    per_order_preds[di].append(res["pred"])
                    if res["chosen_index"] >= 0:
                        chosen_idx.append(res["chosen_index"])
                    if oi == 0:                       # canonical order carries the metrics
                        rec = {"id": doc["id"], "tier": doc["tier"], "gold": doc["label"],
                               "pred": res["pred"], "probabilities": res["probs"],
                               "confidence": res["confidence"] if res["confidence"] is not None else 0.0,
                               "top_prob": res["probs"].get(res["pred"], 0.0),
                               "latency_ms": res["latency_ms"], "passes": res["passes"]}
                        records.append(rec)
                        if want_multilabel and framing in ("noul_per_label", "noul_described"):
                            rec["_raw"] = res.get("raw")
                            rec["_gold_set"] = sorted(C.gold_set_at(doc, labels))

            summary = M.summarise(records)
            summary["k"] = k
            summary["chance"] = round(1 / k, 4)
            summary["lift_over_chance"] = round(summary["accuracy"] * k, 2)
            summary["mean_passes"] = round(sum(r["passes"] for r in records) / len(records), 2)
            summary["wall_seconds"] = round(time.monotonic() - t0, 1)
            if framing == "two_stage":
                summary.pop("ece_confidence", None)      # no menu-wide distribution to calibrate
                summary.pop("ece_top_prob", None)
                summary.pop("brier", None)
                summary.pop("reliability", None)
                summary["mean_confidence"] = None
            single[framing][str(k)] = summary

            tiers = {}
            for tier in C.TIERS:
                sub = [r for r in records if r["tier"] == tier]
                if sub:
                    tiers[tier] = {"n": len(sub),
                                   "accuracy": round(M.accuracy([(r["gold"], r["pred"]) for r in sub]), 4)}
            by_tier[framing][str(k)] = tiers

            if n_orders > 1:
                presentation[framing][str(k)] = {
                    "orders": n_orders,
                    "order_consistency": M.order_consistency(per_order_preds),
                    "position_bias": M.position_bias(chosen_idx, k),
                }

            if want_multilabel and framing in ("noul_per_label", "noul_described") and k == max(rungs):
                golds = [set(r["_gold_set"]) for r in records]
                ml = {}
                for th in THRESHOLDS:
                    preds = [F.multilabel_from({"raw": r["_raw"], "pred": r["pred"]}, th)
                             for r in records]
                    ml[str(th)] = M.multilabel_scores(golds, preds)
                # the ceiling: a choice question can only ever name one label
                ml["argmax_only"] = M.multilabel_scores(golds, [{r["pred"]} for r in records])
                multilabel[framing] = {"k": k, "by_threshold": ml}

            for r in records:
                r.pop("_raw", None)
                r.pop("_gold_set", None)
                r.pop("probabilities", None)
                raw_records.append({**r, "framing": framing, "k": k})

            acc, ch = summary["accuracy"], summary["chance"]
            log(f"  {framing:<15} k={k:<3} n={summary['n']:<4} acc={acc:.3f} "
                f"(chance {ch:.3f}, {summary['lift_over_chance']}x)  "
                f"conf={summary['mean_confidence'] if summary['mean_confidence'] is None else round(summary['mean_confidence'],3)}  "
                f"{summary['wall_seconds']}s")

    return single, by_tier, presentation, multilabel, raw_records


def fixed_subset_curve(svc, docs, tax, rungs, framings):
    """The same documents at every k, so only the menu grows.

    The main curve cannot separate two things: a bigger menu, and a different
    set of documents, because a document is only scored at k when its own gold
    label is on the menu. This holds the documents fixed -- the ones whose gold
    label is on the *smallest* menu, so they are scorable at every rung -- and
    grows only the number of wrong answers on offer. Any fall here is option
    count alone.
    """
    base = C.labels_at(min(rungs), tax)
    here = C.docs_at(docs, base)
    out: dict = {}
    for framing in framings:
        out[framing] = {}
        for k in rungs:
            labels = C.labels_at(k, tax)
            pairs = []
            for doc in here:
                res = classify(svc, framing, C.as_state(doc), labels, tax, labels)
                pairs.append((doc["label"], res["pred"]))
            out[framing][str(k)] = {
                "n": len(pairs),
                "accuracy": round(M.accuracy(pairs), 4),
                "distractors": k - len(base),
            }
        line = "  ".join(f"k={k}:{out[framing][str(k)]['accuracy']:.3f}" for k in rungs)
        log(f"  fixed subset  {framing:<15} n={len(here)}  {line}")
    return {"documents": [d["id"] for d in here], "labels": base, "by_framing": out}


def ablate_fields(svc, docs, tax, k):
    """Does the message furniture help or hurt?

    Not a reimplementation of laya's own `clean_email_body` -- that lives in the
    model package and a second copy of it here would be free to drift. This is
    a plain field ablation instead, which answers the same question honestly:
    how much of the decision rests on the subject line alone.
    """
    labels = C.labels_at(k, tax)
    here = C.docs_at(docs, labels)
    out = {}
    for variant in ("subject+body", "subject_only", "body_only"):
        pairs = []
        for doc in here:
            if variant == "subject_only":
                state = {"subject": doc["subject"]}
            elif variant == "body_only":
                state = {"body": doc["body"]}
            else:
                state = C.as_state(doc)
            res = F.ask_choice(svc, state, labels, tax["descriptions"], described=True)
            pairs.append((doc["label"], res["pred"]))
        out[variant] = {"n": len(pairs), "accuracy": round(M.accuracy(pairs), 4),
                        "macro_f1": round(M.macro_f1(pairs), 4)}
        log(f"  fields {variant:<14} acc={out[variant]['accuracy']:.3f}")
    return out


def check_determinism(svc, docs, tax, k, n=20):
    """Same prompt twice. Any difference here would undermine every other number."""
    labels = C.labels_at(k, tax)
    here = C.docs_at(docs, labels)[:n]
    same = 0
    for doc in here:
        a = F.ask_choice(svc, C.as_state(doc), labels, tax["descriptions"])
        b = F.ask_choice(svc, C.as_state(doc), labels, tax["descriptions"])
        same += a["pred"] == b["pred"] and abs(a["probs"][a["pred"]] - b["probs"][b["pred"]]) < 1e-9
    return {"n": len(here), "identical": same}


def fit_temperature(records_by_k):
    """Post-hoc calibration, the only 'temperature' this model has.

    Fitted on the odd-indexed records and scored on the even ones, so the
    reported improvement is held out. Argmax is invariant under this transform,
    so accuracy cannot move -- only the confidence attached to it.
    """
    out = {}
    for k, recs in records_by_k.items():
        recs = [r for r in recs if r.get("_probs")]
        if len(recs) < 8:
            continue
        fit, held = recs[1::2], recs[0::2]
        best_t, best_e = 1.0, None
        for t in [round(0.5 + 0.1 * i, 2) for i in range(46)]:
            e = M.ece([max(F.temperature_scale(r["_probs"], t).values()) for r in fit],
                      [r["gold"] == r["pred"] for r in fit])
            if best_e is None or e < best_e:
                best_t, best_e = t, e
        before = M.ece([max(r["_probs"].values()) for r in held],
                       [r["gold"] == r["pred"] for r in held])
        after = M.ece([max(F.temperature_scale(r["_probs"], best_t).values()) for r in held],
                      [r["gold"] == r["pred"] for r in held])
        out[str(k)] = {"n_fit": len(fit), "n_held_out": len(held),
                       "temperature": best_t,
                       "ece_top_prob_before": round(before, 4),
                       "ece_top_prob_after": round(after, 4),
                       "accuracy_unchanged": True}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="a few dozen calls: enough to prove the path works on a slow service")
    ap.add_argument("--rungs", default=None, help="comma-separated k values")
    ap.add_argument("--framings", default=None, help="comma-separated framing names")
    ap.add_argument("--orders", type=int, default=3, help="presentation orders per document")
    ap.add_argument("--limit", type=int, default=0, help="cap documents per rung (0 = all)")
    ap.add_argument("--out", default=None, help="result filename stem")
    args = ap.parse_args()

    tax = C.load_taxonomy()
    docs = C.load_documents()

    if args.smoke:
        rungs, framings, orders = [4], ["choice", "noul_per_label"], 2
        docs = docs[:24]
    else:
        rungs = [int(x) for x in (args.rungs or ",".join(map(str, tax["rungs"]))).split(",")]
        framings = (args.framings or ",".join(FRAMINGS)).split(",")
        orders = args.orders
    if args.limit:
        docs = docs[: args.limit]

    t_start = time.monotonic()
    with Service() as svc:
        info = svc.wait_ready()
        log(f"service {svc.url}  backend={info.get('backend')} device={info.get('device')} "
            f"subfolder={info.get('subfolder')}")
        log(f"corpus  {len(docs)} documents, rungs {rungs}, framings {framings}\n")

        single, by_tier, presentation, multilabel, records = sweep(
            svc, docs, tax, rungs, framings, orders, want_multilabel=not args.smoke)

        extras = {}
        if not args.smoke:
            log("\n  option count with the documents held fixed")
            extras["fixed_subset"] = fixed_subset_curve(
                svc, docs, tax, rungs, [f for f in framings if f != "two_stage"])

            log("\n  ablations")
            extras["field_ablation"] = ablate_fields(svc, docs, tax, max(rungs))
            extras["determinism"] = check_determinism(svc, docs, tax, max(rungs))
            log(f"  determinism    {extras['determinism']['identical']}/"
                f"{extras['determinism']['n']} identical on a repeat call")

            log("  post-hoc temperature scaling")
            by_k: dict = {}
            for k in rungs:
                labels = C.labels_at(k, tax)
                here = C.docs_at(docs, labels)
                rs = []
                for doc in here:
                    r = F.ask_choice(svc, C.as_state(doc), labels, tax["descriptions"])
                    rs.append({"gold": doc["label"], "pred": r["pred"], "_probs": r["probs"]})
                by_k[k] = rs
            extras["temperature_scaling"] = fit_temperature(by_k)
            for k, v in extras["temperature_scaling"].items():
                log(f"    k={k:<3} t={v['temperature']:<5} "
                    f"ECE {v['ece_top_prob_before']:.3f} -> {v['ece_top_prob_after']:.3f} (held out)")

        calls, wall = svc.calls, round(time.monotonic() - t_start, 1)

    payload = {
        "generated": dt.datetime.now(dt.UTC).isoformat(),
        "service": info,
        "corpus": C.stats(docs),
        "config": {"rungs": rungs, "framings": framings, "orders": orders,
                   "thresholds": list(THRESHOLDS), "smoke": args.smoke},
        "single_label": single,
        "by_tier": by_tier,
        "presentation": presentation,
        "multilabel": multilabel,
        **extras,
        "cost": {"predict_calls": calls, "wall_seconds": wall,
                 "calls_per_second": round(calls / wall, 1) if wall else None},
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = args.out or run_identity(info, n=len(docs), o=orders) + ("-smoke" if args.smoke else "")
    path = RESULTS / f"{stem}__classify.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    (RESULTS / f"{stem}__records.json").write_text(json.dumps(records, indent=2) + "\n")
    log(f"\n{calls} calls in {wall}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
