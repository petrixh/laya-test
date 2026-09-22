"""Render a sweep result into results/report.md.

Stdlib only, so the report can be regenerated anywhere without the runner image.

  python -m classify.report                 # newest non-smoke result
  python -m classify.report --input X.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

RESULTS = pathlib.Path(__file__).parent / "results"
TIERS = ("explicit", "realistic", "implicit", "adversarial")
FRAMING_BLURB = {
    "choice": "one k-option question, every label with a written description",
    "choice_bare": "the same question, bare label names and no descriptions",
    "noul_per_label": "k independent yes/no questions in one pass, highest score wins",
    "noul_described": "the same, with an explicit true/false reading per question",
    "two_stage": "a 4-option group question, then a question inside that group",
}
# Thresholds the headline is computed against, so the claim is reproducible
# rather than eyeballed off the curve.
GOOD, USABLE = 0.90, 0.75


def pick_input() -> pathlib.Path:
    files = [p for p in RESULTS.glob("*__classify.json") if "smoke" not in p.name]
    if not files:
        files = list(RESULTS.glob("*__classify.json"))
    if not files:
        raise SystemExit(f"no sweep results in {RESULTS}")
    return max(files, key=lambda p: p.stat().st_mtime)


def fmt(x, nd=3, dash="  --  "):
    return dash if x is None else f"{x:.{nd}f}"


def table(rows: list[list[str]], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def breaking_point(series: dict[str, dict]) -> dict:
    """Largest k still at or above each bar, and the first k that falls below.

    Stated as a computed quantity rather than a reading off the chart, so it
    moves if the numbers move.
    """
    ks = sorted(int(k) for k in series)
    good = [k for k in ks if series[str(k)]["accuracy"] >= GOOD]
    usable = [k for k in ks if series[str(k)]["accuracy"] >= USABLE]
    below = [k for k in ks if series[str(k)]["accuracy"] < USABLE]
    return {"last_good": max(good) if good else None,
            "last_usable": max(usable) if usable else None,
            "first_below_usable": min(below) if below else None}


def sparkline(values: list[float]) -> str:
    bars = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    lo, hi = 0.0, 1.0
    return "".join(bars[min(len(bars) - 1, int((v - lo) / (hi - lo) * len(bars)))] for v in values)


def render(d: dict) -> str:
    svc, corpus, cfg = d["service"], d["corpus"], d["config"]
    single, by_tier = d["single_label"], d["by_tier"]
    rungs = [str(k) for k in cfg["rungs"]]
    framings = [f for f in cfg["framings"] if f in single and single[f]]
    L: list[str] = []
    A = L.append

    A("# Classifying documents with Laya\n")
    A("_Part two of this project. Part one put the model in a game; this part gives it "
      "a pile of documents and watches for the point where it stops coping._\n")

    # ---- caveats, before any number
    A("## Read this first\n")
    A(f"- **The corpus is invented.** All {corpus['n']} documents were written for this "
      "repository to be realistic in shape. None is a real message, and no accuracy here "
      "transfers to real traffic without re-measuring on real traffic.")
    A(f"- **One model, one checkpoint, one run.** `{svc.get('checkpoint')}`, subfolder "
      f"`{svc.get('subfolder')}`, backend `{svc.get('backend')}`, {svc.get('runtime')}. "
      "Nothing here is averaged over seeds, because a forward pass is deterministic and "
      "repeats add nothing -- but the corpus is a single sample, and that sampling error "
      "is real.")
    A(f"- **Sample sizes are small.** {corpus['n']} documents, ten per label, forty per "
      "tier. A per-tier cell at k=16 holds forty documents, so a difference under roughly "
      "ten points is not a difference.")
    A("- **Backends are not interchangeable.** The MLX port is an independent FP16 "
      "conversion. Every result file records which backend produced it; do not compare "
      "across them without checking.\n")

    # ---- headline
    A("## The short version\n")
    flat = "choice" if "choice" in single else framings[0]
    bp = breaking_point(single[flat])
    top_k = rungs[-1]

    def bar(v, pct, smallest):
        return (f"never reaches {pct:.0%}, even at k={smallest}" if v is None
                else f"stays at or above {pct:.0%} up to k={v}")

    A(f"Over the whole corpus, a flat k-option question "
      f"{bar(bp['last_good'], GOOD, rungs[0])}, and "
      f"{bar(bp['last_usable'], USABLE, rungs[0])}.\n")

    # The two findings that only fall out of the controls, stated from the numbers.
    fs = d.get("fixed_subset")
    if fs and flat in fs["by_framing"]:
        curve = fs["by_framing"][flat]
        lo, hi = curve[rungs[0]]["accuracy"], curve[top_k]["accuracy"]
        A(f"**That fall is mostly not about option count.** Hold the documents fixed and "
          f"grow only the menu, and `{flat}` goes {lo:.3f} -> {hi:.3f} across "
          f"{curve[top_k]['distractors']} added distractors. What the main curve is "
          f"mostly measuring is that the labels added later are harder and closer "
          f"together, not that the list got longer.\n")

    pres = d.get("presentation", {})
    if flat in pres and top_k in pres[flat]:
        oc_flat = pres[flat][top_k]["order_consistency"]
        others = [(f, p[top_k]["order_consistency"]) for f, p in pres.items()
                  if f != flat and top_k in p]
        A(f"**Where it does break is presentation.** At k={top_k}, `{flat}` gives the same "
          f"answer under a reordered menu only {oc_flat:.0%} of the time"
          + (f", against {max(v for _, v in others):.0%} for "
             f"`{max(others, key=lambda kv: kv[1])[0]}`" if others else "")
          + ". Part one found the same thing on a three-lane game and fixed it the same "
            "way: ask about one thing at a time and there is no list to prefer the front "
            "of.\n")

    A(f"At k={top_k} the framings separate:\n")
    rows = []
    for f in sorted(framings, key=lambda f: -single[f][top_k]["accuracy"]):
        sm = single[f][top_k]
        oc = pres.get(f, {}).get(top_k, {}).get("order_consistency")
        rows.append([f"`{f}`", FRAMING_BLURB.get(f, ""), fmt(sm["accuracy"]),
                     fmt(sm["macro_f1"]), str(sm["mean_passes"]),
                     fmt(oc) if oc is not None else "not measured"])
    A(table(rows, ["framing", "what it asks", "accuracy", "macro F1", "passes",
                   "same answer, reordered"]))
    A("")

    # ---- the curve
    A("## Accuracy against option count\n")
    A(f"Chance is 1/k. The corpus is balanced ten documents per label, so the "
      f"majority-class baseline sits at 1/k too.\n")
    head = ["framing"] + [f"k={k}" for k in rungs] + ["shape"]
    rows = []
    for f in framings:
        cells, vals = [], []
        for k in rungs:
            s = single[f].get(k)
            cells.append(fmt(s["accuracy"]) if s else "  --  ")
            if s:
                vals.append(s["accuracy"])
        rows.append([f"`{f}`"] + cells + [sparkline(vals)])
    rows.append(["_chance_"] + [fmt(1 / int(k)) for k in rungs] + [""])
    A(table(rows, head))
    A("")
    A("Lift over chance, which is the fairer way to read a curve whose baseline is moving:\n")
    rows = []
    for f in framings:
        rows.append([f"`{f}`"] + [(f"{single[f][k]['lift_over_chance']}x" if k in single[f] else "--")
                                  for k in rungs])
    A(table(rows, ["framing"] + [f"k={k}" for k in rungs]))
    A("")

    # ---- option count, documents held fixed
    if d.get("fixed_subset"):
        fs = d["fixed_subset"]
        A("### The same documents, a longer menu\n")
        A(f"The curve above cannot separate two things, because a document is only "
          f"scored at k when its own gold label is on the menu: as k grows, the "
          f"document set grows with it. This holds the documents fixed -- the "
          f"{len(fs['documents'])} whose gold label is `"
          + "` or `".join(fs["labels"]) + "`, so they are scorable at every rung -- "
          "and grows only the number of wrong answers on offer. Any fall here is "
          "option count alone -- with one caveat. These labels are the first in the "
          "nested order, so the gold option always sits at the front of the menu and "
          "the distractors are added behind it: this measures wrong answers added "
          "after the right one, not menu length in general. Menu position itself is "
          "measured in the presentation section below.\n")
        rows = []
        for f, per_k in fs["by_framing"].items():
            rows.append([f"`{f}`"] + [fmt(per_k[k]["accuracy"]) if k in per_k else "  --  "
                                      for k in rungs])
        rows.append(["_distractors added_"] +
                    [str(int(k) - len(fs["labels"])) for k in rungs])
        A(table(rows, ["framing"] + [f"k={k}" for k in rungs]))
        A("")

    # ---- difficulty tiers
    A("## Accuracy by difficulty tier\n")
    A("The tiers, in order: a short message that names its own category; the same thing "
      "buried in greetings, a quoted thread and a signature; one where the category has "
      "to be inferred with none of the giveaway vocabulary present; and one carrying the "
      "surface vocabulary of a *different* category.\n")
    for f in framings:
        if not by_tier.get(f):
            continue
        A(f"**`{f}`**\n")
        rows = []
        for tier in TIERS:
            cells = []
            for k in rungs:
                cell = by_tier[f].get(k, {}).get(tier)
                cells.append(fmt(cell["accuracy"]) if cell else "  --  ")
            rows.append([tier] + cells)
        A(table(rows, ["tier"] + [f"k={k}" for k in rungs]))
        A("")

    # ---- presentation controls
    if d.get("presentation"):
        A("## Does it read the document, or the order of the options?\n")
        A("The control that needs no baseline: ask the same question with the options "
          "rearranged. A model reading the document answers the same way every time. "
          "`max excess` is how far the most-favoured option slot ran above its 1/k share.\n")
        rows = []
        for f, per_k in d["presentation"].items():
            for k in rungs:
                p = per_k.get(k)
                if not p:
                    continue
                rows.append([f"`{f}`", k, str(p["orders"]),
                             fmt(p["order_consistency"]),
                             fmt(p["position_bias"]["max_excess"])])
        A(table(rows, ["framing", "k", "orders", "same answer every order", "max excess"]))
        A("")

    # ---- calibration
    A("## Calibration: what it says it knows\n")
    A("`confidence` here is laya's own, normalised Shannon entropy `1 - H(p)/log(k)` over "
      "the returned distribution. Read the `choice` rows with the checkpoint's calibration "
      "table in mind: `rl_agent_config.json` divides the logits by a temperature chosen by "
      "option count, 1.0 for six to ten options but 0.10 for eleven or more, so every "
      "`choice` answer at k>=11 is sharpened tenfold before it reaches this table. That, "
      "not the `log(k)` denominator, is the step between k=8 and k=12. ECE is the gap "
      "between confidence and realised accuracy; 0 is honest.\n")
    rows = []
    for f in framings:
        if single[f].get(rungs[0], {}).get("mean_confidence") is None:
            continue
        for k in rungs:
            s = single[f].get(k)
            if not s or s.get("mean_confidence") is None:
                continue
            rows.append([f"`{f}`", k, fmt(s["accuracy"]), fmt(s["mean_confidence"]),
                         fmt(s.get("ece_confidence")), fmt(s.get("brier"))])
    A(table(rows, ["framing", "k", "accuracy", "mean confidence", "ECE", "Brier"]))
    A("")

    if d.get("temperature_scaling"):
        A("### Post-hoc temperature scaling\n")
        A("This model has no sampling temperature -- `predict()` takes no such parameter "
          "and one forward pass is deterministic. What it does have is a probability "
          "distribution that can be softened after the fact. The temperature below was "
          "fitted on half the documents and scored on the other half. It changes no "
          "prediction: the argmax is invariant, so accuracy is untouched and only the "
          "confidence attached to it moves.\n")
        rows = [[k, str(v["temperature"]), fmt(v["ece_top_prob_before"]),
                 fmt(v["ece_top_prob_after"]), str(v["n_held_out"])]
                for k, v in d["temperature_scaling"].items()]
        A(table(rows, ["k", "fitted t", "ECE before", "ECE after (held out)", "n held out"]))
        A("")

    # ---- multi-label
    if d.get("multilabel"):
        A("## More than one label at a time\n")
        A(f"{corpus['multi_label']} of the {corpus['n']} documents carry a second gold "
          "label. Only the per-label framing can name more than one, because its scores "
          "are independent; a choice question distributes one unit of probability across "
          "the menu and can only ever name a single winner. `argmax only` is that ceiling, "
          "measured rather than asserted.\n")
        for f, block in d["multilabel"].items():
            A(f"**`{f}`, k={block['k']}**\n")
            rows = []
            for key, s in block["by_threshold"].items():
                name = "argmax only" if key == "argmax_only" else f"threshold {key}"
                rows.append([name, fmt(s["micro_precision"]), fmt(s["micro_recall"]),
                             fmt(s["micro_f1"]), fmt(s["macro_f1"]),
                             fmt(s["exact_set_match"]), fmt(s["at_least_one_correct"]),
                             fmt(s["mean_predicted_labels"], 2)])
            A(table(rows, ["rule", "micro P", "micro R", "micro F1", "macro F1",
                           "exact set", "≥1 right", "labels/doc"]))
            A(f"\nGold averages {block['by_threshold']['argmax_only']['mean_gold_labels']} "
              "labels per document.\n")

    # ---- ablations
    if d.get("field_ablation"):
        A("## What the decision actually rests on\n")
        A("The same question against the subject line alone, the body alone, and both. "
          "Not a reimplementation of laya's own `clean_email_body` -- a second copy of "
          "that here would be free to drift out of sync with the real one -- but it "
          "answers the same question: how much of the message is doing work.\n")
        rows = [[v_name, str(v["n"]), fmt(v["accuracy"]), fmt(v["macro_f1"])]
                for v_name, v in d["field_ablation"].items()]
        A(table(rows, ["state given to the model", "n", "accuracy", "macro F1"]))
        A("")

    if d.get("determinism"):
        det = d["determinism"]
        A(f"**Determinism.** The same prompt sent twice gave an identical distribution "
          f"{det['identical']}/{det['n']} times. Every comparison above assumes this.\n")

    # ---- cost
    A("## Cost\n")
    cost = d["cost"]
    rows = []
    for f in framings:
        s = single[f].get(rungs[-1])
        if s:
            rows.append([f"`{f}`", str(s["mean_passes"]), fmt(s["latency_p50_ms"], 1)])
    A(table(rows, [f"framing (at k={rungs[-1]})", "forward passes", "p50 latency ms"]))
    A("")
    A(f"The whole sweep was {cost['predict_calls']} calls in {cost['wall_seconds']}s "
      f"({cost['calls_per_second']}/s) against `{svc.get('backend')}` on "
      f"`{svc.get('device')}`.\n")
    A("A per-label question is one HTTP call and one forward pass, but it carries k "
      "questions rather than one, so it is not free: at k=16 it costs noticeably more "
      "wall time than the flat question it replaces. The two-stage framing pays two "
      "round trips for two small questions.\n")

    # ---- corpus appendix
    A("## The corpus\n")
    A(f"{corpus['n']} documents, {corpus['multi_label']} of them multi-label. "
      f"Bodies run {corpus['body_chars']['min']}-{corpus['body_chars']['max']} characters, "
      f"mean {corpus['body_chars']['mean']}.\n")
    A("Types: " + ", ".join(f"{k} ({v})" for k, v in corpus["per_type"].items()) + ".\n")
    A("Ten documents per label; forty per tier. `corpus/documents.jsonl` holds them and "
      "`classify/tests/test_corpus.py` enforces the balance.\n")

    A("---\n")
    A(f"_Generated by `python -m classify.report` from `{d.get('_source', 'results/')}` "
      f"on {d['generated'][:10]}._")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = pathlib.Path(args.input) if args.input else pick_input()
    d = json.loads(path.read_text())
    d["_source"] = path.name
    out = pathlib.Path(args.out) if args.out else RESULTS / "report.md"
    out.write_text(render(d))
    print(f"wrote {out} from {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
