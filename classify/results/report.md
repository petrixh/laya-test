# Classifying documents with Laya

_Part two of this project. Part one put the model in a game; this part gives it a pile of documents and watches for the point where it stops coping._

## Read this first

- **The corpus is invented.** All 160 documents were written for this repository to be realistic in shape. None is a real message, and no accuracy here transfers to real traffic without re-measuring on real traffic.
- **One model, one checkpoint, one run.** `convaiinnovations/laya`, subfolder `None`, backend `mlx`, laya-mlx 0.1.0. Nothing here is averaged over seeds, because a forward pass is deterministic and repeats add nothing -- but the corpus is a single sample, and that sampling error is real.
- **Sample sizes are small.** 160 documents, ten per label, forty per tier. A per-tier cell at k=16 holds forty documents, so a difference under roughly ten points is not a difference.
- **Backends are not interchangeable.** The MLX port is an independent FP16 conversion. Every result file records which backend produced it; do not compare across them without checking.

## The short version

Over the whole corpus, a flat k-option question never reaches 90%, even at k=2, and stays at or above 75% up to k=3.

**That fall is mostly not about option count.** Hold the documents fixed and grow only the menu, and `choice` goes 0.850 -> 0.900 across 14 added distractors. What the main curve is mostly measuring is that the labels added later are harder and closer together, not that the list got longer.

**Where it does break is presentation.** At k=16, `choice` gives the same answer under a reordered menu only 64% of the time, against 100% for `noul_per_label`. Part one found the same thing on a three-lane game and fixed it the same way: ask about one thing at a time and there is no list to prefer the front of.

At k=16 the framings separate:

| framing | what it asks | accuracy | macro F1 | passes | same answer, reordered |
|---|---|---|---|---|---|
| `choice_bare` | the same question, bare label names and no descriptions | 0.550 | 0.558 | 1.0 | not measured |
| `choice` | one k-option question, every label with a written description | 0.544 | 0.545 | 1.0 | 0.637 |
| `noul_per_label` | k independent yes/no questions in one pass, highest score wins | 0.512 | 0.510 | 1.0 | 1.000 |
| `noul_described` | the same, with an explicit true/false reading per question | 0.350 | 0.319 | 1.0 | not measured |
| `two_stage` | a 4-option group question, then a question inside that group | 0.306 | 0.269 | 2.0 | not measured |

## Accuracy against option count

Chance is 1/k. The corpus is balanced ten documents per label, so the majority-class baseline sits at 1/k too.

| framing | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 | shape |
|---|---|---|---|---|---|---|---|---|
| `choice` | 0.850 | 0.767 | 0.675 | 0.683 | 0.600 | 0.517 | 0.544 | ▇▇▆▆▅▅▅ |
| `choice_bare` | 0.850 | 0.700 | 0.675 | 0.650 | 0.537 | 0.583 | 0.550 | ▇▆▆▆▅▅▅ |
| `noul_per_label` | 0.900 | 0.800 | 0.775 | 0.700 | 0.588 | 0.567 | 0.512 | █▇▇▆▅▅▅ |
| `noul_described` | 0.800 | 0.800 | 0.725 | 0.550 | 0.438 | 0.383 | 0.350 | ▇▇▆▅▄▄▃ |
| `two_stage` | 0.750 | 0.667 | 0.600 | 0.433 | 0.438 | 0.350 | 0.306 | ▇▆▅▄▄▃▃ |
| _chance_ | 0.500 | 0.333 | 0.250 | 0.167 | 0.125 | 0.083 | 0.062 |  |

Lift over chance, which is the fairer way to read a curve whose baseline is moving:

| framing | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|
| `choice` | 1.7x | 2.3x | 2.7x | 4.1x | 4.8x | 6.2x | 8.7x |
| `choice_bare` | 1.7x | 2.1x | 2.7x | 3.9x | 4.3x | 7.0x | 8.8x |
| `noul_per_label` | 1.8x | 2.4x | 3.1x | 4.2x | 4.7x | 6.8x | 8.2x |
| `noul_described` | 1.6x | 2.4x | 2.9x | 3.3x | 3.5x | 4.6x | 5.6x |
| `two_stage` | 1.5x | 2.0x | 2.4x | 2.6x | 3.5x | 4.2x | 4.9x |

### The same documents, a longer menu

The curve above cannot separate two things, because a document is only scored at k when its own gold label is on the menu: as k grows, the document set grows with it. This holds the documents fixed -- the 20 whose gold label is `billing` or `technical`, so they are scorable at every rung -- and grows only the number of wrong answers on offer. Any fall here is option count alone -- with one caveat. These labels are the first in the nested order, so the gold option always sits at the front of the menu and the distractors are added behind it: this measures wrong answers added after the right one, not menu length in general. Menu position itself is measured in the presentation section below.

| framing | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|
| `choice` | 0.850 | 0.850 | 0.750 | 0.800 | 0.900 | 0.850 | 0.900 |
| `choice_bare` | 0.850 | 0.750 | 0.750 | 0.700 | 0.650 | 0.650 | 0.650 |
| `noul_per_label` | 0.900 | 0.800 | 0.800 | 0.800 | 0.800 | 0.650 | 0.650 |
| `noul_described` | 0.800 | 0.800 | 0.800 | 0.750 | 0.650 | 0.650 | 0.650 |
| _distractors added_ | 0 | 1 | 2 | 4 | 6 | 10 | 14 |

## Accuracy by difficulty tier

The tiers, in order: a short message that names its own category; the same thing buried in greetings, a quoted thread and a signature; one where the category has to be inferred with none of the giveaway vocabulary present; and one carrying the surface vocabulary of a *different* category.

**`choice`**

| tier | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|
| explicit | 1.000 | 0.875 | 1.000 | 0.875 | 0.750 | 0.667 | 0.750 |
| realistic | 1.000 | 1.000 | 0.900 | 0.867 | 0.800 | 0.633 | 0.650 |
| implicit | 0.750 | 0.714 | 0.600 | 0.571 | 0.500 | 0.467 | 0.475 |
| adversarial | 0.600 | 0.500 | 0.200 | 0.400 | 0.350 | 0.300 | 0.300 |

**`choice_bare`**

| tier | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|
| explicit | 1.000 | 0.750 | 0.800 | 0.750 | 0.750 | 0.767 | 0.700 |
| realistic | 1.000 | 1.000 | 0.900 | 0.800 | 0.700 | 0.700 | 0.650 |
| implicit | 0.750 | 0.571 | 0.600 | 0.643 | 0.400 | 0.433 | 0.525 |
| adversarial | 0.600 | 0.500 | 0.400 | 0.400 | 0.300 | 0.433 | 0.325 |

**`noul_per_label`**

| tier | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|
| explicit | 1.000 | 0.875 | 0.900 | 0.875 | 0.850 | 0.733 | 0.750 |
| realistic | 1.000 | 0.857 | 0.700 | 0.667 | 0.600 | 0.633 | 0.550 |
| implicit | 0.750 | 0.857 | 0.800 | 0.714 | 0.550 | 0.500 | 0.500 |
| adversarial | 0.800 | 0.625 | 0.700 | 0.533 | 0.350 | 0.400 | 0.250 |

**`noul_described`**

| tier | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|
| explicit | 1.000 | 1.000 | 0.900 | 0.562 | 0.450 | 0.433 | 0.425 |
| realistic | 1.000 | 1.000 | 0.800 | 0.800 | 0.600 | 0.567 | 0.425 |
| implicit | 0.500 | 0.571 | 0.600 | 0.429 | 0.400 | 0.267 | 0.300 |
| adversarial | 0.600 | 0.625 | 0.600 | 0.400 | 0.300 | 0.267 | 0.250 |

**`two_stage`**

| tier | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|
| explicit | 1.000 | 1.000 | 0.800 | 0.500 | 0.550 | 0.400 | 0.375 |
| realistic | 0.800 | 0.714 | 0.700 | 0.600 | 0.550 | 0.467 | 0.425 |
| implicit | 0.750 | 0.571 | 0.500 | 0.286 | 0.300 | 0.300 | 0.250 |
| adversarial | 0.400 | 0.375 | 0.400 | 0.333 | 0.350 | 0.233 | 0.175 |

## Does it read the document, or the order of the options?

The control that needs no baseline: ask the same question with the options rearranged. A model reading the document answers the same way every time. `max excess` is how far the most-favoured option slot ran above its 1/k share.

| framing | k | orders | same answer every order | max excess |
|---|---|---|---|---|
| `choice` | 2 | 3 | 1.000 | 0.050 |
| `choice` | 3 | 3 | 0.833 | 0.067 |
| `choice` | 4 | 3 | 0.950 | 0.067 |
| `choice` | 6 | 3 | 0.750 | 0.178 |
| `choice` | 8 | 3 | 0.762 | 0.025 |
| `choice` | 12 | 3 | 0.742 | 0.078 |
| `choice` | 16 | 3 | 0.637 | 0.125 |
| `noul_per_label` | 2 | 3 | 1.000 | 0.100 |
| `noul_per_label` | 3 | 3 | 1.000 | 0.011 |
| `noul_per_label` | 4 | 3 | 1.000 | 0.075 |
| `noul_per_label` | 6 | 3 | 1.000 | 0.067 |
| `noul_per_label` | 8 | 3 | 1.000 | 0.050 |
| `noul_per_label` | 12 | 3 | 1.000 | 0.042 |
| `noul_per_label` | 16 | 3 | 1.000 | 0.021 |

## Calibration: what it says it knows

`confidence` here is laya's own, normalised Shannon entropy `1 - H(p)/log(k)` over the returned distribution. Read the `choice` rows with the checkpoint's calibration table in mind: `rl_agent_config.json` divides the logits by a temperature chosen by option count, 1.0 for six to ten options but 0.10 for eleven or more, so every `choice` answer at k>=11 is sharpened tenfold before it reaches this table. That, not the `log(k)` denominator, is the step between k=8 and k=12. ECE is the gap between confidence and realised accuracy; 0 is honest.

| framing | k | accuracy | mean confidence | ECE | Brier |
|---|---|---|---|---|---|
| `choice` | 2 | 0.850 | 0.493 | 0.444 | 0.239 |
| `choice` | 3 | 0.767 | 0.389 | 0.378 | 0.349 |
| `choice` | 4 | 0.675 | 0.327 | 0.373 | 0.445 |
| `choice` | 6 | 0.683 | 0.494 | 0.316 | 0.519 |
| `choice` | 8 | 0.600 | 0.463 | 0.246 | 0.608 |
| `choice` | 12 | 0.517 | 0.968 | 0.451 | 0.905 |
| `choice` | 16 | 0.544 | 0.966 | 0.422 | 0.856 |
| `choice_bare` | 2 | 0.850 | 0.469 | 0.381 | 0.284 |
| `choice_bare` | 3 | 0.700 | 0.399 | 0.374 | 0.424 |
| `choice_bare` | 4 | 0.675 | 0.427 | 0.291 | 0.467 |
| `choice_bare` | 6 | 0.650 | 0.673 | 0.233 | 0.573 |
| `choice_bare` | 8 | 0.537 | 0.612 | 0.210 | 0.723 |
| `choice_bare` | 12 | 0.583 | 0.977 | 0.401 | 0.801 |
| `choice_bare` | 16 | 0.550 | 0.975 | 0.430 | 0.879 |
| `noul_per_label` | 2 | 0.900 | 0.513 | 0.387 | 0.135 |
| `noul_per_label` | 3 | 0.800 | 0.484 | 0.332 | 0.250 |
| `noul_per_label` | 4 | 0.775 | 0.423 | 0.352 | 0.304 |
| `noul_per_label` | 6 | 0.700 | 0.403 | 0.312 | 0.439 |
| `noul_per_label` | 8 | 0.588 | 0.355 | 0.238 | 0.572 |
| `noul_per_label` | 12 | 0.567 | 0.325 | 0.245 | 0.602 |
| `noul_per_label` | 16 | 0.512 | 0.313 | 0.201 | 0.660 |
| `noul_described` | 2 | 0.800 | 0.233 | 0.567 | 0.225 |
| `noul_described` | 3 | 0.800 | 0.194 | 0.606 | 0.345 |
| `noul_described` | 4 | 0.725 | 0.172 | 0.553 | 0.440 |
| `noul_described` | 6 | 0.550 | 0.127 | 0.423 | 0.591 |
| `noul_described` | 8 | 0.438 | 0.105 | 0.332 | 0.717 |
| `noul_described` | 12 | 0.383 | 0.098 | 0.286 | 0.786 |
| `noul_described` | 16 | 0.350 | 0.088 | 0.262 | 0.834 |

### Post-hoc temperature scaling

This model has no sampling temperature -- `predict()` takes no such parameter and one forward pass is deterministic. What it does have is a probability distribution that can be softened after the fact. The temperature below was fitted on half the documents and scored on the other half. It changes no prediction: the argmax is invariant, so accuracy is untouched and only the confidence attached to it moves.

| k | fitted t | ECE before | ECE after (held out) | n held out |
|---|---|---|---|---|
| 2 | 2.2 | 0.084 | 0.186 | 10 |
| 3 | 1.7 | 0.237 | 0.343 | 15 |
| 4 | 2.1 | 0.268 | 0.414 | 20 |
| 6 | 0.5 | 0.180 | 0.216 | 30 |
| 8 | 0.9 | 0.254 | 0.252 | 40 |
| 12 | 5.0 | 0.420 | 0.292 | 60 |
| 16 | 5.0 | 0.399 | 0.292 | 80 |

## More than one label at a time

34 of the 160 documents carry a second gold label. Only the per-label framing can name more than one, because its scores are independent; a choice question distributes one unit of probability across the menu and can only ever name a single winner. `argmax only` is that ceiling, measured rather than asserted.

**`noul_per_label`, k=16**

| rule | micro P | micro R | micro F1 | macro F1 | exact set | ≥1 right | labels/doc |
|---|---|---|---|---|---|---|---|
| threshold 0.3 | 0.467 | 0.660 | 0.547 | 0.565 | 0.338 | 0.731 | 1.71 |
| threshold 0.5 | 0.568 | 0.603 | 0.585 | 0.589 | 0.438 | 0.700 | 1.29 |
| threshold 0.7 | 0.596 | 0.510 | 0.550 | 0.542 | 0.463 | 0.619 | 1.04 |
| argmax only | 0.606 | 0.500 | 0.548 | 0.541 | 0.475 | 0.606 | 1.00 |

Gold averages 1.212 labels per document.

**`noul_described`, k=16**

| rule | micro P | micro R | micro F1 | macro F1 | exact set | ≥1 right | labels/doc |
|---|---|---|---|---|---|---|---|
| threshold 0.3 | 0.221 | 0.608 | 0.325 | 0.347 | 0.131 | 0.644 | 3.33 |
| threshold 0.5 | 0.309 | 0.510 | 0.385 | 0.382 | 0.206 | 0.569 | 2.00 |
| threshold 0.7 | 0.376 | 0.438 | 0.405 | 0.374 | 0.269 | 0.500 | 1.41 |
| argmax only | 0.431 | 0.356 | 0.390 | 0.356 | 0.325 | 0.431 | 1.00 |

Gold averages 1.212 labels per document.

## What the decision actually rests on

The same question against the subject line alone, the body alone, and both. Not a reimplementation of laya's own `clean_email_body` -- a second copy of that here would be free to drift out of sync with the real one -- but it answers the same question: how much of the message is doing work.

| state given to the model | n | accuracy | macro F1 |
|---|---|---|---|
| subject+body | 160 | 0.544 | 0.545 |
| subject_only | 160 | 0.431 | 0.423 |
| body_only | 160 | 0.469 | 0.448 |

**Determinism.** The same prompt sent twice gave an identical distribution 20/20 times. Every comparison above assumes this.

## Cost

| framing (at k=16) | forward passes | p50 latency ms |
|---|---|---|
| `choice` | 1.0 | 54.0 |
| `choice_bare` | 1.0 | 37.1 |
| `noul_per_label` | 1.0 | 289.8 |
| `noul_described` | 1.0 | 340.2 |
| `two_stage` | 2.0 | 84.1 |

The whole sweep was 6588 calls in 2007.7s (3.3/s) against `mlx` on `mlx`.

A per-label question is one HTTP call and one forward pass, but it carries k questions rather than one, so it is not free: at k=16 it costs noticeably more wall time than the flat question it replaces. The two-stage framing pays two round trips for two small questions.

## The corpus

160 documents, 34 of them multi-label. Bodies run 84-950 characters, mean 384.

Types: email (138), chat (9), ticket (5), incident_report (2), form (2), notice (1), contract_clause (1), disclosure (1), application (1).

Ten documents per label; forty per tier. `corpus/documents.jsonl` holds them and `classify/tests/test_corpus.py` enforces the balance.

---

_Generated by `python -m classify.report` from `base-mlx-mlx-n160-o3__classify.json` on 2026-09-22._
