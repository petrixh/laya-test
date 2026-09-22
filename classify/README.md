# Classifying documents with Laya

> **Part two of an afternoon hack, written end to end by an LLM.** Part one put
> the model in a three-lane game, where a wrong decision is visible because the
> reindeer hits a wall. This part is the boring case that actually turns up in
> production: a pile of emails, tickets, notices and chat messages that have to
> land in the right queue.
>
> Everything here — the corpus, the harness, the measurements — was produced by
> Claude Code with a human steering. The corpus is **invented**, the samples are
> **small**, and nothing has been independently verified. Reproduce anything you
> intend to rely on; the whole point of this folder is that you can.

This folder is self-contained. It does not touch the game, and the game does not
touch it.

## The question

Part one found one thing that generalises beyond a game about reindeer:

> **Option count is the sharpest edge on this model.** A two-option question was
> reliable where a three-option one collapsed, and the fix was to *decompose the
> question* rather than to reword it.

A document classifier is exactly a question with as many options as you have
categories. So: how far does that go before it breaks, does decomposing still
rescue it, and what does the failure look like on the way down?

## How it is set up

**A nested taxonomy.** Sixteen business categories in a fixed order, where the
first *k* are the taxonomy at *k*. Every rung is a prefix of the next, so the
curve over *k* is not confounded by *which* labels happened to be drawn. It is
still confounded by which *documents* are scorable at each rung — see the
fixed-subset control below, which is what actually separates the two.

```
k=2   billing technical
k=4   + sales legal
k=8   + security hr product_feedback account_admin
k=16  + logistics partnerships press facilities compliance procurement training data_privacy
```

The later labels are deliberately close neighbours of the earlier ones —
`compliance` against `legal`, `data_privacy` against `security` — because near
neighbours are what make a large menu hard.

**160 documents, four difficulty tiers, forty each.**

| tier | what it is |
|---|---|
| `explicit` | short, and it names its own category in plain words |
| `realistic` | the same signal buried in greetings, a quoted thread and a signature |
| `implicit` | the category has to be inferred; none of the giveaway vocabulary is there |
| `adversarial` | carries the surface vocabulary of a *different* category |

An adversarial document is not a trick for its own sake. It is the message that
says `Re: Invoice #7781` and turns out to be about a download failing — the kind
that a keyword rule gets wrong and a reader gets right.

Exactly ten documents per label and forty per tier, enforced by
`tests/test_corpus.py`. Because the corpus is balanced, chance and the
majority-class baseline are both exactly 1/k, so one number covers both.

Thirty-four documents carry a **second** gold label, for the multi-label half.

**Five framings**, scored identically so the comparison is like for like:

| framing | what it asks |
|---|---|
| `choice` | one k-option question, every label with a written description |
| `choice_bare` | the same question with bare label names, no descriptions |
| `noul_per_label` | k independent yes/no questions in a single forward pass, highest wins |
| `noul_described` | the same, with an explicit true/false reading supplied per question |
| `two_stage` | a 4-option group question, then a question inside that group |

`two_stage` exists to take part one's finding literally: at k=16 it asks a
4-option question and then another 4-option question, where the flat framing
asks one 16-option question.

**Controls,** because a classification number on its own is not worth much:

- **Option order.** The same question with the options rearranged. A model
  reading the document answers the same way; one keyed on position does not.
  This is the control that needs no baseline, it is what caught the lane
  question in part one, and it turned out to be the one that matters most here.
- **A fixed document subset.** The same twenty documents at every rung, with
  only the number of wrong answers growing. Without this the curve over *k*
  cannot tell a longer menu apart from a harder document set, and on this
  corpus it was mostly the latter.
- **Position bias.** How far the most-favoured option slot ran above its 1/k share.
- **Determinism.** The same prompt twice. Every comparison assumes it.
- **Baselines.** Chance and majority, on every row.

## Results

**[`results/report.md`](results/report.md)** — generated, not written by hand.
Regenerate it with `make report` from any result file. There is a rendered
version in [`assets/report.html`](assets/report.html).

The short version: the option-count hypothesis above **did not replicate**.
Accuracy does fall as the menu grows, but hold the documents fixed and grow
only the distractors and a described `choice` question is flat to slightly up
across fourteen of them. What actually breaks is presentation — at k=16 a flat
menu gives a different answer for a third of documents when it is reshuffled,
which no accuracy number reveals — and the model becomes *more* confident as it
becomes less accurate, because its confidence is `1 - H(p)/log(k)` and the
denominator grows with the menu.

The raw numbers sit beside it as JSON. Filenames encode everything that makes
two runs incomparable — checkpoint, backend, device, corpus size, orders —
because on this project an `n=120` run once silently overwrote an `n=150`
baseline that differed only in that.

## Running it

The endpoint is `LAYA_URL` and defaults to a service on this machine.
**No host is hard-coded anywhere in this repository.** To use faster hardware,
pass one in:

```bash
make test                                  # offline: corpus + metrics. No service.
make smoke                                 # + a live slice and a few dozen sweep calls
make sweep                                 # the full sweep
make report                                # render results into report.md
make all                                   # all of the above, stopping on the first failure
```

```bash
make sweep LAYA_URL=http://your-fast-host:8000
```

`make all` runs the tests first and the report last, so a red suite never
produces a report for someone to quote from.

On the CPU container a forward pass is most of a second and the full sweep is a
few thousand calls, so `make smoke` is the one to run there — it is sized to
prove the path works, not to measure anything. The full sweep wants a faster
service.

Everything runs in the `laya-tests` image, which has `pytest` and `httpx` and
deliberately **not** the model package: the suite talks to the service only
through its HTTP contract, so it cannot accidentally test a second local
implementation instead of the real one.

## What is actually tunable

The first question anyone asks is what the knobs are. For this model the honest
answer is that the familiar ones do not exist:

> **There is no temperature, top-p or top-k.** `Agent.predict(state, questions)`
> takes no sampling parameters. It is one forward pass over a non-autoregressive
> model and it is deterministic — the same prompt returns bit-identical scores.
> Nothing is being sampled, so there is nothing to sample differently.

What is real, ordered by how much it actually moved the numbers in
[`results/report.md`](results/report.md):

1. **Ask one question per label rather than one menu.** The largest effect
   found, and it does not show up in accuracy at all: `noul_per_label` gives
   the same answer under every option ordering at every k, where a flat
   `choice` question at k=16 changes its answer on a third of documents from a
   reshuffle alone. Its confidence also falls with k instead of rising.
2. **How hard the documents are.** At k=16 the explicit tier scores 0.75 and
   the adversarial tier 0.30. Writing style moves the number further than
   anything on this list.
3. **Label descriptions.** Barely change overall accuracy, but they are what
   makes `choice` indifferent to menu length: on a fixed document set it holds
   across fourteen added distractors where bare labels decay.
   `laya.render_options` renders a criterion as `label: description` and falls
   back to the bare label when the description is empty.
4. **How many options you ask for.** Much weaker than part one implied, and the
   reason the sweep grew a fixed-subset control: most of the apparent fall over
   k was the document set changing underneath it, not the menu growing.
5. **Post-hoc temperature scaling.** The one honest sense in which this model
   has a temperature — not sampling, but softening the returned distribution
   after the fact. Argmax-invariant, so it fixes the confidence without
   touching a single prediction. It only helps at large k, and made
   calibration worse at every k below 8.
6. **What you put in the state.** Subject and body together beat either alone.

Two knobs that sounded promising and were not. **The true/false reading on a
`noul` question** is undocumented but supported — `render_options` falls back
to "yes, the statement holds" when you omit it — and supplying one dropped
accuracy sharply while making the model over-predict. **Two-stage
decomposition**, a group question then a label question, was the worst framing
tested: a wrong group at stage one cannot be recovered at stage two, so the
errors compound.

`LAYA_SUBFOLDER` also selects a different checkpoint (`typed-decisions`,
`multilingual`), which is a genuine lever — the abandoned `three-lanes` branch
has Banking77 numbers where `typed-decisions` beat the base checkpoint by about
ten points at k=32. Comparing them needs two services, so it is not part of this
sweep.

## Layout

```
corpus/documents.jsonl   160 documents with gold labels, tier and type
corpus/taxonomy.json     the nested label order, descriptions and groups
corpus/README.md         provenance, and how to add a document
corpus.py                loading, taxonomy slicing, corpus statistics
framings.py              the ways to ask, and post-hoc temperature scaling
metrics.py               accuracy, macro-F1, ECE, Brier, multi-label P/R/F1
service.py               the HTTP client; LAYA_URL and nothing hard-coded
sweep.py                 the run
report.py                results -> report.md, stdlib only
tests/test_corpus.py     corpus integrity. No service needed.
tests/test_metrics.py    metric unit tests. No service needed.
tests/test_smoke.py      a live slice against whatever LAYA_URL points at
```

`metrics.py` and most of `test_metrics.py` are recovered from the abandoned
`three-lanes` branch, where they were written for a Banking77 label-budget sweep
and deleted along with it. They came back with their tests.
