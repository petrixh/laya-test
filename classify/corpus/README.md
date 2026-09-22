# The corpus

160 documents in `documents.jsonl`, one JSON object per line.

## Provenance

**Every document here is invented.** They were written for this repository to be
realistic in *shape* — the length, the furniture, the hedging, the way a real
message buries its point in the third paragraph — and none of them is a real
message from anyone. Every person, company, reference number and address is made
up, and the recurring cast is deliberately small so that the corpus reads as the
constructed thing it is.

Nothing in this file is a real person's data. `tests/test_corpus.py` asserts the
mechanical part of that: no address containing `@`, no URL, no IP, no digit run
long enough to be an account or card number, no credential-shaped strings. The
file was also reviewed end to end before it was committed.

This matters for reading any result computed from it: accuracy on invented
documents is evidence about the *shape* of the problem, not a number that
transfers to real traffic. Re-measure on real traffic before relying on it.

## Fields

| field | meaning |
|---|---|
| `id` | `<label>-NN`, unique |
| `tier` | `explicit`, `realistic`, `implicit` or `adversarial` |
| `type` | email, ticket, chat, incident_report, contract_clause, notice, form, application, disclosure |
| `subject` | the subject line, or a short stand-in for formats that have none |
| `body` | the message |
| `label` | the primary gold label |
| `labels` | all gold labels, `labels[0] == label` |

## The balance, and why it is exact

Ten documents per label and forty per tier, from a grid that gives label *i* and
tier *t* three documents where `(i + t) % 4 < 2` and two otherwise. Both margins
come out exactly balanced, which is what lets chance and the majority-class
baseline be quoted as the same number, 1/k.

`tests/test_corpus.py` enforces the grid. If you add a document the tests will
fail until the grid is restored — that is deliberate, because an unbalanced
corpus silently changes what every per-tier and per-label comparison means.

## Adding a document

1. Pick the label and tier cell you want to grow, and grow its opposite too, so
   the margins stay exact.
2. Invent everything. No real names, companies, addresses, numbers or incidents.
3. Write to the tier honestly. An `implicit` document that happens to contain the
   word "invoice" is an `explicit` document wearing a disguise, and it will
   quietly flatter the model.
4. Run `make test`.

## Tiers

- **explicit** — short, and it names its own category in plain words.
- **realistic** — the same signal, buried in greetings, a quoted thread, a
  signature and a disclaimer.
- **implicit** — the category has to be inferred. None of the giveaway vocabulary
  is present.
- **adversarial** — carries the surface vocabulary of a *different* category, or
  sits on the boundary between two neighbours. The message headed `Re: Invoice`
  that turns out to be about a download failing.

Where a document genuinely belongs to two categories it carries both in
`labels`. Where it merely *sounds* like another category it carries one — the
misleading surface is the whole point of that document, and giving it the second
label would defeat it.
