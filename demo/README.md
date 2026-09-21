## `two-stage-lane/` — asking the model to pick the lane, which fails

The same game and the same stage-one classifier, but the lane is chosen by a
second Laya question instead of by the harness rule.

| | |
|---|---|
| stage-one accuracy | 1.00 (120/120) |
| stage-two decisions | 36 |
| **chose a wall** | **25% of waves** |
| crashes | 9 |
| best distance | 167m (against 1798m on the rule) |

The frame shows why: the lane panel reads L 0.92, M 0.06, R 0.02, and it reads
close to that whatever the lanes actually contain. The model is answering
"left", not choosing. A constant "always left" baseline scores 0.825 on the
offline sweep against the best framing's 0.875.

## `three-lanes/` — the lane-aware autopilot

Laya on MLX driving the three-lane game: three lanes, walls that must be gone
around, and a wave across all lanes at once.

| | |
|---|---|
| decisions | 201 |
| accuracy | **1.00** — jump 71/71, duck 60/60, block 70/70 |
| crashes | **0** |
| best distance | **1798m** |
| lane changes | 51 |

Each obstacle gets one forward pass carrying two questions: the validated
two-option ground-versus-air `choice`, plus a `noul` asking whether it is a
solid barrier. Mean P(block) comes out at 0.90 for walls, 0.24 for ground
obstacles and 0.00 for hanging ones.

The state still only names the object. The harness reads which lane it is in
and applies a fixed preference (empty lane beats a manoeuvre, never enter a
barrier); the model supplies the classification.

> **Note:** the two recordings below are from the single-lane game. They are a
> record of the old one, kept because they are what the prompt-triage result was
> measured on.

# Demo runs

Two recorded runs of the Reindeer Jump autopilot. Both are real captures from
`agent/play.mjs`; each folder has `run.webm`, a `frame.png` still, and the
`summary.json` / `trace.json` / `deaths.json` the run produced.

## `laya-guided/` — the same model, after prompt triage

Laya on MLX (M3 Max), 40 decisions with the `guided` framing chosen by
`eval/prompt_sweep.py` and confirmed by `eval/prompt_confirm.py`.

| | |
|---|---|
| accuracy | **1.00** (40/40) |
| ground obstacles | 29 / 29 |
| air obstacles | 11 / 11 |
| crashes | **0** |
| best distance | **721m** |
| mean confidence | 0.057 |

No leak: the state is only `There is a log on the track ahead of the running
reindeer.` The model still decides whether that sits or hangs. What changed is
static and instance-independent -- neutral label names instead of `jump`/`duck`,
criteria giving both examples and mechanism, and two options instead of three.

Note the confidence: 0.057 while scoring 1.00. It is right and does not believe
it, so none of this is gateable on confidence.

## `laya-cpu/` — the same model, first framing

Laya `convaiinnovations/laya` (base checkpoint, fp32, CPU, 4 threads) driving the
game for 40 decisions, before the prompt sweep. Kept as the before-picture.

| | |
|---|---|
| accuracy | **0.35** (chance is 0.33 on three labels) |
| ground obstacles | **0 / 26** |
| air obstacles | **14 / 14** |
| crashes | 10, all attributed to wrong verdicts, none to latency |
| mean confidence | 0.13 |
| latency p50 / p95 | 940ms / 1135ms |

**It answers `duck` to everything.** The 0.35 is just the base rate of air
obstacles. The still frame shows it plainly: `DUCK` at confidence 0.092, with a
near-flat distribution (jump 0.41 / duck 0.45 / run 0.14) and a decision log of
`log → duck`, `snowman → duck`, `snowman → duck`.

The model is not confused, it is *abstaining*: 0.13 mean confidence here versus
0.73 on Banking77 at four labels. Run `python -m eval.game_probes` for the
diagnosis — the short version is that Laya reads text, and this task needs
spatial reasoning it does not have.

## `oracle-reference/` — what the harness can do

The identical harness driven by a perfect oracle at Laya's *measured* CPU latency
(975ms), so only the decision quality differs.

| | |
|---|---|
| accuracy | 1.00 |
| crashes | **0** |
| best distance | 556m |

This is the control. It shows the game, the hook, the decide-early/act-late
timing and the grading are all sound at ~1s inference, and isolates the 10
crashes in the other run to the model's answers.

It also earned its keep: at 975ms an *earlier* version of the autopilot crashed
11 times in 30 even with a perfect oracle, because it only classified the next
obstacle and the verdict arrived after impact. Pipelining the lookahead — every
obstacle inside the 2.6s horizon gets classified as it enters, up to three in
flight — took that to zero.
