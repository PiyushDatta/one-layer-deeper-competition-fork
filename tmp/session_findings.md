# Session findings — what is and is not reachable

Written after a measurement session on 2026-08-31. Everything here is measured,
not argued. Instruments are `tmp/sweep*.py`, `tmp/core_question.py`,
`tmp/compose_check.py`, `tmp/trajectory.py`, `tmp/multi.py`.

## 1. Max T is not reachable, and it is not a tuning problem

`depth_evaluation_exhaustive_x` reserves 38 units before generating anything
else, so the depth cohorts share **0 of 38** x values with training. Certifying
any rung needs `x^2 mod 323` computed for units never seen.

`tmp/core_question.py` strips away every piece of submission machinery: a plain
transformer, digits in, digits out, trained on the 248 seen units and tested on
the 38 reserved.

| width | lr | weight decay | steps | train exact | reserved exact |
|---|---|---|---:|---:|---:|
| 128 | 1e-3 | 1.0 | 39,814 | 1.000 | 0.026 |
| 256 | 1e-3 | 1.0 | 163,076 | 0.624 | 0.000 |
| 512 | 3e-4 | 0.3 | 163,076 | 1.000 | 0.026 |

Train exact reaches 1.000 by **step 89**. Reserved exact never leaves 0.
163,000 steps is grokking-scale, so this is not a step-budget problem and not a
hyperparameter problem.

Why this differs from the grokking literature: Power et al. tokenise each group
element as its own token, so the model learns a structured embedding table.
Here x arrives as three decimal digits, which makes it genuine 3-digit modular
arithmetic. That is the task family MAIN scored 0/10 on.

Across roughly 40 further configurations, width 16 to 12.9M, loops 1 to 8, lr
1e-3 to 1e-2, decay 0.1 to 30, batch 32 to 512, SAM on and off, forced discrete
tape, the best reserved-set numbers were CE 2.069 and token accuracy 0.246,
against a **no-input baseline of CE 1.863 and token accuracy 0.267**. Nothing
beat doing nothing.

## 2. Two different ceilings, and conflating them is a real error

The Easy score is mean exact accuracy over test and ood.

| predictor | test | ood | score |
|---|---:|---:|---:|
| length-AWARE constant | 0.0867 | 0.2200 | **0.1533** |
| best length-BLIND constant, all 1000 searched | 0.0600 | 0.1533 | **0.1067** |
| per-slot modal constant | 0.0600 | 0.1000 | 0.0800 |
| collapsed trained runs | 0.0667 | 0.1000 | 0.0833 |

The model cannot know the answer length, so 0.1533 is not available to it. The
honest ceiling for a non-computing model is **0.1067**, and collapsed runs
already reach 78% of it. An earlier claim in this repo that we sit at "one third
of a trivial baseline" used the length-aware figure and overstated the gap.

ood is high for a constant because it is T=6, and `x^64 mod 323` has only 9
distinct values. The same collapse explains why ID depth rungs T>=4 look easier
than T=1: they have 9 possible answers, T=1 has 32.

## 3. Composition on seen x is blocked at the optimisation level

96% of test rows use an x trained at a different T, so composing a memorised
step map would take test toward 0.9 without any generalisation. It does not
happen, and forcing it makes things worse.

| tape | steps | train | test | note |
|---|---:|---:|---:|---|
| off | 9,122 | 0.549 | 0.033 | exits are independent lookups |
| soft, forced open | 9,835 | 0.548 | 0.033 | no change |
| hard, straight-through | 4,275 | 0.040 | 0.030 | **all 8 exits identical** |

With a hard tape every exit returns the same accuracy to two decimals, meaning
B has collapsed to near-identity. The structure that would make `B^k` a real
iterate is exactly the structure the model cannot optimise in this budget.

## 4. The E1-only win does not survive

A tiny heavily regularised config, `D_MODEL=32`, decay 10 on both optimizers, T
visible, scores 0.0833 on E1 against the shipped 0.0517. It gets there by
collapsing to a constant, train accuracy 0.03 and training loss climbing to
14.4.

| | E1 | E7 | E2 |
|---|---:|---:|---:|
| shipped | 0.0517 | 0.0597 | 0.0110 |
| tiny, decay 10 | **0.0833** | 0.0521 | 0.0000 |

It wins only where it was tuned. Not committed. Any future search should score
on the mean of several datasets, which is what `tmp/multi.py` does.

## 5. What would actually be needed

Nothing in Changes 7 to 16 is wrong, and several are measurably working: the
monotone stride-2 exit ladder exists, SAM narrowed the train/test gap by 1.76
nats, control/data separation stopped the saturation it was aimed at. They are
all downstream of a step map the model cannot learn.

Reaching Max T needs `x^2 mod N` to generalise to unseen units from a few
hundred digit-tokenised examples in 60 seconds. That is the open problem, and
none of the structural work touches it.

## 6. No tuned configuration beats the shipped one

A random search over width, loops, both weight decays, learning rate, SAM and
T-visibility, scored on three Easy datasets, produced three candidates that
looked better than shipped on that mean. The best was `D_MODEL=64`,
`TRAIN_LOOPS=8`, Muon decay 1.0, AdamW decay 3.0.

Re-scored on all ten Easy datasets it loses clearly:

| dataset | shipped | candidate |
|---|---:|---:|
| e1 | 0.0517 | 0.0483 |
| e2 | 0.0171 | 0.0090 |
| e3 | 0.0044 | 0.0163 |
| e4 | 0.0091 | 0.0030 |
| e5 | 0.0046 | 0.0046 |
| e6 | 0.0820 | 0.0588 |
| e7 | 0.0597 | 0.0677 |
| e8 | 0.1009 | 0.0656 |
| e9 | 0.0378 | 0.0322 |
| e10 | 0.1021 | 0.0211 |
| **mean** | **0.0469** | **0.0326** |

Three datasets is not enough to select on. Single-run noise on one dataset is
large enough that the same configuration measured 0.0833 and 0.0333 in two
sweeps an hour apart. Use `tmp/allten.py` for any future selection.

**The shipped configuration stands.** Nothing found in this session beats it.

## 7. The failure is architecture-limited, not data-limited

Two probes in `tmp/data_limit.py`, both outside the submission.

**Coverage does not help.** Vary how many of the 288 units are shown, hold out
the rest, ~22,000 full-batch steps each:

| shown | train n | held n | train exact | held exact |
|---:|---:|---:|---:|---:|
| 50% | 144 | 144 | 1.000 | 0.007 |
| 70% | 201 | 87 | 1.000 | 0.000 |
| 86% | 247 | 41 | 1.000 | 0.024 |
| 93% | 267 | 21 | 1.000 | 0.000 |
| 97% | 279 | 9 | 1.000 | 0.000 |

Showing 279 of 288 entries and holding out 9 still generalises to none of them.
The model is filling a table, not interpolating a function, and E1's 86% is not
the binding constraint.

**Volume does not help either.** Modular multiplication `x*y mod 323` over all
288^2 pairs, same digit tokenisation, 74,649 training pairs, 35,874 steps:

```
train exact 0.844    held exact 0.085
```

300x the data and generalisation is still 8.5%. A transformer over digit tokens
does not learn generalising modular arithmetic. That is the wall, and it sits
below every structural idea in this document.

## 8. The machinery is worth 2.2x, SAM is score-neutral

Ten Easy datasets, `tmp/ablate_all.py`:

| | mean over 10 datasets |
|---|---:|
| shipped | **0.0516** |
| shipped without SAM | 0.0513 |
| minimal: no hypotheses, no history, no tape, no SAM, T visible | 0.0238 |

Changes 7 to 16 collectively more than double the score against a stripped
model, so the structural work was not wasted even though it did not reach Max T.

SAM is a wash on score, 0.0513 against 0.0516, despite cutting test loss from
5.78 to 4.02 in the single-dataset measurement. It buys loss geometry and costs
half the step count. Keep or drop on other grounds; it does not move the metric.

## 9. Correction: it is NOT architecture-limited

Section 7 concluded that a transformer over digit tokens cannot do modular
arithmetic, from a probe reaching 0.085 held-out on `x*y mod 323`. **That probe
was badly designed.** It read the answer off the last three *input* digit
positions. Giving the model three dedicated learned query slots instead, same
data, same tokenisation:

| config | params | steps | train | held |
|---|---:|---:|---:|---:|
| w256 h4 L2 | 1,588,748 | 13,390 | 0.930 | **0.722** |
| w256 h4 L4 | 3,168,268 | 6,760 | 0.953 | **0.812** |
| w256 h4 L8 | 6,327,308 | 3,076 | 0.934 | 0.765 |
| w512 h8 L4 | 12,627,980 | 2,255 | 0.888 | 0.295 |

0.812 against 0.085. Digit-tokenised modular arithmetic is learnable. The
readout was the bottleneck. Section 7's coverage result still stands, 279 of
288 units shown still generalises to none, but its architecture claim does not.

## 10. Answer queries, the first real cross-dataset gain

Ported into the submission: the last `MAX_ANSWER_DIGITS` valid workspace slots
are seeded from dedicated learned queries rather than prompt content. The
prompt stream stays readable so nothing is lost.

| dataset | queries | no queries |
|---|---:|---:|
| e1 | 0.0433 | 0.0517 |
| e2 | 0.0156 | 0.0042 |
| e3 | 0.0094 | 0.0106 |
| e4 | 0.0085 | 0.0076 |
| e5 | 0.0071 | 0.0033 |
| e6 | 0.0786 | 0.0786 |
| e7 | 0.0715 | 0.0597 |
| e8 | 0.1068 | 0.1009 |
| e9 | 0.0311 | 0.0378 |
| e10 | 0.1461 | 0.1141 |
| **mean** | **0.0518** | **0.0468** |

Ahead on 6, behind on 3, tied on 1. On E1 it also cuts test loss 3.82 to 3.46
and ood 4.17 to 3.48, and reaches train exact 0.313 at step 300 against 0.234.

## 11. The 12 correct predictions are real, not a constant

Dumping E1 predictions: test emits **59 distinct answers** across 150 rows with
the modal one appearing 7 times, ood emits 31 distinct across 100. The model is
input-dependent. The recurring 5/150 and 7/100 across configurations is
small-number coincidence near chance, not collapse onto the marginal.

## 12. Medium is harder, not easier

M1, fixed N=10403, 600s: **train exact 0.000 at 1,900 steps**, loss plateaued at
4.37. The extra compute and 45x the data do not help because five-digit N is a
much harder step map. The grokking-regime argument for Medium does not survive
contact with M1.
