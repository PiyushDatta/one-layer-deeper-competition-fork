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
