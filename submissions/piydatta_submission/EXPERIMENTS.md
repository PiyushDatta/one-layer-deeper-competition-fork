# Universal Transformer and ACT experiment log

Last updated: 2026-08-25

This file records the `piydatta_submission` experiments discussed during development so results remain available outside the chat history.

## Common benchmark setup

- Manifest: `benchmark/manifests/h100_easy_e1.json`
- Task: `squaring-mod-easy-e1`
- Seed: `74`
- Training budget: `60` seconds
- Evaluation budget: `30` seconds
- Training and evaluation batch size: `512`
- Data workers: `0`
- Model width: `128`
- Attention heads: `4`
- Vocabulary size: `17`
- Maximum sequence length: `13`
- Test examples: `150`
- OOD examples: `100`
- Unless stated otherwise, ACT uses epsilon `0.01` and ponder weight `0.001`.

Command used:

```powershell
python -m benchmark.runner --manifest benchmark/manifests/h100_easy_e1.json --submission-file submissions/piydatta_submission/submission.py --num-workers 0
```

## Interpretation cautions

- Exact accuracy is full-sequence accuracy. One incorrect output token makes the entire example incorrect.
- For these Easy E1 runs, the **Easy practice score (higher is better)** is the runner's `mean_exact_accuracy`: the unweighted mean of test and OOD exact accuracy, not pooled accuracy over all 250 examples.
- Final test/OOD evaluation losses are pure cross-entropy. ACT training loss includes the ponder penalty; fixed-depth training loss is plain cross-entropy.
- ACT training losses produced with different ponder weights are not directly comparable because the objective itself changes. Final test/OOD cross-entropies remain directly comparable.
- All runs use one seed and contain few correct examples. Differences of one or two examples are weak evidence.
- Runs under **Controlled results** use the reordered constructor: every shared layer is initialized before the ACT-only halting unit. Fixed and ACT runs at the same depth therefore start from identical shared weights.
- No run certified a depth-profile rung.

## Competition scoring

The scoring direction is **higher is better**, not lower.

| Tier | Ranking metric | Direction |
|---|---|---|
| Easy and Medium practice tiers | Mean exact accuracy over every scored split and seed | Higher is better |
| Hard public leaderboard | Largest consecutively certified `Max T`, then largest consecutively certified `OOD N Max T`, then next-rung exact accuracies | Higher is better at each comparison |

All experiments in this file use one Easy E1 seed with two scored splits, so their displayed practice score is:

```text
Easy practice score = (test exact accuracy + OOD exact accuracy) / 2
```

This Easy score is the primary score emitted by the runner for these experiments. It is a useful practice proxy, but it is not the official Hard leaderboard ranking. The current runs therefore cannot establish which architecture would win Hard; none certified even the first depth-profile rung.

## Controlled results

`Correct` is shown as `test + OOD` correct examples. `Steps/s` is completed optimizer steps divided by recorded training seconds.

| ID | Mode | Loops | Ponder weight | Steps | Steps/s | Final train loss | Test acc. | OOD acc. | Easy practice score (higher is better) | Correct | Test loss | OOD loss | Mean loss | Eval seconds | Model state | Optimizer state |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C-F7 | Fixed UT | 7 | 0 | 1,737 | 28.9 | 0.000037 | 1.33% | 2.00% | 1.67% | 2 + 2 | 8.4827 | 8.3219 | 8.4023 | 0.243 | 202,880 | 405,774 |
| C-A7 | ACT | 7 max | 0.001 | 1,337 | 22.3 | 0.009499 | 4.67% | 2.00% | 3.33% | 7 + 2 | 7.0162 | 6.5388 | 6.7775 | 0.335 | 203,009 | 406,034 |
| C-F12 | Fixed UT | 12 | 0 | 1,170 | 19.5 | 0.005907 | 2.00% | 6.00% | **4.00%** | 3 + 6 | 6.2717 | 6.0073 | 6.1395 | 0.321 | 203,520 | 407,054 |
| C-A12 | ACT | 12 max | 0.001 | 944 | 15.7 | 0.025062 | 2.00% | 3.00% | 2.50% | 3 + 3 | 6.0552 | 6.1554 | 6.1053 | 0.464 | 203,649 | 407,314 |
| C-A12-100 | ACT | 12 max | 0.1 | 1,379 | 23.0 | 0.214539 | 3.33% | 4.00% | 3.67% | 5 + 4 | 9.4464 | 9.6546 | 9.5505 | 0.264 | 203,649 | 407,314 |
| C-F16 | Fixed UT | 16 | 0 | 1,090 | 18.2 | 0.016597 | **5.33%** | 1.00% | 3.17% | 8 + 1 | 5.9419 | 6.6195 | 6.2807 | 0.337 | 204,032 | 408,078 |
| C-A16 | ACT | 16 max | 0.001 | 795 | 13.2 | 0.033143 | 4.00% | 1.00% | 2.50% | 6 + 1 | 6.1057 | 6.0994 | 6.1025 | 0.574 | 204,161 | 408,338 |
| C-A16-R2 | ACT | 16 max | 0.001 | 610 | 10.2 | 0.030163 | 4.00% | 1.00% | 2.50% | 6 + 1 | 5.6136 | 5.4787 | 5.5461 | 0.625 | 204,161 | 408,338 |
| C-F32 | Fixed UT | 32 | 0 | 698 | 11.6 | 0.087446 | 4.00% | 1.00% | 2.50% | 6 + 1 | 5.2080 | 5.5695 | 5.3888 | 0.497 | 206,080 | 412,174 |
| C-A32-001 | ACT | 32 max | 0.001 | 525 | 8.7 | 0.066734 | 2.67% | 1.00% | 1.83% | 4 + 1 | 5.3220 | 5.1662 | 5.2441 | 0.794 | 206,209 | 412,434 |
| C-A32-005 | ACT | 32 max | 0.005 | 519 | 8.6 | 0.047847 | 2.00% | 2.00% | 2.00% | 3 + 2 | 5.7093 | 5.1824 | 5.4459 | 0.894 | 206,209 | 412,434 |
| C-A32-010 | ACT | 32 max | 0.01 | 630 | 10.5 | 0.039723 | 2.67% | 4.00% | 3.33% | 4 + 4 | 6.4577 | 5.4972 | 5.9775 | 0.500 | 206,209 | 412,434 |

## Controlled leaderboard by competition rules

This applies the Hard leaderboard ordering to the public Easy depth profiles: certified ID `Max T`, certified OOD-N `Max T`, ID next-rung accuracy, then OOD next-rung accuracy. These remain diagnostic Easy results, not predictions of private Hard performance. All runs have `Max T = None`, so the table is currently ordered by ID T=1 accuracy and then OOD T=1 accuracy. Exact ties remain tied because local runs have no official submission timestamp.

| # | ID | Configuration | In Distribution N progress | Out of Distribution N progress |
|---:|---|---|---:|---:|
| **1** | **C-F16** | **Fixed UT, 16 loops** | **T=1 Acc 2.6316%** | **T=1 Acc 0.5859%** |
| **2=** | C-A12 | ACT max 12, weight 0.001 | T=1 Acc 2.6316% | T=1 Acc 0.3906% |
| **2=** | C-A32-010 | ACT max 32, weight 0.01 | T=1 Acc 2.6316% | T=1 Acc 0.3906% |
| **4** | C-A7 | ACT max 7, weight 0.001 | T=1 Acc 0.0000% | T=1 Acc 1.3672% |
| **5** | **C-A16-R2** | **ACT max 16, weight 0.001, repeat 2** | **T=1 Acc 0.0000%** | **T=1 Acc 0.9766%** |
| **6=** | C-A12-100 | ACT max 12, weight 0.1 | T=1 Acc 0.0000% | T=1 Acc 0.5859% |
| **6=** | C-A32-001 | ACT max 32, weight 0.001 | T=1 Acc 0.0000% | T=1 Acc 0.5859% |
| **8=** | C-F7 | Fixed UT, 7 loops | T=1 Acc 0.0000% | T=1 Acc 0.3906% |
| **8=** | C-A32-005 | ACT max 32, weight 0.005 | T=1 Acc 0.0000% | T=1 Acc 0.3906% |
| **10=** | C-F12 | Fixed UT, 12 loops | T=1 Acc 0.0000% | T=1 Acc 0.1953% |
| **10=** | C-F32 | Fixed UT, 32 loops | T=1 Acc 0.0000% | T=1 Acc 0.1953% |
| **12** | C-A16 | ACT max 16, weight 0.001 | T=1 Acc 0.0000% | T=1 Acc 0.0000% |

The current controlled competition-style leader is **Fixed UT with 16 loops**. No run certified T=1. The new `C-A16-R2` run ranks fifth because its ID T=1 accuracy is zero, but its `5/512` OOD T=1 result beats the other runs tied at zero ID accuracy except `C-A7`.

## Controlled ranking by Easy practice score

| Rank | ID | Method | Test acc. | OOD acc. | Easy practice score | Gap from best |
|---:|---|---|---:|---:|---:|---:|
| 1 | C-F12 | Fixed UT, 12 loops | 2.00% | 6.00% | **4.00%** | - |
| 2 | C-A12-100 | ACT, max 12, weight 0.1 | 3.33% | 4.00% | **3.67%** | 0.33 pp |
| 3 | C-A7 | ACT, max 7, weight 0.001 | 4.67% | 2.00% | **3.33%** | 0.67 pp |
| 3 | C-A32-010 | ACT, max 32, weight 0.01 | 2.67% | 4.00% | **3.33%** | 0.67 pp |
| 5 | C-F16 | Fixed UT, 16 loops | 5.33% | 1.00% | **3.17%** | 0.83 pp |
| 6 | C-A12 | ACT, max 12, weight 0.001 | 2.00% | 3.00% | **2.50%** | 1.50 pp |
| 6 | C-A16 | ACT, max 16, weight 0.001 | 4.00% | 1.00% | **2.50%** | 1.50 pp |
| 6 | C-A16-R2 | ACT, max 16, weight 0.001, repeat 2 | 4.00% | 1.00% | **2.50%** | 1.50 pp |
| 6 | C-F32 | Fixed UT, 32 loops | 4.00% | 1.00% | **2.50%** | 1.50 pp |
| 10 | C-A32-005 | ACT, max 32, weight 0.005 | 2.00% | 2.00% | **2.00%** | 2.00 pp |
| 11 | C-A32-001 | ACT, max 32, weight 0.001 | 2.67% | 1.00% | **1.83%** | 2.17 pp |
| 12 | C-F7 | Fixed UT, 7 loops | 1.33% | 2.00% | **1.67%** | 2.33 pp |

The best controlled Easy practice score is therefore **Fixed UT with 12 loops at 4.00%**. The best ACT practice score is **max 12 with ponder weight 0.1 at 3.67%**. Both solved 9 of the 250 examples, but the practice score weights the two splits equally: Fixed 12's stronger OOD result gives it the higher score.

### ACT max-16 repeat

`C-A16-R2` repeats the same visible configuration and state sizes as `C-A16`. Both produced exactly `4.00%` test accuracy, `1.00%` OOD accuracy, and a `2.50%` Easy practice score. The repeat completed fewer optimizer steps (`610` versus `795`) and evaluated slightly more slowly (`0.625` versus `0.574` seconds), but reduced mean reported loss from `6.1025` to `5.5461`. Its OOD T=1 depth progress improved from `0/512` to `5/512`; ID T=1 remained `0/38` in both runs. This repeat illustrates substantial wall-clock and depth-profile variability even with the same seed and configuration.

## Ponder-weight sweep at ACT max 32

| Metric | Weight 0.001 | Weight 0.005 | Weight 0.01 |
|---|---:|---:|---:|
| Completed steps | 525 | 519 | **630** |
| Approx. steps/s | 8.7 | 8.6 | **10.5** |
| Test exact accuracy | **2.67%** | 2.00% | **2.67%** |
| OOD exact accuracy | 1.00% | 2.00% | **4.00%** |
| Easy practice score (higher is better) | 1.83% | 2.00% | **3.33%** |
| Total correct | 5 | 5 | **8** |
| Evaluation time | 0.794 s | 0.894 s | **0.500 s** |
| Reported mean loss | **5.2441** | 5.4459 | 5.9775 |

The reported mean losses are pure evaluation cross-entropy and are directly comparable across ponder weights. They show that the accuracy gain at weight `0.01` came with worse probabilistic loss; future diagnostics should add raw and weighted ponder values alongside them.

Compared with weight `0.001`, weight `0.01`:

- completed about 20% more optimizer steps;
- cut evaluation time by about 37%;
- increased mean exact accuracy from 1.83% to 3.33%;
- increased total correct examples from 5 to 8.

Weight `0.005` did not produce an intermediate efficiency result. Its evaluation was slower than both other settings, showing that the learned halting policy is not monotonic in the penalty from this single run.

Compared with fixed 32, ACT max 32 at weight `0.01`:

- improved mean exact accuracy from `2.50%` to `3.33%`;
- improved OOD exact accuracy from `1.00%` to `4.00%`;
- reduced test exact accuracy from `4.00%` to `2.67%`;
- produced 8 total correct examples instead of 7;
- completed 630 steps instead of 698;
- had essentially identical evaluation time: `0.500` versus `0.497` seconds.

This is the first max-32 ACT setting that is competitive with fixed 32 on both Easy practice score and wall-clock evaluation time.

## Stronger ponder penalty at ACT max 12

| Metric | ACT 12, weight 0.001 | ACT 12, weight 0.1 | Fixed 12 |
|---|---:|---:|---:|
| Completed steps | 944 | **1,379** | 1,170 |
| Approx. steps/s | 15.7 | **23.0** | 19.5 |
| Test exact accuracy | 2.00% | **3.33%** | 2.00% |
| OOD exact accuracy | 3.00% | 4.00% | **6.00%** |
| Easy practice score (higher is better) | 2.50% | 3.67% | **4.00%** |
| Total correct | 6 | **9** | **9** |
| Evaluation time | 0.464 s | **0.264 s** | 0.321 s |
| Reported mean loss | 6.1053 | 9.5505 | 6.1395 |

The reported evaluation loss is already pure cross-entropy, so its increase to `9.5505` is not caused by adding the larger ponder term to the reported value. The stronger penalty improves exact accuracy and speed while worsening probabilistic loss/calibration.

The final training loss of `0.2145`, combined with nonnegative cross-entropy, bounds mean training ponder time to at most about `2.15`. The model's near-perfect training accuracy makes it likely that most of this final objective is ponder cost, suggesting very shallow learned computation. The ACT diagnostics should confirm this directly on future evaluation splits.

Compared with ACT 12 at weight `0.001`, weight `0.1`:

- completed about 46% more optimizer steps;
- reduced evaluation time by about 43%;
- increased mean exact accuracy from `2.50%` to `3.67%`;
- increased total correct examples from 6 to 9.

Compared with fixed 12, ACT 12 at weight `0.1`:

- completed about 18% more optimizer steps;
- reduced evaluation time by about 18%;
- produced the same 9 total correct examples;
- scored `3.67%` instead of `4.00%` mean exact accuracy because the benchmark weights the higher fixed-model OOD result equally with test accuracy.

This is the first ACT configuration to provide a clear wall-clock computation saving over its same-depth fixed control.

## Fixed versus ACT conclusions by depth

| Depth | Easy-score winner | Easy practice score | Mean-loss winner | Speed winner |
|---:|---|---:|---|---|
| 7 | ACT | 3.33% vs 1.67% | ACT | Fixed |
| 12 | Fixed | 4.00% vs 2.50% | ACT, narrowly | Fixed |
| 12, ACT weight 0.1 | Fixed, narrowly | 4.00% vs 3.67% | Not comparable across objectives | ACT |
| 16 | Fixed | 3.17% vs 2.50% | ACT | Fixed |
| 32, ACT weight 0.001 | Fixed | 2.50% vs 1.83% | ACT | Fixed |
| 32, ACT weight 0.01 | ACT | 3.33% vs 2.50% | Not comparable across objectives | Fixed, narrowly |

Current takeaways:

- **Fixed 16** leads the controlled competition-style Easy depth leaderboard, although it has not certified T=1.
- **Fixed 12** has the best controlled Easy practice score: `4.00%`.
- **Fixed 16** has the best test accuracy: `5.33%`.
- **Fixed 12** has the best OOD accuracy: `6.00%`.
- **ACT 12 at weight 0.1** has the best ACT Easy practice score: `3.67%`.
- **ACT 12 at weight 0.1** is the first adaptive run faster than its same-depth fixed control.
- ACT consistently learns the training task rapidly per optimizer step but usually reduces wall-clock throughput.
- Increasing loop capacity generally reduces cross-entropy while exact accuracy peaks at an intermediate depth.
- At max 32, weight `0.01` is much more promising than `0.001` or `0.005` because it recovers most fixed-32 evaluation speed and improves exact accuracy.

## Legacy and pre-control results

These runs remain useful historical references but should not be mixed with the controlled ablations. Some pre-control models constructed the ACT-only halting unit before shared layers, shifting the random initialization of those shared layers.

| ID | Description | Loops | Ponder weight | Steps | Final train loss | Test acc. | OOD acc. | Easy practice score (higher is better) | Test loss | OOD loss | Mean loss | Eval seconds | Model state |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L-BASE | Original one-block model, no UT | 1 | 0 | 1,859 | 0.037468 | 1.33% | 0.00% | 0.67% | 15.8968 | 20.5906 | 18.2437 | 0.259 | 201,984 |
| L-F7 | Initial fixed Universal Transformer | 7 | 0 | 1,443 | 0.004322 | 2.67% | 3.00% | 2.83% | 7.3000 | 6.7759 | 7.0379 | 0.355 | 202,880 |
| L-A64 | Initial ACT, max 64 | 64 max | 0.001 | 348 | 0.105395 | 2.67% | 4.00% | 3.33% | 4.7053 | 5.0129 | 4.8591 | 1.417 | 210,305 |
| L-A12 | Initial ACT, max 12 | 12 max | 0.001 | 1,034 | 0.016693 | 1.33% | 4.00% | 2.67% | 6.3002 | 6.3013 | 6.3008 | 0.427 | 203,649 |
| L-A7 | ACT max 7 before initialization reorder | 7 max | 0.001 | 1,355 | 0.009649 | 2.67% | 0.00% | 1.33% | 7.0661 | 6.9811 | 7.0236 | 0.330 | 203,009 |
| L-F7B | Fixed 7 before initialization reorder | 7 | 0 | 1,683 | 0.000052 | 2.67% | 4.00% | 3.33% | 8.3911 | 7.5319 | 7.9615 | 0.294 | 202,880 |

### Quarantined run

One run was reported as ACT max 12, but its model state (`202,880`), optimizer state (`405,774`), speed, and completed steps matched a fixed seven-loop model with no halting head. It is retained here only to prevent accidental reuse as ACT evidence.

| Reported configuration | Steps | Test acc. | OOD acc. | Easy practice score (higher is better) | Mean loss | Eval seconds | Model state |
|---|---:|---:|---:|---:|---:|---:|---:|
| "ACT max 12," metadata indicates fixed 7 | 1,754 | 3.33% | 2.00% | 2.67% | 12.9305 | 0.238 | 202,880 |

## ACT diagnostics

For a local diagnostic run, temporarily set `DBUG = True` in `submission.py`. The runner detects that constant and automatically performs the separate ACT diagnostic pass. Restore `DBUG = False` before making a competition submission; the false path creates no diagnostic dictionaries or cap masks. The older `--include-act-diagnostics` option remains available as an explicit override, but it is not needed for this submission.

```powershell
python -m benchmark.runner --manifest benchmark/manifests/h100_easy_e1.json --submission-file submissions/piydatta_submission/submission.py --num-workers 0
```

- pure task cross-entropy, excluding ponder penalty;
- raw mean ponder time;
- weighted ponder contribution;
- mean, median, p90, p95, and p99 token update counts;
- maximum/global loop iterations per batch;
- token and batch cap-hit rates;
- mean remainder, especially for cap-forced tokens;
- percentage of tokens naturally halted and total processing ended after every iteration;
- update counts for correct versus incorrect examples;
- update counts by sequence length and test/OOD split.

Diagnostics run in a second, non-scoring pass after the normal result and depth profile are fixed. Their time is reported separately as `act_diagnostics_seconds` and must not be compared with the ordinary `evaluation_seconds` field.

Wall-clock time is controlled by the slowest active token in a dense-attention batch, so global iterations per batch are more informative than mean token updates alone.

## Recommended next experiments

1. Temporarily set `DBUG = True` for future ACT analysis runs; diagnostics are collected automatically. Restore `DBUG = False` before submission.
2. At max 12, compare ponder weights around the promising `0.1` value, such as `0.03`, `0.05`, `0.15`, and `0.2`.
3. Compare the explicitly labeled evaluation task cross-entropy across ponder weights.
4. Repeat the strongest configurations on additional manifests or seeds before treating one- or two-example differences as reliable.
5. Keep fixed 16 as the current competition-style Easy depth reference, fixed 12 as the Easy practice-score reference, and ACT 12/weight 0.1 as the adaptive practice-score reference.
