# Scratchpads (arXiv 2112.00114) applied to `piydatta_submission`

Proposed changes only — nothing in this document has been applied to
`submissions/piydatta_submission/submission.py`.

All line numbers refer to `submissions/piydatta_submission/submission.py` unless
stated otherwise.

---

## 1. What transfers from the paper, and what doesn't

Nye et al. change the **data**: intermediate algorithm steps are written into the
target text and supervised token-by-token. That is unavailable here.

- Rule 14 bans data augmentation, data inspection, and task-specific solvers.
- The evaluator owns the data, the labels, and the outer loop.
- Computing `x**2 % N` yourself to build intermediate targets is exactly the
  banned "task-specific solver."

There is **no per-example scratchpad** in this dataset and none can be built.
Rows are independent, `_generate_record` (`data/squaring_mod.py:1150`) samples
`x` independently per T setting, rows land in different splits, and pairing rows
by content inside a batch would itself be data inspection.

The scratchpad does two separable things:

| | What it does | Available here? |
|---|---|---|
| **(A)** | supplies per-line targets — supervision *of* the intermediate steps | **No** |
| **(B)** | re-expresses a T-step problem as T applications of one short operation, so the model never represents "raise to the 2^T" as a single map | **Yes** |

(B) is where the depth-extrapolation benefit lives, and it comes from **weight
tying plus depth-aligned exits** — not from data.

### The mechanism, precisely

No pairing and no ordering are needed. The constraint is imposed across the
*population* of rows, with `B` the shared tied block:

```
rows with T=1  ->  head(B^1(x)) = x^2 mod N
rows with T=2  ->  head(B^2(x)) = x^4 mod N
rows with T=3  ->  head(B^3(x)) = x^8 mod N
```

No row references any other row. Each contributes gradient to the same `B`. A
T-specific shortcut can satisfy any one line alone; only `B` = one squaring
satisfies all three at once. That composition constraint is the whole benefit.

Evaluator shuffling is irrelevant to it: `train.jsonl` concatenates all T
settings and the loader shuffles globally, so every batch carries all three
constraints simultaneously.

This is **strictly weaker** than the paper's supervision. It constrains the
shared operator distributionally; it does not tell any example what its own
intermediates are.

### What it depends on — verified on E1

The constraint only bites where the intermediates land inside the region the
T=1 rows cover. Measured on E1 (`p=17, q=19`, N=323):

```
T=1: 250 distinct x    T=2: 250 distinct x    T=3: 250 distinct x
x-set identical across all T: True
units coprime to 323: 288
```

250 of 288 units appear as `x` at every T (pool exhaustion, not deliberate
design). Squaring maps units to units, so the T=1 rows pin down ~87% of the
squaring permutation on Z*_323 pointwise.

**This does not hold everywhere.** M4 uses 14–22-bit moduli with 30k rows per
setting — roughly 1% coverage. There `B` must *generalize* the squaring map
rather than pin it down, and this approach should be expected to be much weaker.
Do not tune on E1 and assume it transfers.

---

## 2. The six changes

| # | Change | Location |
|---|---|---|
| 1 | stop collapsing rows before depth selection | `:119` |
| 2 | per-row soft-min over exits | `:109-136` |
| 3 | drop all-exits-to-final-label | `HYPOTHESIS_ALL_LOSS_WEIGHT`, `:38` / `:132` |
| 4 | T-conditioned exit at eval | `:899`, `:1278-1284` |
| 5 | loop count | `FIXED_LOOPS`, `:24` |
| 6 | re-tokenization between loops | `SynchronizedProcessor.forward`, `:534-558` |

---

## Rows 1–3: `token_training_loss` (one hunk, three fixes)

These land together because they are the same nine lines.

### Current code (`:118-136`)

```python
    rows_with_targets = batch.valid_mask.any(dim=-1)
    candidate_evidence = sequence_losses[rows_with_targets].mean(dim=0)
    depth_cost = HYPOTHESIS_DEPTH_PENALTY * torch.arange(...)
    selection_evidence = candidate_evidence + depth_cost
    survivor_loss = -temperature * torch.logsumexp(
        hypothesis_log_prior - selection_evidence / temperature, dim=0,
    )
    all_candidate_loss = candidate_evidence.mean()
    task_loss = (1.0 - HYPOTHESIS_ALL_LOSS_WEIGHT) * survivor_loss \
              + HYPOTHESIS_ALL_LOSS_WEIGHT * all_candidate_loss
```

### Row 1 — `:119` collapses the batch before selecting a depth

`sequence_losses` arrives as `[batch, K]`, exactly the per-row-per-exit signal
needed, and `.mean(dim=0)` throws the row axis away. Everything after operates
on one `[K]` vector, so the objective can only pick **one exit for all 512
rows**. In a mixed-T batch the depth minimizing mean loss across T=1, 2 and 3 is
a compromise wrong for all three.

### Row 2 — no per-row soft-min

Replacing the batch-mean with a per-row `logsumexp` over the exit axis gives the
whole mechanism: for a T=3 row, whichever exit's readout matches `x^8` dominates
its own logsumexp. The row is never told it is a T=3 row — **its label does the
identification**. This is why the change needs no `input_ids` in the loss and no
T plumbing.

### Row 3 — `all_candidate_loss` demands idempotence

At weight 0.25 it averages all K exits against the *final* label. Per row:

- T=1 row: exit1 = x^2, exit2 = x^2, ... exit7 = x^2  =>  `B(x^2) = x^2`
- T=3 row: exit1 = x^8, exit2 = x^8, ... exit7 = x^8  =>  `B(x^8) = x^8`

Together: **B∘B = B**. An idempotent `B` applied 64 times equals `B` applied
once — precisely and only the operator shape that cannot certify T=64. A quarter
of the loss is pulling there.

`HYPOTHESIS_DEPTH_PENALTY` goes with it. `:120-125` adds cost rising with exit
index, favouring shallow exits — it rewards "answer at exit 1," which is the
shortcut, not the composable solution.

### Diff

```diff
@@ -36,8 +36,6 @@
 USE_LATENT_HYPOTHESES = True
 HYPOTHESIS_TEMPERATURE = 1.0
-HYPOTHESIS_ALL_LOSS_WEIGHT = 0.25
-HYPOTHESIS_DEPTH_PENALTY = 0.01
 NUM_SCRATCHPAD_TOKENS = 4
```

```diff
@@ -118,19 +116,17 @@ def token_training_loss(batch: TokenLossBatch) -> Tensor:
     rows_with_targets = batch.valid_mask.any(dim=-1)
-    candidate_evidence = sequence_losses[rows_with_targets].mean(dim=0)
-    depth_cost = HYPOTHESIS_DEPTH_PENALTY * torch.arange(
-        candidate_count,
-        device=candidate_evidence.device,
-        dtype=candidate_evidence.dtype,
-    )
-    selection_evidence = candidate_evidence + depth_cost
-
-    temperature = HYPOTHESIS_TEMPERATURE
-    survivor_loss = -temperature * torch.logsumexp(
-        hypothesis_log_prior - selection_evidence / temperature,
-        dim=0,
-    )
-    all_candidate_loss = candidate_evidence.mean()
-    task_loss = (
-        1.0 - HYPOTHESIS_ALL_LOSS_WEIGHT
-    ) * survivor_loss + HYPOTHESIS_ALL_LOSS_WEIGHT * all_candidate_loss
+
+    # Per-row soft-min over exits. The label picks the matching depth on its
+    # own: for a T=k row only the exit whose readout equals x^(2^k) has low
+    # loss, so the tied block is trained as one composable step instead of a
+    # T-specific map. Averaging over rows first would force one exit on the
+    # whole batch; scoring every exit against the final label would demand
+    # B(B(x)) == B(x), which is exactly what cannot extrapolate.
+    temperature = HYPOTHESIS_TEMPERATURE
+    row_losses = -temperature * torch.logsumexp(
+        hypothesis_log_prior - sequence_losses / temperature,
+        dim=1,
+    )
+    task_loss = row_losses[rows_with_targets].mean()
     return task_loss + PONDER_WEIGHT * ponder_cost
```

`candidate_count` stays bound by the unpack at `:101`; it just stops being used
here.

### Degradation check

If all exits have equal loss `L`, then `logsumexp(log_w - L/tau) = -L/tau`
because log-softmax weights sum to 1, so `row_losses = L` — plain cross-entropy.
It is a weighted soft-min bounded above by the weighted mean, never a different
objective in disguise.

---

## Row 4: T-conditioned exit (`:899`, `:1278-1284`)

### Current code

```python
self.hypothesis_prior = nn.Parameter(torch.empty(self.max_loops))   # :899
...
hypothesis_log_prior = F.log_softmax(self.hypothesis_prior, dim=0)  # :1278
winning_hypothesis = self.hypothesis_prior.argmax(dim=0)
x = stacked_hypothesis_states[:, winning_hypothesis]
logits = hypothesis_logits[:, winning_hypothesis]
```

A bare parameter. Depth is input-independent at train *and* eval — every row
reads out at the same loop index regardless of its prompt. Rows 1–3 are
untestable without this: the loss would learn per-row depths that eval ignores.

### Diff

```diff
@@ -903,6 +901,14 @@
             self.scratchpad_embedding = nn.Embedding(
                 NUM_SCRATCHPAD_TOKENS,
                 D_MODEL,
             )
             nn.init.normal_(self.scratchpad_embedding.weight, std=init_std)
+            # Same reason as the scratchpad: the exit selector goes last so
+            # every earlier fixed-path parameter keeps its seeded value.
+            if USE_LATENT_HYPOTHESES:
+                self.exit_norm = RMSNorm(D_MODEL)
+                self.exit_head = nn.Linear(D_MODEL, self.max_loops)
+                nn.init.zeros_(self.exit_head.weight)
+                nn.init.zeros_(self.exit_head.bias)
```

```diff
@@ -1278,7 +1284,25 @@
-            hypothesis_log_prior = F.log_softmax(
-                self.hypothesis_prior,
-                dim=0,
-            )
-            winning_hypothesis = self.hypothesis_prior.argmax(dim=0)
-            x = stacked_hypothesis_states[:, winning_hypothesis]
-            logits = hypothesis_logits[:, winning_hypothesis]
+            # Exit depth has to vary with the prompt's T. Pool the T segment
+            # only; position_signal keeps multi-digit T ordered.
+            prompt_features = token_state + position_signal
+            t_weights = (segment_ids.eq(3) & valid_tokens).to(prompt_features.dtype)
+            t_summary = (prompt_features * t_weights[..., None]).sum(dim=1) / (
+                t_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
+            )
+            hypothesis_log_prior = F.log_softmax(
+                self.exit_head(self.exit_norm(t_summary)) + self.hypothesis_prior,
+                dim=-1,
+            )
+            selected = hypothesis_log_prior.argmax(dim=-1)
+            gather_index = selected[:, None, None, None]
+            logits = hypothesis_logits.gather(
+                1,
+                gather_index.expand(-1, 1, *hypothesis_logits.shape[2:]),
+            ).squeeze(1)
+            x = stacked_hypothesis_states.gather(
+                1,
+                gather_index.expand(-1, 1, *stacked_hypothesis_states.shape[2:]),
+            ).squeeze(1)
```

### Notes

**Segment 3 is the T region.** `:1219` seeds it from `T_TOKEN_ID` and `:1226`
cummaxes it forward. Under `separate_input_output` there is no `ANS` token in
`input_ids` (`data/squaring_mod.py:434` returns before appending it), so segment
4 never occurs and segment 3 runs to the end of the valid prompt.

**`hypothesis_prior` survives** as a learned bias on top of the T-conditioned
term, so the selector can still express a T-independent depth preference, and
the `nn.Parameter` ordering at `:899` is untouched.

**Zero-init matters.** `exit_head` starting at zero means the selector *is*
today's global prior at step 0 and only becomes T-dependent as gradient arrives.
The change cannot regress the starting point — worth something when Hard gives
one attempt per day.

**Gradient path.** `argmax` is not differentiable, but the loss never touches
`logits` on this path — it consumes `hypothesis_logits` and
`hypothesis_log_prior`, both in the graph, and the selector gets gradient
through the `logsumexp`. Rule 8 wants an unbroken path from loss to the
parameters responsible for the prediction, and that holds. Still worth checking
deliberately before submitting.

---

## Row 5: loop count (`FIXED_LOOPS`, `:24`)

Decouple training depth from eval depth. Training at 64 loops is ~9x today's
per-step cost inside 60s, and pointless on E1 where nothing supervises past exit
3. Eval at 64 is nearly free — forward-only, no grad, a few hundred rows.

### Diff

```diff
@@ -22,7 +22,8 @@
 PONDER_WEIGHT = 0.005
 USE_ACT = False
-FIXED_LOOPS = 7
+TRAIN_LOOPS = 8
+EVAL_LOOPS = 64
 ACT_MAX_LOOPS = 16
```

```diff
@@ -433,6 +434,7 @@ class SynchronizedProcessor(nn.Module):
         attention_mask: Tensor | None = None,
         *,
+        num_loops: int | None = None,
         segment_ids: Tensor | None = None,
@@ -534,7 +536,8 @@
-        for _ in range(self.num_loops):
+        loops = self.num_loops if num_loops is None else num_loops
+        for _ in range(loops):
             joint_state = torch.cat((prompt_memory, work_state), dim=1)
```

```diff
@@ -848,1 +851,1 @@
-        self.max_loops = ACT_MAX_LOOPS if self.use_act else FIXED_LOOPS
+        self.max_loops = ACT_MAX_LOOPS if self.use_act else EVAL_LOOPS
@@ -893,3 +896,3 @@
             self.processor = SynchronizedProcessor(
                 block,
-                num_loops=self.max_loops,
+                num_loops=TRAIN_LOOPS,
                 num_scratchpad_tokens=NUM_SCRATCHPAD_TOKENS,
             )
```

`max_loops = EVAL_LOOPS` sizes `hypothesis_prior` and `exit_head` at 64 so the
selector *can* address a deep exit. Both `self.processor(...)` call sites in
`_run_processor` (`:976`, `:982`) gain
`num_loops=(TRAIN_LOOPS if self.training else EVAL_LOOPS)`, and `forward` slices
the selector to the loops actually run:

```diff
@@ -1268,3 +1272,4 @@
-            if len(hypothesis_states) != self.max_loops:
+            active_loops = TRAIN_LOOPS if self.training else EVAL_LOOPS
+            if len(hypothesis_states) != active_loops:
                 raise RuntimeError("processor did not produce every hypothesis")
@@
             hypothesis_log_prior = F.log_softmax(
-                self.exit_head(self.exit_norm(t_summary)) + self.hypothesis_prior,
+                (self.exit_head(self.exit_norm(t_summary)) + self.hypothesis_prior)[
+                    :, :active_loops
+                ],
                 dim=-1,
             )
```

### Memory, and a free win

At 64 eval loops `hypothesis_states` holds 64 tensors of `[512, 14, 6400]` —
about 5.9 GB in bf16. And `stacked_hypothesis_states` exists only to produce
`x`, which with `DBUG = False` is used by nothing (`forward` returns
`logits, auxiliary`; `x` only feeds `_build_model_diagnostics`). Guard the
stacking behind `DBUG` and store per-loop logits instead of states — logits are
`[512, 9, 17]`, negligible.

### What this row is and is not

It makes deep exits **reachable**, not **selected**. Entries 9–63 of `exit_head`
never receive gradient on E1, so a T=64 prompt will not route there. This is
plumbing; the countdown-in-the-scratchpad idea is what closes that gap.

It also barely affects the Easy/Medium *score*: E1's scored splits are test
(T in {1,2,3}) and ood (T=6), which `FIXED_LOOPS = 7` already covers. Loop count
matters for the Max T ladder — diagnostic on Easy/Medium, the actual ranking on
Hard.

---

## Row 6: re-tokenization between loops (`:534-558`)

The one with a real mechanism behind it. Today the workspace is a continuous
residual iterated K times with nothing forcing `h_k` to decode to anything. In
the paper each scratchpad line is re-read *as tokens* — a discrete bottleneck
that stops error accumulating off-manifold. Sixty-one unsupervised iterations
without it is a lot of compounding.

`SynchronizedProcessor` has no access to `head` / `token_embedding` /
`final_norm`. Passing a bound method rather than assigning submodules sidesteps
any param re-registration question under the 500M ceiling.

### Diff

```diff
@@ -1,4 +1,5 @@
 from __future__ import annotations
 
+from collections.abc import Callable
 import math
@@ -40,6 +41,8 @@
 NUM_SCRATCHPAD_TOKENS = 4
+RETOKENIZE_TEMPERATURE = 1.0
+RETOKENIZE_GATE_INIT = -2.0
```

```diff
@@ -434,6 +437,7 @@ class SynchronizedProcessor(nn.Module):
         num_loops: int | None = None,
+        retokenize: Callable[[Tensor], Tensor] | None = None,
         segment_ids: Tensor | None = None,
@@ -551,6 +555,16 @@
             work_state = candidate_state[:, prompt_len:]
+            if retokenize is not None:
+                # Force each loop's output back onto the token manifold before
+                # the next application, so many iterations cannot drift into
+                # directions the 17-token vocabulary cannot express.
+                work_state = torch.cat(
+                    (
+                        work_state[:, :1],
+                        retokenize(work_state[:, 1:output_end]),
+                        work_state[:, output_end:],
+                    ),
+                    dim=1,
+                )
             if stage_states is not None:
```

Placed before the `stage_states` / `hypothesis_states` appends, so exit *k* reads
the re-tokenized state. Only the prompt-aligned workspace is constrained — the
control token and the four scratchpad slots stay continuous, since they are
working memory with no reason to live in token space.

On `Model`, after `exit_head`:

```diff
+                self.retokenize_gate = nn.Parameter(
+                    torch.full((), RETOKENIZE_GATE_INIT)
+                )
```

```diff
+    def _retokenize(self, workspace: Tensor) -> Tensor:
+        normed = self.final_norm(workspace)
+        probs = F.softmax(self.head(normed) / RETOKENIZE_TEMPERATURE, dim=-1)
+        # head.weight is token_embedding.weight, so this decodes and re-encodes
+        # through the same tied matrix.
+        token_view = probs @ self.token_embedding.weight
+        # Embeddings are std 0.02; the residual stream is not. Match RMS so the
+        # mix is a direction blend rather than an amplitude collapse.
+        target_rms = workspace.pow(2).mean(-1, keepdim=True).sqrt()
+        token_rms = token_view.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
+        token_view = token_view * (target_rms / token_rms)
+        gate = torch.sigmoid(self.retokenize_gate)
+        return workspace + gate * (token_view - workspace)
```

Both `self.processor(...)` calls pass `retokenize=self._retokenize`.

### Why the whole workspace, not just the answer slots

The answer lives at `target_positions`, which `data/squaring_mod.py:106` sets to
the last `len(result)` valid positions — and `len(result)` is the answer's digit
count, which `forward` has no way to know (it receives only `input_ids` and
`attention_mask`). The model literally cannot identify its own output slots.
Uniform retokenization needs no position knowledge and is the literal reading of
"each line is re-read as tokens."

### Why the gate starts closed

`sigmoid(-2.0) ~= 0.12`, so early training stays close to today's behaviour and
the model dials the bottleneck in as it earns its keep. Same no-regression
property as the zero-init `exit_head`. If it never opens, that is a real result:
the continuous path is winning and the bottleneck is not paying for itself.

---

## 3. Cross-cutting

### Parameter budget

| Item | Elements |
|---|---|
| current model (`:18`) | 491,859,202 |
| `exit_norm` (RMSNorm 6400) | 6,400 |
| `exit_head` (Linear 6400 -> 64) | 409,664 |
| `retokenize_gate` | 1 |
| **total** | **492,275,267** of 500,000,000 |

~7.7M to spare. `exit_head` scales with `EVAL_LOOPS` (6400 x K), so raising K
past ~1,250 would breach the ceiling on its own.

### Tests this breaks

All four in `tests/test_piydatta_debug.py`, all expected:

| Test | Why |
|---|---|
| `:74` shape assertion | `hypothesis_log_prior` is now `[batch, K]`, was `[K]` |
| `:778` `..._uses_only_valid_target_positions` | feeds a `[K]` prior |
| `:815` `test_eval_uses_the_globally_selected_hypothesis` | global selection is the thing being removed |
| `:838` `test_hypothesis_depth_penalty_prefers_the_earlier_exit` | the penalty is gone |

The last two encode behaviour these changes deliberately reverse, so rewrite
rather than patch: one asserting per-row selection, one asserting a T=k row puts
its gradient on exit k.

### Landing order

1. Rows 1–4 — one unit, testable alone.
2. Row 6 — has the mechanism.
3. Row 5 — only helps once something can route to a deep exit.

Measure E7's T=4 holdout after step 1. E7 trains on T in {1,2,3} and holds out
T=4, so it is the nearest local extrapolation test.

---

## 4. The hole none of this closes (stage 2)

`B` still attends to the whole prompt including T (`_run_processor:955` builds
`prompt_memory` from `token_state`, which carries every segment). So the
optimizer retains a second way to satisfy the soft-min: **read T and emit the
final answer at exit 1.**

On E1 that is the *easier* solution — 200 train rows x 3 T settings = 600
(x, T) -> answer pairs against a 492M-parameter block. That is a lookup table,
and lookup tables extrapolate to T=64 not at all.

**The fix:** withhold the T segment from the recurrent block, and give it only
to the exit selector. Then `B` cannot compute `x^(2^T)` in one step because it
does not know T, and the only representable solution is one squaring per loop
with the controller deciding when to stop. This mirrors the paper's structure
directly — each scratchpad line is produced by an operation that does not know
how many lines remain.

Implementation is all in `submission.py`, mostly in the masks
`SynchronizedProcessor.forward` already builds (`:453-481`): zero the T-segment
positions in the key mask over `prompt_memory`. One wrinkle: the workspace is
seeded per-prompt-position from `prompt_memory` (`:961`), so workspace slots
sitting on the T positions would leak it back in and need the same treatment.

### Beyond that: the width/depth allocation

`D_MODEL = 6400` with one tied block puts you at 491,859,202 of 500M — the whole
budget on width, run 7 times. The paper's argument runs the other way: the win
is serial steps in token space, not a wider single step. Under a fixed 60s
clock, 7 loops x 492M is the same FLOPs as 64 loops x ~54M, and only the second
can certify T=64. Params scale as d^2, so `d_model ~= 2048` (~50M/block) buys 64
loops at roughly today's cost.

None of the six changes above touch this. If the depth story works at all, this
is the next and largest lever.

---

# What came next

These six changes were implemented and measured. They did not move the score:
E1 went to 192 training steps at `D_MODEL=1024`, but train accuracy saturates at
100% by step 100 while test accuracy stays at 2.7%. The model memorizes
`(x, T) -> answer` rather than learning a composable step.

The follow-up plan — control/data separation, derived from *Towards Modular
Algorithm Induction* (ICLR 2020, https://arxiv.org/abs/2003.04227) — lives in
its own file:

**→ `tmp/proposed_changes.md`**

It carries the measurement table, the memorization diagnosis, and Changes 7–10.
