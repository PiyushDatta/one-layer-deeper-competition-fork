# Proposed changes — control/data separation for `piydatta_submission`

**Status:** Changes 7, 8a, 11, 12 and the new 13–15 are **implemented and
measured**, all uncommitted. Changes 8b, 9 and 10 are not started. **Start at
§0.0**, which carries the latest session: three structural defects found with
`tmp/probe.py`, all fixed, and why the score still did not move. Sections 1–7
are the original analysis, retained for provenance and now partly superseded.

Companion document: `tmp/scratchpad-changes.md` covers the six scratchpad
changes (Nye et al. 2112.00114) that are already implemented. This file covers
what to do next, after those were measured and found insufficient.

All line numbers refer to `submissions/piydatta_submission/submission.py` unless
stated otherwise.

**Primary source:** *Towards Modular Algorithm Induction*, Abolafia, Singh,
Zaheer, Sutton (Google Brain), ICLR 2020.

- Paper: https://arxiv.org/abs/2003.04227
- PDF: https://arxiv.org/pdf/2003.04227
- OpenReview: https://openreview.net/forum?id=B1lXfA4Ywr
- Local copy: `tmp/modular-algorithm-induction.pdf`, extracted text `tmp/mai.txt`

arxiv.org does not resolve directly from the devserver. Fetch through the
forward proxy:

```bash
curl -x fwdproxy:8080 -sL -o out.pdf https://arxiv.org/pdf/2003.04227
```

---

## 0. Status — read this first

| # | Change | Where | Status |
|---|---|---|---|
| **7** | **Control/data separation** — hide the T digits from the recurrent block `B`, leave them visible only to the exit selector | §3 | **Done, measured** |
| **11** | **Un-zero-init `exit_head`** so the selector is T-sensitive from step 0 | §0.7 | **Done, measured** |
| **12** | **Explicit selector M-step** — train the prior to predict which exit explains the row | §0.8 | **Done, measured** |
| **8a** | **Action history** — "what changed" + "where it changed" from the previous loop | §0.5 | **Done, measured** (no effect) |
| **13** | **Anti-collapse** — evidence-only posterior + `I(T; exit)` mutual information | **§0.0** | **Done, measured** |
| **14** | **Monotone exit ladder** — order-respecting digit reader drives the exit centre | **§0.0** | **Done, measured** |
| **15** | **`EVAL_LOOPS` 64 → 128** — stride 2 means T=64 needs exit 128 | **§0.0** | **Done, measured** |
| **16** | **SAM** — sharpness-aware minimization via `backward_passes_per_step` | **§0.1** | **Proposed, next to try** |
| 8b | Countdown via a second module — a control block that sees T and the counter but never `x` | §0.6 | Not started |
| 9 | Straight-through retokenization; no-decay group for `retokenize_gate` | §5 | Not started |
| 10 | Batch size — 512 against a 600-row split is full-batch | §6 | Not started |

---

## 0.0 Latest session — three structural defects found and fixed

Working file is `submissions/piydatta_submission/submission.py` (+255 lines,
**uncommitted**). Instrument is `tmp/probe.py`: trains the submission for a
wall-clock budget, then inspects per-exit accuracy, the exit histogram by T, and
whether the emitted answer varies with T.

### Why this is not the paper's result

MAIN's four successes — Copy, Reverse, Increment, Filter Even — are tasks where
**the per-step computation is a hand-written module** (`Sum`, `Max`,
`Increment`). The controller only learns *routing*. Their one task where the
controller had to actually compute — multi-digit add, needing carry logic —
scored **0/10** in the full model and 1/10 in a single ablation.

We ask a *learned* block to discover modular squaring, which is harder than
carry logic, in ~200 gradient steps against their 30M RL timesteps, and rule 7
forbids the hand-written modules that carried their good results. The closest
analogue to our task in their paper is the one they largely failed. Reproducing
their table is not an available outcome; making this model learn is.

### Defect 1 — the exit prior had collapsed

Probe on the 8a build:

```
per-exit exact accuracy:          prior argmax histogram by T digit:
  exit 0: 0.2613                    T=1: {0: 157}
  exit 1: 0.0831                    T=2: {0: 163}
  ...                               T=3: {0: 154}
  exit 7: 0.0302                  591 steps, loss still 0.687
```

Every row at every T routed to exit 0, and per-exit accuracy **decayed
monotonically with depth** — the recurrence was actively destructive, not merely
inert. Cause: `posterior ∝ prior * exp(-loss)` is self-reinforcing, so a leading
exit starves the rest of gradient. **Change 12 was accelerating the collapse it
was meant to help.**

This also retires the §0 "1/3 ceiling" reading as *nearly* right for the wrong
reason: the ceiling was real, but it came from collapse, not from a
mis-calibrated selector.

**Change 13.** `POSTERIOR_USES_PRIOR = False` — fit the M step to "which exit
explains this row", not "which exit already won" — plus `EXIT_MI_WEIGHT`,
maximising `I(T; exit) = H(marginal) - E_row[H(exit | row)]`: sharp per row,
spread across the batch. Since the prior is a function of T alone, spreading the
marginal is spreading across T.

Ablation shows **the evidence-only posterior is the load-bearing half**; the MI
term alone changes little.

### Defect 2 — depth was being used as an index

Change 13 worked, and immediately exposed the next problem. Exits differentiated
and train accuracy returned to **1.000000 at step 100** — but the map was:

```
T=1 -> exit 7      T=2 -> exit 4      T=3 -> exit 1
```

**Anti-monotone.** No squaring operator satisfies that: `B^1 = 8th power` and
`B^4 = 4th power` are contradictory. So the loop index was serving as an
*index*, not as iterated computation — `B`'s residual state evolves
distinguishably with loop count, so `B^k` can encode `f(x, k)`, and the prior
makes `k` a proxy for T.

**Change 7 hid T from attention; the exit index leaked it straight back in.**
This is the same memorization by another route, and the anti-monotone histogram
is the tell.

### Defect 3 — the first monotone fix was vacuous

Setting `centre = |stride| * tau(T)` with `tau` a free scalar head changes
nothing: `tau` simply learns to *decrease* in T, reproducing the identical
anti-monotone map. Constraining the stride is useless while the scalar is free.

**Change 14.** Order has to be built into the digit reader
(`Model._ordered_t_value`): digit magnitudes are `cumsum(softplus(...))` over
digit token ids, so the map from digit id to value is increasing by
construction, and place weights are positive. The exit centre is then
`|stride| * value(T) + centre`. Every value stays learned; only the ordering is
imposed.

Result — the first composable ladder in any run:

```
T=1 -> exit 2      T=2 -> exit 4      T=3 -> exit 6
```

Monotone, stride 2, i.e. `B^2 = one squaring`.

**Change 15.** Stride 2 means T=64 needs exit 128, so `EVAL_LOOPS = 64` would
have silently capped the ladder at T=32. Raised to 128; evaluation costs 8.0s of
its 30s budget.

### Results

| | E1 score | E1 test / ood | E7 test / ood |
|---|---|---|---|
| 8a build | 0.0467 | 0.033 / 0.060 | — |
| + Changes 13–15 | 0.0517 | 0.033 / 0.070 | 0.061 / 0.071 |

31 tests pass. **The score did not move.** Everything here is inside noise on
250 evaluation rows. Train accuracy is 0.84 at step 200 — still memorizing.

### Bottom line

Three genuine structural defects found and fixed. The model now has the
composable exit ladder the entire design was aiming at, and it still does not
generalize.

The remaining gap is **not structural**. Learning `y -> y^2 mod 323` as a
generalizing function from 600 examples in ~200 steps is the actual hard
problem, and none of Changes 7–15 touch it. MAIN did not solve it either — they
hand-wrote the module.

### Two things to decide

**A rule 7 judgement call.** The ordered digit reader (Change 14) imposes
monotonicity architecturally — the way a positional encoding imposes order —
while leaving every value learned. I believe it is legal, but it is closer to
the line than anything else here, and the stride-2 ladder depends on it. Set
`EXIT_ORDERED_DIGITS = False` to revert to the free scalar head.

**A local-only speedup, not a submission change.** `num_workers=2` against a
600-row dataset at batch 512 is one batch per epoch, so DataLoader workers
respawn *every training step*. `tmp/probe.py` gets 590 steps at `num_workers=0`
where the runner gets 219. Not actionable for scoring — the evaluator owns the
manifest — but it makes local iteration 3x faster. It also ruled out "not enough
steps" early: at 590 steps the loss was still pinned at 0.69.

### New constants

`POSTERIOR_USES_PRIOR`, `EXIT_MI_WEIGHT`, `EXIT_MONOTONE_IN_T`,
`EXIT_ORDERED_DIGITS`, `EXIT_STRIDE_INIT`, `EXIT_WIDTH_INIT`,
`EXIT_CENTRE_INIT`, `DIGIT_OFFSET`, `NUM_DIGITS`, `MAX_T_DIGITS`. All report in
the CONSTANTS banner; each is ablatable.

---

## 0.1 Change 16 — SAM (sharpness-aware minimization)

**Source:** *Sharpness-Aware Minimization for Efficiently Improving
Generalization*, Foret, Kleiner, Mobahi, Neyshabur, ICLR 2021.
https://arxiv.org/abs/2010.01412

### What it does

Ordinary training minimises the loss at a point. SAM minimises it over a
neighbourhood:

```
min_w  max_{||eps|| <= rho}  L(w + eps)
```

"Find weights where the loss is low *and stays low if you nudge them*," which
biases toward wide flat basins and away from sharp narrow ones. The inner max is
approximated to first order, giving two passes per update:

1. `g = grad L(w)` at the current weights.
2. `eps = rho * g / ||g||` — the direction that locally *increases* loss most.
3. `g' = grad L(w + eps)` — the gradient at the deliberately worsened point.
4. Restore `w`, then update with `g'`: `w <- w - lr * g'`.

The step is computed at the perturbed point and applied at the original one. In
a sharp well `g'` points hard back out and the update is large; in a flat basin
`g'` ~= `g` and little changes. Sharp minima get walked out of.

### Why it is relevant to our failure

Memorising 600 examples individually carves many narrow wells — each memorised
point is a sharp local fit. A compositional circuit is one broad basin. The
grokking literature consistently places the generalising solution at lower
weight norm and flatter geometry. So "prefer flat" is indirectly "prefer the
compositional solution", which is the preference we have been unable to express
through any loss term so far.

### The API fits exactly

From the README's `OptimizerBundle` contract:

> `backward_passes_per_step` — 1–8 evaluator-owned forward/loss/backward passes
> on the same batch before one update. `between_backward_passes` runs under
> `no_grad` and may transform gradients, parameters, or optimizer state; **a
> custom optimizer can restore temporary perturbations when it performs the
> final update.**

| SAM step | Mechanism |
|---|---|
| 1. `g = grad L(w)` | pass 1, evaluator-owned |
| 2. perturb `w += eps` | `between_backward_passes`, stash `eps` |
| 3. `g' = grad L(w + eps)` | pass 2, evaluator-owned |
| 4. restore, update with `g'` | our `optimizer.step()` |

The "restore temporary perturbations" clause describes step 4 almost verbatim.
The rules anticipate this pattern.

### Implementation notes

`between_backward_passes(BackwardPassContext)` receives only
`(completed_steps, pass_index, total_passes)` — **no model or parameter
handle** — so it must be a closure over the parameter list built in
`build_optimizer`. It already runs under `no_grad`.

```python
SAM_RHO = 0.05   # 0.0 ablates SAM back to a single pass


def build_optimizer(model, spec):
    ...
    params = [p for p in model.parameters() if p.requires_grad]
    perturbation: list[Tensor | None] = [None] * len(params)

    def perturb(context: BackwardPassContext) -> None:
        # Only before the final pass. Runs under no_grad already.
        total_sq = torch.zeros((), device=params[0].device)
        for p in params:
            if p.grad is not None:
                total_sq = total_sq + p.grad.pow(2).sum()
        scale = SAM_RHO / total_sq.sqrt().clamp_min(1e-12)
        for i, p in enumerate(params):
            if p.grad is None:
                perturbation[i] = None
                continue
            step = p.grad * scale
            perturbation[i] = step
            p.add_(step)

    optimizer = CombinedOptimizer([muon, adamw], restore=(params, perturbation))
    return OptimizerBundle(
        optimizer,
        scheduler=_build_scheduler(optimizer, spec),
        backward_passes_per_step=2 if SAM_RHO > 0.0 else 1,
        between_backward_passes=perturb if SAM_RHO > 0.0 else None,
    )
```

`CombinedOptimizer.step()` must undo the perturbation *before* delegating, so
the update lands at the original point:

```python
def step(self, closure=None):
    if self._restore is not None:
        params, perturbation = self._restore
        with torch.no_grad():
            for p, eps in zip(params, perturbation):
                if eps is not None:
                    p.sub_(eps)
        perturbation[:] = [None] * len(perturbation)
    for optimizer in self._optimizers:
        optimizer.step(closure)
```

Details worth getting right:

- The evaluator **clips gradients independently on each pass**, so the `g` read
  in `perturb` is already clipped. Fine for a direction, but it means `rho` is
  measured against a clipped norm.
- Global norm across all parameters is the original SAM. Per-parameter scaling
  is ASAM (Kwon et al. 2021) and is the variant to try if plain SAM does
  nothing.
- The scheduler steps once per completed optimizer update, not per pass, so no
  change there.
- Restoration must be exact. Any drift accumulates as a silent parameter offset
  across every step of training.

### Cost, and the honest caveats

**It halves the step count** — two passes per update takes ~219 steps to ~110.
We are already step-starved, and every other item on the list gets *cheaper*
with more steps while this gets more expensive. Pair it with `should_reuse_batch`
if step count becomes the limiter.

**SAM chooses among basins you can reach; it does not cross plateaus.** Our
diagnosis is that the compositional solution sits behind a flat region with no
gradient pointing at it. SAM reshapes preferences among *reachable* solutions.
It will not manufacture signal where there is none — that is what the
self-consistency loss and the depth curriculum are for. This is why it is ranked
below them.

**Untested interaction with Muon.** Muon already orthogonalises its update;
composing that with a SAM-perturbed gradient is not a combination either paper
studied. If results are strange, try SAM with the AdamW group only.

**The theory is contested.** Dinh et al. (2017) showed sharpness is not
reparameterisation-invariant, so "flat implies generalises" is an empirical
regularity, not a theorem. SAM works reliably in practice; the explanation is
less settled than its popularity suggests.

### Success criterion

Train accuracy should saturate *later or lower* while test accuracy holds or
improves — i.e. the train/test gap narrows. If train accuracy still hits ~0.85
by step 100 at half the steps, SAM is not biasing the basin choice and should be
reverted rather than tuned.

---

### What this document originally proposed

Written after the six scratchpad changes were measured and found to have left
the model memorizing `(x, T) -> answer`.

The claim behind Change 7: if `B` cannot see T, it cannot select between `x^2`,
`x^4` and `x^8` for the same `x`, so the only function it can represent that
fits all three T groups is one squaring. Memorization then works *for* us — it
memorizes the 288-entry step map instead of the 600-entry `(x, T)` map.

Its stated success criterion — **train accuracy stops saturating** — was met.

### Latest run — Changes 11 + 12, E1, seed 74

189 tests pass. State unchanged at 12,704,897; neither change adds parameters.

| | Change 7 only | + Changes 11/12 |
|---|---|---|
| completed steps | 196 | 225 |
| step-1 train loss | 3.147706 | **5.080527** |
| step-100 train accuracy | 0.257812 | 0.267578 |
| step-100 train loss | 0.779737 | 0.780819 |
| step-200 train accuracy | — | 0.275391 |
| final_train_loss | 0.694682 | 0.690627 |
| test loss | 5.765 | 5.937 |
| test / ood accuracy | 0.0400 / 0.060 | 0.0400 / 0.060 |
| score | 0.0500 | 0.0500 |

For reference, the pre-Change-7 state was: train accuracy **1.000000** at step
100, `final_train_loss` **0.000269**, test loss 7.782, score 0.0433. Change 7
removed that memorization; Changes 11 and 12 moved almost nothing on top of it.

### The selector diagnosis was wrong

The §0 prediction — that fixing the selector "should sharply raise train accuracy
toward what the soft-min loss implies" — is **falsified**. Train accuracy went
25.8% → 27.5%. Test loss got slightly worse.

**And the selector demonstrably learned.** Step-1 loss is 5.081 against 3.148
before, a delta of 1.93 ≈ `ln(8) = 2.08` — the selector term at a uniform prior
over 8 exits, confirming both correct wiring and a uniform prior at
initialisation. By the end the *total* is 0.6906, and since `row_losses >= 0`
the selector term is at most 0.69, down from 2.08. The prior concentrated, and
it bought ~2 points.

**Where the reasoning broke.** `final_train_loss = 0.695` was read as "some exit
is nearly right for most rows," implying the readout was picking the wrong one.
But that is a *per-token* cross-entropy, and exact match needs every digit
right. A per-token CE near 0.69 is simply not accurate enough for exact match on
2–3 digits regardless of which exit is chosen. The "25.8% is 77% of the 1/3
ceiling" arithmetic fit by coincidence.

**Still unresolved from this run alone:** the reported loss now sums both terms,
so it cannot be decomposed. If `row_losses` stayed ~0.69 the selector term is
~0; if the selector term is ~0.4 then `row_losses` fell to ~0.29, which would be
a large improvement paired with a 2-point accuracy move. The
`SELECTOR_LOSS_WEIGHT = 0.0` ablation settles it in one run.

### The actual bottleneck

Pre-Change-7 train CE was 0.0003 — memorized. It is now 0.69 and stays there.
**The model can no longer fit even the training data.** Change 7 removed the
shortcut and the composable solution has not replaced it. Everything downstream
of "does `B` compute a squaring" is moot until it does.

Two hypotheses, both cheap to test:

1. **Is one block application enough for one squaring?** With T hidden and the
   answer slots seeded blank, exit 1 must equal `x^2` after a *single*
   attention+MLP: read the `x` digits from `prompt_memory`, square mod N, write
   to the answer slots. If one layer cannot do that, exit 1 fails and `B(B(x))`
   cannot be `x^4` either — the ladder fails from the bottom.

   The soft-min already permits the fix with no code change: nothing pins exit
   index to T, so the model is free to discover a **stride** — exit 2 for T=1,
   exit 4 for T=2, exit 6 for T=3 — giving two block applications per squaring,
   with the selector learning the mapping. At stride 2, `EVAL_LOOPS = 64` still
   reaches T=32.

2. **Do the exits differentiate at all?** If `B` is near-identity every exit
   emits the same thing, and accuracy is capped near 1/3 for reasons unrelated
   to the selector. A DBUG run comparing exit-`k` logits pairwise answers this
   and is the most informative measurement currently available.

### Notes carried forward from Change 7

`SynchronizedProcessor.forward` takes a `prompt_key_mask` for the read-only half
of the joint key mask only, so the workspace stays readable and writable —
required because `target_positions` sit on segment 3. The workspace is seeded
from `blank_token + position_signal` at T positions rather than from
`prompt_memory`. `segment_ids` flows to `_run_processor` unconditionally,
**including into the five DBUG counterfactual probes** — without that the probes
would run with T unmasked while the normal forward has it masked, confounding
every counterfactual.

The separation invariant is on the block's **output**, not its input.
`prompt_memory` still physically contains the T tokens; they are merely
unreachable as attention keys. The test asserts that work-stream block output
and every exit's `hypothesis_logits` are bit-identical across two prompts
differing only in the T digit.

### Notes carried forward from Changes 11/12

`SELECTOR_LOSS_WEIGHT = 1.0` is untuned. At K=8 the term is a cross-entropy of
order `ln(8) = 2.08`, comparable to the task loss, so 1.0 is not a small
perturbation — though it fell to ≤0.69 by convergence, so it is not dominating
at the end. It is a constant precisely so `0.0` ablates Change 12 and measures
11 alone.

"Trains the selector only" is slightly loose. Gradient does not reach
`processor.block`, which is the claim that matters, but it does reach
`token_embedding`, `position_embedding` and `segment_embedding` through
`t_summary`. Small, arguably desirable since it lets the T representation
sharpen, but not literally selector-only.

`test_row_loss_is_a_soft_min_bounded_by_the_exit_losses` began failing by
exactly `ln(2) = 0.693` — the selector term at a uniform posterior over two
exits, which is itself confirmation the term is wired correctly. It now pins
`SELECTOR_LOSS_WEIGHT = 0.0` for its duration, with
`test_selector_loss_pulls_the_prior_toward_the_explaining_exit` added to cover
the new behaviour.

---

## 0.5 Change 8a — action history (supersedes §4)

MAIN's ablation removes action history and **every task drops to 0/10**. It
supplies two distinct things, and 8a should supply both:

- **where** the heads touched — one-hot channels in the variable-size `sigma_t`
- **what** was under them — values in the fixed-size `xi_t`

Constants:

```python
HISTORY_RANK = 64          # low-rank; a full D x D projection is 1.05M params and ~8% step cost
HISTORY_GATE_INIT = -2.0   # same convention as retokenize_gate
```

New parameters, last in `__init__` per the existing ordering convention:

```python
self.history_norm = RMSNorm(D_MODEL)
self.history_down = nn.Linear(D_MODEL, HISTORY_RANK, bias=False)
self.history_up = nn.Linear(HISTORY_RANK, D_MODEL, bias=False)
self.history_where = nn.Parameter(torch.zeros(D_MODEL))
self.history_gate = nn.Parameter(torch.full((), HISTORY_GATE_INIT))
nn.init.normal_(self.history_down.weight, std=init_std)
nn.init.zeros_(self.history_up.weight)      # signal starts at zero, the gate opens it
nn.init.normal_(self.history_where, std=init_std)
```

```python
def _history_signal(self, delta: Tensor) -> Tensor:
    # "what changed" — low-rank projection of the previous loop's update
    what = self.history_up(self.history_down(self.history_norm(delta)))
    # "where it changed" — MAIN's landmark channel for recently touched cells
    change_rms = delta.pow(2).mean(-1, keepdim=True).sqrt()
    where = change_rms * self.history_where.view(1, 1, -1)
    return torch.sigmoid(self.history_gate) * (what + where)
```

Injected in `SynchronizedProcessor.forward` **after** the readout is recorded, so
history is scaffolding for the next loop rather than part of the answer:

```diff
             work_state = candidate_state[:, prompt_len:]
             if retokenize is not None:
                 work_state = torch.cat((...), dim=1)
+            readout_state = work_state
             if hypothesis_states is not None:
-                hypothesis_states.append(work_state[:, 1:output_end])
+                hypothesis_states.append(readout_state[:, 1:output_end])
+            if history is not None:
+                work_state = work_state + history(work_state - previous_work_state)
+            previous_work_state = readout_state
```

`previous_work_state` is initialised to the seed before the loop, so the loop-0
delta is zero. Depth-agnostic — no per-index parameters — so it extrapolates
past `TRAIN_LOOPS`. Adds ~132K parameters.

---

## 0.6 Change 8b — countdown via a second module (supersedes §4)

### The tension §4 left open, and how MAIN resolves it

§4 warned that seeding a control register from T puts T back inside `B`'s
attention, undoing Change 7, and left it unresolved. **MAIN answers this and the
original write-up missed it: its modules have disjoint views.** A module sees
only the cells under its own read heads. So use two blocks:

| | reads | never sees |
|---|---|---|
| `block` (compute) | prompt minus T-segment, workspace slots | T, the counter |
| `control_block` | T-segment of prompt, control slots | workspace, `x` |

`B` still cannot shortcut — no access to T or to the counter. `control_block`
cannot compute squaring — it never sees `x`. The counter carries *how many steps
remain* and structurally nothing else.

This also replaces the static `exit_head(t_summary)` selector, which is the part
that cannot extrapolate: it is only ever trained on exits 1–3. A halting
predicate over a decrementing counter can.

```python
NUM_CONTROL_SLOTS = 4
USE_COUNTDOWN = True
```

```python
# last in __init__
self.control_block = Block()                       # own parameters, not tied to self.block
self.control_slots = nn.Embedding(NUM_CONTROL_SLOTS, D_MODEL)
self.halt_head = nn.Linear(D_MODEL, 1)
```

Per loop the control stream advances alongside the compute stream:

```python
# control_memory = prompt_memory gathered at T-segment positions (+ position_signal)
control_joint = torch.cat((control_memory, control_state), dim=1)
control_state = self.control_block(control_joint, control_mask)[:, control_memory.shape[1]:]
halt_logit = self.halt_head(self.final_norm(control_state[:, :1]))   # [B, 1]
```

Collect `halt_logit` per loop into `[B, K]` and use it as the exit distribution,
replacing `exit_head`:

```diff
-            hypothesis_log_prior = F.log_softmax(
-                (self.exit_head(self.exit_norm(t_summary)) + self.hypothesis_prior)[
-                    :, :active_loops
-                ],
-                dim=-1,
-            )
+            hypothesis_log_prior = F.log_softmax(
+                torch.cat(halt_logits, dim=1) + self.hypothesis_prior[:active_loops],
+                dim=-1,
+            )
```

Everything downstream — the per-row soft-min and the Change 12 M-step — works
unchanged, since both consume `hypothesis_log_prior` as `[B, K]`. Change 12
becomes *more* useful here: it now trains a halting predicate rather than a
static lookup.

### Cost

| | params | notes |
|---|---|---|
| current | 12,704,897 | |
| + 8a | ~132,000 | rank-64 projection |
| + 8b | ~12,600,000 | second full-width block |
| **total** | **~25.4M** | 5.1% of the 500M ceiling |

Compute is the real cost, not parameters. The control stream is ~6 positions
against the main ~31, so roughly +19% per loop, plus 8a's projection. Expect
steps to fall from 225 to ~170–190. If that bites, narrow `control_block` to
`D_MODEL // 4` with projections in and out — it only has to count.

### What success looks like

- **8a working** — train CE falls below 0.69. MAIN's ablation says iterative
  computation is impossible without history, so if the delta channel does
  anything it should show up as the block finally fitting.
- **8b working** — the `seen_n` ladder stops being flat across T. That is the
  first sign anything responds to depth. A non-null `Max T` is the real prize
  and is several steps further out.
- **Watch `sigmoid(history_gate)` and `sigmoid(retokenize_gate)`.** Both start at
  0.12. If neither opens, the model is declining the scaffolding, and that is
  the finding.

Land 8a alone first and measure. It is ~130K parameters and one hook, so if a
combined run regresses you will not know which half did it — and 8b touches the
exit path, which is the part that currently works.

---

## 0.7 Change 11 — stop zero-initialising `exit_head` (done)

Zero-init was recommended so the T-conditioned exit "cannot regress the starting
point." That reasoning held while the shortcut existed and T-conditioning was
optional insurance. **Change 7 inverted it.** T-conditioning is now the only
path by which T can affect the output, and zero-init delays the sole working
mechanism inside a 196-step budget.

```diff
-                nn.init.zeros_(self.exit_head.weight)
-                nn.init.zeros_(self.exit_head.bias)
+                nn.init.normal_(self.exit_head.weight, std=init_std)
+                nn.init.zeros_(self.exit_head.bias)
```

Note the consequence for tests: `test_eval_selects_an_exit_per_row_from_the_t_segment`
currently has to randomise `exit_head` by hand to show T reaching the selector,
precisely because zero-init makes the model T-blind at step 0. After this change
it no longer needs to.

Cheapest possible probe of the diagnosis. Watch the loss/accuracy gap.

## 0.8 Change 12 — give the selector its own gradient (done)

The soft-min does push the prior toward the posterior. Differentiating
`-tau * logsumexp(log_w - L/tau)` through the `log_softmax` gives
`tau * (p_k - w_k)`, which is the correct EM signal. But it is bounded by
`tau = 1` and competes with task gradients into a 12.7M-parameter block.

There is also a train/eval objective mismatch: **training is satisfied when
*some* exit is right; evaluation requires the *argmax-prior* exit to be right.**
Nothing penalises a wrong prior directly.

An explicit M-step term closes that:

```diff
+    with torch.no_grad():
+        posterior = F.softmax(
+            hypothesis_log_prior - sequence_losses / temperature, dim=1
+        )
+    # Train the T-conditioned prior to predict which exit actually explains the
+    # row. The soft-min only needs *some* exit to be right; eval needs the
+    # selected one to be right.
+    selector_loss = -(posterior * hypothesis_log_prior).sum(dim=1)
     task_loss = row_losses[rows_with_targets].mean()
+    task_loss = task_loss + SELECTOR_LOSS_WEIGHT * selector_loss[
+        rows_with_targets
+    ].mean()
```

The posterior is detached, so this trains the selector only, not the block.

## 0.9 Updated ordering

| Step | Change | Cost | Success criterion |
|---|---|---|---|
| ~~0~~ | ~~**7** — control/data separation~~ | done | ~~train accuracy stops saturating~~ **met** |
| ~~1~~ | ~~**11** — un-zero-init `exit_head`~~ | done | ~~loss/accuracy gap narrows~~ **not met** (25.8% → 27.5%) |
| ~~2~~ | ~~**12** — explicit selector loss~~ | done | ~~train accuracy rises~~ **not met**; selector learned, bought ~2 points |
| 3 | **8a** — action history (§0.5) | ~1h | train CE falls below 0.69 |
| 4 | **8b** — countdown module (§0.6) | ~half day | `seen_n` ladder stops being flat across T |
| 5 | **10** — batch size | minutes | more steps, same or better test |
| 6 | **9** — straight-through | ~30m | `retokenize_gate` opens |

Two measurements worth taking alongside, both cheap:

- **`SELECTOR_LOSS_WEIGHT = 0.0` ablation** — decomposes the summed loss and
  retires the selector question either way.
- **DBUG: do exits differentiate?** If `B` is near-identity, that is the finding
  and everything reorders around it.

**Retired diagnostic.** The gap between `final_train_loss` and train accuracy
was proposed as a zero-cost measure of selector error. It is not — the gap is
mostly the difference between per-token cross-entropy and exact match. Do not
read it that way.

**Expectation to set.** The binding constraint is now that `B` cannot fit the
training data at all (CE 0.69, versus 0.0003 when it was allowed to memorize).
8a and 8b target that. **Test** accuracy is downstream of it and cannot move
until train CE does — the flat `seen_n` ladder
(`0, 0.053, 0.026, 0.026, 0, 0, 0`) confirms nothing yet responds to depth.

---

## 1. Where we are

Three runs on E1, all `seed=74`:

| | baseline (D=6400) | D=6400 + scratchpad changes | D=1024 + scratchpad changes |
|---|---|---|---|
| model_state_elements | 491,859,202 | 201,810,049 | 12,703,873 |
| completed steps | — | 59 | **192** |
| evaluation_seconds | — | 19.69 / 30 | 5.63 / 30 |
| final_train_loss | — | 1.655 | **0.000269** |
| test loss | — | 2.052 | **7.782** |
| test / ood accuracy | — | 0.0267 / 0.060 | 0.0267 / 0.060 |
| **score** | — | 0.0433 | **0.0433** |

The width cut did what it was meant to — 3x the steps, eval down to a fifth of
budget — and the score did not move.

### The model memorizes

Train accuracy hits 100% at step 100 of 192 with train loss 2.7e-4, while test
accuracy is 2.7% and test loss went *up* to 7.78. With `drop_last=True` at batch
512 on a 600-row train split that is one batch per epoch — 164 full-batch
epochs, half of them on an already-solved training set.

The `seen_n` ladder confirms it. On the 38 reserved `x` values never seen in
training: `0.0, 0.0, 0.053, 0.079, 0.026, 0.0, 0.053` — noise, no depth
structure.

**Diagnosis.** `B` reads T from the prompt and memorizes `(x, T) -> answer`. The
per-row soft-min permits the composable solution but does nothing to prefer it,
and with 12.7M parameters against 600 rows the lookup is far cheaper. The
scratchpad changes made composition *representable*; nothing made it
*preferred*.

### Why the scratchpad alone cannot fix this

The scratchpad's power in Nye et al. comes from **supervised intermediate
targets**, which change the loss landscape so that memorizing an end-to-end map
stops being optimal. Rule 14 denies that. What was implemented is the
architectural composability half, which cannot prevent memorization by itself.

Nye et al. also had three advantages absent here: pretrained LMs, thousands of
training examples, and per-step operations (single-digit addition with carry)
far easier than `y -> y^2 mod N`.

### The reframe

Memorization is not the enemy. Memorizing the **wrong function** is.

At fixed N=323 the squaring map is a permutation over 288 units. A perfect
lookup of `y -> y^2 mod 323`, applied T times, gives exact answers at T=64. You
do not need arithmetic generalization to certify Max T on a seen modulus — you
need the memorization to be of the **one-step map**. The T=1 rows already hand
over 200 of the 288 entries.

Currently the model memorizes 600 `(x, T)` entries, useless at T=4. The target
is 288 `y -> y^2` entries, sufficient for every T. Same mechanism, right
function.

Caveat: this does nothing for the OOD-N profile, where the modulus is unseen and
a lookup is worthless. That needs real arithmetic and is likely out of reach in
these budgets. Max T on seen moduli is the first ranking key and *is* reachable.

---

## 2. What MAIN contributes

MAIN learns algorithms from input-output examples and generalizes from training
length 10 to test length 100. Three design commitments transfer.

### 2.1 Control flow is separated from data flow

Three strictly separated parts:

- **Memory** — a finite tape of *discrete* tokens, length set by the input.
- **Modules** — pre-specified functions, each reading R cells and writing W
  cells. A module sees **only the cells under its read heads**. Never the tape,
  never the task. Modules need not be differentiable.
- **Controller** — a policy that **cannot touch memory**. It only chooses which
  module to run and where its heads go.

```
s_{t+1}[h^(w,j)_t] := m(s_t[h^(r,1)_t], ..., s_t[h^(r,R)_t])[j]
s_{t+1}[i]         := s_t[i]     for all other i
```

Neither half can solve the task alone. The controller decides *what and where*;
the module computes but is blind to everything except its arguments.

**This is the structural version of "withhold T from `B`."** MAIN gets it by
construction — a module physically cannot memorize `(x, T) -> answer` because it
never sees T.

### 2.2 Restricting the compute path's view helped, empirically

Table 2 of the paper — runs out of 10 reaching 100% at length 100:

| Config | Copy | Reverse | Increment | Filter Even | Multi-Digit Add |
|---|---|---|---|---|---|
| Attention encoder (full) | 7 | 7 | 5 | 9 | **0** |
| − No Tape Values | 3 | 6 | 1 | 0 | **1** |
| − No Action History | 0 | 0 | 0 | 0 | 0 |
| − No Action History Tape Values | 7 | 8 | 7 | 5 | 0 |
| Recurrent encoder | 0 | 3 | 0 | 0 | 0 |

Multi-Digit Add — their only genuinely arithmetic task, and the closest analogue
to modular squaring — **never worked in the full model** and succeeded only when
tape contents were *removed* from the controller. Their explanation: hiding the
tape and leaving only head values constrains the argument space and helps the
controller find the computation.

That is published evidence for restricting what the compute path can see, on the
task most like ours.

### 2.3 Action history is load-bearing

Removing it drops **every** task to 0/10. Without a record of what it just did,
the controller cannot tell where it is in an iterative computation.

We have no analogue. The workspace carries no explicit record of which loop it
is on or what changed last iteration.

### Already covered

- **Attention beats recurrence for length generalization** (recurrent is fine at
  length 10, collapses at 100). We already use attention.
- **Landmarks as immutable metadata** — MAIN found the controller overwrote
  landmark tokens, so their positions are supplied separately. We already have
  this: `prompt_memory` is constant across loops, pinned by
  `test_synchronized_processor_keeps_prompt_memory_immutable`.

### What does not transfer

- MAIN's modules are **hand-written functions** (`Sum`, `Max`, `Increment`).
  That is a hard-coded algorithm in the forward pass — rule 7 forbids it. Our
  squaring step must be learned.
- **30M RL timesteps across 50 workers with IMPALA.** We have 60 seconds, the
  evaluator owns the loop, and participant code may not call backward.
- **The halting oracle.** They removed halt-learning as unstable and supplied an
  oracle, at eval too. We partly get this free since T is in the prompt — but
  only via the exit selector.

---

## 3. Change 7 — control/data separation

**The only proposed change that addresses the measured failure.**

Make `B` unable to read the T digits. Only the exit selector sees T. Then `B`
cannot select between `x^2`, `x^4` and `x^8` for the same `x`, and the only
function it can represent that fits all three T groups is one squaring.
Memorization then works *for* us — it memorizes the step map.

### The gotcha: the answer is written on top of the T tokens

Verified on E1 `train.jsonl` row 0 (N=323, x=275, T=1, answer=43):

```
prompt      N   3   2   3   X   2   7   5   T   1
index       0   1   2   3   4   5   6   7   8   9
segment     1   1   1   1   2   2   2   2   3   3
target_positions: [8, 9] -> segments [3, 3]
```

`data/squaring_mod.py:106` sets `target_positions` to the last `len(result)`
valid positions, and `segment_ids` marks segment 3 from the `T` token onward
(`:1219`, `:1226`). **They coincide.** Both answer slots sit on segment 3.

So a naive "mask out segment 3" blanks the output register. The separation must
distinguish two streams:

| stream | contains T? | action |
|---|---|---|
| `prompt_memory` (read-only, immutable across loops) | yes, the T digit tokens | **mask these keys out** |
| workspace (mutable, holds the answer) | seeded from `prompt_memory` | **keep fully readable/writable, seed T slots blank** |

### Sketch

`_run_processor` currently seeds everything from the prompt:

```python
prompt_memory = token_state + position_signal
workspace_state = prompt_memory + self.workspace_token.view(1, 1, -1)
```

```diff
+        # Control/data separation (MAIN 3.1): the recurrent block may not read
+        # the T digits. Only the exit selector sees T.
+        t_positions = segment_ids.eq(3)
+        # Read-only prompt stream: mask T keys out entirely.
+        prompt_key_mask = attention_mask.bool() & ~t_positions
+        # Workspace stays fully readable and writable, because the answer is
+        # read from target_positions, which sit ON segment 3. Only the seed
+        # content is blanked, so loop 0 carries no T.
+        workspace_seed = torch.where(
+            t_positions[..., None],
+            self.blank_token.view(1, 1, -1) + position_signal,
+            prompt_memory,
+        )
+        workspace_state = workspace_seed + self.workspace_token.view(1, 1, -1)
```

`SynchronizedProcessor.forward` takes a new `prompt_key_mask` used for the
read-only half of the joint key mask, while `work_mask` keeps the full valid
mask:

```diff
-        joint_mask = torch.cat((prompt_mask, work_mask), dim=1)
+        joint_mask = torch.cat((prompt_key_mask, work_mask), dim=1)
```

`segment_ids` must now be passed to `_run_processor` unconditionally — today it
is only forwarded when `collect_model_diagnostics` is on (`:1249`). Add
`self.blank_token = nn.Parameter(...)` last in `__init__`, per the existing
ordering convention.

### Residual leakage to accept and name

- **Prompt length** still varies with T's digit count. Single-digit on Easy;
  matters on Medium (T=16).
- **`position_signal`** is still present at the blanked slots.

Neither reveals T's *value* on Easy. Both are worth stating rather than
pretending the separation is airtight.

### What to measure

Train accuracy should **stop** hitting 100%. That is the point — the shortcut is
gone. If train accuracy stays saturated, the separation leaked somewhere.

---

## 4. Change 8 — a progress signal

MAIN's ablation puts every task at 0/10 without action history. Nothing in our
workspace records which loop it is on.

**8a. Delta channel.** Feed `work_state - prev_work_state` through a projection
into the next loop. Depth-agnostic — no per-index parameters, so it extrapolates
past `TRAIN_LOOPS` — and cheap. Closest direct analogue to MAIN's read/write
history.

**8b. Countdown register.** Reserve scratchpad slots as a control register
seeded from the T segment and let `B` learn to decrement it. Halting becomes the
local predicate "is it zero," which extrapolates from {1,2,3} far better than
"map digits(T) to a loop index."

**Tension to resolve before attempting 8b:** it re-introduces T into `B`'s
attention through the control slot, which is exactly what Change 7 removes. A
countdown carries *how many steps remain* rather than *which power to jump to*,
but `B` is a single shared block attending over everything, so nothing
structurally enforces that distinction. Do not attempt 8b until 7 is measured.

**Do 8a first.** It is strictly compatible with Change 7.

---

## 5. Change 9 — harder discreteness

MAIN's tape holds discrete tokens and its modules are not differentiable. Our
retokenization is the soft version, gated to ~0.12.

```python
if RETOKENIZE_STRAIGHT_THROUGH:
    hard = F.one_hot(probabilities.argmax(-1), probabilities.shape[-1]).to(probabilities.dtype)
    probabilities = hard + probabilities - probabilities.detach()
```

Also: the gate sits in an AdamW group with `weight_decay=0.1`, which drifts it
from −2.0 toward 0 at roughly 2e-4/step — toward *opening*, independent of
whether it helps. A no-decay group would make the gate's trajectory
interpretable as evidence.

---

## 6. Change 10 — batch size

512 against a 600-row train split with `drop_last=True` is one batch per epoch:
full-batch gradient descent, a known contributor to memorization.
`Submission(batch_size=...)` is unset, so it inherits the manifest's 512.
Something like 64 gives 9 batches/epoch and real SGD noise.

Caveat: it is a single import-time constant with no spec available, so whatever
is chosen also applies to Hard's hidden dataset, where 512 may be correct.

---

## 7. Order and success criteria

| Step | Change | Cost | Success criterion |
|---|---|---|---|
| 1 | **7** — control/data separation | ~1h | train accuracy stops saturating; train/test gap narrows |
| 2 | **10** — batch size | minutes | more steps, same or better test |
| 3 | **8a** — delta channel | ~1h | E7 T=4 holdout moves off noise |
| 4 | **9** — straight-through | ~30m | `seen_n` ladder stops being flat across T |
| 5 | **8b** — countdown | ~half day | first non-null `Max T` |

Change 7 is the only step that addresses the measured failure. Everything after
it is contingent on it working.

**Primary metric from here is the train/test gap, not `mean_exact_accuracy`.**
At 100% vs 2.7% the score cannot distinguish "learned nothing" from "learned the
wrong function," and only the second is true.

**Secondary diagnostic:** whether `hypothesis_log_prior.argmax(-1)` correlates
with the prompt's T. If the ladder is still flat across T after Change 7 with
500+ steps, the exit selector is not learning and 8b becomes the priority rather
than the last step.

**Test impact.** Change 7 touches the workspace seeding and mask construction,
so expect `test_workspace_starts_from_aligned_prompt_representation` and
`test_synchronized_scratchpad_handles_an_all_padding_prompt` to need updating,
and add one asserting `B` cannot see the T segment (two prompts differing only
in the T digit should produce identical loop-1 workspace states).
