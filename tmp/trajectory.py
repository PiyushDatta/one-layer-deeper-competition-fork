"""Score trajectory: does mean_exact_accuracy peak early and then decay?

A no-input constant predictor scores 0.1533 on E1 (test 0.0867, ood 0.2200).
Every trained run scores about 0.05. If the model passes through the marginal
on its way to confident memorisation, there is an early-stopping point worth
far more than any architectural change, and Submission(max_steps=...) can
capture it.

Usage: python -m tmp.trajectory [seconds] [label] [key=value ...]
"""

from __future__ import annotations

import sys
import time

import torch

from benchmark.api import ModelSpec, OptimizerSpec, TokenLossBatch, BackwardPassContext
from benchmark.batches import prepare_batch
from data.config import DataConfig
from data.factory import make_dataloaders
from tmp.sweep import DEVICE, ROOT, fresh_module, measure


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    label = sys.argv[2] if len(sys.argv) > 2 else "shipped"
    overrides = {}
    for item in sys.argv[3:]:
        key, _, value = item.partition("=")
        overrides[key] = float(value) if "." in value or "e-" in value else int(value)

    module = fresh_module()
    for key, value in overrides.items():
        setattr(module, key, value)
    torch.manual_seed(74)
    batch = getattr(module, "TRAIN_BATCH_SIZE", 64)
    loaders = make_dataloaders(
        DataConfig(data_root=ROOT, batch_size=batch, eval_batch_size=512,
                   seed=45, num_workers=0),
        device=DEVICE,
    )
    model = module.build_model(ModelSpec(17, 13, 500_000_000)).to(DEVICE)
    bundle = module.build_optimizer(
        model, OptimizerSpec(training_time_seconds=budget, device_type="cuda")
    )
    opt, sched = bundle.optimizer, bundle.scheduler
    passes = bundle.backward_passes_per_step

    print(f"{label}  overrides={overrides}  params={sum(p.numel() for p in model.parameters()):,}")
    print("baseline to beat: test 0.0867  ood 0.2200  score 0.1533\n")
    print(f"{'step':>6} {'elapsed':>8} {'train_ex':>9} {'test_ex':>8} "
          f"{'ood_ex':>7} {'score':>7} {'D1_ex':>6}")

    best = (0.0, 0)
    start, step, next_report = time.monotonic(), 0, 25
    model.train()
    while time.monotonic() - start < budget:
        for raw in loaders["train"]:
            if time.monotonic() - start >= budget:
                break
            ids, targets, mask, positions = prepare_batch(raw, DEVICE)
            for pass_index in range(1, passes + 1):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, aux = model(ids, attention_mask=mask)
                    # The runner hands the loss logits already gathered at the
                    # target positions, not the full sequence.
                    rows = torch.arange(logits.shape[0], device=DEVICE)[:, None]
                    at_logits = logits[rows, positions.clamp_min(0)].float()
                    loss = module.token_training_loss(
                        TokenLossBatch(at_logits, targets, targets != -100, positions, aux)
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if pass_index < passes and bundle.between_backward_passes is not None:
                    with torch.no_grad():
                        bundle.between_backward_passes(
                            BackwardPassContext(step, pass_index, passes)
                        )
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            if step >= next_report:
                elapsed = time.monotonic() - start
                tr = measure(module, model, loaders, "train")
                te = measure(module, model, loaders, "test")
                od = measure(module, model, loaders, "ood")
                d1 = measure(module, model, loaders, "depth_t_1")
                score = (te["exact"] + od["exact"]) / 2
                if score > best[0]:
                    best = (score, step)
                print(f"{step:>6} {elapsed:>7.1f}s {tr['exact']:>9.3f} "
                      f"{te['exact']:>8.3f} {od['exact']:>7.3f} {score:>7.4f} "
                      f"{d1['exact']:>6.3f}", flush=True)
                next_report = step + 25
                model.train()
    print(f"\nBEST score {best[0]:.4f} at step {best[1]}")


if __name__ == "__main__":
    main()
