"""Sweep submission constants against the one metric that matters.

depth_t_1 is the 38 reserved units, disjoint from training by construction, so
its exact accuracy is a direct measurement of "can this compute x^2 mod 323 for
an x it has never seen". Everything else is downstream of that.

Usage: python -m tmp.sweep [seconds]
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time

import torch
import torch.nn.functional as F

from benchmark.api import ModelSpec, OptimizerSpec, TokenLossBatch, BackwardPassContext
from benchmark.batches import prepare_batch
from data.config import DataConfig
from data.factory import make_dataloaders

ROOT = "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123"
PATH = "submissions/piydatta_submission/submission.py"
DEVICE = torch.device("cuda")


def fresh_module():
    spec = importlib.util.spec_from_file_location("sub", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PRINT_LOGS = False
    module.DBUG = False
    return module


@torch.no_grad()
def measure(module, model, loaders, split):
    """Exact match is 1/38 granular on the reserved set, so also report token
    accuracy and cross entropy, which have far lower variance."""

    model.eval()
    hits = rows = token_hits = token_total = 0
    loss_sum = 0.0
    for batch in loaders[split]:
        ids, targets, mask, positions = prepare_batch(batch, DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(ids, attention_mask=mask)
        index = torch.arange(logits.shape[0], device=DEVICE)[:, None]
        at_logits = logits[index, positions.clamp_min(0)].float()
        at = at_logits.argmax(-1)
        valid = targets != -100
        hits += int(((at == targets) | ~valid).all(-1).sum())
        rows += logits.shape[0]
        token_hits += int(((at == targets) & valid).sum())
        token_total += int(valid.sum())
        loss_sum += float(
            F.cross_entropy(at_logits[valid], targets[valid], reduction="sum")
        )
    return {
        "exact": hits / rows,
        "token": token_hits / max(token_total, 1),
        "ce": loss_sum / max(token_total, 1),
    }


def run(overrides: dict, budget: float, loaders_cache: dict):
    module = fresh_module()
    for key, value in overrides.items():
        setattr(module, key, value)
    torch.manual_seed(74)

    batch = getattr(module, "TRAIN_BATCH_SIZE", 64)
    if batch not in loaders_cache:
        loaders_cache[batch] = make_dataloaders(
            DataConfig(
                data_root=ROOT,
                batch_size=batch,
                eval_batch_size=512,
                seed=45,
                num_workers=0,
            ),
            device=DEVICE,
        )
    loaders = loaders_cache[batch]

    model = module.build_model(ModelSpec(17, 13, 500_000_000)).to(DEVICE)
    elements = sum(p.numel() for p in model.parameters())
    bundle = module.build_optimizer(
        model, OptimizerSpec(training_time_seconds=budget, device_type="cuda")
    )
    opt, sched = bundle.optimizer, bundle.scheduler
    passes = bundle.backward_passes_per_step

    model.train()
    start, steps = time.monotonic(), 0
    train_hits = train_rows = 0
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
                            BackwardPassContext(steps, pass_index, passes)
                        )
            index = torch.arange(logits.shape[0], device=DEVICE)[:, None]
            at = logits[index, positions.clamp_min(0)].float().argmax(-1)
            valid = targets != -100
            train_hits += int(((at == targets) | ~valid).all(-1).sum())
            train_rows += logits.shape[0]
            opt.step()
            if sched is not None:
                sched.step()
            steps += 1

    out = {"steps": steps, "elements": elements, "loss": float(loss.detach())}
    out["train"] = train_hits / max(train_rows, 1)
    for split in ("test", "ood", "depth_t_1"):
        out[split] = measure(module, model, loaders, split)
    del model
    torch.cuda.empty_cache()
    return out


HEADER = (
    f"{'config':<40} {'params':>9} {'steps':>6} {'train':>6} "
    f"{'te_ex':>6} {'te_tok':>7} | {'D1_ex':>6} {'D1_tok':>7} {'D1_ce':>7}"
)


def show(label, out):
    d1, te = out["depth_t_1"], out["test"]
    print(
        f"{label:<40} {out['elements']:>9,} {out['steps']:>6} {out['train']:>6.3f} "
        f"{te['exact']:>6.3f} {te['token']:>7.3f} | "
        f"{d1['exact']:>6.3f} {d1['token']:>7.3f} {d1['ce']:>7.3f}",
        flush=True,
    )


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    cache: dict = {}
    print(f"budget {budget}s per config. d_t1 is the 38 reserved units.\n")
    print(HEADER)
    # Grokking recipe: small model, strong decay, many steps, on the exact task
    # family grokking was discovered on (Power et al. 2022, modular arithmetic).
    small = {"D_MODEL": 128, "NUM_HEADS": 4, "D_FF": 512, "SAM_RHO": 0.0}
    grid = [
        ("baseline as shipped", {}),
        ("d128 L4", {**small, "TRAIN_LOOPS": 4}),
        ("d128 L2", {**small, "TRAIN_LOOPS": 2}),
        # Grokking recipe: small model, strong decay, as many steps as possible.
        ("d128 L2 wd1.0", {**small, "TRAIN_LOOPS": 2, "MUON_WEIGHT_DECAY": 1.0}),
        ("d128 L2 wd3.0", {**small, "TRAIN_LOOPS": 2, "MUON_WEIGHT_DECAY": 3.0}),
        ("d128 L2 lr3e-3", {**small, "TRAIN_LOOPS": 2, "MUON_LR": 3e-3}),
        ("d128 L2 lr1e-2", {**small, "TRAIN_LOOPS": 2, "MUON_LR": 1e-2}),
        ("d128 L2 bs32", {**small, "TRAIN_LOOPS": 2, "TRAIN_BATCH_SIZE": 32}),
        ("d128 L2 bs256", {**small, "TRAIN_LOOPS": 2, "TRAIN_BATCH_SIZE": 256}),
        ("d128 L2 lr1e-2 wd1.0", {**small, "TRAIN_LOOPS": 2, "MUON_LR": 1e-2,
                                  "MUON_WEIGHT_DECAY": 1.0}),
        ("d64 L2 lr1e-2", {"D_MODEL": 64, "NUM_HEADS": 4, "D_FF": 256,
                           "SAM_RHO": 0.0, "TRAIN_LOOPS": 2, "MUON_LR": 1e-2}),
        ("d256 L2 lr3e-3", {"D_MODEL": 256, "NUM_HEADS": 4, "D_FF": 1024,
                            "SAM_RHO": 0.0, "TRAIN_LOOPS": 2, "MUON_LR": 3e-3}),
    ]
    for label, overrides in grid:
        try:
            show(label, run(overrides, budget, cache))
        except Exception as exc:  # keep the sweep going
            print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
