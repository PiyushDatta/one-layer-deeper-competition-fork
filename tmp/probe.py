"""Standalone probe: train the submission briefly, then ask whether the exits
carry any information at all.

Usage: python -m tmp.probe <submission_file> [seconds]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from benchmark.api import ModelSpec, OptimizerSpec
from benchmark.batches import prepare_batch
from data.config import DataConfig
from data.factory import make_dataloaders

ROOT = "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123"


def load(path: str):
    spec = importlib.util.spec_from_file_location("sub", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    path = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    sub = load(path)
    device = torch.device("cuda:0")
    torch.manual_seed(74)

    cfg = DataConfig(data_root=ROOT, batch_size=512, eval_batch_size=512, seed=45)
    loaders = make_dataloaders(cfg, device=device)

    model = sub.SUBMISSION.build_model(ModelSpec(17, 13, 500_000_000)).to(device)
    bundle = sub.SUBMISSION.build_optimizer(
        model, OptimizerSpec(training_time_seconds=budget, device_type="cuda")
    )
    opt, sched = bundle.optimizer, bundle.scheduler

    # ---- train, mirroring the evaluator's outer loop ----
    model.train()
    start, steps = time.monotonic(), 0
    while time.monotonic() - start < budget:
        for batch in loaders["train"]:
            if time.monotonic() - start >= budget:
                break
            ids, targets, mask, tpos = prepare_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(ids, attention_mask=mask)
                valid = targets != -100
                from benchmark.api import TokenLossBatch

                loss = sub.SUBMISSION.token_training_loss(
                    TokenLossBatch(logits, targets, valid, tpos, aux)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            steps += 1
    print(f"trained steps={steps} final_loss={loss.item():.6f}")

    # ---- probe ----
    # Stay in train() so forward returns the auxiliary dict (eval returns a
    # bare ponder scalar). This also probes the 8 training exits, not all 64.
    model.train()
    per_exit_hits = defaultdict(int)
    prior_by_t = defaultdict(Counter)
    rows = 0
    same_pred_across_t = Counter()
    pred_by_xt: dict[tuple[int, int], tuple] = {}
    exit_spread = []

    with torch.no_grad():
        for split in ("train", "test"):
            for batch in loaders[split]:
                ids, targets, mask, tpos = prepare_batch(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, aux = model(ids, attention_mask=mask)
                hyp = aux["hypothesis_logits"].float()      # [B, K, L, V]
                prior = aux["hypothesis_log_prior"].float()  # [B, K]
                B, K, L, V = hyp.shape
                valid = targets != -100

                # exit spread: how different are exits from each other?
                flat = hyp.reshape(B, K, -1)
                spread = (flat[:, :, None, :] - flat[:, None, :, :]).abs().mean()
                exit_spread.append(spread.item())

                # per-exit exact accuracy at target positions
                gather = tpos.clamp_min(0)[:, None, :, None].expand(B, K, tpos.shape[1], V)
                at_targets = hyp.gather(2, gather).argmax(-1)  # [B, K, Ltgt]
                correct = ((at_targets == targets[:, None, :]) | ~valid[:, None, :]).all(-1)
                for k in range(K):
                    per_exit_hits[k] += correct[:, k].sum().item()
                rows += B

                # prior choice by T
                chosen = prior.argmax(-1).tolist()
                t_tok = ids[:, -1].tolist()  # last prompt token is the T digit
                for c, t in zip(chosen, t_tok):
                    prior_by_t[t - 7].update([c])

                # does the emitted answer depend on T for the same x?
                sel = prior.argmax(-1)
                selected = at_targets[torch.arange(B, device=device), sel]
                id_rows = ids.tolist()
                for r in range(B):
                    key = (str(id_rows[r][4:-2]), t_tok[r] - 7)
                    pred_by_xt[key] = tuple(selected[r].tolist())

    print(f"\nrows={rows}  mean pairwise exit logit distance={sum(exit_spread)/len(exit_spread):.6f}")
    print("per-exit exact accuracy:")
    for k in sorted(per_exit_hits):
        print(f"  exit {k:2d}: {per_exit_hits[k]/rows:.4f}")

    print("\nprior argmax histogram by T digit:")
    for t in sorted(prior_by_t):
        print(f"  T={t}: {dict(sorted(prior_by_t[t].items()))}")

    by_x = defaultdict(dict)
    for (x, t), pred in pred_by_xt.items():
        by_x[x][t] = pred
    shared = [v for v in by_x.values() if len(v) >= 2]
    identical = sum(1 for v in shared if len(set(v.values())) == 1)
    print(
        f"\nx values seen at >=2 different T: {len(shared)}; "
        f"of those, emitting an IDENTICAL answer at every T: {identical} "
        f"({identical/max(len(shared),1):.1%})"
    )


if __name__ == "__main__":
    main()
