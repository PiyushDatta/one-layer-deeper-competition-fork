"""Can the model at least reach the no-input marginal?

A constant predictor scores 0.1533 on E1 (test 0.0867, ood 0.2200). The ood
0.2200 comes from T=6, where x^64 mod 323 has only 9 distinct values. That is a
T-CONDITIONAL fact, and Change 7 hides T from the block, so the block cannot
represent it however long it trains.

Capacity is the other half. With enough parameters the model memorises training
rows instead of learning the label distribution. Small enough and the marginal
is the best available fit.

Usage: python -m tmp.sweep_c [seconds]
"""

from __future__ import annotations

import sys

from tmp.sweep import run, show

HEADER = (
    f"{'config':<40} {'params':>9} {'steps':>6} {'train':>6} "
    f"{'te_ex':>6} {'od_ex':>6} {'score':>7} | {'D1_ex':>6}"
)


def show_score(label, out):
    score = (out["test"]["exact"] + out["ood"]["exact"]) / 2
    flag = "  <-- beats baseline" if score > 0.1533 else ""
    print(
        f"{label:<40} {out['elements']:>9,} {out['steps']:>6} {out['train']:>6.3f} "
        f"{out['test']['exact']:>6.3f} {out['ood']['exact']:>6.3f} {score:>7.4f} | "
        f"{out['depth_t_1']['exact']:>6.3f}{flag}",
        flush=True,
    )


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    cache: dict = {}
    print(f"budget {budget}s. baseline score to beat: 0.1533\n")
    print(HEADER)
    seen = {"HIDE_T_FROM_BLOCK": False}
    grid = [
        ("shipped (T hidden)", {}),
        ("T visible", seen),
        ("T visible d128 L2", {**seen, "D_MODEL": 128, "NUM_HEADS": 4,
                               "D_FF": 512, "TRAIN_LOOPS": 2, "SAM_RHO": 0.0}),
        ("T visible d64 L2", {**seen, "D_MODEL": 64, "NUM_HEADS": 4,
                              "D_FF": 256, "TRAIN_LOOPS": 2, "SAM_RHO": 0.0}),
        ("T visible d32 L2", {**seen, "D_MODEL": 32, "NUM_HEADS": 4,
                              "D_FF": 128, "TRAIN_LOOPS": 2, "SAM_RHO": 0.0}),
        ("T visible d16 L2", {**seen, "D_MODEL": 16, "NUM_HEADS": 4,
                              "D_FF": 64, "TRAIN_LOOPS": 2, "SAM_RHO": 0.0}),
        ("T visible d32 L1", {**seen, "D_MODEL": 32, "NUM_HEADS": 4,
                              "D_FF": 128, "TRAIN_LOOPS": 1, "SAM_RHO": 0.0}),
        ("T visible d32 L2 wd3", {**seen, "D_MODEL": 32, "NUM_HEADS": 4,
                                  "D_FF": 128, "TRAIN_LOOPS": 2, "SAM_RHO": 0.0,
                                  "MUON_WEIGHT_DECAY": 3.0}),
        ("T hidden d32 L2", {"D_MODEL": 32, "NUM_HEADS": 4, "D_FF": 128,
                             "TRAIN_LOOPS": 2, "SAM_RHO": 0.0}),
    ]
    for label, overrides in grid:
        try:
            show_score(label, run(overrides, budget, cache))
        except Exception as exc:
            print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
