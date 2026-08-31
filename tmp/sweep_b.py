"""Does forcing a discrete tape make the exits actually compose?

Measured last session: exit 2 is 1.00 on train T=1 and 0.03 on test T=1, even
though 48 of 50 test rows use an x the model trained on at another T. So the
exits are independent lookups, not iterates of one operator.

MAIN's tape is discrete. Our retokenization is the soft version and its gate
CLOSES when training is free to choose (0.119 -> 0.082), because a continuous
workspace lets each exit carry loop-specific information. Forcing the gate open
makes the workspace a genuine token sequence every loop, so B is a token-to-token
map and B^k is real iteration. That is the structural way to get composition.

Usage: python -m tmp.sweep_b [seconds]
"""

from __future__ import annotations

import sys

from tmp.sweep import HEADER, run, show

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    cache: dict = {}
    small = {"D_MODEL": 128, "NUM_HEADS": 4, "D_FF": 512, "SAM_RHO": 0.0,
             "TRAIN_LOOPS": 2}
    print(f"budget {budget}s. te_ex is the composition target, D1 needs arithmetic.\n")
    print(HEADER)
    grid = [
        ("d128 L2 reference", small),
        # sigmoid(10) is 0.99995, so the tape is effectively hard.
        ("tape forced open", {**small, "RETOKENIZE_GATE_INIT": 10.0}),
        ("tape open, no straight-through", {**small, "RETOKENIZE_GATE_INIT": 10.0,
                                            "RETOKENIZE_STRAIGHT_THROUGH": False}),
        ("tape half open", {**small, "RETOKENIZE_GATE_INIT": 0.0}),
        ("tape open, no history", {**small, "RETOKENIZE_GATE_INIT": 10.0,
                                   "USE_ACTION_HISTORY": False}),
        ("tape open, L4", {**small, "RETOKENIZE_GATE_INIT": 10.0, "TRAIN_LOOPS": 4}),
        ("tape open, L8", {**small, "RETOKENIZE_GATE_INIT": 10.0, "TRAIN_LOOPS": 8}),
        ("tape open, temp 1.0", {**small, "RETOKENIZE_GATE_INIT": 10.0,
                                 "RETOKENIZE_TEMPERATURE": 1.0}),
        ("tape open, lr3e-3", {**small, "RETOKENIZE_GATE_INIT": 10.0,
                               "MUON_LR": 3e-3}),
        ("tape open, T visible", {**small, "RETOKENIZE_GATE_INIT": 10.0,
                                  "HIDE_T_FROM_BLOCK": False}),
    ]
    for label, overrides in grid:
        try:
            show(label, run(overrides, budget, cache))
        except Exception as exc:
            print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
