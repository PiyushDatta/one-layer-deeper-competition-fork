"""Composition on SEEN x is the only large gain left.

96% of test rows use an x the model trained on at a different T. If B were a
genuine step map, memorising its 248 entries is feasible even though computing
it is not, and test would approach 0.9. A hard tape forces B to be a
token-to-token map, which is the structure that makes B^k real iteration. The
question is whether it can be optimised at all.
"""
from __future__ import annotations
import sys
from tmp.sweep import run
from tmp.sweep_c import HEADER, show_score

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    cache: dict = {}
    print(f"budget {budget}s. length-blind constant ceiling 0.1067.\n")
    print(HEADER)
    base = {"NUM_HEADS": 4, "TRAIN_LOOPS": 2, "SAM_RHO": 0.0,
            "D_MODEL": 128, "D_FF": 512}
    grid = [
        ("soft tape open", {**base, "RETOKENIZE_GATE_INIT": 10.0,
                            "RETOKENIZE_STRAIGHT_THROUGH": False}),
        ("soft tape open temp1", {**base, "RETOKENIZE_GATE_INIT": 10.0,
                                  "RETOKENIZE_STRAIGHT_THROUGH": False,
                                  "RETOKENIZE_TEMPERATURE": 1.0}),
        ("hard tape", {**base, "RETOKENIZE_GATE_INIT": 10.0}),
        ("hard tape lr3e-3", {**base, "RETOKENIZE_GATE_INIT": 10.0, "MUON_LR": 3e-3}),
        ("soft tape d256", {**base, "D_MODEL": 256, "D_FF": 1024,
                            "RETOKENIZE_GATE_INIT": 10.0,
                            "RETOKENIZE_STRAIGHT_THROUGH": False}),
        ("no tape control", base),
    ]
    for label, overrides in grid:
        try:
            show_score(label, run(overrides, budget, cache))
        except Exception as exc:
            print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
