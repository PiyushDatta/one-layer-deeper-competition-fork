"""Strip machinery. Fifteen changes have accumulated; test whether they help."""
from __future__ import annotations
import sys
from tmp.sweep import run
from tmp.sweep_c import HEADER, show_score

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    cache: dict = {}
    print(f"budget {budget}s. baseline 0.1533 (test .0867 ood .2200)\n")
    print(HEADER)
    base = {"HIDE_T_FROM_BLOCK": False, "NUM_HEADS": 4, "TRAIN_LOOPS": 2,
            "SAM_RHO": 0.0, "D_MODEL": 32, "D_FF": 128,
            "MUON_WEIGHT_DECAY": 10.0, "ADAMW_WEIGHT_DECAY": 10.0}
    grid = [
        ("d32 wd10 reference", base),
        ("no latent hypotheses", {**base, "USE_LATENT_HYPOTHESES": False}),
        ("no hypotheses, no history", {**base, "USE_LATENT_HYPOTHESES": False,
                                       "USE_ACTION_HISTORY": False}),
        ("no hypotheses L1", {**base, "USE_LATENT_HYPOTHESES": False, "TRAIN_LOOPS": 1}),
        ("no hypotheses d64", {**base, "USE_LATENT_HYPOTHESES": False,
                               "D_MODEL": 64, "D_FF": 256}),
        ("no hypotheses d128", {**base, "USE_LATENT_HYPOTHESES": False,
                                "D_MODEL": 128, "D_FF": 512}),
        ("no hypotheses wd30", {**base, "USE_LATENT_HYPOTHESES": False,
                                "MUON_WEIGHT_DECAY": 30.0, "ADAMW_WEIGHT_DECAY": 30.0}),
        ("no hypotheses wd3", {**base, "USE_LATENT_HYPOTHESES": False,
                               "MUON_WEIGHT_DECAY": 3.0, "ADAMW_WEIGHT_DECAY": 3.0}),
        ("no hypotheses d64 wd30", {**base, "USE_LATENT_HYPOTHESES": False,
                                    "D_MODEL": 64, "D_FF": 256,
                                    "MUON_WEIGHT_DECAY": 30.0,
                                    "ADAMW_WEIGHT_DECAY": 30.0}),
    ]
    for label, overrides in grid:
        try:
            show_score(label, run(overrides, budget, cache))
        except Exception as exc:
            print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
