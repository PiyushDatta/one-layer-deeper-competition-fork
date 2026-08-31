"""Push regularisation hard. Target is the 0.1533 no-input baseline."""
from __future__ import annotations
import sys
from tmp.sweep import run
from tmp.sweep_c import HEADER, show_score

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    cache: dict = {}
    print(f"budget {budget}s. baseline score to beat: 0.1533\n")
    print(HEADER)
    tiny = {"HIDE_T_FROM_BLOCK": False, "NUM_HEADS": 4, "TRAIN_LOOPS": 2,
            "SAM_RHO": 0.0}
    d32 = {**tiny, "D_MODEL": 32, "D_FF": 128}
    d64 = {**tiny, "D_MODEL": 64, "D_FF": 256}
    grid = [
        ("d32 wd3 (best so far)", {**d32, "MUON_WEIGHT_DECAY": 3.0}),
        ("d32 wd3 adamw3", {**d32, "MUON_WEIGHT_DECAY": 3.0, "ADAMW_WEIGHT_DECAY": 3.0}),
        ("d32 wd10 adamw10", {**d32, "MUON_WEIGHT_DECAY": 10.0, "ADAMW_WEIGHT_DECAY": 10.0}),
        ("d32 wd30 adamw30", {**d32, "MUON_WEIGHT_DECAY": 30.0, "ADAMW_WEIGHT_DECAY": 30.0}),
        ("d64 wd3 adamw3", {**d64, "MUON_WEIGHT_DECAY": 3.0, "ADAMW_WEIGHT_DECAY": 3.0}),
        ("d64 wd10 adamw10", {**d64, "MUON_WEIGHT_DECAY": 10.0, "ADAMW_WEIGHT_DECAY": 10.0}),
        ("d128 wd10 adamw10", {**tiny, "D_MODEL": 128, "D_FF": 512,
                               "MUON_WEIGHT_DECAY": 10.0, "ADAMW_WEIGHT_DECAY": 10.0}),
        ("d32 wd10 adamw10 L4", {**d32, "TRAIN_LOOPS": 4,
                                 "MUON_WEIGHT_DECAY": 10.0, "ADAMW_WEIGHT_DECAY": 10.0}),
        ("d32 wd10 adamw10 bs512", {**d32, "TRAIN_BATCH_SIZE": 512,
                                    "MUON_WEIGHT_DECAY": 10.0, "ADAMW_WEIGHT_DECAY": 10.0}),
        ("d32 wd10 adamw10 lr3e-3", {**d32, "MUON_LR": 3e-3,
                                     "MUON_WEIGHT_DECAY": 10.0, "ADAMW_WEIGHT_DECAY": 10.0}),
    ]
    for label, overrides in grid:
        try:
            show_score(label, run(overrides, budget, cache))
        except Exception as exc:
            print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
