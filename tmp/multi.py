"""Score configs on THREE Easy datasets, so tuning cannot overfit to E1.

The 0.0833 config found by E1-only search lost on both E7 and E2, because it
collapses onto E1's answer distribution. Any candidate has to win on the mean.
"""
from __future__ import annotations
import json, subprocess, sys, itertools, random

MANIFESTS = ["h100_easy_e1", "h100_easy_e7", "h100_easy_e2"]
BASE = "submissions/piydatta_submission/submission.py"


def score(overrides, tag):
    src = open(BASE).read()
    for key, value in overrides.items():
        old = [l for l in src.splitlines() if l.startswith(f"{key} = ")]
        if not old:
            raise KeyError(key)
        src = src.replace(old[0], f"{key} = {value!r}", 1)
    path = f"/tmp/multi_{tag}.py"
    open(path, "w").write(src)
    total = []
    for mf in MANIFESTS:
        out = subprocess.run(
            [".venv/bin/python", "-m", "benchmark.runner", "--manifest",
             f"benchmark/manifests/{mf}.json", "--submission-file", path],
            capture_output=True, text=True, timeout=900,
            env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"},
        ).stdout
        line = [l for l in out.splitlines() if l.startswith("RESULT_JSON=")]
        total.append(json.loads(line[0][12:])["score"]["mean_exact_accuracy"] if line else 0.0)
    return total


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    random.seed(0)
    print(f"{'config':<58} {'E1':>7} {'E7':>7} {'E2':>7} {'mean':>7}", flush=True)
    print(f"{'SHIPPED':<58} " + " ".join(f"{v:>7.4f}" for v in score({}, "ship")), flush=True)
    space = {
        "D_MODEL": [64, 128, 256, 512],
        "TRAIN_LOOPS": [2, 4, 8],
        "MUON_WEIGHT_DECAY": [0.1, 1.0, 3.0],
        "ADAMW_WEIGHT_DECAY": [0.1, 1.0, 3.0],
        "MUON_LR": [1e-3, 3e-3],
        "SAM_RHO": [0.0, 0.05],
        "HIDE_T_FROM_BLOCK": [True, False],
    }
    for trial in range(trials):
        pick = {k: random.choice(v) for k, v in space.items()}
        pick["NUM_HEADS"] = 4
        pick["D_FF"] = 4 * pick["D_MODEL"]
        label = " ".join(f"{k.split('_')[0][:4]}{v}" for k, v in pick.items()
                         if k not in ("NUM_HEADS", "D_FF"))
        try:
            vals = score(pick, f"t{trial}")
            mean = sum(vals) / len(vals)
            print(f"{label:<58} " + " ".join(f"{v:>7.4f}" for v in vals)
                  + f" {mean:>7.4f}", flush=True)
        except Exception as exc:
            print(f"{label:<58} FAILED {type(exc).__name__}", flush=True)
