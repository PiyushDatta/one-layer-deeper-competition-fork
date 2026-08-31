"""Shipped vs best candidate across all ten Easy datasets, to beat single-run noise."""
from __future__ import annotations
import json, subprocess, sys
BASE = "submissions/piydatta_submission/submission.py"
MF = [f"h100_easy_e{i}" for i in range(1, 11)]

def build(overrides, tag):
    src = open(BASE).read()
    for k, v in overrides.items():
        old = [l for l in src.splitlines() if l.startswith(f"{k} = ")][0]
        src = src.replace(old, f"{k} = {v!r}", 1)
    p = f"/tmp/allten_{tag}.py"; open(p, "w").write(src); return p

def score(path, mf):
    out = subprocess.run(
        [".venv/bin/python", "-m", "benchmark.runner", "--manifest",
         f"benchmark/manifests/{mf}.json", "--submission-file", path],
        capture_output=True, text=True, timeout=900,
        env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"}).stdout
    line = [l for l in out.splitlines() if l.startswith("RESULT_JSON=")]
    return json.loads(line[0][12:])["score"]["mean_exact_accuracy"] if line else float("nan")

cand = {"D_MODEL": 64, "NUM_HEADS": 4, "D_FF": 256, "TRAIN_LOOPS": 8,
        "MUON_WEIGHT_DECAY": 1.0, "ADAMW_WEIGHT_DECAY": 3.0}
paths = {"shipped": build({}, "ship"), "candidate": build(cand, "cand")}
print(f"{'dataset':<16} {'shipped':>9} {'candidate':>10}", flush=True)
tot = {k: [] for k in paths}
for mf in MF:
    row = {}
    for name, p in paths.items():
        try:
            row[name] = score(p, mf)
        except Exception:
            row[name] = float("nan")
        tot[name].append(row[name])
    print(f"{mf:<16} {row['shipped']:>9.4f} {row['candidate']:>10.4f}", flush=True)
for name, vals in tot.items():
    good = [v for v in vals if v == v]
    print(f"MEAN {name:<11} {sum(good)/len(good):.4f}  over {len(good)} datasets", flush=True)
