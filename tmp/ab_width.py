"""A/B the model width across all ten Easy datasets.

D_MODEL=1024 was chosen while arguing width-for-depth, and that argument did not
survive. A single tied block at width 256 had the best test CE of the session,
3.514 against 4.251, at 1/15th the parameters, but only on one dataset.
"""
from __future__ import annotations
import json, subprocess
BASE = "submissions/piydatta_submission/submission.py"
MF = [f"h100_easy_e{i}" for i in range(1, 11)]

def build(over, tag):
    src = open(BASE).read()
    for k, v in over.items():
        old = [l for l in src.splitlines() if l.startswith(f"{k} = ")][0]
        src = src.replace(old, f"{k} = {v!r}", 1)
    p = f"/tmp/abw_{tag}.py"; open(p, "w").write(src); return p

def score(path, mf):
    out = subprocess.run(
        [".venv/bin/python", "-m", "benchmark.runner", "--manifest",
         f"benchmark/manifests/{mf}.json", "--submission-file", path],
        capture_output=True, text=True, timeout=900,
        env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"}).stdout
    l = [x for x in out.splitlines() if x.startswith("RESULT_JSON=")]
    if not l:
        return float("nan"), float("nan")
    r = json.loads(l[0][12:])
    return r["score"]["mean_exact_accuracy"], r["score"]["mean_loss"]

paths = {
    "d1024": build({}, "d1024"),
    "d256": build({"D_MODEL": 256, "NUM_HEADS": 4, "D_FF": 1024}, "d256"),
    "d512": build({"D_MODEL": 512, "NUM_HEADS": 8, "D_FF": 2048}, "d512"),
}
print(f"{'dataset':<15}" + "".join(f"{k+' ex':>11}{k+' ce':>10}" for k in paths), flush=True)
tot = {k: ([], []) for k in paths}
for mf in MF:
    cells = ""
    for k, p in paths.items():
        try: ex, ce = score(p, mf)
        except Exception: ex = ce = float("nan")
        tot[k][0].append(ex); tot[k][1].append(ce)
        cells += f"{ex:>11.4f}{ce:>10.3f}"
    print(f"{mf:<15}{cells}", flush=True)
mean = lambda v: sum(x for x in v if x == x) / max(1, len([x for x in v if x == x]))
print(f"{'MEAN':<15}" + "".join(
    f"{mean(tot[k][0]):>11.4f}{mean(tot[k][1]):>10.3f}" for k in paths), flush=True)
