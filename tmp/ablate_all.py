"""Is the accumulated machinery net-positive? Ten datasets, three configs."""
from __future__ import annotations
import json, subprocess
BASE = "submissions/piydatta_submission/submission.py"
MF = [f"h100_easy_e{i}" for i in range(1, 11)]

def build(over, tag):
    src = open(BASE).read()
    for k, v in over.items():
        old = [l for l in src.splitlines() if l.startswith(f"{k} = ")][0]
        src = src.replace(old, f"{k} = {v!r}", 1)
    p = f"/tmp/ab_{tag}.py"; open(p, "w").write(src); return p

def score(path, mf):
    out = subprocess.run(
        [".venv/bin/python", "-m", "benchmark.runner", "--manifest",
         f"benchmark/manifests/{mf}.json", "--submission-file", path],
        capture_output=True, text=True, timeout=900,
        env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"}).stdout
    l = [x for x in out.splitlines() if x.startswith("RESULT_JSON=")]
    return json.loads(l[0][12:])["score"]["mean_exact_accuracy"] if l else float("nan")

configs = {
    "shipped": {},
    "no_sam": {"SAM_RHO": 0.0},
    "minimal": {"USE_LATENT_HYPOTHESES": False, "USE_ACTION_HISTORY": False,
                "SAM_RHO": 0.0, "HIDE_T_FROM_BLOCK": False,
                "RETOKENIZE_GATE_INIT": -20.0},
}
paths = {k: build(v, k) for k, v in configs.items()}
print(f"{'dataset':<16}" + "".join(f"{k:>11}" for k in configs), flush=True)
tot = {k: [] for k in configs}
for mf in MF:
    row = {}
    for k, p in paths.items():
        try: row[k] = score(p, mf)
        except Exception: row[k] = float("nan")
        tot[k].append(row[k])
    print(f"{mf:<16}" + "".join(f"{row[k]:>11.4f}" for k in configs), flush=True)
print("MEAN            " + "".join(
    f"{sum(v for v in tot[k] if v == v)/max(1,len([v for v in tot[k] if v == v])):>11.4f}"
    for k in configs), flush=True)
