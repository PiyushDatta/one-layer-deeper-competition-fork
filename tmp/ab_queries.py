"""A/B the dedicated answer queries across all ten Easy datasets."""
from __future__ import annotations
import json, subprocess
BASE = "submissions/piydatta_submission/submission.py"
MF = [f"h100_easy_e{i}" for i in range(1, 11)]

def build(over, tag):
    src = open(BASE).read()
    for k, v in over.items():
        old = [l for l in src.splitlines() if l.startswith(f"{k} = ")][0]
        src = src.replace(old, f"{k} = {v!r}", 1)
    p = f"/tmp/abq_{tag}.py"; open(p, "w").write(src); return p

def score(path, mf):
    out = subprocess.run(
        [".venv/bin/python", "-m", "benchmark.runner", "--manifest",
         f"benchmark/manifests/{mf}.json", "--submission-file", path],
        capture_output=True, text=True, timeout=900,
        env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"}).stdout
    l = [x for x in out.splitlines() if x.startswith("RESULT_JSON=")]
    return json.loads(l[0][12:])["score"]["mean_exact_accuracy"] if l else float("nan")

paths = {"queries": build({}, "on"), "no_queries": build({"USE_ANSWER_QUERIES": False}, "off")}
print(f"{'dataset':<16}{'queries':>10}{'no_queries':>12}", flush=True)
tot = {k: [] for k in paths}
for mf in MF:
    row = {}
    for k, p in paths.items():
        try: row[k] = score(p, mf)
        except Exception: row[k] = float("nan")
        tot[k].append(row[k])
    print(f"{mf:<16}{row['queries']:>10.4f}{row['no_queries']:>12.4f}", flush=True)
print("MEAN            " + "".join(
    f"{sum(v for v in tot[k] if v==v)/max(1,len([v for v in tot[k] if v==v])):>10.4f}"
    if k=="queries" else
    f"{sum(v for v in tot[k] if v==v)/max(1,len([v for v in tot[k] if v==v])):>12.4f}"
    for k in paths), flush=True)
