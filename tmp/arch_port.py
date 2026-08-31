"""Port the probe architecture. Metric: 3-digit exact, which has NEVER moved off 0.

E1 exact accuracy only ever counts 1- and 2-digit answers, so it cannot
distinguish anything. 114 of 150 test rows have a 3-digit answer and no
configuration has ever solved one. That is the number to watch.
"""
from __future__ import annotations
import json, sys, time, torch, torch.nn.functional as F
from pathlib import Path
from benchmark.api import ModelSpec, OptimizerSpec, TokenLossBatch, BackwardPassContext
from benchmark.batches import prepare_batch
from data.config import DataConfig
from data.factory import make_dataloaders
from tmp.sweep import DEVICE, ROOT, fresh_module

def run(over, budget):
    m = fresh_module()
    for k, v in over.items(): setattr(m, k, v)
    torch.manual_seed(74)
    bs = getattr(m, "TRAIN_BATCH_SIZE", 64)
    ld = make_dataloaders(DataConfig(data_root=ROOT, batch_size=bs,
        eval_batch_size=512, seed=45, num_workers=0), device=DEVICE)
    model = m.build_model(ModelSpec(17, 13, 500_000_000)).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    b = m.build_optimizer(model, OptimizerSpec(budget, "cuda"))
    opt, sch, np_ = b.optimizer, b.scheduler, b.backward_passes_per_step
    model.train(); t0 = time.monotonic(); steps = 0
    while time.monotonic() - t0 < budget:
        for raw in ld["train"]:
            if time.monotonic() - t0 >= budget: break
            ids, tg, mk, tp = prepare_batch(raw, DEVICE)
            for pi in range(1, np_ + 1):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, aux = model(ids, attention_mask=mk)
                    r = torch.arange(lg.shape[0], device=DEVICE)[:, None]
                    loss = m.token_training_loss(TokenLossBatch(
                        lg[r, tp.clamp_min(0)].float(), tg, tg != -100, tp, aux))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if pi < np_ and b.between_backward_passes is not None:
                    with torch.no_grad():
                        b.between_backward_passes(BackwardPassContext(steps, pi, np_))
            opt.step()
            if sch is not None: sch.step()
            steps += 1
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ("test", "depth_t_1"):
            long_hit = long_n = hit = tot = 0; ce = ntok = 0.0
            for raw in ld[split]:
                ids, tg, mk, tp = prepare_batch(raw, DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = model(ids, attention_mask=mk)
                r = torch.arange(lg.shape[0], device=DEVICE)[:, None]
                al = lg[r, tp.clamp_min(0)].float(); at = al.argmax(-1)
                v = tg != -100
                ok = ((at == tg) | ~v).all(-1)
                hit += int(ok.sum()); tot += lg.shape[0]
                three = v.sum(-1) == 3
                long_hit += int((ok & three).sum()); long_n += int(three.sum())
                ce += float(F.cross_entropy(al[v], tg[v], reduction="sum")); ntok += int(v.sum())
            out[split] = (hit/tot, long_hit/max(long_n,1), long_n, ce/ntok)
    del model; torch.cuda.empty_cache()
    return n, steps, out

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    print(f"{'config':<34}{'params':>10}{'steps':>7}"
          f"{'te_ex':>7}{'te_3dig':>8}{'te_ce':>7}{'D1_3dig':>8}", flush=True)
    grid = [
        ("current NUM_BLOCKS=1", {}),
        ("blocks=2", {"NUM_BLOCKS": 2}),
        ("blocks=4", {"NUM_BLOCKS": 4}),
        ("blocks=4 d256", {"NUM_BLOCKS": 4, "D_MODEL": 256, "NUM_HEADS": 4, "D_FF": 1024}),
        ("blocks=4 d256 L4", {"NUM_BLOCKS": 4, "D_MODEL": 256, "NUM_HEADS": 4,
                              "D_FF": 1024, "TRAIN_LOOPS": 4}),
        ("blocks=8 d256 L8", {"NUM_BLOCKS": 8, "D_MODEL": 256, "NUM_HEADS": 4,
                              "D_FF": 1024}),
        ("blocks=4 d256 L4 noSAM", {"NUM_BLOCKS": 4, "D_MODEL": 256, "NUM_HEADS": 4,
                                    "D_FF": 1024, "TRAIN_LOOPS": 4, "SAM_RHO": 0.0}),
        ("d256 blocks=1", {"D_MODEL": 256, "NUM_HEADS": 4, "D_FF": 1024}),
    ]
    for label, over in grid:
        try:
            n, steps, o = run(over, budget)
            te, d1 = o["test"], o["depth_t_1"]
            mark = "  <-- 3-DIGIT!" if te[1] > 0 else ""
            print(f"{label:<34}{n:>10,}{steps:>7}{te[0]:>7.3f}{te[1]:>8.3f}"
                  f"{te[3]:>7.3f}{d1[1]:>8.3f}{mark}", flush=True)
        except Exception as exc:
            print(f"{label:<34} FAILED {type(exc).__name__}: {exc}", flush=True)
