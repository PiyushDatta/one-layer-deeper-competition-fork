"""With a hard tape, can B^2 memorise x -> x^2 on the 248 SEEN units?

If yes, composition is an optimisation problem and worth more effort. If no,
the step map is unlearnable even where memorisation is allowed, and every
downstream idea is blocked.
"""
from __future__ import annotations
import sys, time
from collections import defaultdict
import torch
from benchmark.api import ModelSpec, OptimizerSpec, TokenLossBatch
from benchmark.batches import prepare_batch
from data.config import DataConfig
from data.factory import make_dataloaders
from tmp.sweep import DEVICE, ROOT, fresh_module

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
hard = sys.argv[2] == "hard" if len(sys.argv) > 2 else True

m = fresh_module()
m.D_MODEL, m.NUM_HEADS, m.D_FF = 128, 4, 512
m.TRAIN_LOOPS, m.SAM_RHO = 8, 0.0
if hard:
    m.RETOKENIZE_GATE_INIT = 10.0
torch.manual_seed(74)
loaders = make_dataloaders(
    DataConfig(data_root=ROOT, batch_size=64, eval_batch_size=512, seed=45,
               num_workers=0), device=DEVICE)
model = m.build_model(ModelSpec(17, 13, 500_000_000)).to(DEVICE)
b = m.build_optimizer(model, OptimizerSpec(budget, "cuda"))
opt, sched = b.optimizer, b.scheduler
model.train(); start = time.monotonic(); steps = 0
while time.monotonic() - start < budget:
    for raw in loaders["train"]:
        if time.monotonic() - start >= budget: break
        ids, tg, mk, tp = prepare_batch(raw, DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, aux = model(ids, attention_mask=mk)
            r = torch.arange(lg.shape[0], device=DEVICE)[:, None]
            loss = m.token_training_loss(
                TokenLossBatch(lg[r, tp.clamp_min(0)].float(), tg, tg != -100, tp, aux))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None: sched.step()
        steps += 1
print(f"tape={'hard' if hard else 'off'} steps={steps} loss={float(loss.detach()):.4f}\n")

@torch.no_grad()
def table(split):
    model.train()
    hits = defaultdict(lambda: defaultdict(int)); tot = defaultdict(int)
    for raw in loaders[split]:
        ids, tg, mk, tp = prepare_batch(raw, DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(ids, attention_mask=mk)
        hyp = aux["hypothesis_logits"].float(); B, K, L, V = hyp.shape
        g = tp.clamp_min(0)[:, None, :, None].expand(B, K, tp.shape[1], V)
        ok = ((hyp.gather(2, g).argmax(-1) == tg[:, None, :]) | ~(tg != -100)[:, None, :]).all(-1)
        for r, t in enumerate((ids[:, -1] - 7).tolist()):
            tot[t] += 1
            for k in range(K):
                if ok[r, k]: hits[t][k] += 1
    print(f"{split}: per-exit exact by T")
    print("   T   n  " + "".join(f"  exit{k}" for k in range(8)))
    for t in sorted(tot):
        if t < 0: continue
        print(f"  {t:2d} {tot[t]:3d} " + "".join(f"  {hits[t][k]/tot[t]:5.2f}" for k in range(8)))
table("train"); table("test")
