"""Is the failure data-limited or architecture-limited?

Two probes, both outside the submission, both purely diagnostic.

fraction  Vary how many of the 288 units are shown. E1 shows 86%. If
          generalisation never appears even at 97%, the model is not
          interpolating a function, it is memorising a table, and no amount of
          coverage inside this task will help.

pairs     Modular multiplication x*y mod 323 over all 288^2 = 82,944 pairs.
          Same arithmetic, same digit tokenisation, 300x the data. If this is
          learnable the architecture can do modular arithmetic and our problem
          is purely a data-count problem. If it is not, digit-tokenised modular
          arithmetic is out of reach regardless.

Usage: python -m tmp.data_limit fraction|pairs [seconds]
"""

from __future__ import annotations

import math
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

N = 323
DEVICE = torch.device("cuda")
DIGIT = 2
VOCAB = DIGIT + 10


def units():
    return [x for x in range(1, N) if math.gcd(x, N) == 1]


class Tiny(nn.Module):
    def __init__(self, width=256, heads=4, layers=2, positions=6):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(positions, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, VOCAB)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        return self.head(self.body(self.embed(ids) + self.position(pos)))


def digits_of(value, width):
    return [DIGIT + int(c) for c in f"{value:0{width}d}"]


def encode(pairs, in_width, out_width):
    ids = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=DEVICE)
    out = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=DEVICE)
    return ids, out


def train_eval(train_pairs, test_pairs, budget, in_width, out_width, batch=4096):
    torch.manual_seed(0)
    model = Tiny(positions=in_width).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                            betas=(0.9, 0.98))
    tr_x, tr_y = encode(train_pairs, in_width, out_width)
    te_x, te_y = encode(test_pairs, in_width, out_width)

    @torch.no_grad()
    def acc(x, y):
        model.eval()
        pred = model(x)[:, -out_width:].argmax(-1)
        model.train()
        return float((pred == y).all(-1).float().mean())

    start, step = time.monotonic(), 0
    while time.monotonic() - start < budget:
        if len(train_pairs) > batch:
            idx = torch.randint(0, tr_x.shape[0], (batch,), device=DEVICE)
            bx, by = tr_x[idx], tr_y[idx]
        else:
            bx, by = tr_x, tr_y
        logits = model(bx)[:, -out_width:]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
    return step, acc(tr_x, tr_y), acc(te_x, te_y)


def run_fraction(budget):
    us = units()
    print(f"{len(us)} units. varying how many are shown.\n")
    print(f"{'shown':>7} {'train n':>8} {'held n':>7} {'steps':>7} "
          f"{'train_ex':>9} {'held_ex':>8}")
    for frac in (0.50, 0.70, 0.86, 0.93, 0.97):
        rng = random.Random(0)
        shuffled = us[:]
        rng.shuffle(shuffled)
        cut = int(len(us) * frac)
        make = lambda xs: [
            (digits_of(x, 3), digits_of(pow(x, 2, N), 3)) for x in xs
        ]
        tr, te = make(shuffled[:cut]), make(shuffled[cut:])
        steps, a, b = train_eval(tr, te, budget, 3, 3)
        print(f"{frac:>7.0%} {len(tr):>8} {len(te):>7} {steps:>7} "
              f"{a:>9.3f} {b:>8.3f}", flush=True)


def run_pairs(budget):
    us = units()
    allp = [
        (digits_of(a, 3) + digits_of(b, 3), digits_of(a * b % N, 3))
        for a in us for b in us
    ]
    rng = random.Random(0)
    rng.shuffle(allp)
    cut = int(len(allp) * 0.9)
    tr, te = allp[:cut], allp[cut:]
    print(f"modular multiplication, {len(allp):,} pairs, "
          f"{len(tr):,} train / {len(te):,} held out\n")
    steps, a, b = train_eval(tr, te, budget, 6, 3)
    print(f"steps {steps}  train_exact {a:.3f}  held_exact {b:.3f}")
    print("\nIf held_exact is high, the architecture CAN do digit-tokenised")
    print("modular arithmetic and our problem is purely data count.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fraction"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    (run_fraction if mode == "fraction" else run_pairs)(budget)
