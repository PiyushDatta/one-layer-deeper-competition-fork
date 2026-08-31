"""Find an architecture that does digit-tokenised modular arithmetic at all.

The vanilla transformer reaches held-out exact 0.085 on x*y mod 323 with 74,649
training pairs. That is the wall under everything else. This searches for an
architecture that clears it, using the data-rich task as the testbed so the
data-count question is removed. Anything that works here becomes a candidate to
port into the submission and re-test at 248 examples.

Usage: python -m tmp.arch_search [seconds_per_config]
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


def dataset():
    us = [x for x in range(1, N) if math.gcd(x, N) == 1]
    rows = [
        ([DIGIT + int(c) for c in f"{a:03d}{b:03d}"],
         [DIGIT + int(c) for c in f"{a * b % N:03d}"])
        for a in us for b in us
    ]
    random.Random(0).shuffle(rows)
    cut = int(len(rows) * 0.9)
    to = lambda part: (
        torch.tensor([r[0] for r in part], device=DEVICE),
        torch.tensor([r[1] for r in part], device=DEVICE),
    )
    return to(rows[:cut]), to(rows[cut:])


class Net(nn.Module):
    """Encoder over 6 input digits plus 3 learned query slots for the answer."""

    def __init__(self, width, heads, layers, ff_mult=4, place=False):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(9, width)
        self.query = nn.Parameter(torch.randn(3, width) * 0.02)
        self.place = place
        if place:
            # Significance features: which power of ten this digit carries, and
            # which operand it belongs to. An input representation, like a
            # positional encoding, not an algorithm.
            self.place_embed = nn.Embedding(3, width)
            self.operand = nn.Embedding(2, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, ff_mult * width, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, VOCAB)

    def forward(self, ids):
        batch = ids.shape[0]
        x = self.embed(ids)
        if self.place:
            idx = torch.arange(6, device=ids.device)
            x = x + self.place_embed(idx % 3) + self.operand(idx // 3)
        x = torch.cat([x, self.query.expand(batch, -1, -1)], dim=1)
        x = x + self.position(torch.arange(x.shape[1], device=ids.device))
        return self.head(self.body(x))[:, -3:]


def train(cfg, train_set, test_set, budget, batch=2048):
    torch.manual_seed(0)
    model = Net(**cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1,
                            betas=(0.9, 0.98))
    tr_x, tr_y = train_set
    te_x, te_y = test_set

    @torch.no_grad()
    def acc(x, y, cap=8192):
        model.eval()
        pred = model(x[:cap]).argmax(-1)
        model.train()
        return float((pred == y[:cap]).all(-1).float().mean())

    start, step = time.monotonic(), 0
    while time.monotonic() - start < budget:
        idx = torch.randint(0, tr_x.shape[0], (batch,), device=DEVICE)
        loss = F.cross_entropy(
            model(tr_x[idx]).reshape(-1, VOCAB), tr_y[idx].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
    return step, acc(tr_x, tr_y), acc(te_x, te_y), sum(
        p.numel() for p in model.parameters()
    )


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    train_set, test_set = dataset()
    print(f"x*y mod {N}: {train_set[0].shape[0]:,} train / "
          f"{test_set[0].shape[0]:,} held out. baseline to beat: 0.085\n")
    print(f"{'config':<40} {'params':>10} {'steps':>7} {'train':>7} {'held':>7}")
    grid = [
        ("w256 h4 L2", dict(width=256, heads=4, layers=2)),
        ("w256 h4 L4", dict(width=256, heads=4, layers=4)),
        ("w256 h4 L8", dict(width=256, heads=4, layers=8)),
        ("w512 h8 L4", dict(width=512, heads=8, layers=4)),
        ("w256 h4 L4 ff8", dict(width=256, heads=4, layers=4, ff_mult=8)),
        ("w256 h4 L4 +place", dict(width=256, heads=4, layers=4, place=True)),
        ("w512 h8 L8 +place", dict(width=512, heads=8, layers=8, place=True)),
        ("w128 h4 L6 +place", dict(width=128, heads=4, layers=6, place=True)),
    ]
    for label, cfg in grid:
        try:
            steps, tr, te, n = train(cfg, train_set, test_set, budget)
            mark = "  <-- clears baseline" if te > 0.15 else ""
            print(f"{label:<40} {n:>10,} {steps:>7} {tr:>7.3f} {te:>7.3f}{mark}",
                  flush=True)
        except Exception as exc:
            print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
