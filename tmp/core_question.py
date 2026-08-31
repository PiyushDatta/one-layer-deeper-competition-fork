"""Is x^2 mod 323 on UNSEEN units learnable at all, and how many steps does it take?

Strips away every piece of submission machinery. A plain small transformer,
digits in, digits out, trained on the units that appear in E1 training and
tested on the 38 reserved ones. If this cannot grok inside a plausible step
budget, nothing built on top of it can certify Max T, and effort should move to
the test/ood score instead.

Usage: python -m tmp.core_question [seconds] [d_model] [lr] [weight_decay]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

N = 323
ROOT = Path("data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123")
DEVICE = torch.device("cuda")
PAD, BOS = 0, 1
DIGIT = 2  # digits occupy DIGIT .. DIGIT+9
VOCAB = DIGIT + 10


def splits():
    load = lambda name: [
        json.loads(line)
        for line in (ROOT / f"{name}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    seen = sorted({r["x"] for r in load("train")} | {r["x"] for r in load("test")})
    reserved = sorted({r["x"] for r in load("depth_t_1")})
    assert not (set(seen) & set(reserved))
    return seen, reserved


def encode(values):
    ids = torch.full((len(values), 3), PAD, dtype=torch.long)
    out = torch.full((len(values), 3), PAD, dtype=torch.long)
    for row, x in enumerate(values):
        for i, ch in enumerate(f"{x:03d}"):
            ids[row, i] = DIGIT + int(ch)
        for i, ch in enumerate(f"{pow(x, 2, N):03d}"):
            out[row, i] = DIGIT + int(ch)
    return ids.to(DEVICE), out.to(DEVICE)


class Tiny(nn.Module):
    def __init__(self, width, heads, layers):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(3, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, VOCAB)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        return self.head(self.body(self.embed(ids) + self.position(pos)))


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    decay = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    seen, reserved = splits()
    print(f"train units {len(seen)}  reserved units {len(reserved)}  "
          f"coverage {len(seen)/288:.1%}")
    print(f"width {width} lr {lr} weight_decay {decay} budget {budget}s\n")
    torch.manual_seed(0)

    train_x, train_y = encode(seen)
    test_x, test_y = encode(reserved)
    model = Tiny(width, 4, 2).to(DEVICE)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay, betas=(0.9, 0.98))

    @torch.no_grad()
    def evaluate(ids, tgt):
        model.eval()
        pred = model(ids).argmax(-1)
        model.train()
        return float((pred == tgt).all(-1).float().mean()), float(
            (pred == tgt).float().mean()
        )

    print(f"{'step':>7} {'elapsed':>8} {'loss':>8} {'tr_ex':>6} {'te_ex':>6} {'te_tok':>7}")
    start, step, next_report = time.monotonic(), 0, 1
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= budget:
            break
        logits = model(train_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), train_y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step >= next_report:
            tr, _ = evaluate(train_x, train_y)
            te, te_tok = evaluate(test_x, test_y)
            print(f"{step:>7} {elapsed:>7.1f}s {float(loss):>8.4f} "
                  f"{tr:>6.3f} {te:>6.3f} {te_tok:>7.3f}", flush=True)
            next_report = max(step + 1, int(step * 1.6))
    tr, _ = evaluate(train_x, train_y)
    te, te_tok = evaluate(test_x, test_y)
    print(f"\nFINAL steps={step} train_exact={tr:.3f} "
          f"reserved_exact={te:.3f} reserved_token={te_tok:.3f}")


if __name__ == "__main__":
    main()
