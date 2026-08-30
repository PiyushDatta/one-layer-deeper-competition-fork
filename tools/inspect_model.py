"""Per-example, per-loop inspection of a trained submission model.

The evaluator discards the model after scoring, so this trains one locally,
optionally checkpoints it, and attaches forward hooks to read out what each
recurrent loop did to each token position.

Nothing here is part of the submission. Checkpointing is a local debug
convenience and torch.load is not allowed inside submission.py.

    python -m tools.inspect_model --d-model 256 --seconds 25 --save /tmp/m.pt
    python -m tools.inspect_model --load /tmp/m.pt --example 3

The default command trains or loads, then runs both reports. Use train, lens,
or trace to run one part on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch
from benchmark.batches import prepare_batch
from benchmark.manifest import load_manifest
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = ROOT / "submissions" / "piydatta_submission" / "submission.py"
DEFAULT_MANIFEST = ROOT / "benchmark" / "manifests" / "h100_easy_e1.json"
MARKERS = {0: "PAD", 1: "BOS", 2: "N", 3: "X", 4: "T", 5: "ANS", 6: "EOS"}


def token_name(value: int) -> str:
    return MARKERS.get(value, str(value - 7))


def decode(input_ids: torch.Tensor) -> str:
    return " ".join(token_name(int(v)) for v in input_ids)


def load_submission(path: Path, d_model: int | None):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DBUG = False
    if d_model is not None:
        heads = next(h for h in (8, 4, 2, 1) if d_model % h == 0)
        module.D_MODEL = d_model
        module.NUM_HEADS = heads
        module.D_FF = 4 * d_model
    return module


class Tracer:
    """Capture the state entering and leaving the shared block on every loop."""

    def __init__(self, model) -> None:
        self.block = model.processor.block
        self.calls: list[dict] = []
        self._handles = []

    def __enter__(self):
        self.calls.clear()

        def pre(_module, args):
            state, mask = args[0], (args[1] if len(args) > 1 else None)
            self.calls.append(
                {
                    "input": state.detach().float(),
                    "mask": None if mask is None else mask.detach(),
                }
            )

        def post(_module, _args, output):
            self.calls[-1]["output"] = output.detach().float()

        def qkv(_module, _args, output):
            self.calls[-1]["qkv"] = output.detach().float()

        self._handles = [
            self.block.register_forward_pre_hook(pre),
            self.block.register_forward_hook(post),
            self.block.qkv.register_forward_hook(qkv),
        ]
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def attention(self, call: int, heads: int, row: int) -> torch.Tensor:
        """Recompute attention probabilities that SDPA does not expose."""

        record = self.calls[call]
        q, k, _ = record["qkv"][row : row + 1].chunk(3, dim=-1)
        length = q.shape[1]
        q = q.view(1, length, heads, -1).transpose(1, 2)
        k = k.view(1, length, heads, -1).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        mask = record["mask"]
        if mask is not None:
            scores = scores.masked_fill(
                ~mask[row : row + 1, None, None, :].bool(), float("-inf")
            )
        return scores.softmax(dim=-1)[0].mean(dim=0)


def build(module, manifest, device):
    spec = ModelSpec(
        vocab_size=infer_vocab_size(manifest.data),
        max_seq_len=infer_max_seq_len(manifest.data),
        maximum_model_state_elements=manifest.model_state.maximum_elements,
    )
    model = module.build_model(spec).to(device=device, dtype=torch.float32)
    return model, spec


def train(module, model, manifest, loaders, device, seconds):
    bundle = module.build_optimizer(
        model, OptimizerSpec(training_time_seconds=seconds, device_type=device.type)
    )
    model.train()
    deadline = time.monotonic() + seconds
    loader = loaders["train"]
    iterator = iter(loader)
    step = 0
    while time.monotonic() < deadline:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        input_ids, targets, mask, positions = prepare_batch(batch, device)
        bundle.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, aux = model(input_ids, attention_mask=mask)
            rows = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[rows, positions.clamp_min(0)].float()
            loss = module.token_training_loss(
                TokenLossBatch(token_logits, targets, targets != -100, positions, aux)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), manifest.runtime.grad_clip)
        bundle.optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()
        step += 1
    print(f"trained {step} steps, final loss {loss.item():.4f}")
    return model


def first_batch(loaders, split, device):
    return prepare_batch(next(iter(loaders[split])), device)


@torch.no_grad()
def run_trace(module, model, loaders, device, split, index):
    input_ids, targets, mask, positions = first_batch(loaders, split, device)
    row = index
    prompt_len = int(mask[row].sum())
    answer_slots = positions[row][targets[row] != -100]
    model.eval()

    with Tracer(model) as tracer:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, _ = model(input_ids, attention_mask=mask)

    predicted = logits[row, answer_slots].argmax(dim=-1)
    expected = targets[row][targets[row] != -100]
    print(f"\nprompt   {decode(input_ids[row][:prompt_len])}")
    print(f"answer   at positions {answer_slots.tolist()}")
    print(f"expected {[token_name(int(v)) for v in expected]}")
    print(f"predicted {[token_name(int(v)) for v in predicted]}")

    print(f"\n{len(tracer.calls)} block calls, joint length {tracer.calls[0]['input'].shape[1]}")
    print("(positions 0..P-1 are frozen prompt memory, then control, workspace, scratchpad)")

    print("\n--- residual stream RMS at the answer slots, per loop ---")
    offset = prompt_len + 1
    for call, record in enumerate(tracer.calls, start=1):
        state = record["output"][row]
        rms = state.square().mean(dim=-1).sqrt()
        work = [f"{float(rms[offset + int(p)]):.3f}" for p in answer_slots]
        print(f"  loop {call}: workspace {work}  control {float(rms[prompt_len]):.3f}")

    print("\n--- logit lens at the answer slots, per loop ---")
    head, norm = model.head, model.final_norm
    for call, record in enumerate(tracer.calls, start=1):
        work_state = record["output"][row, prompt_len:]
        stage_logits = head(norm(work_state.to(head.weight.dtype)))
        for slot, want in zip(answer_slots, expected):
            probs = stage_logits[1 + int(slot)].float().softmax(dim=-1)
            top = probs.topk(3)
            guesses = ", ".join(
                f"{token_name(int(i))}={float(p):.2f}"
                for p, i in zip(top.values, top.indices)
            )
            print(
                f"  loop {call} slot {int(slot)} want {token_name(int(want))}: {guesses}"
            )

    print("\n--- attention from each answer slot, per loop (mass >= 0.05) ---")
    heads = module.NUM_HEADS
    names = [f"{i}:{token_name(int(v))}" for i, v in enumerate(input_ids[row][:prompt_len])]
    names += ["CTL"] + [f"w{i}" for i in range(prompt_len)]
    names += [f"s{i}" for i in range(module.NUM_SCRATCHPAD_TOKENS)]
    for call in range(len(tracer.calls)):
        probs = tracer.attention(call, heads, row)
        for slot in answer_slots:
            query = offset + int(slot)
            weights = probs[query]
            top = [
                f"{names[i]}={float(weights[i]):.2f}"
                for i in weights.argsort(descending=True)[:6]
                if float(weights[i]) >= 0.05
            ]
            print(f"  loop {call + 1} slot {int(slot)} -> {' '.join(top)}")


@torch.no_grad()
def run_lens(module, model, loaders, device, split, count):
    input_ids, targets, mask, positions = first_batch(loaders, split, device)
    model.eval()
    with torch.autocast(device.type, dtype=torch.bfloat16):
        logits, _ = model(input_ids, attention_mask=mask)
    rows = torch.arange(logits.shape[0], device=device)[:, None]
    token_logits = logits[rows, positions.clamp_min(0)].float()
    valid = targets != -100
    correct = ((token_logits.argmax(dim=-1) == targets) | ~valid).all(dim=1)

    print(f"\nsplit={split} exact accuracy {float(correct.float().mean()):.4f}")
    per_slot = ((token_logits.argmax(dim=-1) == targets) & valid).sum(dim=0)
    totals = valid.sum(dim=0)
    print("per answer-digit accuracy:")
    for i, (hit, total) in enumerate(zip(per_slot, totals)):
        if int(total):
            print(f"  digit {i}: {int(hit)}/{int(total)} = {float(hit) / float(total):.3f}")

    print(f"\nfirst {count} examples:")
    for row in range(min(count, input_ids.shape[0])):
        length = int(mask[row].sum())
        want = [token_name(int(v)) for v in targets[row][valid[row]]]
        got = [
            token_name(int(v))
            for v in token_logits[row].argmax(dim=-1)[valid[row]]
        ]
        flag = "ok " if correct[row] else "   "
        print(f"  {flag}[{row:3}] {decode(input_ids[row][:length])} -> {got} want {want}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "train", "trace", "lens"),
    )
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--d-model", type=int, help="shrink the model for fast iteration")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = load_manifest(args.manifest)
    module = load_submission(args.submission, args.d_model)
    loaders = make_dataloaders(manifest.data, device=device)
    model, _ = build(module, manifest, device)

    if args.load:
        payload = torch.load(args.load, map_location=device, weights_only=True)
        model.load_state_dict(payload)
        print(f"loaded {args.load}")
    else:
        train(module, model, manifest, loaders, device, args.seconds)
        if args.save:
            torch.save(model.state_dict(), args.save)
            print(f"saved {args.save}")

    if args.command in ("all", "lens"):
        run_lens(module, model, loaders, device, args.split, args.count)
    if args.command in ("all", "trace"):
        run_trace(module, model, loaders, device, args.split, args.example)


if __name__ == "__main__":
    main()
