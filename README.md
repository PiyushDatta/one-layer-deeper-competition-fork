# One Layer Deeper
An architecture-and-optimizer competition from **Core Automation × Tilde Research**.

Build the best function-composition model under a fixed persistent-state ceiling and H100 training-time budget. Participants control architecture, depth, optimizer, learning-rate schedule, and training loss. The evaluator controls data, the outer loop, and final evaluation.

> **Submission deadline:** August 31 at 10:00 PM PT. The service will not accept submissions after this time.

For competition updates, join [discord.gg/gpumode](https://discord.gg/gpumode) and follow the `#one-layer-deeper` channel.

We are grateful to [Modal](https://modal.com/) for supporting the GPU evaluation infrastructure and to [Northflank](https://northflank.com/) for supporting the competition service and leaderboard. Thank you both for helping make this research competition possible.


## Install

#### CLI only for remote GPU use

```bash
uv tool install git+https://github.com/tilde-research/one-layer-deeper.git
one-layer --help
```

[See the full CLI instructions.](#cli)


#### CLI and local and remote GPU use

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git
cd one-layer-deeper
uv venv .venv
source .venv/bin/activate
uv sync
python -m unittest discover -s tests
```

[See the full CLI](#cli) and [local development](#local-development) instructions.


## Rules

1. Submit exactly one UTF-8 file named `submission.py`. It exports one `benchmark.Submission` with model and optimizer factories and an optional training loss.
2. The submission must be self-contained. It may import the public `benchmark` API and pinned evaluator dependencies, but it may not depend on repository `model` or `optim` modules, extra files, package installation, or external services.
3. Participant code defines the model, optimizer bundle, optional learning-rate scheduler, optional loss, training and evaluation batch sizes, and maximum training steps. Recurrence, adaptive computation, and depth curricula are allowed.
4. The evaluator fixes data, sampling, the one-forward/one-backward loop, gradient clipping, optimizer cadence, seeds, deadline, final evaluation, and aggregation. Participants may choose the training and evaluation batch size and a lower maximum step count; evaluator ceilings still apply.
5. The model may contain at most 500,000,000 scalar parameters and persistent buffers. Shared state counts once; frozen state still counts.
6. Optimizer state, activations, and temporary workspace may use remaining VRAM. OOM or timeout fails the run.
7. Easy provides 60 H100 training seconds, Medium 600 seconds, and Hard 3,600 seconds. Model construction, submission import, and compilation consume the budget.
8. A custom training loss receives final logits, labels, and the model's auxiliary output and returns one differentiable finite scalar. The evaluator performs backward.
9. Each final checkpoint is evaluated once with a separate time budget equal to half its training allowance. The evaluator uses fixed loss and exact accuracy, and the score is mean exact accuracy across fixed evaluation splits and seeds.
10. Data inspection, task-specific solvers, custom training loops, participant-controlled backward passes, and manifest overrides are not allowed.
11. The metric recorded for a Hard run must not be exploited. Any attempt to exploit it will result in an immediate ban.

Depth is deliberately unconstrained. Fixed stacks, tied recurrence, iterative refinement, routing, adaptive halting, memory tokens, and parameter-free work are all valid if the model-state ceiling is respected. A deeper forward completes fewer optimizer updates under the same clock.

### Submission contract

The file is limited to 256 KiB. `build_model(spec)` receives `vocab_size`, `max_seq_len`, and `maximum_model_state_elements`. It returns a `torch.nn.Module` whose `config` exposes the first two matching fields. The model accepts evaluator tensor arguments and returns `(logits, auxiliary_value)`.

The evaluator calls `model.train()` for optimization and `model.eval()` for final evaluation. If the model should behave differently during evaluation, use PyTorch's inherited `self.training` flag inside `forward` (for example, `if self.training: ... else: ...`).

`build_optimizer(model, spec)` receives the per-seed time allowance and device type. It returns an `OptimizerBundle`; its optimizer must include every trainable parameter exactly once. An optional scheduler is stepped after every update.

```python
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

def build_model(spec: ModelSpec):
    model = MyModel(spec)
    assert_model_state(model, spec)
    return model

def build_optimizer(model, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(MyOptimizer(model.parameters()))

SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,       # optional; training
    eval_batch_size=1024, # optional; evaluation
    max_steps=20_000,     # optional; cannot exceed the evaluator ceiling
)
```

If omitted, `batch_size` and `max_steps` use the evaluator manifest defaults.
Evaluation uses `eval_batch_size` when provided, then an explicit participant
`batch_size`, then the evaluator manifest's evaluation batch size, and finally
the manifest's training batch size. A participant `max_steps` can end training
early. The evaluator's wall-clock deadline and absolute step ceiling always remain
enforced. An optional scheduler returned in `OptimizerBundle` is stepped after
every completed optimizer update.

The website offers one basic, non-recurrent Transformer using `torch.optim.AdamW`. Its standalone `submission.py` lives under `submissions/baseline_adamw`.

### Compute tiers

The public Easy and Medium datasets provide separate prompt and output tensors.
The evaluator supplies a padding mask, not a causal mask, so models can attend
bidirectionally over the complete prompt in those practice tiers. Hard uses a
private hidden evaluator.

- **Easy:** datasets `e1`–`e5`, 60 training seconds, 60 accepted attempts per UTC day.
- **Medium:** datasets `m1`–`m5`, 600 training seconds, 6 accepted attempts per UTC day.
- **Hard:** dataset `h1`, 3,600 training seconds, 1 accepted attempt per UTC day.

Easy and Medium are practice tiers. The public leaderboard ranks only each participant's best successful Hard submission. Failed evaluations count after acceptance; authentication and validation rejections do not. Source and detailed results remain private.

## CLI

### Install the CLI

This installs the lightweight submission CLI only. To run evaluations locally, see [Local development](#local-development).

Install [uv](https://docs.astral.sh/uv/) and then install the command directly from GitHub:

```bash
uv tool install git+https://github.com/tilde-research/one-layer-deeper.git
one-layer --help
```

### Example workflow

```bash
one-layer login
one-layer validate submissions/baseline_adamw/submission.py
one-layer submit submissions/baseline_adamw/submission.py --tier easy --dataset e1 --wait
one-layer jobs
one-layer status <submission-id>
one-layer metrics <submission-id> --output metrics.jsonl
one-layer leaderboard
```

`one-layer login` opens GitHub authentication, receives a generated `old_…` API key through a temporary localhost callback, and saves it to `~/.config/one-layer/config.json` with user-only permissions. Signing in again rotates a lost key. The service stores the GitHub identity plus only the key's SHA-256 digest and short support prefix. By default, one evaluation may be queued or running per GitHub account.

`one-layer jobs` lists the signed-in participant's queued and running submissions,
including the submission IDs accepted by `one-layer status <submission-id>`. Use
`one-layer jobs --all` to include completed and failed submissions, or `--json`
for machine-readable output.

After a successful evaluation, `one-layer metrics <submission-id>` downloads a
bounded JSONL history containing evaluator-selected training, evaluation, and
summary metrics. Raw submission stdout, stderr, and exception text are not
included in participant-facing status responses or metric downloads, and are
deleted from the service database 24 hours after the run finishes.


## Local development

### Install locally

Clone the repository and install its dependencies:

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git
cd one-layer-deeper
uv venv .venv
source .venv/bin/activate
uv sync
python -m unittest discover -s tests
```

### Example of running a submission locally

Modal is not required for local evaluation. The runner takes an evaluator-owned
manifest and one standalone submission file. Start with the short CPU smoke test:

```bash
python -m benchmark.runner \
  --manifest benchmark/manifests/smoke_cpu.json \
  --submission-file submissions/baseline_adamw/submission.py
```

The smoke manifest creates its small dataset automatically. Before running a
public Easy or Medium manifest, generate the full datasets referenced by those
manifests:

```bash
bash scripts/generate_datasets.sh
```

The script writes the datasets under `data/generated/`. You only need to run it
again if those generated files are removed. For a tier-faithful run on a local
H100, first find an idle GPU and expose only that device. The manifest's
`cuda:0` will then refer to the selected physical GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmark.runner \
  --manifest benchmark/manifests/h100_easy_e1.json \
  --submission-file submissions/baseline_adamw/submission.py
```

Hard evaluation is available only through hosted submission. The final `RESULT_JSON=...` line contains aggregate and split metrics.

## License

Licensed under the [Apache License 2.0](LICENSE).
