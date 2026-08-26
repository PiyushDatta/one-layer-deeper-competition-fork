"""Evaluator-owned runner for the One Layer Deeper competition."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data import (
    infer_max_seq_len,
    infer_vocab_size,
    make_dataloaders,
)
from .act_diagnostics import ActDiagnosticsAccumulator
from .api import (
    BackwardPassContext,
    BatchReuseContext,
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
)
from .batches import prepare_batch
from .manifest import BenchmarkManifest, load_manifest
from .metrics import MetricRecorder
from .validation import (
    assert_state_versions_unchanged,
    capture_state_versions,
    lint_submission_source,
    validate_model_state,
    validate_optimizer,
    validate_submission,
)


EVALUATION_TIME_FRACTION = 0.5
MAX_BACKWARD_PASSES_PER_STEP = 8
MAX_OPTIMIZER_STEPS_PER_BATCH = 8
SCORING_SPLIT_PRIORITY = ("test", "ood", "ood_t", "ood_n_t")
NON_SCORING_SPLITS = frozenset(("train", "eval"))
DEPTH_SPLIT_PREFIX = "depth_t_"
OOD_N_DEPTH_SPLIT_PREFIX = "depth_ood_n_t_"


def _scoring_split_names(dataloaders) -> tuple[str, ...]:
    """Return deterministic scored splits for final measurement."""

    available = {
        name
        for name in set(dataloaders) - NON_SCORING_SPLITS
        if not name.startswith(
            (DEPTH_SPLIT_PREFIX, OOD_N_DEPTH_SPLIT_PREFIX)
        )
    }
    prioritized = [name for name in SCORING_SPLIT_PRIORITY if name in available]
    remaining = sorted(available - set(prioritized))
    return tuple((*prioritized, *remaining))


def _depth_split_names(
    dataloaders, prefix: str = DEPTH_SPLIT_PREFIX
) -> tuple[str, ...]:
    names = [
        name for name in dataloaders if name.startswith(prefix)
    ]
    return tuple(
        sorted(names, key=lambda name: int(name.removeprefix(prefix)))
    )


def _format_profile_progress(
    result: dict,
    *,
    label: str,
    ladder_key: str,
    max_t_key: str,
    rungs_key: str,
) -> str:
    profile = result["depth_profile"]
    ladder = [int(time_steps) for time_steps in profile.get(ladder_key, ())]
    if not ladder:
        return f"{label}: N/A"

    max_t = profile.get(max_t_key)
    certified_t = int(max_t) if max_t is not None else 0
    max_t_text = str(certified_t) if certified_t else "None"
    next_t = next(
        (time_steps for time_steps in ladder if time_steps > certified_t),
        None,
    )
    if next_t is None:
        return f"{label}: MaxT={max_t_text}, Certified"

    measurements = []
    seed_results = result.get("seeds", ())
    for seed_result in seed_results:
        seed_profile = seed_result.get("depth_profile") or {}
        rung = next(
            (
                item
                for item in seed_profile.get(rungs_key, ())
                if int(item["time_steps"]) == next_t
            ),
            None,
        )
        if rung is None:
            return (
                f"{label}: MaxT={max_t_text}, Next=T={next_t}, "
                "Acc=unavailable"
            )
        correct = rung.get("correct_examples")
        total = rung.get("example_count")
        if correct is not None and total:
            accuracy = int(correct) / int(total)
        else:
            exact_accuracy = rung.get("exact_accuracy")
            if exact_accuracy is None:
                return (
                    f"{label}: MaxT={max_t_text}, Next=T={next_t}, "
                    "Acc=unavailable"
                )
            accuracy = float(exact_accuracy)
        measurements.append((accuracy, correct, total))

    if not measurements:
        return (
            f"{label}: MaxT={max_t_text}, Next=T={next_t}, "
            "Acc=unavailable"
        )

    accuracy, correct, total = min(measurements, key=lambda item: item[0])
    count_text = (
        f" ({int(correct)}/{int(total)})"
        if correct is not None and total
        else ""
    )
    return (
        f"{label}: MaxT={max_t_text}, Next=T={next_t}, "
        f"Acc={100.0 * accuracy:.4f}%{count_text}"
    )


def _format_competition_progress(result: dict) -> str | None:
    profile = result.get("depth_profile")
    if not profile or not profile.get("ladder"):
        return None
    id_progress = _format_profile_progress(
        result,
        label="ID",
        ladder_key="ladder",
        max_t_key="max_certified_time_steps",
        rungs_key="rungs",
    )
    ood_progress = _format_profile_progress(
        result,
        label="OOD N",
        ladder_key="ood_n_ladder",
        max_t_key="ood_n_max_certified_time_steps",
        rungs_key="ood_n_rungs",
    )
    return f"COMPETITION_PROGRESS | {id_progress} | {ood_progress}"


def _format_act_diagnostics(result: dict) -> str | None:
    """Format local ACT telemetry without changing official score fields."""

    lines = []
    for seed_result in result.get("seeds", ()):
        seed = seed_result["seed"]
        payload = seed_result.get("act_diagnostics") or {}
        metric_groups = list(payload.get("scoring_splits", {}).items())
        metric_groups.extend(
            (f"{label}/T={rung['time_steps']}", rung["diagnostics"])
            for label, rung in payload.get(
                "first_uncertified_depth_rungs", {}
            ).items()
        )
        for split, diagnostics in metric_groups:
            updates = diagnostics["token_update_counts"]
            global_loops = diagnostics["global_iterations_per_batch"]
            caps = diagnostics["cap_hits"]
            remainders = diagnostics["remainders"]
            forced_remainder = remainders["mean_cap_forced"]
            forced_remainder_text = (
                "N/A" if forced_remainder is None else f"{forced_remainder:.3f}"
            )
            lines.append(
                f"ACT_SUMMARY | seed={seed} split={split} | "
                f"eval_task_CE="
                f"{diagnostics['evaluation_task_cross_entropy']:.6f} | "
                f"ponder={diagnostics['raw_mean_ponder_time']:.3f} | "
                f"weighted_ponder={diagnostics['weighted_ponder_contribution']:.6f} | "
                f"updates mean/median/p90/p95/p99/max="
                f"{updates['mean']:.3f}/{updates['median']:.3f}/"
                f"{updates['p90']:.0f}/{updates['p95']:.0f}/"
                f"{updates['p99']:.0f}/{updates['maximum']:.0f} | "
                f"global_loops mean/max="
                f"{global_loops['mean']:.3f}/{global_loops['maximum']:.0f} | "
                f"cap_reached token/batch="
                f"{100.0 * caps['token_reached_cap_rate']:.2f}%/"
                f"{100.0 * caps['batch_reached_cap_rate']:.2f}% | "
                f"cap_forced token/batch="
                f"{100.0 * caps['token_forced_cap_rate']:.2f}%/"
                f"{100.0 * caps['batch_forced_cap_rate']:.2f}% | "
                f"remainder mean/forced="
                f"{remainders['mean']:.3f}/{forced_remainder_text}"
            )
            endings = " ".join(
                f"{item['iteration']}:"
                f"{item['ended_at_percentage']:.1f}%/"
                f"{item['ended_by_percentage']:.1f}%"
                for item in diagnostics["iteration_end_percentages"]
            )
            if diagnostics["iteration_detail_truncated"]:
                endings += (
                    " ... "
                    f"{diagnostics['iteration_detail_unreported_token_percentage']:.1f}% "
                    "of tokens ended after the detail limit"
                )
            lines.append(
                f"ACT_PROCESSING_ENDED_AT/BY | seed={seed} split={split} | "
                f"{endings}"
            )
            correctness = diagnostics["by_correctness"]
            lines.append(
                f"ACT_BY_CORRECTNESS | seed={seed} split={split} | "
                + " | ".join(
                    f"{label}: n={group['example_count']}, "
                    f"updates={group['mean_updates_per_example']:.3f}"
                    if group["mean_updates_per_example"] is not None
                    else f"{label}: n=0, updates=N/A"
                    for label, group in correctness.items()
                )
            )
            by_length = diagnostics["by_sequence_length"]
            lines.append(
                f"ACT_BY_INPUT_LENGTH | seed={seed} split={split} | "
                + " | ".join(
                    f"L={length}: n={group['example_count']}, "
                    f"acc={100.0 * group['exact_accuracy']:.2f}%, "
                    f"updates={group['mean_updates_per_example']:.3f}"
                    for length, group in by_length.items()
                )
            )
    return "\n".join(lines) if lines else None


def _deny_dataset_file_access(data_root: str | Path) -> None:
    """Prevent uploaded code from reopening evaluator-owned dataset files."""

    protected_root = Path(data_root).resolve()

    def audit(event: str, args: tuple) -> None:
        if event != "open" or not args:
            return
        candidate = args[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(candidate)).resolve()
        if path == protected_root or protected_root in path.parents:
            raise PermissionError("submission may not access evaluator-owned dataset files")

    sys.addaudithook(audit)


def _configure_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _with_batch_size(
    dataloaders,
    manifest: BenchmarkManifest,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
):
    """Rebatch already-loaded datasets without reopening evaluator data files."""

    resized = {}
    for split_name, original in dataloaders.items():
        is_train = split_name == "train"
        generator = (
            torch.Generator(device="cpu").manual_seed(seed) if is_train else None
        )
        loader = DataLoader(
            original.dataset,
            batch_size=batch_size if is_train else eval_batch_size,
            shuffle=(
                manifest.data.shuffle_train
                if is_train
                else manifest.data.shuffle_eval
            ),
            collate_fn=original.collate_fn,
            num_workers=manifest.data.num_workers,
            pin_memory=original.pin_memory,
            drop_last=manifest.data.drop_last if is_train else False,
            generator=generator,
        )
        if is_train and len(loader) == 0:
            raise ValueError(
                f"submission batch_size={batch_size} produces no complete training batches"
            )
        resized[split_name] = loader
    return resized


def _resolve_batch_sizes(
    submission: Submission,
    manifest: BenchmarkManifest,
) -> tuple[int, int]:
    batch_size = submission.batch_size or manifest.data.batch_size
    eval_batch_size = (
        submission.eval_batch_size
        or submission.batch_size
        or manifest.data.eval_batch_size
        or manifest.data.batch_size
    )
    return batch_size, eval_batch_size


def _resolve_device(manifest: BenchmarkManifest) -> torch.device:
    device = torch.device(manifest.runtime.device)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("manifest requires CUDA, but CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "official execution requires exactly one visible CUDA device; "
            f"found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(device)
    return device


def _make_model_spec(manifest: BenchmarkManifest) -> ModelSpec:
    return ModelSpec(
        vocab_size=infer_vocab_size(manifest.data),
        max_seq_len=infer_max_seq_len(manifest.data),
        maximum_model_state_elements=manifest.model_state.maximum_elements,
    )


def _validate_model_interface(model: nn.Module, spec: ModelSpec) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model must expose a config object")
    expected = {
        "vocab_size": spec.vocab_size,
        "max_seq_len": spec.max_seq_len,
    }
    for field, value in expected.items():
        if getattr(config, field, None) != value:
            raise ValueError(f"model config {field} must equal {value}")


def _autocast(manifest: BenchmarkManifest, device: torch.device):
    if not manifest.runtime.amp:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, manifest.runtime.dtype),
    )


def _compile_model(model: nn.Module, manifest: BenchmarkManifest) -> nn.Module:
    return torch.compile(model, dynamic=True) if manifest.runtime.compile else model


def _next_batch(iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def _loss_and_accuracy(
    model: nn.Module,
    batch,
    manifest: BenchmarkManifest,
    device: torch.device,
    *,
    training_loss=None,
    token_training_loss=None,
    act_diagnostics: ActDiagnosticsAccumulator | None = None,
) -> tuple[torch.Tensor, float, int, int]:
    input_ids, targets, attention_mask, target_positions = prepare_batch(
        batch,
        device,
    )

    with _autocast(manifest, device):
        logits, auxiliary = model(
            input_ids,
            attention_mask=attention_mask,
        )
        if (
            logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
            or logits.shape[-1] != model.config.vocab_size
        ):
            raise ValueError(
                "language-model logits must have shape "
                "(batch, sequence, vocab_size)"
            )
        if target_positions is None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "causal language-model targets must match the input shape"
                )
            token_logits = logits[:, :-1, :].float()
            token_targets = targets[:, 1:]
        else:
            if target_positions.shape != targets.shape:
                raise ValueError(
                    "target_positions must have the same shape as targets"
                )
            valid_positions = target_positions[targets != -100]
            if (
                (valid_positions < 0).any().item()
                or (valid_positions >= input_ids.shape[1]).any().item()
            ):
                raise ValueError("target position is outside the input sequence")
            batch_indices = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[
                batch_indices,
                target_positions.clamp_min(0),
            ].float()
            token_targets = targets

        valid = token_targets != -100
        if not valid.any().item():
            raise ValueError("batch contains no valid language-model targets")
        if token_training_loss is not None:
            loss = token_training_loss(
                TokenLossBatch(
                    logits=token_logits,
                    labels=token_targets,
                    valid_mask=valid,
                    target_positions=target_positions,
                    auxiliary=auxiliary,
                )
            )
        else:
            loss_logits = token_logits[valid]
            loss_labels = token_targets[valid]
            if training_loss is None:
                loss = F.cross_entropy(loss_logits, loss_labels)
            else:
                loss = training_loss(loss_logits, loss_labels, auxiliary)

        token_predictions = token_logits.argmax(dim=-1)
        rows_with_targets = valid.any(dim=1)
        exact_rows = ((token_predictions == token_targets) | ~valid).all(dim=1)
        if act_diagnostics is not None:
            act_diagnostics.add(
                auxiliary=auxiliary,
                attention_mask=attention_mask,
                exact_rows=exact_rows,
                rows_with_targets=rows_with_targets,
            )
        exact_rows = exact_rows[rows_with_targets]
        example_count = int(rows_with_targets.sum().item())
        loss_weight = int(valid.sum().item())

        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise TypeError("training_loss must return one scalar tensor")
        if loss.device != device:
            raise ValueError(f"training_loss must return a tensor on {device}")
        if (
            training_loss is not None or token_training_loss is not None
        ) and not loss.requires_grad:
            raise ValueError("training_loss result must be differentiable")

    exact_accuracy = exact_rows.float().mean().item()
    return loss, exact_accuracy, example_count, loss_weight


def _train(
    *,
    raw_model: nn.Module,
    train_model: nn.Module,
    training_loss,
    bundle: OptimizerBundle,
    dataloader,
    manifest: BenchmarkManifest,
    device: torch.device,
    started_at: float,
    deadline: float,
    budget_seconds: float,
    max_steps: int,
    seed: int,
    token_training_loss=None,
    metric_recorder: MetricRecorder | None = None,
) -> tuple[float | None, int, float, int]:
    optimizer = bundle.optimizer
    raw_model.train()
    validate_optimizer(bundle, raw_model, device)
    iterator = iter(dataloader)
    final_loss = None
    final_accuracy = None
    completed_steps = 0
    last_metric_step = 0
    optimizer_state_elements = 0
    batch = None
    reuse_batch = False
    current_batch_uses = 0
    if bundle.backward_passes_per_step > MAX_BACKWARD_PASSES_PER_STEP:
        raise ValueError(
            "OptimizerBundle.backward_passes_per_step exceeds the evaluator "
            f"maximum of {MAX_BACKWARD_PASSES_PER_STEP}"
        )

    for step in range(1, max_steps + 1):
        if time.monotonic() >= deadline:
            break
        validate_model_state(raw_model, manifest.model_state, device)
        if not reuse_batch:
            batch, iterator = _next_batch(iterator, dataloader)
            current_batch_uses = 0
        current_batch_uses += 1

        for pass_index in range(1, bundle.backward_passes_per_step + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy, _, _ = _loss_and_accuracy(
                train_model,
                batch,
                manifest,
                device,
                training_loss=training_loss,
                token_training_loss=token_training_loss,
            )
            if not torch.isfinite(loss).all().item():
                raise FloatingPointError(
                    f"non-finite training loss at step {step}, pass {pass_index}"
                )
            loss.backward()
            if manifest.runtime.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), manifest.runtime.grad_clip
                )
            if (
                pass_index < bundle.backward_passes_per_step
                and bundle.between_backward_passes is not None
            ):
                with torch.no_grad():
                    bundle.between_backward_passes(
                        BackwardPassContext(
                            completed_steps=completed_steps,
                            pass_index=pass_index,
                            total_passes=bundle.backward_passes_per_step,
                        )
                    )
        optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()

        final_loss = float(loss.item())
        final_accuracy = accuracy
        completed_steps = step
        reuse_batch = False
        if (
            bundle.should_reuse_batch is not None
            and current_batch_uses < MAX_OPTIMIZER_STEPS_PER_BATCH
        ):
            with torch.no_grad():
                reuse_decision = bundle.should_reuse_batch(
                    BatchReuseContext(
                        completed_steps=completed_steps,
                        current_batch_uses=current_batch_uses,
                        loss=final_loss,
                    )
                )
            if type(reuse_decision) is not bool:
                raise TypeError(
                    "OptimizerBundle.should_reuse_batch must return bool"
                )
            reuse_batch = reuse_decision

        if step == 1:
            optimizer_state_elements = validate_optimizer(bundle, raw_model, device)
        if step == 1 or step % manifest.runtime.log_every == 0:
            elapsed = time.monotonic() - started_at
            print(
                f"step={step} loss={final_loss:.6f} accuracy={accuracy:.6f} "
                f"elapsed={elapsed:.1f}s budget={budget_seconds:.1f}s",
                flush=True,
            )
            if metric_recorder is not None:
                metric_recorder.record_training(
                    seed=seed,
                    step=step,
                    elapsed_seconds=elapsed,
                    loss=final_loss,
                    exact_accuracy=accuracy,
                )
                last_metric_step = step

    elapsed = time.monotonic() - started_at
    validate_model_state(raw_model, manifest.model_state, device)
    if (
        metric_recorder is not None
        and completed_steps > 0
        and completed_steps != last_metric_step
    ):
        metric_recorder.record_training(
            seed=seed,
            step=completed_steps,
            elapsed_seconds=elapsed,
            loss=final_loss,
            exact_accuracy=final_accuracy,
        )
    return final_loss, completed_steps, elapsed, optimizer_state_elements


def _evaluate(
    model: nn.Module,
    dataloader,
    manifest: BenchmarkManifest,
    device: torch.device,
    *,
    deadline: float,
    budget_seconds: float,
    include_act_diagnostics: bool = False,
) -> dict:
    model.eval()
    versions = capture_state_versions(model)
    loss_sum = 0.0
    correct_sum = 0.0
    example_count = 0
    loss_count = 0
    act_diagnostics = (
        ActDiagnosticsAccumulator() if include_act_diagnostics else None
    )
    with torch.no_grad():
        for batch in dataloader:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"evaluation exhausted its {budget_seconds:.1f}s time budget"
                )
            loss, accuracy, batch_examples, batch_loss_weight = _loss_and_accuracy(
                model,
                batch,
                manifest,
                device,
                act_diagnostics=act_diagnostics,
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"evaluation exhausted its {budget_seconds:.1f}s time budget"
                )
            loss_sum += float(loss.item()) * batch_loss_weight
            correct_sum += accuracy * batch_examples
            example_count += batch_examples
            loss_count += batch_loss_weight
    if time.monotonic() >= deadline:
        raise TimeoutError(
            f"evaluation exhausted its {budget_seconds:.1f}s time budget"
        )
    assert_state_versions_unchanged(model, versions)
    model.train()
    if example_count == 0 or loss_count == 0:
        raise ValueError("evaluation split contains no labels")
    accuracy = correct_sum / example_count
    metrics = {
        "loss": loss_sum / loss_count,
        "exact_accuracy": accuracy,
        "correct_examples": int(round(correct_sum)),
        "example_count": example_count,
    }
    if act_diagnostics is not None:
        metrics["act_diagnostics"] = act_diagnostics.summary(
            evaluation_task_cross_entropy=metrics["loss"]
        )
    return metrics



def _evaluate_depth_profile(
    *,
    model: nn.Module,
    dataloaders,
    manifest: BenchmarkManifest,
    device: torch.device,
    deadline: float,
    budget_seconds: float,
    seed: int,
    prefix: str,
    label: str,
) -> dict:
    split_names = _depth_split_names(dataloaders, prefix)
    ladder = [int(name.removeprefix(prefix)) for name in split_names]
    rungs = []
    prefix_solved = True
    max_certified_time_steps = None
    for split_name in split_names:
        time_steps = int(split_name.removeprefix(prefix))
        try:
            metrics = _evaluate(
                model,
                dataloaders[split_name],
                manifest,
                device,
                deadline=deadline,
                budget_seconds=budget_seconds,
            )
        except TimeoutError:
            rungs.append(
                {
                    "time_steps": time_steps,
                    "status": "not_completed",
                    "correct_examples": 0,
                    "example_count": len(dataloaders[split_name].dataset),
                    "exact_accuracy": None,
                }
            )
            break
        solved = metrics["correct_examples"] == metrics["example_count"]
        prefix_solved = prefix_solved and solved
        if prefix_solved:
            max_certified_time_steps = time_steps
        rungs.append(
            {
                "time_steps": time_steps,
                "status": (
                    "certified"
                    if prefix_solved
                    else "passed_uncertified"
                    if solved
                    else "failed"
                ),
                "correct_examples": metrics["correct_examples"],
                "example_count": metrics["example_count"],
                "exact_accuracy": metrics["exact_accuracy"],
            }
        )
        print(
            f"seed={seed} profile={label} depth_t={time_steps} "
            f"exact_accuracy={metrics['exact_accuracy']:.6f} "
            f"certified={prefix_solved}",
            flush=True,
        )
    return {
        "ladder": ladder,
        "max_certified_time_steps": max_certified_time_steps,
        "rungs": rungs,
    }

def _run_seed(
    submission: Submission,
    manifest: BenchmarkManifest,
    model_spec: ModelSpec,
    device: torch.device,
    seed: int,
    budget_seconds: float,
    submission_load_seconds: float,
    dataloaders=None,
    metric_recorder: MetricRecorder | None = None,
    include_act_diagnostics: bool = False,
) -> dict:
    _configure_seed(seed, device)
    batch_size, eval_batch_size = _resolve_batch_sizes(submission, manifest)
    if dataloaders is None:
        dataloaders = make_dataloaders(
            replace(
                manifest.data,
                seed=seed,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
            ),
            device=device,
        )
    elif (
        batch_size != manifest.data.batch_size
        or eval_batch_size
        != (manifest.data.eval_batch_size or manifest.data.batch_size)
    ):
        dataloaders = _with_batch_size(
            dataloaders,
            manifest,
            batch_size,
            eval_batch_size,
            seed,
        )
    max_steps = min(
        manifest.runtime.max_steps,
        submission.max_steps or manifest.runtime.max_steps,
    )

    started_at = time.monotonic() - submission_load_seconds
    deadline = started_at + budget_seconds
    if time.monotonic() >= deadline:
        raise TimeoutError("submission import exhausted the training-time budget")
    model = submission.build_model(model_spec)
    if not isinstance(model, nn.Module):
        raise TypeError("build_model must return torch.nn.Module")
    model_dtype = (
        torch.float32
        if manifest.runtime.amp
        else getattr(
            torch,
            manifest.runtime.dtype,
        )
    )
    model = model.to(device=device, dtype=model_dtype)
    _validate_model_interface(model, model_spec)
    state_elements = validate_model_state(model, manifest.model_state, device)

    bundle = submission.build_optimizer(
        model,
        OptimizerSpec(
            training_time_seconds=budget_seconds,
            device_type=device.type,
        ),
    )
    validate_optimizer(bundle, model, device)
    train_model = _compile_model(model, manifest)
    final_loss, steps, training_seconds, optimizer_state_elements = _train(
        raw_model=model,
        train_model=train_model,
        training_loss=submission.training_loss,
        token_training_loss=submission.token_training_loss,
        bundle=bundle,
        dataloader=dataloaders["train"],
        manifest=manifest,
        device=device,
        started_at=started_at,
        deadline=deadline,
        budget_seconds=budget_seconds,
        max_steps=max_steps,
        seed=seed,
        metric_recorder=metric_recorder,
    )

    evaluation = {}
    evaluation_budget_seconds = budget_seconds * EVALUATION_TIME_FRACTION
    evaluation_started_at = time.monotonic()
    evaluation_deadline = evaluation_started_at + evaluation_budget_seconds
    for split_name in _scoring_split_names(dataloaders):
        dataloader = dataloaders[split_name]
        metrics = _evaluate(
            model,
            dataloader,
            manifest,
            device,
            deadline=evaluation_deadline,
            budget_seconds=evaluation_budget_seconds,
        )
        evaluation[split_name] = metrics
        print(
            f"seed={seed} split={split_name} loss={metrics['loss']:.6f} "
            f"exact_accuracy={metrics['exact_accuracy']:.6f}",
            flush=True,
        )
        if metric_recorder is not None:
            metric_recorder.record_evaluation(
                seed=seed,
                split=split_name,
                loss=metrics["loss"],
                exact_accuracy=metrics["exact_accuracy"],
            )
    depth_profile = _evaluate_depth_profile(
        model=model,
        dataloaders=dataloaders,
        manifest=manifest,
        device=device,
        deadline=evaluation_deadline,
        budget_seconds=evaluation_budget_seconds,
        seed=seed,
        prefix=DEPTH_SPLIT_PREFIX,
        label="seen_n",
    )
    ood_n_depth_profile = _evaluate_depth_profile(
        model=model,
        dataloaders=dataloaders,
        manifest=manifest,
        device=device,
        deadline=evaluation_deadline,
        budget_seconds=evaluation_budget_seconds,
        seed=seed,
        prefix=OOD_N_DEPTH_SPLIT_PREFIX,
        label="ood_n",
    )
    depth_profile.update(
        {
            "depth_factor": depth_profile["max_certified_time_steps"] or 0,
            "ood_n_ladder": ood_n_depth_profile["ladder"],
            "ood_n_max_certified_time_steps": ood_n_depth_profile[
                "max_certified_time_steps"
            ],
            "ood_n_rungs": ood_n_depth_profile["rungs"],
        }
    )
    evaluation_seconds = time.monotonic() - evaluation_started_at

    act_diagnostics_seconds = None
    act_diagnostics_result = None
    if include_act_diagnostics and hasattr(model, "collect_act_diagnostics"):
        diagnostics_started_at = time.monotonic()
        act_diagnostics_result = {
            "scoring_splits": {},
            "first_uncertified_depth_rungs": {},
        }
        previous_diagnostics_setting = getattr(
            model, "collect_act_diagnostics", None
        )
        model.collect_act_diagnostics = True

        def collect_act(dataloader) -> dict | None:
            local_metrics = _evaluate(
                model,
                dataloader,
                manifest,
                device,
                deadline=float("inf"),
                budget_seconds=float("inf"),
                include_act_diagnostics=True,
            )
            return local_metrics.get("act_diagnostics")

        try:
            for split_name in _scoring_split_names(dataloaders):
                diagnostics = collect_act(dataloaders[split_name])
                if diagnostics is not None:
                    act_diagnostics_result["scoring_splits"][
                        split_name
                    ] = diagnostics

            def attach_first_uncertified(
                profile: dict,
                *,
                label: str,
                split_prefix: str,
            ) -> None:
                certified = profile["max_certified_time_steps"] or 0
                rung = next(
                    (
                        item
                        for item in profile["rungs"]
                        if item["time_steps"] > certified
                        and item["status"] != "not_completed"
                    ),
                    None,
                )
                if rung is None:
                    return
                split_name = f"{split_prefix}{rung['time_steps']}"
                diagnostics = collect_act(dataloaders[split_name])
                if diagnostics is not None:
                    depth_diagnostics = act_diagnostics_result[
                        "first_uncertified_depth_rungs"
                    ]
                    depth_diagnostics[label] = {
                        "time_steps": rung["time_steps"],
                        "diagnostics": diagnostics,
                    }

            attach_first_uncertified(
                depth_profile,
                label="seen_n",
                split_prefix=DEPTH_SPLIT_PREFIX,
            )
            attach_first_uncertified(
                ood_n_depth_profile,
                label="ood_n",
                split_prefix=OOD_N_DEPTH_SPLIT_PREFIX,
            )
        finally:
            model.collect_act_diagnostics = previous_diagnostics_setting
        act_diagnostics_seconds = time.monotonic() - diagnostics_started_at
    elif include_act_diagnostics:
        print(
            "ACT_DIAGNOSTICS_UNAVAILABLE | submission debug telemetry is "
            "disabled; set DBUG=True in the submission",
            flush=True,
        )

    result = {
        "seed": seed,
        "model_state_elements": state_elements,
        "optimizer_state_elements_after_first_step": optimizer_state_elements,
        "final_train_loss": final_loss,
        "completed_training_steps": steps,
        "training_batch_size": batch_size,
        "evaluation_batch_size": eval_batch_size,
        "max_training_steps": max_steps,
        "training_seconds": training_seconds,
        "evaluation_budget_seconds": evaluation_budget_seconds,
        "evaluation_seconds": evaluation_seconds,
        "evaluation": evaluation,
        "depth_profile": depth_profile,
    }
    if act_diagnostics_seconds is not None:
        result["act_diagnostics_seconds"] = act_diagnostics_seconds
        result["act_diagnostics"] = act_diagnostics_result
    return result


def _load_submission_file(path: str | Path) -> Submission:
    submission_path = Path(path).resolve()
    if submission_path.suffix != ".py" or not submission_path.is_file():
        raise ValueError("submission must be one existing .py file")
    if submission_path.stat().st_size > 256 * 1024:
        raise ValueError("submission file exceeds the 256 KiB limit")
    lint_submission_source(submission_path)
    module_spec = importlib.util.spec_from_file_location(
        f"uploaded_submission_{submission_path.stat().st_mtime_ns}",
        submission_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load submission from {submission_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    submission = getattr(module, "SUBMISSION", None)
    if not isinstance(submission, Submission):
        raise TypeError("submission.py must export benchmark.Submission as SUBMISSION")
    return submission


def _submission_requests_act_diagnostics(submission: Submission) -> bool:
    """Return whether the submission module has explicitly enabled debug mode."""

    module_globals = getattr(submission.build_model, "__globals__", None)
    return (
        isinstance(module_globals, dict)
        and module_globals.get("DBUG") is True
    )


def run_submission_file(
    submission_path: str | Path,
    manifest_path: str | Path,
    *,
    include_structured_metrics: bool = False,
    include_act_diagnostics: bool = False,
    num_workers: int | None = None,
) -> dict:
    manifest = load_manifest(manifest_path)
    if num_workers is not None:
        manifest = replace(
            manifest,
            data=replace(manifest.data, num_workers=num_workers),
        )
    device = _resolve_device(manifest)
    model_spec = _make_model_spec(manifest)
    preloaded_dataloaders = {}
    if manifest.data.data_root is not None:
        preloaded_dataloaders = {
            seed: make_dataloaders(
                replace(manifest.data, seed=seed),
                device=device,
            )
            for seed in manifest.runtime.seeds
        }
        _deny_dataset_file_access(manifest.data.data_root)
    submission_load_started = time.monotonic()
    submission = _load_submission_file(submission_path)
    validate_submission(submission)
    submission_debug_enabled = _submission_requests_act_diagnostics(submission)
    if submission_debug_enabled and not include_act_diagnostics:
        print(
            "ACT_DIAGNOSTICS_ENABLED | submission DBUG=True",
            flush=True,
        )
    include_act_diagnostics = (
        include_act_diagnostics or submission_debug_enabled
    )
    submission_load_seconds = time.monotonic() - submission_load_started
    budget_per_seed = manifest.runtime.total_training_time_seconds / len(
        manifest.runtime.seeds
    )
    evaluation_budget_per_seed = budget_per_seed * EVALUATION_TIME_FRACTION
    metric_recorder = MetricRecorder() if include_structured_metrics else None
    batch_size, eval_batch_size = _resolve_batch_sizes(submission, manifest)

    print(
        json.dumps(
            {
                "manifest": manifest.name,
                "model_spec": asdict(model_spec),
                "training_batch_size": batch_size,
                "evaluation_batch_size": eval_batch_size,
                "num_workers": manifest.data.num_workers,
                "max_training_steps": min(
                    manifest.runtime.max_steps,
                    submission.max_steps or manifest.runtime.max_steps,
                ),
                "total_training_time_seconds": manifest.runtime.total_training_time_seconds,
                "training_time_seconds_per_seed": budget_per_seed,
                "evaluation_time_seconds_per_seed": evaluation_budget_per_seed,
                "seeds": manifest.runtime.seeds,
            },
            indent=2,
        ),
        flush=True,
    )

    seed_results = [
        _run_seed(
            submission,
            manifest,
            model_spec,
            device,
            seed,
            budget_per_seed,
            submission_load_seconds / len(manifest.runtime.seeds),
            preloaded_dataloaders.get(seed),
            metric_recorder,
            include_act_diagnostics,
        )
        for seed in manifest.runtime.seeds
    ]
    measurements = [
        metrics
        for seed_result in seed_results
        for metrics in seed_result["evaluation"].values()
    ]
    result = {
        "manifest": manifest.name,
        "score": {
            "primary_metric": "mean_exact_accuracy",
            "mean_exact_accuracy": statistics.fmean(
                metrics["exact_accuracy"] for metrics in measurements
            ),
            "mean_loss": statistics.fmean(metrics["loss"] for metrics in measurements),
            "num_measurements": len(measurements),
        },
        "seeds": seed_results,
    }
    if any(seed_result["depth_profile"]["ladder"] for seed_result in seed_results):
        certified_time_steps = min(
            seed_result["depth_profile"]["max_certified_time_steps"] or 0
            for seed_result in seed_results
        )
        ood_n_certified_time_steps = min(
            seed_result["depth_profile"]["ood_n_max_certified_time_steps"] or 0
            for seed_result in seed_results
        )
        result["depth_profile"] = {
            "ladder": seed_results[0]["depth_profile"]["ladder"],
            "max_certified_time_steps": certified_time_steps or None,
            "ood_n_ladder": seed_results[0]["depth_profile"]["ood_n_ladder"],
            "ood_n_profile_available": bool(
                seed_results[0]["depth_profile"]["ood_n_ladder"]
            ),
            "ood_n_max_certified_time_steps": (
                ood_n_certified_time_steps or None
            ),
            "depth_factor": min(
                seed_result["depth_profile"]["depth_factor"]
                for seed_result in seed_results
            ),
        }
    if metric_recorder is not None:
        metric_recorder.record_summary(
            completed_steps=sum(
                seed_result["completed_training_steps"]
                for seed_result in seed_results
            ),
            training_seconds=sum(
                seed_result["training_seconds"] for seed_result in seed_results
            ),
            mean_exact_accuracy=result["score"]["mean_exact_accuracy"],
        )
        result["structured_metrics"] = metric_recorder.snapshot()
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    act_diagnostics = _format_act_diagnostics(result)
    if act_diagnostics is not None:
        print("\n" + act_diagnostics, flush=True)
    competition_progress = _format_competition_progress(result)
    if competition_progress is not None:
        print("\n" + competition_progress, flush=True)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submission-file", required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        help="override the manifest data-loader worker count",
    )
    parser.add_argument("--include-structured-metrics", action="store_true")
    parser.add_argument(
        "--include-act-diagnostics",
        action="store_true",
        help="collect local ACT telemetry during final test/OOD evaluation",
    )
    args = parser.parse_args()
    run_submission_file(
        args.submission_file,
        args.manifest,
        include_structured_metrics=args.include_structured_metrics,
        include_act_diagnostics=args.include_act_diagnostics,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    cli()
