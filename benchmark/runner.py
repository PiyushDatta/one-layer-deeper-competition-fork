"""Evaluator-owned runner for the One Layer Deeper competition."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
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
from .model_diagnostics import (
    capture_parameter_snapshot,
    ModelDiagnosticsAccumulator,
    summarize_parameter_gradients,
    summarize_parameter_updates,
    TrainingModelDiagnosticsTrajectory,
    validate_training_credit,
)
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
            tail_remainder = remainders["mean_tail_forced"]
            tail_remainder_text = (
                "N/A" if tail_remainder is None else f"{tail_remainder:.3f}"
            )
            tail_halt_fraction = diagnostics["tail_halt_fraction"]
            tail_cutoff_text = (
                "off"
                if tail_halt_fraction is None
                else f"{100.0 * tail_halt_fraction:.2f}%"
            )
            lines.append(
                f"ACT_SUMMARY | seed={seed} split={split} | "
                f"eval_task_CE="
                f"{diagnostics['evaluation_task_cross_entropy']:.6f} | "
                f"ponder={diagnostics['raw_mean_ponder_time']:.3f} | "
                f"weighted_ponder={diagnostics['weighted_ponder_contribution']:.6f} | "
                f"tail_cutoff={tail_cutoff_text} | "
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
                f"tail_forced token/example/batch="
                f"{100.0 * caps['token_tail_forced_rate']:.2f}%/"
                f"{100.0 * caps['example_tail_forced_rate']:.2f}%/"
                f"{100.0 * caps['batch_tail_forced_rate']:.2f}% | "
                f"remainder mean/cap/tail="
                f"{remainders['mean']:.3f}/{forced_remainder_text}/"
                f"{tail_remainder_text}"
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


def _format_model_diagnostics(result: dict) -> str | None:
    """Format local model telemetry without changing official score fields."""

    counterfactual_descriptions = {
        "zero": "all_segment_signal_removed",
        "permuted": "all_field_roles_permuted",
        "zero_nx": "N_X_signal_removed_T_preserved",
        "zero_t": "T_signal_removed_N_X_preserved",
        "swap_n_x": "N_X_roles_swapped_T_preserved",
    }
    lines = []
    for seed_result in result.get("seeds", ()):
        seed = seed_result["seed"]
        payload = seed_result.get("model_diagnostics") or {}
        metric_groups = list(payload.get("scoring_splits", {}).items())
        metric_groups.extend(
            (f"{label}/T={rung['time_steps']}", rung["diagnostics"])
            for label, rung in payload.get(
                "first_uncertified_depth_rungs", {}
            ).items()
        )
        for split, diagnostics in metric_groups:
            names = diagnostics["segment_names"]

            segment_counts = diagnostics["segment_token_counts"]

            def segment_values(
                values,
                *,
                precision: int = 3,
                mark_inactive: bool = False,
            ) -> str:
                return "/".join(
                    f"{name}="
                    + (
                        "N/A"
                        if value is None
                        or (mark_inactive and segment_counts[index] == 0)
                        else f"{value:.{precision}f}"
                    )
                    for index, (name, value) in enumerate(zip(names, values))
                )

            counts = "/".join(
                f"{name}={count}"
                for name, count in zip(
                    names,
                    segment_counts,
                )
            )
            lines.append(
                f"MODEL_SEGMENTS | seed={seed} split={split} | "
                f"tokens {counts} | final_state_rms "
                f"{segment_values(diagnostics['final_state_rms_by_segment'])} | "
                f"logit_entropy "
                f"{segment_values(diagnostics['final_logit_entropy_by_segment'])}"
            )

            embedding = diagnostics["segment_embedding"]["last_batch"]
            active_indices = [
                index
                for index, count in enumerate(segment_counts)
                if count > 0
            ]
            pairwise = []
            cosine_matrix = embedding["cosine_matrix"]
            for left_offset, left in enumerate(active_indices):
                for right in active_indices[left_offset + 1 :]:
                    pairwise.append(
                        f"{names[left]}-{names[right]}="
                        f"{cosine_matrix[left][right]:.3f}"
                    )
            lines.append(
                f"MODEL_WEIGHTS | seed={seed} split={split} | "
                f"group=segment_embedding | norms "
                f"{segment_values(embedding['norms'], mark_inactive=True)} | deltas "
                f"{segment_values(embedding['delta_norms'], mark_inactive=True)} | "
                f"relative {segment_values(embedding['relative_deltas'], mark_inactive=True)} | "
                f"cos_to_init {segment_values(embedding['initial_cosines'], mark_inactive=True)} | "
                f"pair_cos {'/'.join(pairwise)}"
            )
            parameter_stats = diagnostics["parameter_stats"]
            most_changed = sorted(
                parameter_stats.items(),
                key=lambda item: item[1]["last_batch"]["relative_delta"],
                reverse=True,
            )[:5]
            lines.append(
                f"MODEL_WEIGHTS | seed={seed} split={split} | "
                "top_relative_delta "
                + " | ".join(
                    f"{name}={snapshots['last_batch']['relative_delta']:.3f} "
                    f"(delta={snapshots['last_batch']['delta_norm']:.3f}, "
                    f"norm={snapshots['last_batch']['norm']:.3f})"
                    for name, snapshots in most_changed
                )
            )

            for layer in diagnostics["layers"]:
                lines.append(
                    f"MODEL_LAYER | seed={seed} split={split} "
                    f"step={layer['step']} | queries={layer['valid_query_count']} | "
                    f"rms in/attn_delta/mlp_delta/out="
                    f"{layer['input_rms']:.3f}/"
                    f"{layer['attention_update_rms']:.3f}/"
                    f"{layer['mlp_update_rms']:.3f}/"
                    f"{layer['output_rms']:.3f} | "
                    f"update_ratio attn/mlp="
                    f"{layer['attention_update_ratio']:.3f}/"
                    f"{layer['mlp_update_ratio']:.3f} | "
                    f"input_output_cos={layer['input_output_cosine']:.3f} | "
                    f"attention entropy/effective_keys="
                    f"{layer['attention_entropy']:.3f}/"
                    f"{layer['effective_attended_tokens']:.3f} | "
                    f"state_delta_by_segment "
                    f"{segment_values(layer['state_change_rms_by_segment'])}"
                )
                query_counts = layer["segment_query_counts"]
                for query_index, query_count in enumerate(query_counts):
                    if query_count == 0:
                        continue
                    key_mass = layer["attention_mass_by_segment"][query_index]
                    stream_mass = layer.get("attention_mass_by_stream")
                    stream_text = ""
                    if stream_mass is not None:
                        stream_text = " | stream_mass " + "/".join(
                            f"{name}={value:.3f}"
                            for name, value in zip(
                                layer["stream_names"],
                                stream_mass[query_index],
                            )
                        )
                    lines.append(
                        f"MODEL_ATTENTION | seed={seed} split={split} "
                        f"step={layer['step']} query={names[query_index]} "
                        f"n={query_count} | raw_key_mass "
                        f"{segment_values(key_mass)}"
                        f"{stream_text}"
                    )

            for stage in diagnostics.get("stage_predictions", ()):
                lines.append(
                    f"MODEL_STAGE | seed={seed} split={split} "
                    f"step={stage['step']} | "
                    f"CE={stage['task_cross_entropy']:.6f} | "
                    f"exact={100.0 * stage['exact_accuracy']:.4f}% "
                    f"({stage['correct_examples']}/{stage['example_count']})"
                )
            for name, counterfactual in diagnostics.get(
                "segment_counterfactuals", {}
            ).items():
                lines.append(
                    f"MODEL_COUNTERFACTUAL | seed={seed} split={split} "
                    f"probe={name} "
                    f"meaning={counterfactual_descriptions.get(name, 'unknown')} | "
                    f"CE={counterfactual['task_cross_entropy']:.6f} | "
                    f"exact={100.0 * counterfactual['exact_accuracy']:.4f}% "
                    f"({counterfactual['correct_examples']}/"
                    f"{counterfactual['example_count']}) | "
                    f"token_prediction_flip="
                    f"{100.0 * counterfactual['token_prediction_flip_rate']:.2f}% | "
                    f"example_prediction_flip="
                    f"{100.0 * counterfactual['example_prediction_flip_rate']:.2f}%"
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
    model_diagnostics: ModelDiagnosticsAccumulator | None = None,
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
        if model_diagnostics is not None:
            model_diagnostics.add(
                auxiliary=auxiliary,
                logits=logits,
                targets=targets,
                target_positions=target_positions,
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


def _collect_training_model_checkpoint(
    *,
    model: nn.Module,
    batch,
    manifest: BenchmarkManifest,
    device: torch.device,
) -> tuple[dict[str, object], float, float]:
    """Collect one post-update checkpoint without mutating model state."""

    if not hasattr(model, "collect_model_diagnostics"):
        raise TypeError("model does not expose debug model diagnostics")
    previous_setting = model.collect_model_diagnostics
    was_training = model.training
    versions = capture_state_versions(model)
    accumulator = ModelDiagnosticsAccumulator()
    model.collect_model_diagnostics = True
    model.eval()
    try:
        with torch.no_grad():
            loss, accuracy, _, _ = _loss_and_accuracy(
                model,
                batch,
                manifest,
                device,
                model_diagnostics=accumulator,
            )
        assert_state_versions_unchanged(model, versions)
    finally:
        model.collect_model_diagnostics = previous_setting
        model.train(was_training)
    summary = accumulator.summary()
    if summary is None:
        raise ValueError(
            "collect_model_diagnostics=True produced no model diagnostics"
        )
    return summary, float(loss.item()), accuracy


def _format_training_model_diagnostic(
    *,
    seed: int,
    record: dict[str, object],
) -> str:
    """Format one bounded post-update training checkpoint."""

    step = record["step"]
    model = record["model"]
    optimization = record["optimization"]
    stages = " ".join(
        f"{stage['step']}:{100.0 * stage['exact_accuracy']:.2f}%"
        for stage in model["stage_predictions"]
    )
    lines = [
        f"MODEL_TRAIN_CHECKPOINT | seed={seed} step={step} | "
        f"pre_update_backward_loss={record['training_loss']:.6f} "
        f"pre_update_backward_acc="
        f"{100.0 * record['training_exact_accuracy']:.2f}% | "
        f"post_update_same_batch_CE="
        f"{record['diagnostic_task_cross_entropy']:.6f} "
        f"post_update_same_batch_acc="
        f"{100.0 * record['diagnostic_exact_accuracy']:.2f}% | "
        f"stage_exact {stages}"
    ]

    segment_names = model["segment_names"]
    embedding = model["segment_embedding"]["last_batch"]
    segment_counts = model["segment_token_counts"]
    segment_deltas = "/".join(
        f"{name}="
        + (
            "N/A"
            if segment_counts[index] == 0
            else f"{embedding['relative_deltas'][index]:.3f}"
        )
        for index, name in enumerate(segment_names)
    )
    layer_dynamics = " ".join(
        f"{layer['step']}:attn={layer['attention_update_ratio']:.3f},"
        f"mlp={layer['mlp_update_ratio']:.3f},"
        f"cos={layer['input_output_cosine']:.3f}"
        for layer in model["layers"]
    )
    counterfactuals = " ".join(
        f"{name}={100.0 * stats['example_prediction_flip_rate']:.1f}%"
        for name, stats in model["segment_counterfactuals"].items()
    )
    lines.append(
        f"MODEL_TRAIN_DYNAMICS | seed={seed} step={step} | "
        f"segment_relative_delta {segment_deltas} | "
        f"layers {layer_dynamics} | counterfactual_example_flip "
        f"{counterfactuals}"
    )

    before_clip = optimization["gradient_before_clipping"]
    after_clip = optimization["gradient_after_clipping"]
    update = optimization["optimizer_update"]
    top_gradients = sorted(
        after_clip["parameters"].items(),
        key=lambda item: item[1]["relative_gradient_norm"],
        reverse=True,
    )[:5]
    gradient_text = "/".join(
        f"{name}={stats['relative_gradient_norm']:.3g}"
        for name, stats in top_gradients
    )
    if update["available"]:
        top_updates = sorted(
            update["parameters"].items(),
            key=lambda item: item[1]["relative_update_norm"],
            reverse=True,
        )[:5]
        update_text = "/".join(
            f"{name}={stats['relative_update_norm']:.3g}"
            for name, stats in top_updates
        )
        update_summary = (
            f"update_norm={update['update_norm']:.6g} "
            f"relative_update={update['relative_update_norm']:.6g} | "
            f"top_relative_update {update_text}"
        )
    else:
        update_summary = f"update_unavailable={update['reason']}"
    clip_threshold = optimization["gradient_clip_threshold"]
    clip_scale = optimization["gradient_clip_scale"]
    optimizer_groups = "/".join(
        f"g{group['group']}:lr={group['learning_rate']:.6g},wd="
        + (
            "N/A"
            if group["weight_decay"] is None
            else f"{group['weight_decay']:.6g}"
        )
        for group in optimization["optimizer_parameter_groups"]
    )
    lines.append(
        f"MODEL_TRAIN_OPTIMIZER | seed={seed} step={step} "
        f"gradient_phase=final_backward | grad_norm pre/post_clip="
        f"{before_clip['gradient_norm']:.6g}/"
        f"{after_clip['gradient_norm']:.6g} | clip_threshold="
        + ("off" if clip_threshold is None else f"{clip_threshold:.6g}")
        + " clip_scale="
        + ("N/A" if clip_scale is None else f"{clip_scale:.6g}")
        + f" | optimizer_groups {optimizer_groups} | "
        f"missing_grad_params={after_clip['missing_gradient_parameter_count']} | "
        f"top_relative_gradient {gradient_text} | {update_summary}"
    )

    credit = record.get("training_credit")
    segment_parameter_gradient = after_clip["parameters"].get(
        "segment_embedding.weight",
        {},
    )
    segment_parameter_update = (
        update.get("parameters", {}).get("segment_embedding.weight", {})
        if update["available"]
        else {}
    )
    segment_counts_for_credit = (
        credit["segment_token_counts"]
        if credit is not None
        else model["segment_token_counts"]
    )

    def optional_segment_values(values) -> str:
        return "/".join(
            f"{name}="
            + (
                "N/A"
                if values is None
                or index >= len(values)
                or segment_counts_for_credit[index] == 0
                else f"{values[index]:.3g}"
            )
            for index, name in enumerate(segment_names)
        )

    lines.append(
        f"MODEL_TRAIN_SEGMENT | seed={seed} step={step} | counts "
        + "/".join(
            f"{name}={count}"
            for name, count in zip(segment_names, segment_counts_for_credit)
        )
        + " | activation_grad_rms="
        + (
            "N/A"
            if credit is None
            else f"{credit['segment_signal_grad_rms']:.3g}"
        )
        + " | activation_grad_by_segment "
        + optional_segment_values(
            None
            if credit is None
            else credit["segment_signal_grad_rms_by_segment"]
        )
        + " | clipped_parameter_row_grad "
        + optional_segment_values(
            segment_parameter_gradient.get("row_gradient_norms")
        )
        + " | row_update "
        + optional_segment_values(segment_parameter_update.get("row_update_norms"))
        + " | row_relative_update "
        + optional_segment_values(
            segment_parameter_update.get("row_relative_update_norms")
        )
    )

    def combined_parameter_norm(
        stats: dict[str, dict[str, object]],
        *,
        suffixes: tuple[str, ...],
        field: str,
    ) -> float | None:
        squared = 0.0
        found = False
        for name, parameter in stats.items():
            if name.endswith(suffixes) and field in parameter:
                value = float(parameter[field])
                squared += value * value
                found = True
        return math.sqrt(squared) if found else None

    def qkv_component_norm(
        stats: dict[str, dict[str, object]],
        *,
        field: str,
        component: str,
    ) -> float | None:
        squared = 0.0
        found = False
        for name, parameter in stats.items():
            if name.endswith((".qkv.weight", ".qkv.bias")):
                components = parameter.get(field)
                if components is not None:
                    value = float(components[component])
                    squared += value * value
                    found = True
        return math.sqrt(squared) if found else None

    gradient_parameters = after_clip["parameters"]
    update_parameters = update.get("parameters", {}) if update["available"] else {}
    branch_specs = (
        (
            "q",
            qkv_component_norm(
                gradient_parameters,
                field="component_gradient_norms",
                component="q",
            ),
            qkv_component_norm(
                update_parameters,
                field="component_update_norms",
                component="q",
            ),
        ),
        (
            "k",
            qkv_component_norm(
                gradient_parameters,
                field="component_gradient_norms",
                component="k",
            ),
            qkv_component_norm(
                update_parameters,
                field="component_update_norms",
                component="k",
            ),
        ),
        (
            "v",
            qkv_component_norm(
                gradient_parameters,
                field="component_gradient_norms",
                component="v",
            ),
            qkv_component_norm(
                update_parameters,
                field="component_update_norms",
                component="v",
            ),
        ),
        (
            "attn_out",
            combined_parameter_norm(
                gradient_parameters,
                suffixes=(".out.weight", ".out.bias"),
                field="gradient_norm",
            ),
            combined_parameter_norm(
                update_parameters,
                suffixes=(".out.weight", ".out.bias"),
                field="update_norm",
            ),
        ),
        (
            "mlp_up",
            combined_parameter_norm(
                gradient_parameters,
                suffixes=(".up.weight", ".up.bias"),
                field="gradient_norm",
            ),
            combined_parameter_norm(
                update_parameters,
                suffixes=(".up.weight", ".up.bias"),
                field="update_norm",
            ),
        ),
        (
            "mlp_down",
            combined_parameter_norm(
                gradient_parameters,
                suffixes=(".down.weight", ".down.bias"),
                field="gradient_norm",
            ),
            combined_parameter_norm(
                update_parameters,
                suffixes=(".down.weight", ".down.bias"),
                field="update_norm",
            ),
        ),
    )

    def branch_values(index: int) -> str:
        return "/".join(
            f"{name}=" + ("N/A" if values[index] is None else f"{values[index]:.3g}")
            for name, *values in branch_specs
        )

    lines.append(
        f"MODEL_TRAIN_BRANCH | seed={seed} step={step} "
        f"scope=shared_block_combined_across_recurrent_calls | "
        f"clipped_grad {branch_values(0)} | update {branch_values(1)}"
    )

    if credit is not None:
        credit_stages = " ".join(
            f"{stage['step']}:rms={stage['state_grad_rms']:.3g},"
            f"rel_final={stage['relative_to_final']:.3g},"
            f"N={stage['state_grad_rms_by_segment'][1]:.3g},"
            f"X={stage['state_grad_rms_by_segment'][2]:.3g},"
            f"T={stage['state_grad_rms_by_segment'][3]:.3g},"
            f"control="
            + (
                "N/A"
                if stage["control_grad_rms"] is None
                else f"{stage['control_grad_rms']:.3g}"
            )
            for stage in credit["stages"]
        )
        lines.append(
            f"MODEL_TRAIN_CREDIT | seed={seed} step={step} | "
            f"retained_stage_gradients {credit_stages}"
        )
    return "\n".join(lines)


def _format_training_model_diagnostics(result: dict) -> str | None:
    """Format all retained training checkpoints across seeds."""

    lines = []
    for seed_result in result.get("seeds", ()):
        trajectory = seed_result.get("training_model_diagnostics") or {}
        for record in trajectory.get("records", ()):
            lines.append(
                _format_training_model_diagnostic(
                    seed=seed_result["seed"],
                    record=record,
                )
            )
    return "\n".join(lines) if lines else None


def _read_experiment_number(experiments_path: str | Path) -> int | None:
    """Read the one authoritative experiment counter from its Markdown log."""

    path = Path(experiments_path)
    if not path.is_file():
        return None
    marker = "Next experiment number:"
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.casefold().startswith(marker.casefold()):
            values.append(stripped[len(marker) :].strip())
    if len(values) > 1:
        raise ValueError(
            f"{path} contains more than one '{marker}' marker"
        )
    if not values:
        return None
    if not values[0].isdecimal() or int(values[0]) < 1:
        raise ValueError(
            f"{path} has an invalid '{marker}' value: {values[0]!r}"
        )
    return int(values[0])


def _write_debug_metrics_report(
    result: dict,
    *,
    experiment_number: int | None,
    experiments_path: str | Path,
    submission_path: str | Path,
    submission_source: str,
    directory: str | Path | None = None,
) -> Path:
    """Flush one complete in-memory debug report to the OS temp folder."""

    resolved_experiments_path = Path(experiments_path).resolve()
    resolved_submission_path = Path(submission_path).resolve()
    experiment_label = (
        "unknown" if experiment_number is None else str(experiment_number)
    )
    sections = [
        "\n".join(
            (
                f"EXPERIMENT_NUMBER={experiment_label}",
                f"EXPERIMENTS_FILE={resolved_experiments_path}",
                f"SUBMISSION_FILE={resolved_submission_path}",
            )
        )
    ]
    training_diagnostics = _format_training_model_diagnostics(result)
    if training_diagnostics is not None:
        sections.append(
            "[TRAINING_DIAGNOSTICS]\n" + training_diagnostics
        )
    act_diagnostics = _format_act_diagnostics(result)
    if act_diagnostics is not None:
        sections.append("[ACT_DIAGNOSTICS]\n" + act_diagnostics)
    model_diagnostics = _format_model_diagnostics(result)
    if model_diagnostics is not None:
        sections.append("[MODEL_DIAGNOSTICS]\n" + model_diagnostics)
    competition_progress = _format_competition_progress(result)
    if competition_progress is not None:
        sections.append(
            "[COMPETITION_PROGRESS]\n" + competition_progress
        )
    sections.append(
        "[RESULT_JSON]\n" + json.dumps(result, indent=2, sort_keys=True)
    )
    # Keep the exact captured source as the final report section so a reader
    # can audit the model without having to infer which working-tree version
    # produced these metrics.
    sections.append("[SUBMISSION_SOURCE]\n" + submission_source)

    temp_directory = None if directory is None else str(Path(directory))
    report_contents = "\n\n".join(sections) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f"one-layer-deeper-experiment-{experiment_label}-",
        suffix=".txt",
        dir=temp_directory,
        delete=False,
    ) as handle:
        handle.write(report_contents)
        report_path = Path(handle.name).resolve()
    return report_path


def _compact_result_for_stdout(result: dict) -> dict:
    """Remove only verbose diagnostic payloads from the console result."""

    verbose_seed_keys = {
        "act_diagnostics",
        "model_diagnostics",
        "training_model_diagnostics",
    }
    compact_result = dict(result)
    compact_result["seeds"] = [
        {
            key: value
            for key, value in seed_result.items()
            if key not in verbose_seed_keys
        }
        for seed_result in result.get("seeds", ())
    ]
    return compact_result


def _emit_final_metrics(
    result: dict,
    *,
    debug_metrics_to_file: bool,
    experiment_number: int | None = None,
    experiments_path: str | Path | None = None,
    submission_path: str | Path | None = None,
    submission_source: str | None = None,
) -> Path | None:
    """Print the compact result or redirect verbose debug output to a file."""

    competition_progress = _format_competition_progress(result)
    if not debug_metrics_to_file:
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
        act_diagnostics = _format_act_diagnostics(result)
        if act_diagnostics is not None:
            print("\n" + act_diagnostics, flush=True)
        model_diagnostics = _format_model_diagnostics(result)
        if model_diagnostics is not None:
            print("\n" + model_diagnostics, flush=True)
        if competition_progress is not None:
            print("\n" + competition_progress, flush=True)
        return None

    if (
        experiments_path is None
        or submission_path is None
        or submission_source is None
    ):
        raise ValueError(
            "debug metrics require experiment and submission source context"
        )
    report_path = _write_debug_metrics_report(
        result,
        experiment_number=experiment_number,
        experiments_path=experiments_path,
        submission_path=submission_path,
        submission_source=submission_source,
    )
    print(
        "RESULT_JSON="
        + json.dumps(_compact_result_for_stdout(result), sort_keys=True),
        flush=True,
    )
    if competition_progress is not None:
        print("\n" + competition_progress, flush=True)
    # Keep this as the final line of a debug run so it remains easy to find.
    print(f"DEBUG_METRICS_FILE | path={report_path}", flush=True)
    return report_path


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
    model_diagnostics_trajectory: TrainingModelDiagnosticsTrajectory | None = None,
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
    latest_optimization = None
    latest_training_credit = None
    last_model_diagnostic_step = 0
    supports_training_credit = (
        model_diagnostics_trajectory is not None
        and hasattr(raw_model, "collect_training_diagnostics")
        and callable(
            getattr(raw_model, "consume_training_grad_diagnostics", None)
        )
    )
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
        latest_training_credit = None
        gradient_before_clipping = None
        gradient_after_clipping = None

        for pass_index in range(1, bundle.backward_passes_per_step + 1):
            final_backward_pass = pass_index == bundle.backward_passes_per_step
            # Debug mode intentionally retains recurrent-state gradients on
            # every final backward pass. The training loop can end at its
            # deadline after any step, so sampling only scheduled log steps
            # would make the true terminal step lose its credit trace.
            collect_training_credit = (
                supports_training_credit and final_backward_pass
            )
            previous_training_diagnostics_setting = None
            if supports_training_credit:
                previous_training_diagnostics_setting = (
                    raw_model.collect_training_diagnostics
                )
                raw_model.collect_training_diagnostics = (
                    collect_training_credit
                )
            optimizer.zero_grad(set_to_none=True)
            try:
                forward_model = (
                    raw_model if collect_training_credit else train_model
                )
                loss, accuracy, _, _ = _loss_and_accuracy(
                    forward_model,
                    batch,
                    manifest,
                    device,
                    training_loss=training_loss,
                    token_training_loss=token_training_loss,
                )
                if not torch.isfinite(loss).all().item():
                    raise FloatingPointError(
                        f"non-finite training loss at step {step}, "
                        f"pass {pass_index}"
                    )
                loss.backward()
                if collect_training_credit:
                    latest_training_credit = validate_training_credit(
                        raw_model.consume_training_grad_diagnostics()
                    )
            finally:
                if supports_training_credit:
                    raw_model.collect_training_diagnostics = (
                        previous_training_diagnostics_setting
                    )
            if model_diagnostics_trajectory is not None and final_backward_pass:
                gradient_before_clipping = summarize_parameter_gradients(
                    raw_model
                )
            if manifest.runtime.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), manifest.runtime.grad_clip
                )
            if model_diagnostics_trajectory is not None and final_backward_pass:
                gradient_after_clipping = summarize_parameter_gradients(
                    raw_model
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
        parameter_snapshot = (
            capture_parameter_snapshot(raw_model)
            if model_diagnostics_trajectory is not None
            else None
        )
        optimizer.step()
        if model_diagnostics_trajectory is not None:
            assert gradient_before_clipping is not None
            assert gradient_after_clipping is not None
            pre_clip_norm = gradient_before_clipping["gradient_norm"]
            post_clip_norm = gradient_after_clipping["gradient_norm"]
            latest_optimization = {
                "gradient_before_clipping": gradient_before_clipping,
                "gradient_after_clipping": gradient_after_clipping,
                "gradient_clip_threshold": manifest.runtime.grad_clip,
                "gradient_clip_scale": (
                    post_clip_norm / pre_clip_norm
                    if pre_clip_norm > 0.0
                    else None
                ),
                "optimizer_parameter_groups": [
                    {
                        "group": index,
                        "learning_rate": float(group["lr"]),
                        "weight_decay": (
                            None
                            if "weight_decay" not in group
                            else float(group["weight_decay"])
                        ),
                    }
                    for index, group in enumerate(optimizer.param_groups)
                ],
                "optimizer_update": summarize_parameter_updates(
                    raw_model,
                    parameter_snapshot,
                ),
            }
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
            if model_diagnostics_trajectory is not None:
                diagnostics_started_at = time.monotonic()
                model_summary, diagnostic_loss, diagnostic_accuracy = (
                    _collect_training_model_checkpoint(
                        model=raw_model,
                        batch=batch,
                        manifest=manifest,
                        device=device,
                    )
                )
                assert latest_optimization is not None
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "collection_seconds": (
                        time.monotonic() - diagnostics_started_at
                    ),
                    "training_loss": final_loss,
                    "training_exact_accuracy": accuracy,
                    "diagnostic_task_cross_entropy": diagnostic_loss,
                    "diagnostic_exact_accuracy": diagnostic_accuracy,
                    "model": model_summary,
                    "optimization": latest_optimization,
                    "training_credit": latest_training_credit,
                }
                model_diagnostics_trajectory.record(record)
                last_model_diagnostic_step = step

    elapsed = time.monotonic() - started_at
    if (
        model_diagnostics_trajectory is not None
        and completed_steps > 0
        and completed_steps != last_model_diagnostic_step
    ):
        diagnostics_started_at = time.monotonic()
        model_summary, diagnostic_loss, diagnostic_accuracy = (
            _collect_training_model_checkpoint(
                model=raw_model,
                batch=batch,
                manifest=manifest,
                device=device,
            )
        )
        assert latest_optimization is not None
        terminal_elapsed = time.monotonic() - started_at
        terminal_record = {
            "step": completed_steps,
            "elapsed_seconds": terminal_elapsed,
            "collection_seconds": time.monotonic() - diagnostics_started_at,
            "training_loss": final_loss,
            "training_exact_accuracy": final_accuracy,
            "diagnostic_task_cross_entropy": diagnostic_loss,
            "diagnostic_exact_accuracy": diagnostic_accuracy,
            "model": model_summary,
            "optimization": latest_optimization,
            "training_credit": latest_training_credit,
        }
        model_diagnostics_trajectory.record(terminal_record)
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
    include_model_diagnostics: bool = False,
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
    model_diagnostics = (
        ModelDiagnosticsAccumulator() if include_model_diagnostics else None
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
                model_diagnostics=model_diagnostics,
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
    if model_diagnostics is not None:
        metrics["model_diagnostics"] = model_diagnostics.summary()
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
    training_model_diagnostics = (
        TrainingModelDiagnosticsTrajectory()
        if include_act_diagnostics
        and hasattr(model, "collect_model_diagnostics")
        else None
    )
    # Retained intermediate gradients and one-shot Python debug contexts are
    # intentionally eager-only. Keep every DBUG training step on that same
    # execution path instead of alternating compiled and eager forwards.
    train_model = (
        model
        if training_model_diagnostics is not None
        else _compile_model(model, manifest)
    )
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
        model_diagnostics_trajectory=training_model_diagnostics,
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

    debug_diagnostics_seconds = None
    act_diagnostics_result = None
    model_diagnostics_result = None
    supports_act_diagnostics = hasattr(model, "collect_act_diagnostics")
    supports_model_diagnostics = hasattr(model, "collect_model_diagnostics")
    if include_act_diagnostics and (
        supports_act_diagnostics or supports_model_diagnostics
    ):
        diagnostics_started_at = time.monotonic()
        if supports_act_diagnostics:
            act_diagnostics_result = {
                "scoring_splits": {},
                "first_uncertified_depth_rungs": {},
            }
        if supports_model_diagnostics:
            model_diagnostics_result = {
                "scoring_splits": {},
                "first_uncertified_depth_rungs": {},
            }
        previous_act_setting = getattr(model, "collect_act_diagnostics", None)
        previous_model_setting = getattr(
            model,
            "collect_model_diagnostics",
            None,
        )
        if supports_act_diagnostics:
            model.collect_act_diagnostics = True
        if supports_model_diagnostics:
            model.collect_model_diagnostics = True

        def collect_diagnostics(dataloader) -> tuple[dict | None, dict | None]:
            local_metrics = _evaluate(
                model,
                dataloader,
                manifest,
                device,
                deadline=float("inf"),
                budget_seconds=float("inf"),
                include_act_diagnostics=supports_act_diagnostics,
                include_model_diagnostics=supports_model_diagnostics,
            )
            return (
                local_metrics.get("act_diagnostics"),
                local_metrics.get("model_diagnostics"),
            )

        def store_diagnostics(
            *,
            group: str,
            key: str,
            act: dict | None,
            model_payload: dict | None,
            time_steps: int | None = None,
        ) -> None:
            if act is not None and act_diagnostics_result is not None:
                value = (
                    act
                    if time_steps is None
                    else {"time_steps": time_steps, "diagnostics": act}
                )
                act_diagnostics_result[group][key] = value
            if (
                model_payload is not None
                and model_diagnostics_result is not None
            ):
                value = (
                    model_payload
                    if time_steps is None
                    else {
                        "time_steps": time_steps,
                        "diagnostics": model_payload,
                    }
                )
                model_diagnostics_result[group][key] = value

        try:
            for split_name in _scoring_split_names(dataloaders):
                act, model_payload = collect_diagnostics(
                    dataloaders[split_name]
                )
                store_diagnostics(
                    group="scoring_splits",
                    key=split_name,
                    act=act,
                    model_payload=model_payload,
                )

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
                act, model_payload = collect_diagnostics(
                    dataloaders[split_name]
                )
                store_diagnostics(
                    group="first_uncertified_depth_rungs",
                    key=label,
                    act=act,
                    model_payload=model_payload,
                    time_steps=rung["time_steps"],
                )

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
            if supports_act_diagnostics:
                model.collect_act_diagnostics = previous_act_setting
            if supports_model_diagnostics:
                model.collect_model_diagnostics = previous_model_setting
        debug_diagnostics_seconds = time.monotonic() - diagnostics_started_at
    elif include_act_diagnostics:
        print(
            "DEBUG_DIAGNOSTICS_UNAVAILABLE | submission debug telemetry is "
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
    if debug_diagnostics_seconds is not None:
        # ACT and model telemetry are collected in the same forward pass, so
        # report one shared duration rather than two values that would be
        # mistaken for independent timings and double-counted.
        result["debug_diagnostics_seconds"] = debug_diagnostics_seconds
    if act_diagnostics_result is not None:
        result["act_diagnostics"] = act_diagnostics_result
    if model_diagnostics_result is not None:
        result["model_diagnostics"] = model_diagnostics_result
    if training_model_diagnostics is not None:
        training_summary = training_model_diagnostics.summary()
        if training_summary is not None:
            result["training_model_diagnostics"] = training_summary
    return result


def _load_submission_file_with_source(
    path: str | Path,
) -> tuple[Submission, str]:
    """Load a submission and capture the exact stable source around import."""

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
    source = submission_path.read_text(encoding="utf-8")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if submission_path.read_text(encoding="utf-8") != source:
        raise RuntimeError("submission.py changed while it was being imported")
    submission = getattr(module, "SUBMISSION", None)
    if not isinstance(submission, Submission):
        raise TypeError("submission.py must export benchmark.Submission as SUBMISSION")
    return submission, source


def _load_submission_file(path: str | Path) -> Submission:
    submission, _ = _load_submission_file_with_source(path)
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
    resolved_submission_path = Path(submission_path).resolve()
    experiments_path = resolved_submission_path.with_name("EXPERIMENTS.md")
    submission_load_started = time.monotonic()
    submission, submission_source = _load_submission_file_with_source(
        resolved_submission_path
    )
    validate_submission(submission)
    submission_debug_enabled = _submission_requests_act_diagnostics(submission)
    if submission_debug_enabled and not include_act_diagnostics:
        print(
            "DEBUG_DIAGNOSTICS_ENABLED | submission DBUG=True | "
            "training + evaluation telemetry",
            flush=True,
        )
    include_act_diagnostics = (
        include_act_diagnostics or submission_debug_enabled
    )
    if include_act_diagnostics:
        experiment_number = _read_experiment_number(experiments_path)
    else:
        experiment_number = None
        submission_source = None
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
    _emit_final_metrics(
        result,
        debug_metrics_to_file=include_act_diagnostics,
        experiment_number=experiment_number,
        experiments_path=(experiments_path if include_act_diagnostics else None),
        submission_path=(
            resolved_submission_path if include_act_diagnostics else None
        ),
        submission_source=submission_source,
    )
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
        help="collect local ACT, model, and training debug telemetry",
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
