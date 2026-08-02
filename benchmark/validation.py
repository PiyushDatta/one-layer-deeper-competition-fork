"""Structural checks for official submissions and their runtime state."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import nn

from .api import OptimizerBundle, Submission, model_state_tensors
from .manifest import ModelStateSpec
from submission_validation import validate_submission_source


def lint_submission_source(path: Path) -> None:
    """Apply the shared source policy before importing an evaluator temp file."""

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("submission.py must be UTF-8") from exc
    validate_submission_source(path.name, source, 256 * 1024, required_filename=None)


def validate_submission(submission: Submission) -> None:
    if not callable(submission.build_model) or not callable(submission.build_optimizer):
        raise TypeError("submission factories must be callable")
    if submission.training_loss is not None and not callable(submission.training_loss):
        raise TypeError("submission training_loss must be callable when provided")
    if submission.token_training_loss is not None and not callable(
        submission.token_training_loss
    ):
        raise TypeError(
            "submission token_training_loss must be callable when provided"
        )
    if (
        submission.training_loss is not None
        and submission.token_training_loss is not None
    ):
        raise ValueError(
            "submission cannot define both training_loss and token_training_loss"
        )
    for name in ("batch_size", "eval_batch_size", "max_steps"):
        value = getattr(submission, name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"submission {name} must be a positive integer when provided")


def validate_model_state(
    model: nn.Module,
    state_spec: ModelStateSpec,
    device: torch.device,
) -> int:
    state = list(model_state_tensors(model))
    if not state:
        raise ValueError("model must contain persistent parameter or buffer state")
    wrong_device = [name for name, value in state if value.device != device]
    if wrong_device:
        raise ValueError(f"model state is not on {device}: {wrong_device[:5]}")
    elements = sum(value.numel() for _, value in state)
    if elements > state_spec.maximum_elements:
        raise ValueError(
            f"model persistent state ({elements:,}) exceeds maximum ({state_spec.maximum_elements:,})"
        )
    return elements


def validate_optimizer(
    bundle: OptimizerBundle,
    model: nn.Module,
    device: torch.device,
) -> int:
    if not isinstance(bundle, OptimizerBundle):
        raise TypeError("build_optimizer must return benchmark.OptimizerBundle")
    optimizer = bundle.optimizer
    for method_name in ("zero_grad", "step", "state_dict"):
        if not callable(getattr(optimizer, method_name, None)):
            raise TypeError(
                "OptimizerBundle.optimizer must implement zero_grad, step, and state_dict"
            )
    if not isinstance(getattr(optimizer, "param_groups", None), list):
        raise TypeError("OptimizerBundle.optimizer must expose a param_groups list")
    if bundle.scheduler is not None and not callable(
        getattr(bundle.scheduler, "step", None)
    ):
        raise TypeError("OptimizerBundle.scheduler must expose step()")

    expected = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    actual = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected_ids = Counter(id(parameter) for parameter in expected)
    actual_ids = Counter(id(parameter) for parameter in actual)
    if expected_ids != actual_ids:
        raise ValueError(
            "optimizer must contain every trainable model parameter exactly once"
        )

    child_optimizers = getattr(optimizer, "optimizers", [optimizer])
    state_elements = 0
    for child_optimizer in child_optimizers:
        child_state = getattr(child_optimizer, "state", {})
        for state in child_state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    if value.device != device:
                        raise ValueError(f"optimizer state tensor is not on {device}")
                    state_elements += value.numel()
    return state_elements


def capture_state_versions(model: nn.Module) -> dict[int, int]:
    return {id(value): value._version for _, value in model_state_tensors(model)}


def assert_state_versions_unchanged(model: nn.Module, before: dict[int, int]) -> None:
    after = capture_state_versions(model)
    if after != before:
        raise ValueError(
            "model parameters or persistent buffers changed during evaluation"
        )
