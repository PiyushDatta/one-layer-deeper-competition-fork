"""The complete public API available to an official submission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class Scheduler(Protocol):
    def step(self) -> None: ...


@dataclass(frozen=True)
class ModelSpec:
    """Public shape and state limits for the current benchmark task."""

    vocab_size: int
    max_seq_len: int
    maximum_model_state_elements: int


def model_state_tensors(model: nn.Module):
    """Yield every distinct parameter and persistent buffer that counts."""

    seen: set[int] = set()
    for name, value in model.named_parameters():
        if id(value) not in seen:
            seen.add(id(value))
            yield f"parameter:{name}", value
    for module_name, child in model.named_modules():
        for name, value in child._buffers.items():
            if value is None or name in child._non_persistent_buffers_set:
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            full_name = f"{module_name}.{name}" if module_name else name
            yield f"buffer:{full_name}", value


def count_model_state_elements(model: nn.Module) -> int:
    """Count scalar elements in all inference-persistent model state."""

    return sum(value.numel() for _, value in model_state_tensors(model))


def assert_model_state(model: nn.Module, spec: ModelSpec) -> int:
    """Assert the public state budget and return the measured element count.

    Submissions should call this at the end of ``build_model`` for an immediate,
    readable failure.  The evaluator repeats the count independently after
    moving the model to the evaluation device.
    """

    elements = count_model_state_elements(model)
    if elements > spec.maximum_model_state_elements:
        raise AssertionError(
            f"model persistent state ({elements:,}) exceeds maximum "
            f"({spec.maximum_model_state_elements:,})"
        )
    return elements


@dataclass(frozen=True)
class OptimizerSpec:
    """Public, data-independent information available to an optimizer."""

    training_time_seconds: float
    device_type: str


@dataclass(frozen=True)
class OptimizerBundle:
    """Participant optimizer and optional participant-defined LR scheduler."""

    optimizer: torch.optim.Optimizer
    scheduler: Scheduler | None = None


@dataclass(frozen=True)
class TokenLossBatch:
    """Boundary-preserving inputs for a participant-defined token loss.

    Invalid target slots remain padded. ``valid_mask`` is authoritative and
    must be used to exclude their labels, logits, and target positions.
    """

    logits: torch.Tensor
    labels: torch.Tensor
    valid_mask: torch.Tensor
    target_positions: torch.Tensor | None
    auxiliary: object


@dataclass(frozen=True)
class Submission:
    """Participant-controlled components exported as ``SUBMISSION``."""

    build_model: Callable[[ModelSpec], nn.Module]
    build_optimizer: Callable[[nn.Module, OptimizerSpec], OptimizerBundle]
    training_loss: (
        Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor] | None
    ) = None
    batch_size: int | None = None
    max_steps: int | None = None
    eval_batch_size: int | None = None
    token_training_loss: Callable[[TokenLossBatch], torch.Tensor] | None = None
