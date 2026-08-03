"""Strict architecture-and-optimizer benchmark interface."""

from .api import (
    BackwardPassContext,
    BatchReuseContext,
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
    count_model_state_elements,
)

__all__ = [
    "BackwardPassContext",
    "BatchReuseContext",
    "ModelSpec",
    "OptimizerBundle",
    "OptimizerSpec",
    "Submission",
    "TokenLossBatch",
    "assert_model_state",
    "count_model_state_elements",
]
