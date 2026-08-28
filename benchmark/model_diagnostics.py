"""Local-only aggregation for model and segment diagnostics.

The submission computes small, detached summaries while the evaluator owns
aggregation and serialization.  Keeping raw activations out of the auxiliary
payload bounds debug-memory use even when a split spans many batches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


SEGMENT_NAMES = ("OTHER", "N", "X", "T", "ANS")
NUM_SEGMENTS = len(SEGMENT_NAMES)
STREAM_NAMES = ("prompt", "control", "work")
NUM_STREAMS = len(STREAM_NAMES)
COUNTERFACTUAL_NAMES = frozenset(
    {"zero", "permuted", "zero_nx", "zero_t", "swap_n_x"}
)
MAX_TRAINING_DIAGNOSTIC_RECORDS = 64
MAX_UPDATE_SNAPSHOT_ELEMENTS = 25_000_000

_SEGMENT_EMBEDDING_VECTOR_KEYS = (
    "norms",
    "delta_norms",
    "relative_deltas",
    "initial_cosines",
)
_PARAMETER_STAT_KEYS = ("norm", "delta_norm", "relative_delta")
_LAYER_SCALAR_KEYS = (
    "input_rms",
    "attention_update_rms",
    "attention_update_ratio",
    "mlp_update_rms",
    "mlp_update_ratio",
    "output_rms",
    "input_output_cosine",
    "attention_entropy",
    "effective_attended_tokens",
)
_NONNEGATIVE_LAYER_KEYS = frozenset(_LAYER_SCALAR_KEYS) - {
    "input_output_cosine"
}
_RMS_LAYER_KEYS = frozenset(
    {
        "input_rms",
        "attention_update_rms",
        "mlp_update_rms",
        "output_rms",
    }
)


def _as_scalar(value: object, *, name: str) -> float:
    if torch.is_tensor(value):
        if value.requires_grad:
            raise ValueError(f"model diagnostic {name} must be detached")
        if value.numel() != 1:
            raise ValueError(f"model diagnostic {name} must be scalar")
        result = float(value.detach().item())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        raise TypeError(f"model diagnostic {name} must be numeric")
    if not math.isfinite(result):
        raise ValueError(f"model diagnostic {name} must be finite")
    return result


def _as_nonnegative_scalar(value: object, *, name: str) -> float:
    result = _as_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"model diagnostic {name} must be nonnegative")
    return result


def _as_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"model diagnostic {name} must be a tensor")
    if value.requires_grad:
        raise ValueError(f"model diagnostic {name} must be detached")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"model diagnostic {name} must have shape {shape}"
        )
    result = value.detach().double().cpu()
    if not torch.isfinite(result).all().item():
        raise ValueError(f"model diagnostic {name} must be finite")
    return result


def _as_counts(value: object, *, name: str) -> torch.Tensor:
    result = _as_tensor(value, name=name, shape=(NUM_SEGMENTS,))
    if (result < 0).any().item() or (result != result.round()).any().item():
        raise ValueError(
            f"model diagnostic {name} must contain nonnegative whole numbers"
        )
    return result.long()


def _validate_nonnegative(values: torch.Tensor, *, name: str) -> None:
    if (values < 0).any().item():
        raise ValueError(f"model diagnostic {name} must be nonnegative")


def _validate_cosines(values: torch.Tensor, *, name: str) -> None:
    tolerance = 1e-5
    if ((values < -1.0 - tolerance) | (values > 1.0 + tolerance)).any().item():
        raise ValueError(f"model diagnostic {name} must be in [-1, 1]")


def _tensor_to_json(value: torch.Tensor) -> Any:
    return value.tolist()


class _WeightedMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0

    def add(self, value: float, weight: int) -> None:
        if weight < 0:
            raise ValueError("diagnostic aggregation weights must be nonnegative")
        self.total += value * weight
        self.weight += weight

    def value(self) -> float | None:
        if self.weight == 0:
            return None
        return self.total / self.weight


class _WeightedRms(_WeightedMean):
    """Pool per-group RMS values exactly from their group counts."""

    def add(self, value: float, weight: int) -> None:
        super().add(value * value, weight)

    def value(self) -> float | None:
        mean_square = super().value()
        return None if mean_square is None else math.sqrt(mean_square)


class _LayerAggregate:
    def __init__(self, step: int) -> None:
        self.step = step
        self.batch_count = 0
        self.valid_query_count = 0
        self.segment_query_counts = torch.zeros(NUM_SEGMENTS, dtype=torch.long)
        self.scalars = {
            key: (_WeightedRms() if key in _RMS_LAYER_KEYS else _WeightedMean())
            for key in _LAYER_SCALAR_KEYS
        }
        self.state_change = [_WeightedRms() for _ in range(NUM_SEGMENTS)]
        self.attention_mass = [
            [_WeightedMean() for _ in range(NUM_SEGMENTS)]
            for _ in range(NUM_SEGMENTS)
        ]
        self.attention_stream_mass: list[list[_WeightedMean]] | None = None

    def add(self, layer: dict[str, object]) -> None:
        prefix = f"layers[{self.step}]"
        valid_query_value = _as_nonnegative_scalar(
            layer["valid_query_count"],
            name=f"{prefix}.valid_query_count",
        )
        if not valid_query_value.is_integer() or valid_query_value < 1:
            raise ValueError(
                "model diagnostic layer valid_query_count must be a positive "
                "whole number"
            )
        valid_query_count = int(valid_query_value)
        segment_query_counts = _as_counts(
            layer["segment_query_counts"],
            name=f"{prefix}.segment_query_counts",
        )
        if int(segment_query_counts.sum().item()) != valid_query_count:
            raise ValueError(
                "model diagnostic segment_query_counts must sum to "
                "valid_query_count"
            )

        scalar_values = {}
        for key in _LAYER_SCALAR_KEYS:
            value = _as_scalar(layer[key], name=f"{prefix}.{key}")
            if key in _NONNEGATIVE_LAYER_KEYS and value < 0.0:
                raise ValueError(
                    f"model diagnostic {prefix}.{key} must be nonnegative"
                )
            if key == "input_output_cosine" and not -1.00001 <= value <= 1.00001:
                raise ValueError(
                    "model diagnostic input_output_cosine must be in [-1, 1]"
                )
            scalar_values[key] = value

        state_change = _as_tensor(
            layer["state_change_rms_by_segment"],
            name=f"{prefix}.state_change_rms_by_segment",
            shape=(NUM_SEGMENTS,),
        )
        _validate_nonnegative(
            state_change,
            name=f"{prefix}.state_change_rms_by_segment",
        )
        attention_mass = _as_tensor(
            layer["attention_mass_by_segment"],
            name=f"{prefix}.attention_mass_by_segment",
            shape=(NUM_SEGMENTS, NUM_SEGMENTS),
        )
        _validate_nonnegative(
            attention_mass,
            name=f"{prefix}.attention_mass_by_segment",
        )
        row_sums = attention_mass.sum(dim=1)
        populated = segment_query_counts > 0
        if populated.any().item() and not torch.allclose(
            row_sums[populated],
            torch.ones_like(row_sums[populated]),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError(
                "model diagnostic attention rows with queries must sum to one"
            )
        if (~populated).any().item() and not torch.allclose(
            attention_mass[~populated],
            torch.zeros_like(attention_mass[~populated]),
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError(
                "model diagnostic attention rows without queries must be zero"
            )
        stream_payload = layer.get("attention_mass_by_stream")
        attention_stream_mass = None
        if stream_payload is not None:
            attention_stream_mass = _as_tensor(
                stream_payload,
                name=f"{prefix}.attention_mass_by_stream",
                shape=(NUM_SEGMENTS, NUM_STREAMS),
            )
            _validate_nonnegative(
                attention_stream_mass,
                name=f"{prefix}.attention_mass_by_stream",
            )
            stream_row_sums = attention_stream_mass.sum(dim=1)
            if populated.any().item() and not torch.allclose(
                stream_row_sums[populated],
                torch.ones_like(stream_row_sums[populated]),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError(
                    "model diagnostic attention stream rows with queries must "
                    "sum to one"
                )
            if (~populated).any().item() and not torch.allclose(
                attention_stream_mass[~populated],
                torch.zeros_like(attention_stream_mass[~populated]),
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    "model diagnostic attention stream rows without queries "
                    "must be zero"
                )
        if self.batch_count and (
            (self.attention_stream_mass is None)
            != (attention_stream_mass is None)
        ):
            raise ValueError(
                "model diagnostic attention stream telemetry changed within a step"
            )
        if attention_stream_mass is not None and self.attention_stream_mass is None:
            self.attention_stream_mass = [
                [_WeightedMean() for _ in range(NUM_STREAMS)]
                for _ in range(NUM_SEGMENTS)
            ]

        self.batch_count += 1
        self.valid_query_count += valid_query_count
        self.segment_query_counts += segment_query_counts
        for key, value in scalar_values.items():
            self.scalars[key].add(value, valid_query_count)
        for query_segment in range(NUM_SEGMENTS):
            weight = int(segment_query_counts[query_segment].item())
            self.state_change[query_segment].add(
                float(state_change[query_segment].item()),
                weight,
            )
            for key_segment in range(NUM_SEGMENTS):
                self.attention_mass[query_segment][key_segment].add(
                    float(attention_mass[query_segment, key_segment].item()),
                    weight,
                )
            if attention_stream_mass is not None:
                assert self.attention_stream_mass is not None
                for stream in range(NUM_STREAMS):
                    self.attention_stream_mass[query_segment][stream].add(
                        float(attention_stream_mass[query_segment, stream].item()),
                        weight,
                    )

    def summary(self) -> dict[str, object]:
        stream_summary = None
        if self.attention_stream_mass is not None:
            stream_summary = [
                [cell.value() for cell in row]
                for row in self.attention_stream_mass
            ]
        return {
            "step": self.step,
            "batch_count": self.batch_count,
            "valid_query_count": self.valid_query_count,
            **{key: value.value() for key, value in self.scalars.items()},
            "segment_query_counts": self.segment_query_counts.tolist(),
            "attention_mass_by_segment": [
                [cell.value() for cell in row] for row in self.attention_mass
            ],
            "stream_names": list(STREAM_NAMES),
            "attention_mass_by_stream": stream_summary,
            "state_change_rms_by_segment": [
                value.value() for value in self.state_change
            ],
        }


class _PredictionAggregate:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.correct_examples = 0
        self.example_count = 0
        self.target_token_count = 0
        self.flipped_predictions = 0
        self.flipped_examples = 0

    def add(
        self,
        *,
        token_logits: torch.Tensor,
        token_targets: torch.Tensor,
        normal_token_predictions: torch.Tensor,
    ) -> None:
        valid = token_targets != -100
        rows_with_targets = valid.any(dim=1)
        if not valid.any().item():
            raise ValueError("model diagnostics batch contains no targets")
        predictions = token_logits.argmax(dim=-1)
        exact = ((predictions == token_targets) | ~valid).all(dim=1)
        self.loss_sum += float(
            F.cross_entropy(
                token_logits[valid].float(),
                token_targets[valid],
                reduction="sum",
            ).item()
        )
        self.correct_examples += int(exact[rows_with_targets].sum().item())
        self.example_count += int(rows_with_targets.sum().item())
        self.target_token_count += int(valid.sum().item())
        self.flipped_predictions += int(
            ((predictions != normal_token_predictions) & valid).sum().item()
        )
        self.flipped_examples += int(
            (
                ((predictions != normal_token_predictions) & valid).any(dim=1)
                & rows_with_targets
            ).sum().item()
        )

    def summary(self, *, include_flip_rate: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "task_cross_entropy": self.loss_sum / self.target_token_count,
            "exact_accuracy": self.correct_examples / self.example_count,
            "correct_examples": self.correct_examples,
            "example_count": self.example_count,
            "target_token_count": self.target_token_count,
        }
        if include_flip_rate:
            result["token_prediction_flip_rate"] = (
                self.flipped_predictions / self.target_token_count
            )
            result["example_prediction_flip_rate"] = (
                self.flipped_examples / self.example_count
            )
        return result


def _select_target_logits(
    logits: torch.Tensor,
    *,
    targets: torch.Tensor,
    target_positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_positions is None:
        return logits[:, :-1, :], targets[:, 1:]
    batch_indices = torch.arange(logits.shape[0], device=logits.device)[:, None]
    return logits[batch_indices, target_positions.clamp_min(0)], targets


class ModelDiagnosticsAccumulator:
    """Aggregate detached model telemetry for one evaluator-owned split."""

    def __init__(self) -> None:
        self._batch_count = 0
        self._segment_token_counts = torch.zeros(
            NUM_SEGMENTS,
            dtype=torch.long,
        )
        self._final_state_rms = [_WeightedRms() for _ in range(NUM_SEGMENTS)]
        self._final_logit_entropy = [
            _WeightedMean() for _ in range(NUM_SEGMENTS)
        ]
        self._layers: dict[int, _LayerAggregate] = {}
        self._segment_embedding_first: dict[str, torch.Tensor] | None = None
        self._segment_embedding_last: dict[str, torch.Tensor] | None = None
        self._parameter_stats_first: dict[str, dict[str, float]] | None = None
        self._parameter_stats_last: dict[str, dict[str, float]] | None = None
        self._stage_predictions: dict[int, _PredictionAggregate] = {}
        self._counterfactuals: dict[str, _PredictionAggregate] = {}

    @property
    def has_data(self) -> bool:
        return self._batch_count > 0

    def add(
        self,
        *,
        auxiliary: object,
        logits: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        target_positions: torch.Tensor | None = None,
    ) -> None:
        """Validate and add one batch's compact diagnostics payload."""

        if not isinstance(auxiliary, dict):
            return
        payload = auxiliary.get("model_diagnostics")
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise TypeError(
                "auxiliary['model_diagnostics'] must be a dictionary or None"
            )

        required = {
            "segment_token_counts",
            "segment_embedding",
            "parameter_stats",
            "final_state_rms_by_segment",
            "final_logit_entropy_by_segment",
            "layers",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(
                "model diagnostics are missing: " + ", ".join(sorted(missing))
            )

        counts = _as_counts(
            payload["segment_token_counts"],
            name="segment_token_counts",
        )
        if int(counts.sum().item()) < 1:
            raise ValueError(
                "model diagnostic segment_token_counts must include a valid token"
            )
        final_state_rms = _as_tensor(
            payload["final_state_rms_by_segment"],
            name="final_state_rms_by_segment",
            shape=(NUM_SEGMENTS,),
        )
        _validate_nonnegative(
            final_state_rms,
            name="final_state_rms_by_segment",
        )
        final_logit_entropy = _as_tensor(
            payload["final_logit_entropy_by_segment"],
            name="final_logit_entropy_by_segment",
            shape=(NUM_SEGMENTS,),
        )
        _validate_nonnegative(
            final_logit_entropy,
            name="final_logit_entropy_by_segment",
        )

        segment_embedding = payload["segment_embedding"]
        if not isinstance(segment_embedding, dict):
            raise TypeError("model diagnostic segment_embedding must be a dictionary")
        embedding_snapshot = {}
        for key in _SEGMENT_EMBEDDING_VECTOR_KEYS:
            values = _as_tensor(
                segment_embedding.get(key),
                name=f"segment_embedding.{key}",
                shape=(NUM_SEGMENTS,),
            )
            if key in {"norms", "delta_norms", "relative_deltas"}:
                _validate_nonnegative(
                    values,
                    name=f"segment_embedding.{key}",
                )
            else:
                _validate_cosines(
                    values,
                    name=f"segment_embedding.{key}",
                )
            embedding_snapshot[key] = values
        cosine_matrix = _as_tensor(
            segment_embedding.get("cosine_matrix"),
            name="segment_embedding.cosine_matrix",
            shape=(NUM_SEGMENTS, NUM_SEGMENTS),
        )
        _validate_cosines(
            cosine_matrix,
            name="segment_embedding.cosine_matrix",
        )
        embedding_snapshot["cosine_matrix"] = cosine_matrix

        parameter_stats = payload["parameter_stats"]
        if not isinstance(parameter_stats, dict) or not parameter_stats:
            raise TypeError(
                "model diagnostic parameter_stats must be a nonempty dictionary"
            )
        parameter_snapshot = {}
        for name, stats in parameter_stats.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "model diagnostic parameter_stats names must be nonempty strings"
                )
            if not isinstance(stats, dict):
                raise TypeError(
                    f"model diagnostic parameter_stats[{name!r}] must be a dictionary"
                )
            parameter_snapshot[name] = {
                key: _as_nonnegative_scalar(
                    stats.get(key),
                    name=f"parameter_stats.{name}.{key}",
                )
                for key in _PARAMETER_STAT_KEYS
            }
        if (
            self._parameter_stats_first is not None
            and set(parameter_snapshot) != set(self._parameter_stats_first)
        ):
            raise ValueError(
                "model diagnostic parameter_stats names changed within a split"
            )

        layers = payload["layers"]
        if not isinstance(layers, (list, tuple)):
            raise TypeError("model diagnostic layers must be a list or tuple")
        seen_steps = set()
        validated_layers = []
        required_layer_keys = {
            "step",
            "valid_query_count",
            "segment_query_counts",
            "attention_mass_by_segment",
            "state_change_rms_by_segment",
            *_LAYER_SCALAR_KEYS,
        }
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise TypeError(
                    f"model diagnostic layers[{index}] must be a dictionary"
                )
            missing_layer = required_layer_keys - set(layer)
            if missing_layer:
                raise ValueError(
                    f"model diagnostics layers[{index}] are missing: "
                    + ", ".join(sorted(missing_layer))
                )
            step = layer["step"]
            if type(step) is not int or step < 1:
                raise ValueError(
                    "model diagnostic layer step must be a positive integer"
                )
            if step in seen_steps:
                raise ValueError(
                    "model diagnostic layer steps must be unique within a batch"
                )
            seen_steps.add(step)
            validated_layers.append((step, layer))

        stage_payload = payload.get("stage_logits")
        counterfactual_payload = payload.get("segment_counterfactual_logits")
        validated_stage_logits: list[tuple[int, torch.Tensor]] = []
        validated_counterfactual_logits: dict[str, torch.Tensor] = {}
        if stage_payload is not None or counterfactual_payload is not None:
            if logits is None or targets is None:
                raise ValueError(
                    "prediction diagnostics require logits and targets from the runner"
                )
            if logits.ndim != 3:
                raise ValueError("normal diagnostic logits must be rank three")
        if stage_payload is not None:
            if not isinstance(stage_payload, (list, tuple)):
                raise TypeError("model diagnostic stage_logits must be a list or tuple")
            seen_prediction_steps = set()
            for index, record in enumerate(stage_payload):
                if not isinstance(record, dict):
                    raise TypeError(
                        f"model diagnostic stage_logits[{index}] must be a dictionary"
                    )
                step = record.get("step")
                if type(step) is not int or step < 0:
                    raise ValueError(
                        "model diagnostic stage-logit step must be a "
                        "nonnegative integer"
                    )
                if step in seen_prediction_steps:
                    raise ValueError(
                        "model diagnostic stage-logit steps must be unique "
                        "within a batch"
                    )
                seen_prediction_steps.add(step)
                stage_logits = record.get("logits")
                if not torch.is_tensor(stage_logits):
                    raise TypeError(
                        f"model diagnostic stage_logits[{index}].logits must be a tensor"
                    )
                if stage_logits.requires_grad:
                    raise ValueError("model diagnostic stage logits must be detached")
                if stage_logits.shape != logits.shape:
                    raise ValueError(
                        "model diagnostic stage logits must match normal logits shape"
                    )
                if not torch.isfinite(stage_logits).all().item():
                    raise ValueError("model diagnostic stage logits must be finite")
                validated_stage_logits.append((step, stage_logits.detach()))
        if counterfactual_payload is not None:
            if not isinstance(counterfactual_payload, dict):
                raise TypeError(
                    "model diagnostic segment_counterfactual_logits must be a dictionary"
                )
            if set(counterfactual_payload) != COUNTERFACTUAL_NAMES:
                raise ValueError(
                    "model diagnostic segment counterfactuals must be exactly "
                    "'zero', 'permuted', 'zero_nx', 'zero_t', and 'swap_n_x'"
                )
            for name, counterfactual_logits in counterfactual_payload.items():
                if not torch.is_tensor(counterfactual_logits):
                    raise TypeError(
                        f"model diagnostic counterfactual {name} logits must be a tensor"
                    )
                if counterfactual_logits.requires_grad:
                    raise ValueError(
                        "model diagnostic counterfactual logits must be detached"
                    )
                if counterfactual_logits.shape != logits.shape:
                    raise ValueError(
                        "model diagnostic counterfactual logits must match normal "
                        "logits shape"
                    )
                if not torch.isfinite(counterfactual_logits).all().item():
                    raise ValueError(
                        "model diagnostic counterfactual logits must be finite"
                    )
                validated_counterfactual_logits[name] = (
                    counterfactual_logits.detach()
                )

        if self._segment_embedding_first is None:
            self._segment_embedding_first = embedding_snapshot
        if self._parameter_stats_first is None:
            self._parameter_stats_first = parameter_snapshot
        self._segment_embedding_last = embedding_snapshot
        self._parameter_stats_last = parameter_snapshot
        self._batch_count += 1
        self._segment_token_counts += counts
        for segment in range(NUM_SEGMENTS):
            weight = int(counts[segment].item())
            self._final_state_rms[segment].add(
                float(final_state_rms[segment].item()),
                weight,
            )
            self._final_logit_entropy[segment].add(
                float(final_logit_entropy[segment].item()),
                weight,
            )
        for step, layer in validated_layers:
            aggregate = self._layers.setdefault(step, _LayerAggregate(step))
            aggregate.add(layer)
        if validated_stage_logits or validated_counterfactual_logits:
            assert logits is not None
            assert targets is not None
            _, token_targets = _select_target_logits(
                logits,
                targets=targets,
                target_positions=target_positions,
            )
            normal_token_logits, _ = _select_target_logits(
                logits,
                targets=targets,
                target_positions=target_positions,
            )
            normal_predictions = normal_token_logits.argmax(dim=-1)
            for step, stage_logits in validated_stage_logits:
                stage_token_logits, _ = _select_target_logits(
                    stage_logits,
                    targets=targets,
                    target_positions=target_positions,
                )
                self._stage_predictions.setdefault(
                    step,
                    _PredictionAggregate(),
                ).add(
                    token_logits=stage_token_logits,
                    token_targets=token_targets,
                    normal_token_predictions=normal_predictions,
                )
            for name, counterfactual_logits in validated_counterfactual_logits.items():
                counterfactual_token_logits, _ = _select_target_logits(
                    counterfactual_logits,
                    targets=targets,
                    target_positions=target_positions,
                )
                self._counterfactuals.setdefault(
                    name,
                    _PredictionAggregate(),
                ).add(
                    token_logits=counterfactual_token_logits,
                    token_targets=token_targets,
                    normal_token_predictions=normal_predictions,
                )

    def summary(self) -> dict[str, object] | None:
        if not self.has_data:
            return None
        assert self._segment_embedding_first is not None
        assert self._segment_embedding_last is not None
        assert self._parameter_stats_first is not None
        assert self._parameter_stats_last is not None

        def embedding_json(snapshot: dict[str, torch.Tensor]) -> dict[str, Any]:
            return {
                key: _tensor_to_json(value) for key, value in snapshot.items()
            }

        return {
            "segment_names": list(SEGMENT_NAMES),
            "batch_count": self._batch_count,
            "valid_token_count": int(self._segment_token_counts.sum().item()),
            "segment_token_counts": self._segment_token_counts.tolist(),
            "segment_embedding": {
                "first_batch": embedding_json(self._segment_embedding_first),
                "last_batch": embedding_json(self._segment_embedding_last),
            },
            "parameter_stats": {
                name: {
                    "first_batch": self._parameter_stats_first[name],
                    "last_batch": self._parameter_stats_last[name],
                }
                for name in self._parameter_stats_first
            },
            "final_state_rms_by_segment": [
                value.value() for value in self._final_state_rms
            ],
            "final_logit_entropy_by_segment": [
                value.value() for value in self._final_logit_entropy
            ],
            "layers": [
                self._layers[step].summary() for step in sorted(self._layers)
            ],
            "stage_predictions": [
                {
                    "step": step,
                    **self._stage_predictions[step].summary(
                        include_flip_rate=False
                    ),
                }
                for step in sorted(self._stage_predictions)
            ],
            "segment_counterfactuals": {
                name: accumulator.summary(include_flip_rate=True)
                for name, accumulator in sorted(self._counterfactuals.items())
            },
        }


def _squared_norm(value: torch.Tensor) -> float:
    if value.is_sparse:
        value = value.coalesce().values()
    return float(value.detach().float().square().sum().item())


def _row_norms(value: torch.Tensor) -> list[float]:
    if value.is_sparse:
        value = value.to_dense()
    rows = value.detach().float().reshape(value.shape[0], -1)
    return rows.norm(dim=1).cpu().tolist()


def _qkv_component_norms(value: torch.Tensor) -> dict[str, float] | None:
    if value.ndim < 1 or value.shape[0] % 3:
        return None
    if value.is_sparse:
        value = value.to_dense()
    q, k, v = value.detach().float().chunk(3, dim=0)
    return {
        "q": float(q.norm().item()),
        "k": float(k.norm().item()),
        "v": float(v.norm().item()),
    }


def summarize_parameter_gradients(model: nn.Module) -> dict[str, object]:
    """Return detached gradient magnitudes without retaining gradient tensors."""

    parameter_stats = {}
    parameter_norm_squared = 0.0
    gradient_norm_squared = 0.0
    trainable_parameter_count = 0
    gradient_parameter_count = 0
    trainable_element_count = 0
    gradient_element_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_parameter_count += 1
        trainable_element_count += parameter.numel()
        parameter_squared = _squared_norm(parameter)
        parameter_norm_squared += parameter_squared
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_parameter_count += 1
        gradient_element_count += gradient.numel()
        gradient_squared = _squared_norm(gradient)
        gradient_norm_squared += gradient_squared
        parameter_norm = math.sqrt(parameter_squared)
        gradient_norm = math.sqrt(gradient_squared)
        stats = {
            "gradient_norm": gradient_norm,
            "parameter_norm": parameter_norm,
            "relative_gradient_norm": (
                gradient_norm / max(parameter_norm, 1e-12)
            ),
        }
        if name == "segment_embedding.weight":
            stats["row_gradient_norms"] = _row_norms(gradient)
        if name.endswith(".qkv.weight") or name.endswith(".qkv.bias"):
            component_norms = _qkv_component_norms(gradient)
            if component_norms is not None:
                stats["component_gradient_norms"] = component_norms
        parameter_stats[name] = stats

    parameter_norm = math.sqrt(parameter_norm_squared)
    gradient_norm = math.sqrt(gradient_norm_squared)
    return {
        "gradient_norm": gradient_norm,
        "parameter_norm": parameter_norm,
        "relative_gradient_norm": gradient_norm / max(parameter_norm, 1e-12),
        "trainable_parameter_count": trainable_parameter_count,
        "gradient_parameter_count": gradient_parameter_count,
        "missing_gradient_parameter_count": (
            trainable_parameter_count - gradient_parameter_count
        ),
        "trainable_element_count": trainable_element_count,
        "gradient_element_count": gradient_element_count,
        "parameters": parameter_stats,
    }


@dataclass
class ParameterSnapshot:
    values: dict[str, torch.Tensor]
    element_count: int


def capture_parameter_snapshot(
    model: nn.Module,
    *,
    maximum_elements: int = MAX_UPDATE_SNAPSHOT_ELEMENTS,
) -> ParameterSnapshot | None:
    """Clone trainable parameters for one debug-only optimizer delta."""

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    element_count = sum(parameter.numel() for _, parameter in named_parameters)
    if element_count > maximum_elements:
        return None
    return ParameterSnapshot(
        values={
            name: parameter.detach().clone()
            for name, parameter in named_parameters
        },
        element_count=element_count,
    )


def summarize_parameter_updates(
    model: nn.Module,
    snapshot: ParameterSnapshot | None,
) -> dict[str, object]:
    """Compare post-step parameters with a pre-step debug snapshot."""

    if snapshot is None:
        return {
            "available": False,
            "reason": (
                "trainable model exceeds the debug update-snapshot element cap"
            ),
            "maximum_snapshot_elements": MAX_UPDATE_SNAPSHOT_ELEMENTS,
        }

    parameter_stats = {}
    before_norm_squared = 0.0
    after_norm_squared = 0.0
    update_norm_squared = 0.0
    current_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if current_names != set(snapshot.values):
        raise ValueError("trainable parameter names changed during optimizer.step")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        before = snapshot.values[name]
        if before.shape != parameter.shape:
            raise ValueError(
                f"trainable parameter {name!r} changed shape during optimizer.step"
            )
        before_squared = _squared_norm(before)
        after_squared = _squared_norm(parameter)
        update_squared = _squared_norm(parameter.detach() - before)
        before_norm_squared += before_squared
        after_norm_squared += after_squared
        update_norm_squared += update_squared
        before_norm = math.sqrt(before_squared)
        update_norm = math.sqrt(update_squared)
        stats = {
            "before_norm": before_norm,
            "after_norm": math.sqrt(after_squared),
            "update_norm": update_norm,
            "relative_update_norm": update_norm / max(before_norm, 1e-12),
        }
        update = parameter.detach() - before
        if name == "segment_embedding.weight":
            row_update_norms = _row_norms(update)
            row_before_norms = _row_norms(before)
            stats["row_update_norms"] = row_update_norms
            stats["row_relative_update_norms"] = [
                update_norm / max(before_norm, 1e-12)
                for update_norm, before_norm in zip(
                    row_update_norms,
                    row_before_norms,
                )
            ]
        if name.endswith(".qkv.weight") or name.endswith(".qkv.bias"):
            component_norms = _qkv_component_norms(update)
            if component_norms is not None:
                stats["component_update_norms"] = component_norms
        parameter_stats[name] = stats

    before_norm = math.sqrt(before_norm_squared)
    update_norm = math.sqrt(update_norm_squared)
    return {
        "available": True,
        "element_count": snapshot.element_count,
        "before_parameter_norm": before_norm,
        "after_parameter_norm": math.sqrt(after_norm_squared),
        "update_norm": update_norm,
        "relative_update_norm": update_norm / max(before_norm, 1e-12),
        "parameters": parameter_stats,
    }


def validate_training_credit(payload: object) -> dict[str, object] | None:
    """Validate and JSON-normalize optional retained-state gradient credit."""

    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("training gradient diagnostics must be a dictionary")
    required = {
        "segment_token_counts",
        "segment_signal_grad_rms",
        "segment_signal_grad_rms_by_segment",
        "stages",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "training gradient diagnostics are missing: "
            + ", ".join(sorted(missing))
        )
    segment_counts = _as_counts(
        payload["segment_token_counts"],
        name="training_credit.segment_token_counts",
    )
    segment_signal_rms = _as_nonnegative_scalar(
        payload["segment_signal_grad_rms"],
        name="training_credit.segment_signal_grad_rms",
    )
    segment_signal = _as_tensor(
        payload["segment_signal_grad_rms_by_segment"],
        name="training_credit.segment_signal_grad_rms_by_segment",
        shape=(NUM_SEGMENTS,),
    )
    _validate_nonnegative(
        segment_signal,
        name="training_credit.segment_signal_grad_rms_by_segment",
    )
    stages = payload["stages"]
    if not isinstance(stages, (list, tuple)):
        raise TypeError("training gradient diagnostic stages must be a list or tuple")
    normalized_stages = []
    seen_steps = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise TypeError(
                f"training gradient diagnostic stages[{index}] must be a dictionary"
            )
        required_stage = {
            "step",
            "state_grad_rms",
            "relative_to_final",
            "state_grad_rms_by_segment",
            "control_grad_rms",
        }
        missing_stage = required_stage - set(stage)
        if missing_stage:
            raise ValueError(
                f"training gradient diagnostics stages[{index}] are missing: "
                + ", ".join(sorted(missing_stage))
            )
        step = stage["step"]
        if type(step) is not int or step < 0:
            raise ValueError(
                "training gradient diagnostic stage step must be nonnegative"
            )
        if step in seen_steps:
            raise ValueError(
                "training gradient diagnostic stage steps must be unique"
            )
        seen_steps.add(step)
        state_grad_rms = _as_nonnegative_scalar(
            stage["state_grad_rms"],
            name=f"training_credit.stages[{index}].state_grad_rms",
        )
        relative_to_final = _as_nonnegative_scalar(
            stage["relative_to_final"],
            name=f"training_credit.stages[{index}].relative_to_final",
        )
        state_by_segment = _as_tensor(
            stage["state_grad_rms_by_segment"],
            name=(
                f"training_credit.stages[{index}]."
                "state_grad_rms_by_segment"
            ),
            shape=(NUM_SEGMENTS,),
        )
        _validate_nonnegative(
            state_by_segment,
            name=(
                f"training_credit.stages[{index}]."
                "state_grad_rms_by_segment"
            ),
        )
        control = stage["control_grad_rms"]
        control_grad_rms = (
            None
            if control is None
            else _as_nonnegative_scalar(
                control,
                name=f"training_credit.stages[{index}].control_grad_rms",
            )
        )
        normalized_stages.append(
            {
                "step": step,
                "state_grad_rms": state_grad_rms,
                "relative_to_final": relative_to_final,
                "state_grad_rms_by_segment": state_by_segment.tolist(),
                "control_grad_rms": control_grad_rms,
            }
        )
    return {
        "segment_names": list(SEGMENT_NAMES),
        "segment_token_counts": segment_counts.tolist(),
        "segment_signal_grad_rms": segment_signal_rms,
        "segment_signal_grad_rms_by_segment": segment_signal.tolist(),
        "stages": sorted(normalized_stages, key=lambda item: item["step"]),
    }


class TrainingModelDiagnosticsTrajectory:
    """Bounded post-update model checkpoints and optimizer-step telemetry."""

    def __init__(
        self,
        max_records: int = MAX_TRAINING_DIAGNOSTIC_RECORDS,
    ) -> None:
        if max_records < 2:
            raise ValueError("training diagnostic max_records must be at least two")
        self._max_records = max_records
        self._records: list[tuple[int, dict[str, object]]] = []
        self._seen = 0
        self._stride = 1
        self._last_record: tuple[int, dict[str, object]] | None = None

    @property
    def has_data(self) -> bool:
        return self._last_record is not None

    def record(self, record: dict[str, object]) -> None:
        step = record.get("step")
        if type(step) is not int or step < 1:
            raise ValueError("training model diagnostic step must be positive")
        if self._last_record is not None:
            previous_step = self._last_record[1]["step"]
            if step <= previous_step:
                raise ValueError(
                    "training model diagnostic steps must strictly increase"
                )
        self._seen += 1
        indexed = (self._seen, record)
        self._last_record = indexed
        if self._seen == 1 or self._seen % self._stride == 0:
            self._records.append(indexed)
        while len(self._records) > self._max_records:
            self._stride *= 2
            self._records = [
                item
                for item in self._records
                if item[0] == 1 or item[0] % self._stride == 0
            ]

    def summary(self) -> dict[str, object] | None:
        if self._last_record is None:
            return None
        records = list(self._records)
        if not records or records[-1][0] != self._last_record[0]:
            if len(records) >= self._max_records:
                records.pop(-1)
            records.append(self._last_record)
        return {
            "checkpoint_phase": "post_optimizer_on_training_batch",
            "gradient_phase": "same_step_final_backward_pre_and_post_clip",
            "optimizer_update_phase": "same_step_pre_to_post_optimizer",
            "max_records": self._max_records,
            "observed_checkpoints": self._seen,
            "records": [record for _, record in records],
        }
