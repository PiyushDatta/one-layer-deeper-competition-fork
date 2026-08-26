"""Local-only aggregation for adaptive-computation diagnostics."""

from __future__ import annotations

import math

import torch


_ACT_AUXILIARY_KEYS = frozenset(
    {
        "update_counts",
        "remainders",
        "cap_forced_mask",
        "max_loops",
        "global_iterations",
        "ponder_weight",
    }
)
MAX_ITERATION_DETAIL = 256


def _nearest_rank(values: torch.Tensor, quantile: float) -> float | None:
    if values.numel() == 0:
        return None
    ordered = values.sort().values
    index = max(0, math.ceil(quantile * ordered.numel()) - 1)
    return float(ordered[index].item())


def _median(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    ordered = values.sort().values
    midpoint = ordered.numel() // 2
    if ordered.numel() % 2:
        return float(ordered[midpoint].item())
    return float(
        ordered[midpoint - 1 : midpoint + 1].double().mean().item()
    )


def _distribution(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "mean": float(values.double().mean().item()),
        "median": _median(values),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "maximum": float(values.max().item()),
    }


class ActDiagnosticsAccumulator:
    """Collect detached ACT telemetry for one evaluator-owned split."""

    def __init__(self) -> None:
        self._update_counts: list[torch.Tensor] = []
        self._remainders: list[torch.Tensor] = []
        self._cap_forced: list[torch.Tensor] = []
        self._per_example_updates: list[torch.Tensor] = []
        self._per_example_ponder: list[torch.Tensor] = []
        self._per_example_reached_cap: list[torch.Tensor] = []
        self._per_example_forced_cap: list[torch.Tensor] = []
        self._lengths: list[torch.Tensor] = []
        self._correct: list[torch.Tensor] = []
        self._global_iterations: list[int] = []
        self._batch_reached_cap: list[bool] = []
        self._batch_forced_cap: list[bool] = []
        self._max_loops: int | None = None
        self._ponder_weight: float | None = None

    @property
    def has_data(self) -> bool:
        return bool(self._update_counts)

    def add(
        self,
        *,
        auxiliary: object,
        attention_mask: torch.Tensor,
        exact_rows: torch.Tensor,
        rows_with_targets: torch.Tensor,
    ) -> None:
        """Validate and retain one batch without holding an autograd graph."""

        if not isinstance(auxiliary, dict) or auxiliary.get("act") is None:
            return
        act = auxiliary["act"]
        if not isinstance(act, dict):
            raise TypeError("ACT diagnostics auxiliary['act'] must be a dictionary")
        missing = _ACT_AUXILIARY_KEYS - set(act)
        if missing:
            raise ValueError(
                "ACT diagnostics are missing: " + ", ".join(sorted(missing))
            )

        max_loops = act["max_loops"]
        global_iterations = act["global_iterations"]
        ponder_weight = act["ponder_weight"]
        if type(max_loops) is not int or max_loops < 1:
            raise ValueError("ACT max_loops must be one positive integer")
        if (
            type(global_iterations) is not int
            or not 1 <= global_iterations <= max_loops
        ):
            raise ValueError(
                "ACT global_iterations must be between one and max_loops"
            )
        if not isinstance(ponder_weight, (int, float)) or not math.isfinite(
            float(ponder_weight)
        ):
            raise ValueError("ACT ponder_weight must be finite")
        if self._max_loops is not None and max_loops != self._max_loops:
            raise ValueError("ACT max_loops changed within one evaluation split")
        if (
            self._ponder_weight is not None
            and float(ponder_weight) != self._ponder_weight
        ):
            raise ValueError("ACT ponder_weight changed within one evaluation split")
        self._max_loops = max_loops
        self._ponder_weight = float(ponder_weight)

        expected_shape = attention_mask.shape
        tensors = {
            "update_counts": act["update_counts"],
            "remainders": act["remainders"],
            "cap_forced_mask": act["cap_forced_mask"],
        }
        for name, value in tensors.items():
            if not torch.is_tensor(value) or value.shape != expected_shape:
                raise ValueError(
                    f"ACT {name} must have shape {tuple(expected_shape)}"
                )
            if value.requires_grad:
                raise ValueError(f"ACT {name} must be detached")
        if tensors["cap_forced_mask"].dtype != torch.bool:
            raise TypeError("ACT cap_forced_mask must have boolean dtype")
        if exact_rows.shape != rows_with_targets.shape:
            raise ValueError("exact_rows and rows_with_targets must align by batch row")

        valid = attention_mask.bool()
        updates = tensors["update_counts"].detach()
        remainders = tensors["remainders"].detach()
        cap_forced = tensors["cap_forced_mask"].detach()
        if not valid.any().item():
            return
        valid_updates = updates[valid].float()
        valid_remainders = remainders[valid].float()
        valid_cap_forced = cap_forced[valid].bool()
        if (
            not torch.isfinite(valid_updates).all().item()
            or not torch.isfinite(valid_remainders).all().item()
        ):
            raise ValueError("ACT update counts and remainders must be finite")
        if (
            (valid_updates < 1).any().item()
            or (valid_updates > max_loops).any().item()
            or (valid_updates != valid_updates.round()).any().item()
        ):
            raise ValueError(
                "ACT update counts must be whole numbers from one through max_loops"
            )
        if (
            (valid_remainders < 0).any().item()
            or (valid_remainders > 1).any().item()
        ):
            raise ValueError("ACT remainders must be between zero and one")
        if (valid_cap_forced & (valid_updates != max_loops)).any().item():
            raise ValueError("ACT cap-forced tokens must reach max_loops")
        derived_global_iterations = int(valid_updates.max().item())
        if global_iterations != derived_global_iterations:
            raise ValueError(
                "ACT global_iterations must equal the largest valid-token "
                "update count"
            )

        selected_rows = rows_with_targets.bool()
        selected_valid = valid[selected_rows]
        selected_updates = updates[selected_rows]
        selected_remainders = remainders[selected_rows]
        selected_cap_forced = cap_forced[selected_rows]
        correct = exact_rows[selected_rows].detach()
        lengths = selected_valid.sum(dim=1)
        if (lengths == 0).any().item():
            raise ValueError("labeled ACT examples must contain a valid input token")
        ponder = selected_updates + selected_remainders
        per_example_updates = (
            (selected_updates * selected_valid).sum(dim=1) / lengths
        )
        per_example_ponder = (
            (ponder * selected_valid).sum(dim=1) / lengths
        )
        reached_cap = (updates == max_loops) & valid
        forced_cap = cap_forced & valid
        selected_reached_cap = (
            (selected_updates == max_loops) & selected_valid
        )
        selected_forced_cap = selected_cap_forced & selected_valid

        self._update_counts.append(valid_updates.cpu())
        self._remainders.append(valid_remainders.cpu())
        self._cap_forced.append(valid_cap_forced.cpu())
        self._per_example_updates.append(per_example_updates.float().cpu())
        self._per_example_ponder.append(per_example_ponder.float().cpu())
        self._per_example_reached_cap.append(
            selected_reached_cap.sum(dim=1).cpu()
        )
        self._per_example_forced_cap.append(
            selected_forced_cap.sum(dim=1).cpu()
        )
        self._lengths.append(lengths.cpu())
        self._correct.append(correct.bool().cpu())
        self._global_iterations.append(derived_global_iterations)
        self._batch_reached_cap.append(bool(reached_cap.any().item()))
        self._batch_forced_cap.append(bool(forced_cap.any().item()))

    def summary(self, *, evaluation_task_cross_entropy: float) -> dict | None:
        if not self.has_data:
            return None
        updates = torch.cat(self._update_counts)
        remainders = torch.cat(self._remainders)
        cap_forced = torch.cat(self._cap_forced)
        example_updates = torch.cat(self._per_example_updates)
        example_ponder = torch.cat(self._per_example_ponder)
        example_reached = torch.cat(self._per_example_reached_cap)
        example_forced = torch.cat(self._per_example_forced_cap)
        lengths = torch.cat(self._lengths)
        correct = torch.cat(self._correct)
        max_loops = self._max_loops
        ponder_weight = self._ponder_weight
        assert max_loops is not None
        assert ponder_weight is not None

        ponder_time = updates + remainders
        reached_cap = updates == max_loops
        natural_halt = ~cap_forced
        token_count = updates.numel()
        observed_iterations = int(updates.max().item())
        reported_iterations = min(observed_iterations, MAX_ITERATION_DETAIL)
        within_detail = updates <= reported_iterations
        histogram = torch.bincount(
            updates[within_detail].long(),
            minlength=reported_iterations + 1,
        )[1 : reported_iterations + 1]
        natural_within_detail = natural_halt & within_detail
        natural_histogram = torch.bincount(
            updates[natural_within_detail].long(),
            minlength=reported_iterations + 1,
        )[1 : reported_iterations + 1]
        forced_histogram = histogram - natural_histogram
        cumulative = histogram.cumsum(dim=0)
        natural_cumulative = natural_histogram.cumsum(dim=0)
        forced_cumulative = forced_histogram.cumsum(dim=0)
        endings = [
            {
                "iteration": iteration,
                "ended_at_percentage": (
                    100.0 * int(histogram[iteration - 1]) / token_count
                ),
                "ended_by_percentage": (
                    100.0 * int(cumulative[iteration - 1]) / token_count
                ),
                "naturally_halted_at_percentage": (
                    100.0 * int(natural_histogram[iteration - 1]) / token_count
                ),
                "naturally_halted_by_percentage": (
                    100.0 * int(natural_cumulative[iteration - 1]) / token_count
                ),
                "cap_forced_at_percentage": (
                    100.0 * int(forced_histogram[iteration - 1]) / token_count
                ),
                "cap_forced_by_percentage": (
                    100.0 * int(forced_cumulative[iteration - 1]) / token_count
                ),
            }
            for iteration in range(1, reported_iterations + 1)
        ]

        def mean_or_none(values: torch.Tensor) -> float | None:
            return float(values.double().mean().item()) if values.numel() else None

        def group(mask: torch.Tensor) -> dict:
            return {
                "example_count": int(mask.sum().item()),
                "mean_updates_per_example": mean_or_none(example_updates[mask]),
                "median_updates_per_example": _median(example_updates[mask]),
                "p95_updates_per_example": _nearest_rank(
                    example_updates[mask], 0.95
                ),
                "mean_ponder_time_per_example": mean_or_none(
                    example_ponder[mask]
                ),
            }

        by_length = {}
        for length in sorted(set(int(value) for value in lengths.tolist())):
            mask = lengths == length
            total_tokens = int(lengths[mask].sum().item())
            by_length[str(length)] = {
                "example_count": int(mask.sum().item()),
                "exact_accuracy": float(correct[mask].float().mean().item()),
                "mean_updates_per_example": mean_or_none(example_updates[mask]),
                "median_updates_per_example": _median(example_updates[mask]),
                "p95_updates_per_example": _nearest_rank(
                    example_updates[mask], 0.95
                ),
                "mean_ponder_time_per_example": mean_or_none(
                    example_ponder[mask]
                ),
                "token_reached_cap_rate": (
                    int(example_reached[mask].sum().item()) / total_tokens
                ),
                "token_forced_cap_rate": (
                    int(example_forced[mask].sum().item()) / total_tokens
                ),
            }

        return {
            "evaluation_task_cross_entropy": float(
                evaluation_task_cross_entropy
            ),
            "raw_mean_ponder_time": float(ponder_time.double().mean().item()),
            "ponder_weight": ponder_weight,
            "weighted_ponder_contribution": (
                ponder_weight * float(ponder_time.double().mean().item())
            ),
            "token_update_counts": _distribution(updates),
            "global_iterations_per_batch": _distribution(
                torch.tensor(self._global_iterations)
            ),
            "cap_hits": {
                "token_reached_cap_rate": float(reached_cap.float().mean().item()),
                "token_forced_cap_rate": float(cap_forced.float().mean().item()),
                "batch_reached_cap_rate": (
                    sum(self._batch_reached_cap) / len(self._batch_reached_cap)
                ),
                "batch_forced_cap_rate": (
                    sum(self._batch_forced_cap) / len(self._batch_forced_cap)
                ),
            },
            "remainders": {
                "mean": mean_or_none(remainders),
                "mean_natural_halt": mean_or_none(remainders[natural_halt]),
                "mean_cap_forced": mean_or_none(remainders[cap_forced]),
            },
            "iteration_end_percentages": endings,
            "iteration_detail_truncated": (
                observed_iterations > MAX_ITERATION_DETAIL
            ),
            "iteration_detail_unreported_token_percentage": (
                100.0 * float((~within_detail).float().mean().item())
            ),
            "observed_global_iterations": observed_iterations,
            "by_correctness": {
                "correct": group(correct),
                "incorrect": group(~correct),
            },
            "by_sequence_length": by_length,
            "valid_token_count": token_count,
            "example_count": int(lengths.numel()),
            "batch_count": len(self._global_iterations),
            "max_loops": max_loops,
        }
