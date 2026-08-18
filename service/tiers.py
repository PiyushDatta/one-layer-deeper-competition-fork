"""Server-owned compute tier and dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass


RESULT_WAIT_GRACE_SECONDS = 60


@dataclass(frozen=True)
class DatasetOption:
    id: str
    label: str
    manifest_filename: str


@dataclass(frozen=True)
class ComputeTier:
    id: str
    label: str
    training_seconds: int
    daily_attempts: int
    evaluator_timeout_seconds: int
    datasets: tuple[DatasetOption, ...]

    @property
    def evaluation_seconds(self) -> int:
        return self.training_seconds // 2

    @property
    def run_deadline_seconds(self) -> int:
        return self.evaluator_timeout_seconds + RESULT_WAIT_GRACE_SECONDS


TIERS: tuple[ComputeTier, ...] = (
    ComputeTier(
        id="easy",
        label="Easy",
        training_seconds=60,
        daily_attempts=60,
        evaluator_timeout_seconds=390,
        datasets=(
            DatasetOption("e6", "E6 · Fixed N=247, T=1/2/3/4", "h100_easy_e6.json"),
            DatasetOption("e7", "E7 · Fixed N=287, T=1/2/3", "h100_easy_e7.json"),
            DatasetOption("e1", "E1 · Fixed N=323, T=1/2/3", "h100_easy_e1.json"),
            DatasetOption("e8", "E8 · Fixed N=287, T=1/2/3", "h100_easy_e8.json"),
            DatasetOption("e9", "E9 · Fixed N=299, T=1/2/3", "h100_easy_e9.json"),
            DatasetOption("e10", "E10 · Fixed N=403, T=1/2/3", "h100_easy_e10.json"),
            DatasetOption("e2", "E2 · Fixed N=899, T=1/2/4", "h100_easy_e2.json"),
            DatasetOption("e3", "E3 · 10–11 bit N, fixed T=2", "h100_easy_e3.json"),
            DatasetOption("e5", "E5 · 10–11 bit N, T=1/2/3", "h100_easy_e5.json"),
            DatasetOption("e4", "E4 · 11–12 bit N, fixed T=2", "h100_easy_e4.json"),
        ),
    ),
    ComputeTier(
        id="medium",
        label="Medium",
        training_seconds=600,
        daily_attempts=6,
        evaluator_timeout_seconds=1200,
        datasets=(
            DatasetOption("m6", "M6 · Fixed N=1,517, T=1/2/4", "h100_medium_m6.json"),
            DatasetOption("m7", "M7 · Fixed N=1,763, T=1/2/4", "h100_medium_m7.json"),
            DatasetOption("m8", "M8 · Fixed N=1,333, T=1/2/4", "h100_medium_m8.json"),
            DatasetOption("m9", "M9 · Fixed N=1,927, T=1/2/4", "h100_medium_m9.json"),
            DatasetOption("m10", "M10 · Fixed N=1,739, T=1/2/4", "h100_medium_m10.json"),
            DatasetOption("m2", "M2 · Fixed N=38,021, T=4/8/16", "h100_medium_m2.json"),
            DatasetOption("m3", "M3 · 11/13/15 bit N, fixed T=2", "h100_medium_m3.json"),
            DatasetOption("m5", "M5 · 12/14/16 bit N, T=2/4/8", "h100_medium_m5.json"),
            DatasetOption("m4", "M4 · 14/18/22 bit N, fixed T=8", "h100_medium_m4.json"),
            DatasetOption("m1", "M1 · Fixed N=10,403, T=4/8/16", "h100_medium_m1.json"),
        ),
    ),
    ComputeTier(
        id="hard",
        label="Hard",
        training_seconds=3600,
        daily_attempts=1,
        evaluator_timeout_seconds=6120,
        datasets=(
            DatasetOption(
                "h1",
                "H1 · Hidden evaluation",
                "h100_hard_h1.json",
            ),
        ),
    ),
)

TIER_BY_ID = {tier.id: tier for tier in TIERS}


def submission_manifest_timeouts() -> dict[str, int]:
    return {
        dataset.manifest_filename: tier.evaluator_timeout_seconds
        for tier in TIERS
        for dataset in tier.datasets
    }


def resolve_tier_dataset(tier_id: str, dataset_id: str | None) -> tuple[ComputeTier, DatasetOption]:
    tier = TIER_BY_ID.get(tier_id.strip().lower())
    if tier is None:
        raise ValueError(f"unknown tier {tier_id!r}; choose easy, medium, or hard")

    normalized_dataset = (dataset_id or "").strip().lower()
    if tier.id == "hard" and not normalized_dataset:
        normalized_dataset = "h1"
    for dataset in tier.datasets:
        if dataset.id == normalized_dataset:
            return tier, dataset
    choices = ", ".join(dataset.id for dataset in tier.datasets)
    raise ValueError(f"dataset for {tier.label} must be one of: {choices}")


def tier_public_payload(tier: ComputeTier) -> dict:
    return {
        "id": tier.id,
        "label": tier.label,
        "training_seconds": tier.training_seconds,
        "evaluation_seconds": tier.evaluation_seconds,
        "daily_attempts": tier.daily_attempts,
        "datasets": [
            {"id": dataset.id, "label": dataset.label}
            for dataset in tier.datasets
        ],
    }
