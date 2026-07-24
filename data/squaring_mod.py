"""Tokenized repeated modular-squaring dataset generation and loading."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
import math
import random
from pathlib import Path
from typing import Any, Protocol

import torch

from .counting import (
    Record,
    TokenizedCountingDataset,
    collate_tokenized_counting,
    compute_split_counts,
    digit_token as counting_digit_token,
    load_counting_dataset_config,
    number_tokens as counting_number_tokens,
    write_dataset_config,
    write_split_files,
)


TOKEN_IDS: dict[str, int] = {
    "PAD": 0,
    "BOS": 1,
    "N": 2,
    "X": 3,
    "T": 4,
    "ANS": 5,
    "EOS": 6,
}
DIGIT_OFFSET = 7
NUM_DIGITS = 10
VOCAB_SIZE = DIGIT_OFFSET + NUM_DIGITS

# A deliberately small, deterministic suite used when DataConfig.data_root is
# unset. It is intended for end-to-end evaluator testing, not as the eventual
# scored squaring-mod benchmark.
SMOKE_FIXED_P = 11
SMOKE_FIXED_Q = 13
SMOKE_TIME_STEPS = (1, 2, 3)
SMOKE_OOD_TIME_STEPS = (4,)
SMOKE_EXAMPLES_PER_SETTING = 100
# N=143, x<=142, and one-digit T produce at most ten prompt tokens in the
# separate-input/output representation.
SMOKE_MAX_SEQ_LEN = 10
ID_SPLITS: tuple[str, ...] = ("train", "test")
Transition = Callable[[int, int, int, int], int]


class Tokenizer(Protocol):
    def __call__(
        self,
        modulus: int,
        x: int,
        time_steps: int,
        result: int,
        *,
        separate_input_output: bool = False,
    ) -> tuple[list[int], list[int]]: ...


class SquaringModTokenizedDataset(TokenizedCountingDataset):
    """JSONL-backed repeated modular-squaring dataset."""


def load_squaring_mod_dataset_config(root: str | Path) -> dict[str, Any]:
    return load_counting_dataset_config(root)


def collate_squaring_mod(batch: list[dict[str, Any]]) -> dict[str, Any]:
    uses_separate_output = [
        len(item["labels"]) != len(item["input_ids"])
        for item in batch
    ]
    if not any(uses_separate_output):
        return collate_tokenized_counting(batch, TOKEN_IDS["PAD"])
    if not all(uses_separate_output):
        raise ValueError("squaring_mod batch cannot mix causal_lm and separate_input_output rows")

    max_input_len = max(len(item["input_ids"]) for item in batch)
    max_target_len = max(len(item["labels"]) for item in batch)
    input_ids = torch.full(
        (len(batch), max_input_len), TOKEN_IDS["PAD"], dtype=torch.long
    )
    labels = torch.full((len(batch), max_target_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_input_len), dtype=torch.bool)
    target_positions = torch.full(
        (len(batch), max_target_len), -1, dtype=torch.long
    )

    for row, item in enumerate(batch):
        item_input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        item_labels = torch.tensor(item["labels"], dtype=torch.long)
        input_len = item_input_ids.numel()
        target_len = item_labels.numel()
        if target_len > input_len:
            raise ValueError("squaring_mod output cannot be longer than its input")
        input_ids[row, :input_len] = item_input_ids
        labels[row, :target_len] = item_labels
        attention_mask[row, :input_len] = True
        target_positions[row, :target_len] = torch.arange(
            input_len - target_len, input_len, dtype=torch.long
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "target_positions": target_positions,
    }


@dataclass(frozen=True)
class SquaringModGenerationConfig:
    output_dir: str
    modulus_bits: list[int] = field(default_factory=lambda: [32])
    fixed_p: int | None = None
    fixed_q: int | None = None
    time_steps: list[int] = field(default_factory=lambda: [16])
    fixed_time_steps: int | None = None
    examples_per_setting: int = 100
    seed: int = 45
    train_fraction: float = 0.8
    test_fraction: float = 0.2
    ood_time_steps: list[int] = field(default_factory=list)
    ood_examples_per_setting: int | None = None
    generator_family: str = "rsa_repeated_squaring"
    separate_input_output: bool = False
    split_group: str = "prompt"
    factor_modulus: int | None = None
    factor_remainder: int | None = None
    separate_ood_splits: bool = False
    depth_evaluation_time_steps: list[int] = field(default_factory=list)
    depth_evaluation_examples_per_setting: int | None = None
    depth_evaluation_exhaustive_x: bool = False
    ood_n_depth_evaluation_modulus_bits: list[int] = field(default_factory=list)
    ood_n_depth_evaluation_examples_per_setting: int | None = None

    def __post_init__(self) -> None:
        fixed_values = (self.fixed_p, self.fixed_q)
        if self.split_group not in ("prompt", "x", "modulus"):
            raise ValueError("split_group must be one of: prompt, x, modulus")
        if self.split_group == "x" and self.fixed_p is None:
            raise ValueError("split_group=x requires fixed_p and fixed_q")
        if self.split_group == "modulus" and self.fixed_p is not None:
            raise ValueError("split_group=modulus requires sampled moduli")
        factor_values = (self.factor_modulus, self.factor_remainder)
        if any(value is None for value in factor_values) and any(
            value is not None for value in factor_values
        ):
            raise ValueError("factor_modulus and factor_remainder must be provided together")
        if self.factor_modulus is not None:
            if self.fixed_p is not None:
                raise ValueError("factor congruence constraints require sampled moduli")
            if self.factor_modulus < 2:
                raise ValueError("factor_modulus must be at least 2")
            if not 0 <= self.factor_remainder < self.factor_modulus:
                raise ValueError("factor_remainder must be in [0, factor_modulus)")
        if self.separate_ood_splits and self.split_group == "prompt":
            raise ValueError("separate_ood_splits requires split_group=x or split_group=modulus")
        if any(value is None for value in fixed_values) and any(value is not None for value in fixed_values):
            raise ValueError("fixed_p and fixed_q must be provided together")
        if self.fixed_p is None:
            if not self.modulus_bits:
                raise ValueError("modulus_bits must contain at least one value when fixed_p/fixed_q are not set")
            if any(value < 4 for value in self.modulus_bits):
                raise ValueError("all modulus_bits values must be at least 4")
        else:
            if self.fixed_p == self.fixed_q:
                raise ValueError("fixed_p and fixed_q must be distinct")
            if not is_probable_prime(self.fixed_p) or not is_probable_prime(self.fixed_q):
                raise ValueError("fixed_p and fixed_q must be prime")
        if self.fixed_time_steps is None:
            if not self.time_steps:
                raise ValueError("time_steps must contain at least one value when fixed_time_steps is not set")
            if any(value < 0 for value in self.time_steps):
                raise ValueError("all time_steps values must be non-negative")
        elif self.fixed_time_steps < 0:
            raise ValueError("fixed_time_steps must be non-negative")
        if any(value < 0 for value in self.ood_time_steps):
            raise ValueError("all ood_time_steps values must be non-negative")
        if self.examples_per_setting < 1:
            raise ValueError("examples_per_setting must be positive")
        if self.ood_examples_per_setting is not None and self.ood_examples_per_setting < 1:
            raise ValueError("ood_examples_per_setting must be positive when provided")
        split_total = self.train_fraction + self.test_fraction
        if not math.isclose(split_total, 1.0):
            raise ValueError("train_fraction + test_fraction must equal 1")
        if min(self.train_fraction, self.test_fraction) < 0:
            raise ValueError("split fractions must be non-negative")
        in_distribution_time_steps = (
            [self.fixed_time_steps]
            if self.fixed_time_steps is not None
            else self.time_steps
        )
        if len(set(in_distribution_time_steps)) != len(in_distribution_time_steps):
            raise ValueError("time_steps must not contain duplicates")
        if len(set(self.ood_time_steps)) != len(self.ood_time_steps):
            raise ValueError("ood_time_steps must not contain duplicates")
        if len(set(self.depth_evaluation_time_steps)) != len(
            self.depth_evaluation_time_steps
        ):
            raise ValueError("depth_evaluation_time_steps must not contain duplicates")
        if any(value < 1 for value in self.depth_evaluation_time_steps):
            raise ValueError("depth_evaluation_time_steps must be positive")
        if len(set(self.ood_n_depth_evaluation_modulus_bits)) != len(
            self.ood_n_depth_evaluation_modulus_bits
        ):
            raise ValueError(
                "ood_n_depth_evaluation_modulus_bits must not contain duplicates"
            )
        if any(value < 4 for value in self.ood_n_depth_evaluation_modulus_bits):
            raise ValueError(
                "all ood_n_depth_evaluation_modulus_bits values must be at least 4"
            )
        id_modulus_bits = (
            {int((self.fixed_p * self.fixed_q).bit_length())}
            if self.fixed_p is not None and self.fixed_q is not None
            else set(self.modulus_bits)
        )
        overlapping_ood_n_bits = (
            id_modulus_bits & set(self.ood_n_depth_evaluation_modulus_bits)
        )
        if overlapping_ood_n_bits:
            raise ValueError(
                "ood_n_depth_evaluation_modulus_bits must not overlap "
                "in-distribution modulus sizes; overlapping values: "
                f"{sorted(overlapping_ood_n_bits)}"
            )
        if (
            self.depth_evaluation_examples_per_setting is not None
            and self.depth_evaluation_examples_per_setting < 1
        ):
            raise ValueError(
                "depth_evaluation_examples_per_setting must be positive when provided"
            )
        if (
            self.ood_n_depth_evaluation_examples_per_setting is not None
            and self.ood_n_depth_evaluation_examples_per_setting < 1
        ):
            raise ValueError(
                "ood_n_depth_evaluation_examples_per_setting must be positive when provided"
            )
        if self.depth_evaluation_time_steps and self.split_group not in ("prompt", "modulus"):
            raise ValueError(
                "depth_evaluation_time_steps requires split_group=prompt or modulus"
            )
        if self.ood_n_depth_evaluation_modulus_bits and self.split_group not in ("prompt", "modulus"):
            raise ValueError(
                "ood_n_depth_evaluation_modulus_bits requires split_group=prompt or modulus"
            )
        if self.ood_n_depth_evaluation_modulus_bits and not self.depth_evaluation_time_steps:
            raise ValueError(
                "ood_n_depth_evaluation_modulus_bits requires depth_evaluation_time_steps"
            )
        if self.ood_n_depth_evaluation_modulus_bits and (
            self.ood_n_depth_evaluation_examples_per_setting is None
        ):
            raise ValueError(
                "ood_n_depth_evaluation_examples_per_setting is required with "
                "ood_n_depth_evaluation_modulus_bits"
            )
        if self.depth_evaluation_exhaustive_x and (
            self.fixed_p is None or self.fixed_q is None
        ):
            raise ValueError(
                "depth_evaluation_exhaustive_x requires fixed_p and fixed_q"
            )
        if self.depth_evaluation_time_steps and (
            self.depth_evaluation_examples_per_setting is None
            and not self.depth_evaluation_exhaustive_x
        ):
            raise ValueError(
                "depth_evaluation_examples_per_setting is required with "
                "depth_evaluation_time_steps"
            )
        if set(in_distribution_time_steps) & set(self.ood_time_steps):
            raise ValueError("ood_time_steps must not overlap training time steps")
        if self.fixed_p is None:
            for modulus_bits in self.modulus_bits:
                prompt_capacity = _exact_sampled_modulus_prompt_capacity(
                    modulus_bits,
                    factor_modulus=self.factor_modulus,
                    factor_remainder=self.factor_remainder,
                )
                if prompt_capacity is not None and self.examples_per_setting > prompt_capacity:
                    raise ValueError(
                        "examples_per_setting exceeds the number of unique sampled-modulus prompts "
                        f"for modulus_bits={modulus_bits} (capacity={prompt_capacity})"
                    )
                if (
                    prompt_capacity is not None
                    and self.ood_time_steps
                    and self.effective_ood_examples_per_setting > prompt_capacity
                ):
                    raise ValueError(
                        "ood_examples_per_setting exceeds the number of unique sampled-modulus prompts "
                        f"for modulus_bits={modulus_bits} (capacity={prompt_capacity})"
                    )
        if self.fixed_p is not None and self.fixed_q is not None:
            prompt_capacity = (self.fixed_p - 1) * (self.fixed_q - 1)
            if self.examples_per_setting > prompt_capacity:
                raise ValueError(
                    "examples_per_setting exceeds the number of unique x values "
                    "for the fixed modulus"
                )
            if self.ood_time_steps and self.effective_ood_examples_per_setting > prompt_capacity:
                raise ValueError(
                    "ood_examples_per_setting exceeds the number of unique x values "
                    "for the fixed modulus"
                )

    @property
    def effective_ood_examples_per_setting(self) -> int:
        return self.ood_examples_per_setting or self.examples_per_setting


def generate_squaring_mod_smoke_dataset(
    output_dir: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Generate the built-in tiny evaluator smoke dataset."""

    return generate_squaring_mod_dataset(
        SquaringModGenerationConfig(
            output_dir=str(output_dir),
            fixed_p=SMOKE_FIXED_P,
            fixed_q=SMOKE_FIXED_Q,
            time_steps=list(SMOKE_TIME_STEPS),
            examples_per_setting=SMOKE_EXAMPLES_PER_SETTING,
            seed=seed,
            train_fraction=0.8,
            test_fraction=0.2,
            ood_time_steps=list(SMOKE_OOD_TIME_STEPS),
            ood_examples_per_setting=SMOKE_EXAMPLES_PER_SETTING,
            separate_input_output=True,
        )
    )


def _exact_sampled_modulus_prompt_capacity(
    modulus_bits: int,
    *,
    factor_modulus: int | None = None,
    factor_remainder: int | None = None,
) -> int | None:
    """Return exact unit-prompt capacity when the factor ranges are tractable."""
    factor_pairs = _enumerate_sampled_factor_pairs(
        modulus_bits,
        factor_modulus=factor_modulus,
        factor_remainder=factor_remainder,
    )
    if factor_pairs is None:
        return None
    return sum((p - 1) * (q - 1) for p, q in factor_pairs)


def _enumerate_sampled_factor_pairs(
    modulus_bits: int,
    *,
    factor_modulus: int | None = None,
    factor_remainder: int | None = None,
) -> list[tuple[int, int]] | None:
    p_bits = modulus_bits // 2
    q_bits = modulus_bits - p_bits
    if max(p_bits, q_bits) > 10:
        return None

    def eligible(value: int) -> bool:
        if not is_probable_prime(value):
            return False
        if factor_modulus is None:
            return True
        return value % factor_modulus == factor_remainder

    p_candidates = [
        value
        for value in range(1 << (p_bits - 1), 1 << p_bits)
        if eligible(value)
    ]
    q_candidates = [
        value
        for value in range(1 << (q_bits - 1), 1 << q_bits)
        if eligible(value)
    ]
    factor_pairs = {
        p * q: (p, q)
        for p in p_candidates
        for q in q_candidates
        if p != q and (p * q).bit_length() == modulus_bits
    }
    return list(factor_pairs.values())


def digit_token(digit: int) -> int:
    return counting_digit_token(digit, digit_offset=DIGIT_OFFSET)


def number_tokens(value: int) -> list[int]:
    return counting_number_tokens(value, digit_offset=DIGIT_OFFSET)


def trapdoor_squaring_mod(x: int, time_steps: int, p: int, q: int) -> int:
    if time_steps < 0:
        raise ValueError("time_steps must be non-negative")
    modulus = p * q
    phi = (p - 1) * (q - 1)
    exponent = pow(2, time_steps, phi)
    return pow(x, exponent, modulus)


def tokenize_squaring_mod_with_result(
    modulus: int,
    x: int,
    time_steps: int,
    result: int,
    *,
    separate_input_output: bool = False,
) -> tuple[list[int], list[int]]:
    input_ids = [TOKEN_IDS["N"]]
    input_ids.extend(number_tokens(modulus))
    input_ids.append(TOKEN_IDS["X"])
    input_ids.extend(number_tokens(x))
    input_ids.append(TOKEN_IDS["T"])
    input_ids.extend(number_tokens(time_steps))

    result_tokens = number_tokens(result)
    if separate_input_output:
        return input_ids, result_tokens

    input_ids.insert(0, TOKEN_IDS["BOS"])
    input_ids.append(TOKEN_IDS["ANS"])
    input_ids.extend(result_tokens)
    input_ids.append(TOKEN_IDS["EOS"])

    labels = [-100] * len(input_ids)
    answer_start = len(input_ids) - len(result_tokens) - 1
    for offset, token in enumerate(result_tokens):
        labels[answer_start + offset] = token
    labels[-1] = TOKEN_IDS["EOS"]
    return input_ids, labels


def generate_squaring_mod_dataset(
    config: SquaringModGenerationConfig,
    *,
    transition: Transition = trapdoor_squaring_mod,
    tokenizer: Tokenizer = tokenize_squaring_mod_with_result,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    if config.split_group == "x":
        records = _generate_x_grouped_records(
            config=config, rng=rng, transition=transition, tokenizer=tokenizer
        )
    elif config.split_group == "modulus":
        records = _generate_modulus_grouped_records(
            config=config, rng=rng, transition=transition, tokenizer=tokenizer
        )
    else:
        records = _generate_prompt_grouped_records(
            config=config, rng=rng, transition=transition, tokenizer=tokenizer
        )

    write_split_files(output_dir, records)
    dataset_config = _dataset_config(config, records)
    write_dataset_config(output_dir, dataset_config)
    return dataset_config


def _id_split_counts(config: SquaringModGenerationConfig) -> dict[str, int]:
    return compute_split_counts(
        config.examples_per_setting,
        {
            "train": config.train_fraction,
            "test": config.test_fraction,
        },
    )


def _generate_prompt_grouped_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    records: list[Record] = []
    seen_prompts: set[tuple[int, int, int]] = set()
    if (
        config.fixed_p is not None
        and config.fixed_q is not None
        and config.depth_evaluation_time_steps
    ):
        assert config.fixed_p is not None and config.fixed_q is not None
        modulus = config.fixed_p * config.fixed_q
        units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
        required_per_time_step = max(
            config.examples_per_setting,
            config.effective_ood_examples_per_setting,
        )
        reserve_count = (
            len(units) - required_per_time_step
            if config.depth_evaluation_exhaustive_x
            else config.depth_evaluation_examples_per_setting
        )
        assert reserve_count is not None
        if reserve_count < 1:
            raise ValueError(
                "fixed-N depth evaluation requires at least one reserved unit"
            )
        if required_per_time_step + reserve_count > len(units):
            raise ValueError(
                "fixed-N depth evaluation lacks enough unique x values for "
                "the ordinary and reserved profile cohorts"
            )
        reserved_x = set(rng.sample(units, reserve_count))
        for time_steps in (*_time_settings(config), *config.ood_time_steps):
            seen_prompts.update((modulus, x, time_steps) for x in reserved_x)
    for modulus_bits in _modulus_settings(config):
        for time_steps in _time_settings(config):
            records.extend(
                _generate_setting_records(
                    config=config,
                    rng=rng,
                    modulus_bits=modulus_bits,
                    time_steps=time_steps,
                    start_index=len(records),
                    seen_prompts=seen_prompts,
                    transition=transition,
                    tokenizer=tokenizer,
                )
            )
    for modulus_bits in _modulus_settings(config):
        for time_steps in config.ood_time_steps:
            records.extend(
                _generate_ood_records(
                    config=config,
                    rng=rng,
                    modulus_bits=modulus_bits,
                    time_steps=time_steps,
                    start_index=len(records),
                    seen_prompts=seen_prompts,
                    transition=transition,
                    tokenizer=tokenizer,
                )
            )
    if config.depth_evaluation_time_steps:
        records.extend(
            _generate_prompt_depth_evaluation_records(


                config=config,
                rng=rng,
                existing_records=records,
                transition=transition,
                tokenizer=tokenizer,
            )
        )
        records.extend(
            _generate_ood_n_depth_evaluation_records(
                config=config,
                rng=rng,
                existing_records=records,
                transition=transition,
                tokenizer=tokenizer,
            )
        )
    return records


def _generate_prompt_depth_evaluation_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    existing_records: list[Record],
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    """Build matched Easy depth cohorts for fixed- or sampled-modulus data."""

    matched_by_bits: dict[int | None, list[tuple[int, int, int]]] = {}
    if config.depth_evaluation_exhaustive_x:
        assert config.fixed_p is not None and config.fixed_q is not None
        modulus = config.fixed_p * config.fixed_q
        used_pairs = {
            (int(record["modulus"]), int(record["x"]))
            for record in existing_records
        }
        matched_by_bits[None] = [
            (config.fixed_p, config.fixed_q, x)
            for x in range(1, modulus)
            if math.gcd(x, modulus) == 1
            and (modulus, x) not in used_pairs
        ]
        if not matched_by_bits[None]:
            raise ValueError(
                "depth_evaluation_exhaustive_x found no held-out fixed-N prompts"
            )
    else:
        count = config.depth_evaluation_examples_per_setting
        assert count is not None
        used_pairs = {
            (int(record["modulus"]), int(record["x"]))
            for record in existing_records
        }
        if config.fixed_p is not None and config.fixed_q is not None:
            factor_pools: dict[int | None, list[tuple[int, int]]] = {
                None: [(config.fixed_p, config.fixed_q)]
            }
        else:
            factor_pools = {}
            for modulus_bits in config.modulus_bits:
                training_moduli = {
                    int(record["modulus"])
                    for record in existing_records
                    if record["split"] == "train"
                    and record["configured_modulus_bits"] == modulus_bits
                }
                selected_moduli = rng.sample(
                    sorted(training_moduli), min(len(training_moduli), count)
                )
                factor_pools[modulus_bits] = [
                    _factor_semiprime(modulus) for modulus in selected_moduli
                ]
        for modulus_bits, factor_pool in factor_pools.items():
            if not factor_pool:
                raise ValueError(
                    f"no training moduli available for depth profile at {modulus_bits} bits"
                )
            weights = [(p - 1) * (q - 1) for p, q in factor_pool]
            matched: list[tuple[int, int, int]] = []
            for _ in range(count):
                for _ in range(10_000):
                    p, q = rng.choices(factor_pool, weights=weights, k=1)[0]
                    modulus = p * q
                    x = _sample_unit(modulus=modulus, rng=rng)
                    pair = (modulus, x)
                    if pair not in used_pairs:
                        used_pairs.add(pair)
                        matched.append((p, q, x))
                        break
                else:
                    raise ValueError(
                        "could not sample a fresh seen-N depth pair after 10,000 attempts"
                    )
            matched_by_bits[modulus_bits] = matched

    records: list[Record] = []
    for time_steps in config.depth_evaluation_time_steps:
        split = f"depth_t_{time_steps}"
        for modulus_bits, matched in matched_by_bits.items():
            for p, q, x in matched:
                records.append(
                    _build_record(
                        config=config,
                        p=p,
                        q=q,
                        x=x,
                        time_steps=time_steps,
                        split=split,
                        index=len(existing_records) + len(records),
                        modulus_bits=modulus_bits,
                        transition=transition,
                        tokenizer=tokenizer,
                    )
                )
    return records
def _generate_x_grouped_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    if config.fixed_p is None or config.fixed_q is None:
        raise ValueError("split_group=x requires fixed factors")
    p, q = config.fixed_p, config.fixed_q
    modulus = p * q
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    rng.shuffle(units)
    if config.examples_per_setting > len(units):
        raise ValueError("fixed modulus lacks enough unique x groups")

    split_counts = _id_split_counts(config)
    id_x = units[: config.examples_per_setting]
    x_by_split: dict[str, list[int]] = {}
    start = 0
    for split in ID_SPLITS:
        stop = start + split_counts[split]
        x_by_split[split] = id_x[start:stop]
        start = stop

    records: list[Record] = []
    for time_steps in _time_settings(config):
        for split in ID_SPLITS:
            for x in x_by_split[split]:
                records.append(
                    _build_record(
                        config=config,
                        p=p,
                        q=q,
                        x=x,
                        time_steps=time_steps,
                        split=split,
                        index=len(records),
                        modulus_bits=None,
                        transition=transition,
                        tokenizer=tokenizer,
                    )
                )

    ood_count = config.effective_ood_examples_per_setting
    unused_x = units[config.examples_per_setting :]
    for time_steps in config.ood_time_steps:
        if config.separate_ood_splits:
            if ood_count > len(x_by_split["train"]):
                raise ValueError("ood_t_seen_x exceeds the number of training x groups")
            if ood_count > len(unused_x):
                raise ValueError("ood_t_unseen_x exceeds the unused fixed-modulus x capacity")
            ood_groups = (
                ("ood_t_seen_x", rng.sample(x_by_split["train"], ood_count)),
                ("ood_t_unseen_x", rng.sample(unused_x, ood_count)),
            )
        else:
            ood_groups = (("ood", rng.sample(units, ood_count)),)
        for split, x_values in ood_groups:
            for x in x_values:
                records.append(
                    _build_record(
                        config=config,
                        p=p,
                        q=q,
                        x=x,
                        time_steps=time_steps,
                        split=split,
                        index=len(records),
                        modulus_bits=None,
                        transition=transition,
                        tokenizer=tokenizer,
                    )
                )
    return records


def _generate_modulus_grouped_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    split_counts = _id_split_counts(config)
    factor_pools_by_bits: dict[int, dict[str, list[tuple[int, int]]]] = {}
    for modulus_bits in config.modulus_bits:
        factor_pairs = _enumerate_sampled_factor_pairs(
            modulus_bits,
            factor_modulus=config.factor_modulus,
            factor_remainder=config.factor_remainder,
        )
        if factor_pairs is None:
            raise ValueError(
                "split_group=modulus requires exactly enumerable factor ranges "
                f"for modulus_bits={modulus_bits}"
            )
        factor_pools_by_bits[modulus_bits] = _partition_factor_pairs(
            factor_pairs=factor_pairs,
            split_counts=split_counts,
            config=config,
            rng=rng,
            modulus_bits=modulus_bits,
        )

    records: list[Record] = []
    seen_prompts: set[tuple[int, int, int]] = set()
    for modulus_bits in config.modulus_bits:
        factor_pools = factor_pools_by_bits[modulus_bits]
        for time_steps in _time_settings(config):
            for split in ID_SPLITS:
                records.extend(
                    _generate_records_from_factor_pool(
                        config=config,
                        rng=rng,
                        factor_pool=factor_pools[split],
                        count=split_counts[split],
                        modulus_bits=modulus_bits,
                        time_steps=time_steps,
                        split=split,
                        start_index=len(records),
                        seen_prompts=seen_prompts,
                        transition=transition,
                        tokenizer=tokenizer,
                    )
                )

    ood_count = config.effective_ood_examples_per_setting
    for modulus_bits in config.modulus_bits:
        factor_pools = factor_pools_by_bits[modulus_bits]
        heldout_pool = factor_pools["test"]
        for time_steps in config.ood_time_steps:
            if config.separate_ood_splits:
                ood_pools = (
                    ("ood_t", factor_pools["train"]),
                    ("ood_n_t", heldout_pool),
                )
            else:
                ood_pools = (("ood", sum(factor_pools.values(), [])),)
            for split, factor_pool in ood_pools:
                records.extend(
                    _generate_records_from_factor_pool(
                        config=config,
                        rng=rng,
                        factor_pool=factor_pool,
                        count=ood_count,
                        modulus_bits=modulus_bits,
                        time_steps=time_steps,
                        split=split,
                        start_index=len(records),
                        seen_prompts=seen_prompts,
                        transition=transition,
                        tokenizer=tokenizer,
                    )
                )
    if config.depth_evaluation_time_steps:
        records.extend(
            _generate_matched_depth_evaluation_records(
                config=config,
                rng=rng,
                factor_pools_by_bits=factor_pools_by_bits,
                existing_records=records,
                transition=transition,
                tokenizer=tokenizer,
            )
        )
        records.extend(
            _generate_ood_n_depth_evaluation_records(
                config=config,
                rng=rng,
                existing_records=records,
                transition=transition,
                tokenizer=tokenizer,
            )
        )
    return records



def _generate_matched_depth_evaluation_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    factor_pools_by_bits: dict[int, dict[str, list[tuple[int, int]]]],
    existing_records: list[Record],
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    """Build matched, evaluator-only depth cohorts from training-pool moduli."""

    count = config.depth_evaluation_examples_per_setting
    if count is None:
        return []
    used_pairs = {
        (int(record["modulus"]), int(record["x"])) for record in existing_records
    }
    matched_by_bits: dict[int, list[tuple[int, int, int]]] = {}
    for modulus_bits in config.modulus_bits:
        training_moduli = {
            int(record["modulus"])
            for record in existing_records
            if record["split"] == "train"
            and record["configured_modulus_bits"] == modulus_bits
        }
        factor_pool = [
            pair
            for pair in factor_pools_by_bits[modulus_bits]["train"]
            if pair[0] * pair[1] in training_moduli
        ]
        weights = [(p - 1) * (q - 1) for p, q in factor_pool]
        matched: list[tuple[int, int, int]] = []
        for _ in range(count):
            for _ in range(10_000):
                p, q = rng.choices(factor_pool, weights=weights, k=1)[0]
                modulus = p * q
                x = _sample_unit(modulus=modulus, rng=rng)
                pair = (modulus, x)
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    matched.append((p, q, x))
                    break
            else:
                raise ValueError(
                    "could not sample a fresh matched depth-evaluation prompt "
                    "after 10,000 attempts"
                )
        matched_by_bits[modulus_bits] = matched

    records: list[Record] = []
    for time_steps in config.depth_evaluation_time_steps:
        split = f"depth_t_{time_steps}"
        for modulus_bits in config.modulus_bits:
            for p, q, x in matched_by_bits[modulus_bits]:
                records.append(
                    _build_record(
                        config=config,
                        p=p,
                        q=q,
                        x=x,
                        time_steps=time_steps,
                        split=split,
                        index=len(existing_records) + len(records),
                        modulus_bits=modulus_bits,
                        transition=transition,
                        tokenizer=tokenizer,
                    )
                )
    return records


def _generate_ood_n_depth_evaluation_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    existing_records: list[Record],
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    """Build matched depth cohorts at unseen interpolation/extrapolation N sizes."""

    count = config.ood_n_depth_evaluation_examples_per_setting
    if not config.ood_n_depth_evaluation_modulus_bits or count is None:
        return []
    existing_moduli = {
        int(record["modulus"])
        for record in existing_records
    }
    used_pairs = {
        (int(record["modulus"]), int(record["x"]))
        for record in existing_records
    }
    matched_by_bits: dict[int, list[tuple[int, int, int]]] = {}
    for modulus_bits in config.ood_n_depth_evaluation_modulus_bits:
        matched: list[tuple[int, int, int]] = []
        for _ in range(count):
            for _ in range(10_000):
                p, q = _sample_rsa_factors(
                    modulus_bits=modulus_bits,
                    rng=rng,
                    factor_modulus=config.factor_modulus,
                    factor_remainder=config.factor_remainder,
                )
                modulus = p * q
                x = _sample_unit(modulus=modulus, rng=rng)
                pair = (modulus, x)
                if modulus not in existing_moduli and pair not in used_pairs:
                    used_pairs.add(pair)
                    matched.append((p, q, x))
                    break
            else:
                raise ValueError(
                    "could not sample a fresh OOD-N depth-evaluation prompt "
                    "after 10,000 attempts"
                )
        matched_by_bits[modulus_bits] = matched

    records: list[Record] = []
    for time_steps in config.depth_evaluation_time_steps:
        split = f"depth_ood_n_t_{time_steps}"
        for modulus_bits in config.ood_n_depth_evaluation_modulus_bits:
            for p, q, x in matched_by_bits[modulus_bits]:
                records.append(
                    _build_record(
                        config=config,
                        p=p,
                        q=q,
                        x=x,
                        time_steps=time_steps,
                        split=split,
                        index=len(existing_records) + len(records),
                        modulus_bits=modulus_bits,
                        transition=transition,
                        tokenizer=tokenizer,
                    )
                )
    return records

def _partition_factor_pairs(
    *,
    factor_pairs: list[tuple[int, int]],
    split_counts: dict[str, int],
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int,
) -> dict[str, list[tuple[int, int]]]:
    modulus_counts = compute_split_counts(
        len(factor_pairs),
        {
            "train": config.train_fraction,
            "test": config.test_fraction,
        },
    )
    for _ in range(10_000):
        shuffled = list(factor_pairs)
        rng.shuffle(shuffled)
        pools: dict[str, list[tuple[int, int]]] = {}
        start = 0
        for split in ID_SPLITS:
            stop = start + modulus_counts[split]
            pools[split] = shuffled[start:stop]
            start = stop
        if all(
            _factor_pool_capacity(pools[split]) >= split_counts[split]
            for split in ID_SPLITS
        ):
            return pools
    raise ValueError(
        "could not partition modulus identities with enough prompt capacity "
        f"for modulus_bits={modulus_bits}; increase the bit size or reduce rows"
    )


def _factor_pool_capacity(factor_pool: list[tuple[int, int]]) -> int:
    return sum((p - 1) * (q - 1) for p, q in factor_pool)


def _generate_records_from_factor_pool(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    factor_pool: list[tuple[int, int]],
    count: int,
    modulus_bits: int,
    time_steps: int,
    split: str,
    start_index: int,
    seen_prompts: set[tuple[int, int, int]],
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    if not factor_pool:
        raise ValueError(f"empty factor pool for split={split}, modulus_bits={modulus_bits}")
    if count > _factor_pool_capacity(factor_pool):
        raise ValueError(
            f"split={split}, modulus_bits={modulus_bits} lacks unique prompt capacity"
        )

    weights = [(p - 1) * (q - 1) for p, q in factor_pool]
    records: list[Record] = []
    for offset in range(count):
        for _ in range(10_000):
            p, q = rng.choices(factor_pool, weights=weights, k=1)[0]
            modulus = p * q
            x = _sample_unit(modulus=modulus, rng=rng)
            prompt = (modulus, x, time_steps)
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                break
        else:
            raise ValueError(
                "could not sample a unique grouped-modulus prompt after 10,000 attempts"
            )
        records.append(
            _build_record(
                config=config,
                p=p,
                q=q,
                x=x,
                time_steps=time_steps,
                split=split,
                index=start_index + offset,
                modulus_bits=modulus_bits,
                transition=transition,
                tokenizer=tokenizer,
            )
        )
    return records


def _generate_setting_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
    time_steps: int,
    start_index: int,
    seen_prompts: set[tuple[int, int, int]],
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    split_counts = compute_split_counts(
        config.examples_per_setting,
        {
            "train": config.train_fraction,
            "test": config.test_fraction,
        },
    )
    records: list[Record] = []
    for split in ID_SPLITS:
        for _ in range(split_counts[split]):
            records.append(
                _generate_record(
                    config=config,
                    rng=rng,
                    modulus_bits=modulus_bits,
                    time_steps=time_steps,
                    split=split,
                    index=start_index + len(records),
                    seen_prompts=seen_prompts,
                    transition=transition,
                    tokenizer=tokenizer,
                )
            )
    return records


def _generate_ood_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
    time_steps: int,
    start_index: int,
    seen_prompts: set[tuple[int, int, int]],
    transition: Transition,
    tokenizer: Tokenizer,
) -> list[Record]:
    return [
        _generate_record(
            config=config,
            rng=rng,
            modulus_bits=modulus_bits,
            time_steps=time_steps,
            split="ood",
            index=start_index + offset,
            seen_prompts=seen_prompts,
            transition=transition,
            tokenizer=tokenizer,
        )
        for offset in range(config.effective_ood_examples_per_setting)
    ]


def _generate_record(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
    time_steps: int,
    split: str,
    index: int,
    seen_prompts: set[tuple[int, int, int]],
    transition: Transition,
    tokenizer: Tokenizer,
) -> Record:
    for _ in range(10_000):
        p, q = _sample_or_fixed_factors(
            config=config,
            rng=rng,
            modulus_bits=modulus_bits,
        )
        modulus = p * q
        x = _sample_unit(modulus=modulus, rng=rng)
        prompt = (modulus, x, time_steps)
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            break
    else:
        raise ValueError(
            "could not generate a unique (modulus, x, time_steps) prompt "
            "after 10,000 attempts"
        )
    return _build_record(
        config=config,
        p=p,
        q=q,
        x=x,
        time_steps=time_steps,
        split=split,
        index=index,
        modulus_bits=modulus_bits,
        transition=transition,
        tokenizer=tokenizer,
    )


def _build_record(
    *,
    config: SquaringModGenerationConfig,
    p: int,
    q: int,
    x: int,
    time_steps: int,
    split: str,
    index: int,
    modulus_bits: int | None,
    transition: Transition,
    tokenizer: Tokenizer,
) -> Record:
    modulus = p * q
    result = transition(x, time_steps, p, q)
    input_ids, labels = tokenizer(
        modulus,
        x,
        time_steps,
        result,
        separate_input_output=config.separate_input_output,
    )
    bit_label = "fixed" if modulus_bits is None else str(modulus_bits)
    return {
        "instance_id": f"squaring_mod_b{bit_label}_t{time_steps}_s{config.seed}_{index:08d}",
        "seed": config.seed,
        "modulus": modulus,
        "modulus_bits": modulus.bit_length(),
        "configured_modulus_bits": modulus_bits,
        "x": x,
        "time_steps": time_steps,
        "result": result,
        "generator_family": config.generator_family,
        "label_exact": True,
        "label_method": "trapdoor_phi",
        "split": split,
        "input_ids": input_ids,
        "labels": labels,
    }


def _sample_or_fixed_factors(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
) -> tuple[int, int]:
    if config.fixed_p is not None and config.fixed_q is not None:
        return config.fixed_p, config.fixed_q
    if modulus_bits is None:
        raise ValueError("modulus_bits is required when fixed_p/fixed_q are not set")
    return _sample_rsa_factors(
        modulus_bits=modulus_bits,
        rng=rng,
        factor_modulus=config.factor_modulus,
        factor_remainder=config.factor_remainder,
    )


def _sample_rsa_factors(
    *,
    modulus_bits: int,
    rng: random.Random,
    factor_modulus: int | None = None,
    factor_remainder: int | None = None,
) -> tuple[int, int]:
    p_bits = modulus_bits // 2
    q_bits = modulus_bits - p_bits
    for _ in range(10_000):
        p = _sample_prime(bits=p_bits, rng=rng)
        q = _sample_prime(bits=q_bits, rng=rng)
        factors_match = (
            factor_modulus is None
            or (p % factor_modulus == factor_remainder and q % factor_modulus == factor_remainder)
        )
        if p != q and (p * q).bit_length() == modulus_bits and factors_match:
            return p, q
    raise ValueError(f"could not sample a {modulus_bits}-bit RSA modulus")


def _sample_prime(*, bits: int, rng: random.Random) -> int:
    if bits < 2:
        raise ValueError("prime bit length must be at least 2")
    for _ in range(10_000):
        value = rng.getrandbits(bits)
        value |= 1
        value |= 1 << (bits - 1)
        if is_probable_prime(value):
            return value
    raise ValueError(f"could not sample a {bits}-bit prime")



def _factor_semiprime(modulus: int) -> tuple[int, int]:
    """Recover evaluator-generated prime factors for a seen modulus identity."""

    for p in range(2, math.isqrt(modulus) + 1):
        if modulus % p == 0:
            q = modulus // p
            if is_probable_prime(p) and is_probable_prime(q):
                return p, q
    raise ValueError(f"could not recover prime factors for modulus {modulus}")

def _sample_unit(*, modulus: int, rng: random.Random) -> int:
    while True:
        x = rng.randrange(1, modulus)
        if math.gcd(x, modulus) == 1:
            return x


def is_probable_prime(value: int) -> bool:
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if value in small_primes:
        return True
    if any(value % prime == 0 for prime in small_primes):
        return False

    d = value - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for base in _miller_rabin_bases(value):
        if base % value == 0:
            continue
        witness = pow(base, d, value)
        if witness in (1, value - 1):
            continue
        for _ in range(s - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _miller_rabin_bases(value: int) -> tuple[int, ...]:
    if value < 2_152_302_898_747:
        return (2, 3, 5, 7, 11)
    if value < 3_474_749_660_383:
        return (2, 3, 5, 7, 11, 13)
    if value < 341_550_071_728_321:
        return (2, 3, 5, 7, 11, 13, 17)
    return (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _modulus_settings(config: SquaringModGenerationConfig) -> list[int | None]:
    if config.fixed_p is not None and config.fixed_q is not None:
        return [None]
    return config.modulus_bits


def _time_settings(config: SquaringModGenerationConfig) -> list[int]:
    if config.fixed_time_steps is not None:
        return [config.fixed_time_steps]
    return config.time_steps


def _dataset_config(config: SquaringModGenerationConfig, records: list[Record]) -> dict[str, Any]:
    split_names = sorted({str(record["split"]) for record in records})
    split_counts = {
        split: sum(1 for record in records if record["split"] == split)
        for split in split_names
    }
    return {
        "dataset_kind": "squaring_mod",
        "generator_config": _public_generator_config(config),
        "split_group": config.split_group,
        "factor_congruence": (
            None
            if config.factor_modulus is None
            else {
                "modulus": config.factor_modulus,
                "remainder": config.factor_remainder,
            }
        ),
        "token_ids": TOKEN_IDS | {"DIGIT_OFFSET": DIGIT_OFFSET},
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": max(len(record["input_ids"]) for record in records),
        "max_modulus_bits": max(record["modulus_bits"] for record in records),
        "max_time_steps": max(record["time_steps"] for record in records),
        "num_examples": len(records),
        "split_counts": split_counts,
        "data_format": (
            "separate_input_output" if config.separate_input_output else "causal_lm"
        ),
        "label_format": (
            "tail_aligned_decimal_residue"
            if config.separate_input_output
            else "next_token_decimal_residue"
        ),
        "label_method": "trapdoor_phi",
    }


def _public_generator_config(config: SquaringModGenerationConfig) -> dict[str, Any]:
    config_dict = asdict(config)
    if config.fixed_p is not None and config.fixed_q is not None:
        config_dict["fixed_modulus"] = config.fixed_p * config.fixed_q
    config_dict["fixed_p"] = None if config.fixed_p is None else "<redacted>"
    config_dict["fixed_q"] = None if config.fixed_q is None else "<redacted>"
    return config_dict


def cli() -> None:
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_class_arguments(SquaringModGenerationConfig)
    parsed = parser.instantiate(parser.parse_args())
    if isinstance(parsed, SquaringModGenerationConfig):
        config = parsed
    else:
        config = SquaringModGenerationConfig(
            **{field.name: getattr(parsed, field.name) for field in fields(SquaringModGenerationConfig)}
        )
    dataset_config = generate_squaring_mod_dataset(config)
    print(f"wrote {dataset_config['num_examples']} examples to {config.output_dir}")


if __name__ == "__main__":
    cli()
