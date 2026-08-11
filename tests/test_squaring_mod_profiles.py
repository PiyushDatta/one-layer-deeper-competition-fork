from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data.squaring_mod import (
    SquaringModGenerationConfig,
    generate_squaring_mod_dataset,
)


def _rows(root: Path, split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / f"{split}.jsonl").read_text().splitlines()
    ]


class SquaringModProfileTests(unittest.TestCase):
    def test_fixed_n_exhaustive_profile_reserves_fresh_x_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SquaringModGenerationConfig(
                output_dir=directory, fixed_p=11, fixed_q=13,
                time_steps=[1, 2], examples_per_setting=100,
                ood_time_steps=[4], ood_examples_per_setting=40,
                train_fraction=0.8, test_fraction=0.2, split_group="prompt",
                depth_evaluation_time_steps=[1, 2, 4],
                depth_evaluation_exhaustive_x=True, seed=19,
                separate_input_output=True,
            )
            dataset_config = generate_squaring_mod_dataset(config)
            root = Path(directory)
            ordinary_pairs = {
                (row["modulus"], row["x"])
                for split in ("train", "test", "ood")
                for row in _rows(root, split)
            }
            expected_pairs = None
            for time_steps in (1, 2, 4):
                rows = _rows(root, f"depth_t_{time_steps}")
                pairs = {(row["modulus"], row["x"]) for row in rows}
                self.assertTrue(pairs)
                self.assertTrue(pairs.isdisjoint(ordinary_pairs))
                if expected_pairs is None:
                    expected_pairs = pairs
                else:
                    self.assertEqual(pairs, expected_pairs)
            self.assertNotIn("depth_ood_n_t_1", dataset_config["split_counts"])

    def test_variable_n_profiles_use_seen_and_unseen_dataset_specific_moduli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SquaringModGenerationConfig(
                output_dir=directory,
                modulus_bits=[10, 11],
                fixed_time_steps=2,
                examples_per_setting=30,
                ood_time_steps=[4],
                ood_examples_per_setting=6,
                train_fraction=0.8,
                test_fraction=0.2,
                split_group="prompt",
                depth_evaluation_time_steps=[1, 2, 4],
                depth_evaluation_examples_per_setting=4,
                ood_n_depth_evaluation_modulus_bits=[12, 13],
                ood_n_depth_evaluation_examples_per_setting=3,
                seed=19,
                separate_input_output=True,
            )
            generate_squaring_mod_dataset(config)
            root = Path(directory)
            train = _rows(root, "train")
            train_moduli = {row["modulus"] for row in train}
            training_pairs = {(row["modulus"], row["x"]) for row in train}

            seen_expected_pairs = None
            ood_expected_pairs = None
            for time_steps in (1, 2, 4):
                seen_rows = _rows(root, f"depth_t_{time_steps}")
                self.assertEqual(len(seen_rows), 8)
                self.assertTrue(
                    {row["modulus"] for row in seen_rows} <= train_moduli
                )
                seen_pairs = {(row["modulus"], row["x"]) for row in seen_rows}
                self.assertTrue(seen_pairs.isdisjoint(training_pairs))
                if seen_expected_pairs is None:
                    seen_expected_pairs = seen_pairs
                else:
                    self.assertEqual(seen_pairs, seen_expected_pairs)

                ood_rows = _rows(root, f"depth_ood_n_t_{time_steps}")
                self.assertEqual(len(ood_rows), 6)
                self.assertEqual(
                    {
                        bits: sum(
                            row["configured_modulus_bits"] == bits
                            for row in ood_rows
                        )
                        for bits in (12, 13)
                    },
                    {12: 3, 13: 3},
                )
                self.assertTrue(
                    {row["modulus"] for row in ood_rows}.isdisjoint(train_moduli)
                )
                ood_pairs = {(row["modulus"], row["x"]) for row in ood_rows}
                if ood_expected_pairs is None:
                    ood_expected_pairs = ood_pairs
                else:
                    self.assertEqual(ood_pairs, ood_expected_pairs)

    def test_fixed_n_profile_reuses_only_its_dataset_modulus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SquaringModGenerationConfig(
                output_dir=directory,
                fixed_p=11,
                fixed_q=13,
                time_steps=[1, 2],
                examples_per_setting=100,
                ood_time_steps=[4],
                ood_examples_per_setting=100,
                train_fraction=0.8,
                test_fraction=0.2,
                split_group="prompt",
                depth_evaluation_time_steps=[1, 2, 4],
                depth_evaluation_examples_per_setting=10,
                seed=19,
                separate_input_output=True,
            )
            dataset_config = generate_squaring_mod_dataset(config)
            root = Path(directory)
            ordinary_pairs = {
                (row["modulus"], row["x"])
                for split in ("train", "test", "ood")
                for row in _rows(root, split)
            }
            for time_steps in (1, 2, 4):
                rows = _rows(root, f"depth_t_{time_steps}")
                self.assertEqual(len(rows), 10)
                self.assertEqual({row["modulus"] for row in rows}, {143})
                self.assertTrue(
                    {
                        (row["modulus"], row["x"]) for row in rows
                    }.isdisjoint(ordinary_pairs)
                )
            self.assertNotIn("depth_ood_n_t_1", dataset_config["split_counts"])


if __name__ == "__main__":
    unittest.main()
