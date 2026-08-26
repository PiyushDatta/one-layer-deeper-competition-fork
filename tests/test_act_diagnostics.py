from __future__ import annotations

import unittest

import torch

from benchmark.act_diagnostics import ActDiagnosticsAccumulator
from benchmark.runner import _format_act_diagnostics


class ActDiagnosticsAccumulatorTests(unittest.TestCase):
    @staticmethod
    def _auxiliary(
        update_counts: torch.Tensor,
        remainders: torch.Tensor,
        cap_forced_mask: torch.Tensor,
        *,
        global_iterations: int,
        max_loops: int = 20,
    ) -> dict:
        return {
            "act": {
                "update_counts": update_counts,
                "remainders": remainders,
                "cap_forced_mask": cap_forced_mask,
                "max_loops": max_loops,
                "global_iterations": global_iterations,
                "ponder_weight": 0.01,
            }
        }

    def test_summarizes_two_batches_and_excludes_padding(self) -> None:
        accumulator = ActDiagnosticsAccumulator()

        first_mask = torch.tensor(
            [
                [True, True, True, True, False, False],
                [True, True, True, True, True, True],
            ]
        )
        first_updates = torch.tensor(
            [
                [1, 2, 3, 4, 99, 99],
                [5, 6, 7, 8, 9, 20],
            ],
            dtype=torch.float32,
        )
        first_remainders = torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1, -5.0, -5.0],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )
        first_forced = torch.tensor(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, False, False],
            ]
        )
        accumulator.add(
            auxiliary=self._auxiliary(
                first_updates,
                first_remainders,
                first_forced,
                global_iterations=20,
            ),
            attention_mask=first_mask,
            exact_rows=torch.tensor([True, False]),
            rows_with_targets=torch.tensor([True, True]),
        )

        second_mask = torch.tensor(
            [
                [True, True, True, False, False, False, False],
                [True, True, True, True, True, True, True],
            ]
        )
        second_updates = torch.tensor(
            [
                [10, 11, 12, 99, 99, 99, 99],
                [13, 14, 15, 16, 17, 18, 20],
            ],
            dtype=torch.float32,
        )
        second_remainders = torch.tensor(
            [
                [0.1, 0.1, 0.1, -5.0, -5.0, -5.0, -5.0],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        second_forced = torch.tensor(
            [
                [False, False, False, True, True, True, True],
                [False, False, False, False, False, False, True],
            ]
        )
        accumulator.add(
            auxiliary=self._auxiliary(
                second_updates,
                second_remainders,
                second_forced,
                global_iterations=20,
            ),
            attention_mask=second_mask,
            exact_rows=torch.tensor([True, False]),
            rows_with_targets=torch.tensor([True, True]),
        )

        summary = accumulator.summary(evaluation_task_cross_entropy=1.25)
        self.assertIsNotNone(summary)
        assert summary is not None

        self.assertEqual(summary["valid_token_count"], 20)
        self.assertEqual(summary["example_count"], 4)
        self.assertEqual(summary["batch_count"], 2)
        self.assertEqual(summary["max_loops"], 20)
        self.assertEqual(summary["evaluation_task_cross_entropy"], 1.25)

        # The valid update counts are 1..18, 20, 20. These assertions pin a
        # midpoint median and discrete nearest-rank upper quantiles.
        updates = summary["token_update_counts"]
        self.assertAlmostEqual(updates["mean"], 10.55)
        self.assertEqual(updates["median"], 10.5)
        self.assertEqual(updates["p90"], 18.0)
        self.assertEqual(updates["p95"], 20.0)
        self.assertEqual(updates["p99"], 20.0)
        self.assertEqual(updates["maximum"], 20.0)

        global_iterations = summary["global_iterations_per_batch"]
        self.assertEqual(global_iterations["mean"], 20.0)
        self.assertEqual(global_iterations["median"], 20.0)
        self.assertEqual(global_iterations["maximum"], 20.0)

        cap_hits = summary["cap_hits"]
        self.assertAlmostEqual(cap_hits["token_reached_cap_rate"], 2 / 20)
        self.assertAlmostEqual(cap_hits["token_forced_cap_rate"], 1 / 20)
        self.assertAlmostEqual(cap_hits["batch_reached_cap_rate"], 1.0)
        self.assertAlmostEqual(cap_hits["batch_forced_cap_rate"], 0.5)

        remainders = summary["remainders"]
        self.assertAlmostEqual(remainders["mean"], 0.14, places=6)
        self.assertAlmostEqual(remainders["mean_natural_halt"], 0.1, places=6)
        self.assertAlmostEqual(remainders["mean_cap_forced"], 0.9, places=6)
        self.assertAlmostEqual(summary["raw_mean_ponder_time"], 10.69, places=6)
        self.assertAlmostEqual(
            summary["weighted_ponder_contribution"], 0.1069, places=6
        )

        endings = summary["iteration_end_percentages"]
        self.assertEqual(len(endings), 20)
        self.assertAlmostEqual(endings[0]["ended_at_percentage"], 5.0)
        self.assertAlmostEqual(endings[0]["ended_by_percentage"], 5.0)
        self.assertAlmostEqual(endings[9]["ended_by_percentage"], 50.0)
        self.assertAlmostEqual(endings[17]["ended_by_percentage"], 90.0)
        self.assertAlmostEqual(endings[18]["ended_at_percentage"], 0.0)
        self.assertAlmostEqual(endings[18]["ended_by_percentage"], 90.0)
        self.assertAlmostEqual(endings[19]["ended_at_percentage"], 10.0)
        self.assertAlmostEqual(endings[19]["ended_by_percentage"], 100.0)
        self.assertAlmostEqual(
            endings[19]["naturally_halted_at_percentage"], 5.0
        )
        self.assertAlmostEqual(endings[19]["cap_forced_at_percentage"], 5.0)
        self.assertAlmostEqual(
            endings[19]["naturally_halted_by_percentage"], 95.0
        )
        self.assertAlmostEqual(endings[19]["cap_forced_by_percentage"], 5.0)

        by_correctness = summary["by_correctness"]
        correct = by_correctness["correct"]
        self.assertEqual(correct["example_count"], 2)
        self.assertAlmostEqual(correct["mean_updates_per_example"], 6.75)
        self.assertAlmostEqual(correct["median_updates_per_example"], 6.75)
        self.assertAlmostEqual(correct["p95_updates_per_example"], 11.0)
        self.assertAlmostEqual(
            correct["mean_ponder_time_per_example"], 6.85, places=5
        )

        incorrect = by_correctness["incorrect"]
        self.assertEqual(incorrect["example_count"], 2)
        self.assertAlmostEqual(
            incorrect["mean_updates_per_example"], ((55 / 6) + (113 / 7)) / 2,
            delta=1e-5,
        )
        self.assertAlmostEqual(
            incorrect["median_updates_per_example"],
            ((55 / 6) + (113 / 7)) / 2,
            delta=1e-5,
        )
        self.assertAlmostEqual(
            incorrect["p95_updates_per_example"], 113 / 7, delta=1e-5
        )
        self.assertAlmostEqual(
            incorrect["mean_ponder_time_per_example"],
            ((55.6 / 6) + (114.5 / 7)) / 2,
            delta=1e-5,
        )

        by_length = summary["by_sequence_length"]
        self.assertEqual(set(by_length), {"3", "4", "6", "7"})
        self.assertEqual(by_length["3"]["example_count"], 1)
        self.assertEqual(by_length["3"]["exact_accuracy"], 1.0)
        self.assertAlmostEqual(by_length["3"]["mean_updates_per_example"], 11.0)
        self.assertEqual(by_length["6"]["exact_accuracy"], 0.0)
        self.assertAlmostEqual(by_length["6"]["token_reached_cap_rate"], 1 / 6)
        self.assertAlmostEqual(by_length["6"]["token_forced_cap_rate"], 0.0)
        self.assertEqual(by_length["7"]["exact_accuracy"], 0.0)
        self.assertAlmostEqual(
            by_length["7"]["mean_updates_per_example"], 113 / 7, delta=1e-5
        )
        self.assertAlmostEqual(by_length["7"]["token_reached_cap_rate"], 1 / 7)
        self.assertAlmostEqual(by_length["7"]["token_forced_cap_rate"], 1 / 7)

    def test_returns_none_when_auxiliary_has_no_act_payload(self) -> None:
        accumulator = ActDiagnosticsAccumulator()
        mask = torch.ones((1, 2), dtype=torch.bool)
        exact_rows = torch.tensor([True])
        rows_with_targets = torch.tensor([True])

        for auxiliary in (None, {}, {"act": None}):
            accumulator.add(
                auxiliary=auxiliary,
                attention_mask=mask,
                exact_rows=exact_rows,
                rows_with_targets=rows_with_targets,
            )

        self.assertFalse(accumulator.has_data)
        self.assertIsNone(
            accumulator.summary(evaluation_task_cross_entropy=0.5)
        )

    def test_no_cap_hits_report_none_instead_of_nan(self) -> None:
        accumulator = ActDiagnosticsAccumulator()
        accumulator.add(
            auxiliary=self._auxiliary(
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([[0.2, 0.3]]),
                torch.tensor([[False, False]]),
                global_iterations=2,
            ),
            attention_mask=torch.tensor([[True, True]]),
            exact_rows=torch.tensor([True]),
            rows_with_targets=torch.tensor([True]),
        )

        summary = accumulator.summary(evaluation_task_cross_entropy=0.5)
        assert summary is not None
        self.assertEqual(summary["cap_hits"]["token_reached_cap_rate"], 0.0)
        self.assertEqual(summary["cap_hits"]["batch_reached_cap_rate"], 0.0)
        self.assertEqual(summary["cap_hits"]["token_forced_cap_rate"], 0.0)
        self.assertEqual(summary["cap_hits"]["batch_forced_cap_rate"], 0.0)
        self.assertIsNone(summary["remainders"]["mean_cap_forced"])

    def test_iteration_detail_is_bounded_for_a_large_cap(self) -> None:
        accumulator = ActDiagnosticsAccumulator()
        accumulator.add(
            auxiliary=self._auxiliary(
                torch.tensor([[1_000.0]]),
                torch.tensor([[0.8]]),
                torch.tensor([[True]]),
                global_iterations=1_000,
                max_loops=1_000,
            ),
            attention_mask=torch.tensor([[True]]),
            exact_rows=torch.tensor([False]),
            rows_with_targets=torch.tensor([True]),
        )

        summary = accumulator.summary(evaluation_task_cross_entropy=0.5)
        assert summary is not None
        self.assertTrue(summary["iteration_detail_truncated"])
        self.assertEqual(len(summary["iteration_end_percentages"]), 256)
        self.assertEqual(
            summary["iteration_detail_unreported_token_percentage"], 100.0
        )

    def test_formats_scored_and_depth_diagnostics(self) -> None:
        accumulator = ActDiagnosticsAccumulator()
        accumulator.add(
            auxiliary=self._auxiliary(
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([[0.2, 0.3]]),
                torch.tensor([[False, False]]),
                global_iterations=2,
            ),
            attention_mask=torch.tensor([[True, True]]),
            exact_rows=torch.tensor([True]),
            rows_with_targets=torch.tensor([True]),
        )
        summary = accumulator.summary(evaluation_task_cross_entropy=0.5)
        result = {
            "seeds": [
                {
                    "seed": 74,
                    "act_diagnostics": {
                        "scoring_splits": {"test": summary},
                        "first_uncertified_depth_rungs": {
                            "seen_n": {
                                "time_steps": 1,
                                "diagnostics": summary,
                            }
                        },
                    },
                }
            ]
        }

        output = _format_act_diagnostics(result)
        assert output is not None
        self.assertIn("seed=74 split=test", output)
        self.assertIn("seed=74 split=seen_n/T=1", output)
        self.assertIn("eval_task_CE=0.500000", output)
        self.assertIn("ACT_PROCESSING_ENDED_AT/BY", output)
        self.assertNotIn("ACT_HALTING_AT/BY", output)

    def test_unlabeled_rows_still_count_toward_global_compute(self) -> None:
        accumulator = ActDiagnosticsAccumulator()
        accumulator.add(
            auxiliary=self._auxiliary(
                torch.tensor([[1.0], [20.0]]),
                torch.tensor([[0.2], [0.8]]),
                torch.tensor([[False], [True]]),
                global_iterations=20,
            ),
            attention_mask=torch.tensor([[True], [True]]),
            exact_rows=torch.tensor([True, False]),
            rows_with_targets=torch.tensor([True, False]),
        )

        summary = accumulator.summary(evaluation_task_cross_entropy=0.5)
        assert summary is not None
        self.assertEqual(summary["valid_token_count"], 2)
        self.assertEqual(summary["example_count"], 1)
        self.assertEqual(
            summary["global_iterations_per_batch"]["maximum"], 20.0
        )
        self.assertEqual(summary["cap_hits"]["token_forced_cap_rate"], 0.5)
        self.assertEqual(
            summary["by_correctness"]["correct"]["example_count"], 1
        )
        self.assertEqual(
            summary["by_correctness"]["incorrect"]["example_count"], 0
        )

    def test_rejects_non_detached_or_misshapen_payload_tensors(self) -> None:
        accumulator = ActDiagnosticsAccumulator()
        mask = torch.ones((1, 2), dtype=torch.bool)
        exact_rows = torch.tensor([True])
        rows_with_targets = torch.tensor([True])
        remainders = torch.full((1, 2), 0.1)
        forced = torch.zeros((1, 2), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "update_counts must be detached"):
            accumulator.add(
                auxiliary=self._auxiliary(
                    torch.ones((1, 2), requires_grad=True),
                    remainders,
                    forced,
                    global_iterations=1,
                ),
                attention_mask=mask,
                exact_rows=exact_rows,
                rows_with_targets=rows_with_targets,
            )

        with self.assertRaisesRegex(
            ValueError, r"remainders must have shape \(1, 2\)"
        ):
            accumulator.add(
                auxiliary=self._auxiliary(
                    torch.ones((1, 2)),
                    torch.ones((1, 3)),
                    forced,
                    global_iterations=1,
                ),
                attention_mask=mask,
                exact_rows=exact_rows,
                rows_with_targets=rows_with_targets,
            )

        with self.assertRaisesRegex(ValueError, "largest valid-token"):
            accumulator.add(
                auxiliary=self._auxiliary(
                    torch.tensor([[1.0, 2.0]]),
                    remainders,
                    forced,
                    global_iterations=1,
                ),
                attention_mask=mask,
                exact_rows=exact_rows,
                rows_with_targets=rows_with_targets,
            )


if __name__ == "__main__":
    unittest.main()
