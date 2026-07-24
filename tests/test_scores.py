from __future__ import annotations

import unittest

from client.cli import _max_t, _score


class ScoreFormattingTests(unittest.TestCase):
    def test_time_budgeted_score_is_primary(self) -> None:
        result = {
            "score": {
                "primary_metric": "mean_exact_accuracy",
                "mean_exact_accuracy": 0.75,
            }
        }
        self.assertEqual(_score({"result": result}), "75.00%")

    def test_sanitized_public_score_does_not_require_result_json(self) -> None:
        self.assertEqual(_score({"score": 0.625}), "62.50%")


    def test_max_t_formats_public_and_private_results(self) -> None:
        self.assertEqual(
            _max_t({"max_certified_time_steps": 32}, "max_certified_time_steps"),
            "32",
        )
        self.assertEqual(
            _max_t({"status": "succeeded"}, "max_certified_time_steps"),
            "<1",
        )
        self.assertEqual(
            _max_t(
                {"status": "succeeded", "ood_n_profile_available": False},
                "ood_n_max_certified_time_steps",
            ),
            "N/A",
        )
        self.assertEqual(_max_t({}, "max_certified_time_steps"), "—")

if __name__ == "__main__":
    unittest.main()
