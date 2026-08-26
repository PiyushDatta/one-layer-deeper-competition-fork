from __future__ import annotations

import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec
from benchmark.runner import _load_submission_file


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = (
    ROOT / "submissions" / "piydatta_submission" / "submission.py"
)


class PiydattaDebugBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.submission = _load_submission_file(SUBMISSION_PATH)
        self.namespace = self.submission.build_model.__globals__
        self.model_class = self.namespace["Model"]
        self.spec = ModelSpec(17, 4, 1_000_000)
        self.input_ids = torch.tensor([[1, 2, 3, 0]])
        self.attention_mask = torch.tensor([[True, True, True, False]])

    def tearDown(self) -> None:
        self.namespace["DBUG"] = False

    def test_final_path_returns_only_the_scalar_ponder_cost(self) -> None:
        self.assertIs(self.namespace["DBUG"], False)
        model = self.model_class(self.spec, use_act=True)
        self.assertFalse(hasattr(model, "collect_act_diagnostics"))

        logits, auxiliary = model(self.input_ids, self.attention_mask)

        self.assertEqual(logits.shape, (1, 4, 17))
        self.assertTrue(torch.is_tensor(auxiliary))
        self.assertEqual(auxiliary.ndim, 0)

        fixed_model = self.model_class(self.spec, use_act=False)
        _, fixed_auxiliary = fixed_model(
            self.input_ids, self.attention_mask
        )
        self.assertFalse(hasattr(fixed_model, "collect_act_diagnostics"))
        self.assertTrue(torch.is_tensor(fixed_auxiliary))
        self.assertEqual(fixed_auxiliary.ndim, 0)

    def test_debug_path_exposes_diagnostics_only_when_enabled(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=True)
        self.assertIs(model.collect_act_diagnostics, False)
        model.collect_act_diagnostics = True

        _, auxiliary = model(self.input_ids, self.attention_mask)

        self.assertIsInstance(auxiliary, dict)
        diagnostics = auxiliary["act"]
        self.assertIsNotNone(diagnostics)
        self.assertEqual(
            diagnostics["cap_forced_mask"].shape,
            self.attention_mask.shape,
        )
        self.assertFalse(diagnostics["update_counts"].requires_grad)


if __name__ == "__main__":
    unittest.main()
