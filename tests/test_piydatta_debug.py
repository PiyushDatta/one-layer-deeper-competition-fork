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
        self.original_debug_setting = self.namespace["DBUG"]
        self.model_class = self.namespace["Model"]
        self.spec = ModelSpec(17, 4, 1_000_000)
        self.input_ids = torch.tensor([[1, 2, 3, 0]])
        self.attention_mask = torch.tensor([[True, True, True, False]])

    def tearDown(self) -> None:
        self.namespace["DBUG"] = self.original_debug_setting

    def test_final_path_returns_only_the_scalar_ponder_cost(self) -> None:
        self.namespace["DBUG"] = False
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
        self.assertEqual(
            diagnostics["tail_forced_mask"].shape,
            self.attention_mask.shape,
        )
        self.assertFalse(diagnostics["update_counts"].requires_grad)

    def test_tail_cutoff_is_counted_per_example(self) -> None:
        self.namespace["DBUG"] = True
        width = self.namespace["D_MODEL"]
        processor_class = self.namespace["UniversalProcessor"]

        class IdentityBlock(torch.nn.Module):
            def forward(self, x, attention_mask):
                return x

        class FirstChannelHaltingUnit(torch.nn.Module):
            def forward(self, x):
                return x[..., :1]

        time_embedding = torch.nn.Embedding(2, width)
        torch.nn.init.zeros_(time_embedding.weight)
        processor = processor_class(
            IdentityBlock(),
            time_embedding,
            use_act=True,
            max_loops=2,
            halting_unit=FirstChannelHaltingUnit(),
            tail_halt_fraction=0.89,
        )
        x = torch.zeros(2, 10, width)
        x[0, :9, 0] = 10.0
        x[0, 9, 0] = -10.0
        x[1, :7, 0] = 10.0
        x[1, 7, 0] = -10.0
        attention_mask = torch.tensor(
            [
                [True] * 10,
                [True] * 8 + [False] * 2,
            ]
        )

        _, _, diagnostics = processor(
            x,
            torch.zeros(10, width),
            attention_mask,
            collect_act_diagnostics=True,
        )

        assert diagnostics is not None
        self.assertTrue(diagnostics["tail_forced_mask"][0, 9])
        self.assertFalse(diagnostics["tail_forced_mask"][1].any())
        self.assertTrue(diagnostics["cap_forced_mask"][1, 7])
        self.assertEqual(diagnostics["global_iterations"], 2)

        one_step_embedding = torch.nn.Embedding(1, width)
        torch.nn.init.zeros_(one_step_embedding.weight)
        one_step_processor = processor_class(
            IdentityBlock(),
            one_step_embedding,
            use_act=True,
            max_loops=1,
            halting_unit=FirstChannelHaltingUnit(),
            tail_halt_fraction=0.89,
        )
        _, _, final_step_diagnostics = one_step_processor(
            x[:1],
            torch.zeros(10, width),
            attention_mask[:1],
            collect_act_diagnostics=True,
        )
        assert final_step_diagnostics is not None
        self.assertFalse(final_step_diagnostics["tail_forced_mask"].any())
        self.assertTrue(final_step_diagnostics["cap_forced_mask"][0, 9])


if __name__ == "__main__":
    unittest.main()
