from __future__ import annotations

import unittest

import torch

from benchmark.model_diagnostics import (
    capture_parameter_snapshot,
    ModelDiagnosticsAccumulator,
    summarize_parameter_gradients,
    summarize_parameter_updates,
    TrainingModelDiagnosticsTrajectory,
    validate_training_credit,
)
from benchmark.runner import (
    _format_model_diagnostics,
    _format_training_model_diagnostic,
)


class ModelDiagnosticsAccumulatorTests(unittest.TestCase):
    @staticmethod
    def _payload(
        *,
        counts: torch.Tensor | None = None,
        input_rms: float = 2.0,
        n_attention_mass: tuple[float, ...] = (0.0, 0.6, 0.2, 0.2, 0.0),
        include_predictions: bool = False,
    ) -> dict:
        if counts is None:
            counts = torch.tensor([0, 2, 1, 1, 0])
        attention = torch.zeros((5, 5))
        layer_counts = counts.clone()
        layer_counts[0] = 0
        stream_attention = torch.zeros((5, 3))
        if layer_counts[1] > 0:
            attention[1] = torch.tensor(n_attention_mass)
        if layer_counts[2] > 0:
            attention[2, 2] = 1.0
        if layer_counts[3] > 0:
            attention[3, 1] = 1.0
        if layer_counts[4] > 0:
            attention[4, 4] = 1.0
        stream_attention[layer_counts > 0] = torch.tensor([0.4, 0.1, 0.5])
        layer_count = int(layer_counts.sum().item())
        payload = {
            "segment_token_counts": counts,
            "segment_embedding": {
                "norms": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
                "delta_norms": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]),
                "relative_deltas": torch.tensor(
                    [0.0, 0.01, 0.02, 0.03, 0.04]
                ),
                "initial_cosines": torch.tensor([0.0, 0.9, 0.8, 0.7, 0.6]),
                "cosine_matrix": torch.eye(5),
            },
            "parameter_stats": {
                "attention": {
                    "norm": torch.tensor(10.0),
                    "delta_norm": torch.tensor(1.0),
                    "relative_delta": torch.tensor(0.1),
                }
            },
            "final_state_rms_by_segment": torch.tensor(
                [0.0, 1.0, 2.0, 3.0, 0.0]
            ),
            "final_logit_entropy_by_segment": torch.tensor(
                [0.0, 0.5, 0.6, 0.7, 0.0]
            ),
            "layers": [
                {
                    "step": 1,
                    "valid_query_count": layer_count,
                    "input_rms": torch.tensor(input_rms),
                    "attention_update_rms": torch.tensor(0.5),
                    "attention_update_ratio": torch.tensor(0.25),
                    "mlp_update_rms": torch.tensor(0.4),
                    "mlp_update_ratio": torch.tensor(0.2),
                    "output_rms": torch.tensor(2.5),
                    "input_output_cosine": torch.tensor(0.8),
                    "attention_entropy": torch.tensor(1.2),
                    "effective_attended_tokens": torch.tensor(3.3),
                    "segment_query_counts": layer_counts,
                    "attention_mass_by_segment": attention,
                    "attention_mass_by_stream": stream_attention,
                    "state_change_rms_by_segment": torch.tensor(
                        [0.0, 0.2, 0.3, 0.4, 0.0]
                    ),
                }
            ],
        }
        if include_predictions:
            # Normal prediction is class 1.  Stage 1 and the permuted
            # counterfactual predict another class; zero preserves it.
            stage = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
            zero = torch.tensor([[[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]])
            permuted = torch.tensor([[[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]])
            payload["stage_logits"] = [{"step": 1, "logits": stage}]
            payload["segment_counterfactual_logits"] = {
                "zero": zero,
                "permuted": permuted,
                "zero_nx": zero.clone(),
                "zero_t": zero.clone(),
                "swap_n_x": permuted.clone(),
            }
        return {"model_diagnostics": payload}

    def test_aggregates_batches_with_the_intended_weights(self) -> None:
        accumulator = ModelDiagnosticsAccumulator()
        accumulator.add(auxiliary=self._payload())
        accumulator.add(
            auxiliary=self._payload(
                counts=torch.tensor([0, 1, 3, 0, 0]),
                input_rms=4.0,
                n_attention_mass=(0.0, 0.2, 0.6, 0.2, 0.0),
            )
        )

        summary = accumulator.summary()
        assert summary is not None
        self.assertEqual(summary["batch_count"], 2)
        self.assertEqual(summary["valid_token_count"], 8)
        self.assertEqual(summary["segment_token_counts"], [0, 3, 4, 1, 0])
        self.assertAlmostEqual(
            summary["final_state_rms_by_segment"][1],
            1.0,
        )
        layer = summary["layers"][0]
        self.assertEqual(layer["valid_query_count"], 8)
        self.assertAlmostEqual(layer["input_rms"], (10.0) ** 0.5)
        self.assertAlmostEqual(
            layer["attention_mass_by_segment"][1][1],
            (2 * 0.6 + 1 * 0.2) / 3,
            places=6,
        )
        self.assertIsNone(layer["attention_mass_by_segment"][0][0])
        self.assertAlmostEqual(layer["attention_mass_by_stream"][1][2], 0.5)
        self.assertEqual(
            summary["parameter_stats"]["attention"]["last_batch"]["norm"],
            10.0,
        )

    def test_computes_stage_and_segment_counterfactual_predictions(self) -> None:
        accumulator = ModelDiagnosticsAccumulator()
        normal_logits = torch.tensor(
            [[[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]]
        )
        targets = torch.tensor([[1, -100]])
        target_positions = torch.tensor([[0, -1]])
        accumulator.add(
            auxiliary=self._payload(include_predictions=True),
            logits=normal_logits,
            targets=targets,
            target_positions=target_positions,
        )

        summary = accumulator.summary()
        assert summary is not None
        stage = summary["stage_predictions"][0]
        self.assertEqual(stage["step"], 1)
        self.assertEqual(stage["correct_examples"], 0)
        self.assertEqual(stage["target_token_count"], 1)
        counterfactuals = summary["segment_counterfactuals"]
        self.assertEqual(
            set(counterfactuals),
            {"zero", "permuted", "zero_nx", "zero_t", "swap_n_x"},
        )
        self.assertEqual(counterfactuals["zero"]["exact_accuracy"], 1.0)
        self.assertEqual(
            counterfactuals["zero"]["token_prediction_flip_rate"],
            0.0,
        )
        self.assertEqual(
            counterfactuals["zero"]["example_prediction_flip_rate"],
            0.0,
        )
        self.assertEqual(counterfactuals["permuted"]["exact_accuracy"], 0.0)
        self.assertEqual(
            counterfactuals["permuted"]["token_prediction_flip_rate"],
            1.0,
        )
        self.assertEqual(
            counterfactuals["permuted"]["example_prediction_flip_rate"],
            1.0,
        )

    def test_ignores_missing_payload_and_rejects_invalid_attention_rows(self) -> None:
        accumulator = ModelDiagnosticsAccumulator()
        for auxiliary in (None, {}, {"model_diagnostics": None}):
            accumulator.add(auxiliary=auxiliary)
        self.assertFalse(accumulator.has_data)
        self.assertIsNone(accumulator.summary())

        payload = self._payload()["model_diagnostics"]
        payload["layers"][0]["attention_mass_by_segment"][1].zero_()
        with self.assertRaisesRegex(ValueError, "rows with queries must sum"):
            accumulator.add(auxiliary={"model_diagnostics": payload})

    def test_formats_concise_model_rows(self) -> None:
        accumulator = ModelDiagnosticsAccumulator()
        accumulator.add(auxiliary=self._payload())
        summary = accumulator.summary()
        result = {
            "seeds": [
                {
                    "seed": 74,
                    "model_diagnostics": {
                        "scoring_splits": {"test": summary},
                        "first_uncertified_depth_rungs": {},
                    },
                }
            ]
        }

        output = _format_model_diagnostics(result)
        assert output is not None
        self.assertIn("MODEL_SEGMENTS | seed=74 split=test", output)
        self.assertIn("MODEL_WEIGHTS | seed=74 split=test", output)
        self.assertIn("MODEL_LAYER | seed=74 split=test step=1", output)
        self.assertIn(
            "MODEL_ATTENTION | seed=74 split=test step=1 query=N",
            output,
        )

    def test_formats_all_counterfactual_probes_with_meanings(self) -> None:
        accumulator = ModelDiagnosticsAccumulator()
        accumulator.add(
            auxiliary=self._payload(include_predictions=True),
            logits=torch.tensor(
                [[[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]]
            ),
            targets=torch.tensor([[1, -100]]),
            target_positions=torch.tensor([[0, -1]]),
        )
        result = {
            "seeds": [
                {
                    "seed": 74,
                    "model_diagnostics": {
                        "scoring_splits": {"test": accumulator.summary()},
                        "first_uncertified_depth_rungs": {},
                    },
                }
            ]
        }

        output = _format_model_diagnostics(result)
        assert output is not None
        self.assertEqual(output.count("MODEL_COUNTERFACTUAL"), 5)
        self.assertIn("probe=zero_nx meaning=N_X_signal_removed_T_preserved", output)
        self.assertIn("probe=zero_t meaning=T_signal_removed_N_X_preserved", output)
        self.assertIn("probe=swap_n_x meaning=N_X_roles_swapped_T_preserved", output)

    def test_summarizes_gradients_and_actual_optimizer_updates(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        loss = model(torch.tensor([[3.0, 4.0]])).sum()
        loss.backward()

        gradients = summarize_parameter_gradients(model)
        self.assertAlmostEqual(gradients["gradient_norm"], 5.0)
        self.assertEqual(gradients["missing_gradient_parameter_count"], 0)
        snapshot = capture_parameter_snapshot(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        updates = summarize_parameter_updates(model, snapshot)
        self.assertTrue(updates["available"])
        self.assertAlmostEqual(updates["update_norm"], 0.5, places=6)
        self.assertAlmostEqual(
            updates["parameters"]["weight"]["relative_update_norm"],
            0.5 / (5.0**0.5),
            places=6,
        )

    def test_reports_segment_rows_and_qkv_components(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.segment_embedding = torch.nn.Embedding(5, 2)
                self.processor = torch.nn.Module()
                self.processor.block = torch.nn.Module()
                self.processor.block.qkv = torch.nn.Linear(2, 6)

        model = TinyModel()
        sum(parameter.sum() for parameter in model.parameters()).backward()
        gradients = summarize_parameter_gradients(model)["parameters"]
        segment = gradients["segment_embedding.weight"]
        self.assertEqual(len(segment["row_gradient_norms"]), 5)
        self.assertAlmostEqual(segment["row_gradient_norms"][1], 2.0**0.5)
        qkv_weight = gradients["processor.block.qkv.weight"]
        self.assertEqual(
            set(qkv_weight["component_gradient_norms"]),
            {"q", "k", "v"},
        )
        self.assertAlmostEqual(
            qkv_weight["component_gradient_norms"]["q"],
            2.0,
        )

        snapshot = capture_parameter_snapshot(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.1)
        updates = summarize_parameter_updates(model, snapshot)["parameters"]
        self.assertEqual(
            len(updates["segment_embedding.weight"]["row_update_norms"]),
            5,
        )
        self.assertIn(
            "component_update_norms",
            updates["processor.block.qkv.bias"],
        )

    def test_training_trajectory_is_bounded_and_keeps_terminal_record(self) -> None:
        trajectory = TrainingModelDiagnosticsTrajectory(max_records=4)
        for step in range(1, 21):
            trajectory.record({"step": step, "marker": step})

        summary = trajectory.summary()
        assert summary is not None
        self.assertLessEqual(len(summary["records"]), 4)
        self.assertEqual(summary["records"][0]["step"], 1)
        self.assertEqual(summary["records"][-1]["step"], 20)
        self.assertEqual(summary["observed_checkpoints"], 20)
        self.assertEqual(
            summary["checkpoint_phase"],
            "post_optimizer_on_training_batch",
        )

    def test_validates_training_credit_payload(self) -> None:
        payload = {
            "segment_token_counts": torch.tensor([0, 4, 3, 2, 0]),
            "segment_signal_grad_rms": torch.tensor(0.25),
            "segment_signal_grad_rms_by_segment": torch.tensor(
                [0.0, 0.1, 0.2, 0.3, 0.0]
            ),
            "stages": [
                {
                    "step": 0,
                    "state_grad_rms": torch.tensor(0.4),
                    "relative_to_final": torch.tensor(0.5),
                    "state_grad_rms_by_segment": torch.tensor(
                        [0.0, 0.1, 0.2, 0.3, 0.0]
                    ),
                    "control_grad_rms": None,
                },
                {
                    "step": 1,
                    "state_grad_rms": torch.tensor(0.8),
                    "relative_to_final": torch.tensor(1.0),
                    "state_grad_rms_by_segment": torch.tensor(
                        [0.1, 0.2, 0.3, 0.4, 0.0]
                    ),
                    "control_grad_rms": torch.tensor(0.2),
                },
            ],
        }

        normalized = validate_training_credit(payload)
        assert normalized is not None
        self.assertEqual(normalized["segment_names"][0], "OTHER")
        self.assertEqual(normalized["segment_token_counts"], [0, 4, 3, 2, 0])
        self.assertAlmostEqual(normalized["segment_signal_grad_rms"], 0.25)
        self.assertEqual(normalized["stages"][0]["control_grad_rms"], None)
        self.assertAlmostEqual(
            normalized["stages"][1]["relative_to_final"],
            1.0,
        )

    def test_formats_real_training_telemetry_schema_end_to_end(self) -> None:
        accumulator = ModelDiagnosticsAccumulator()
        accumulator.add(
            auxiliary=self._payload(include_predictions=True),
            logits=torch.tensor(
                [[[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]]
            ),
            targets=torch.tensor([[1, -100]]),
            target_positions=torch.tensor([[0, -1]]),
        )
        model_summary = accumulator.summary()
        assert model_summary is not None

        model = torch.nn.Linear(2, 1, bias=False)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        gradients = summarize_parameter_gradients(model)
        snapshot = capture_parameter_snapshot(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        updates = summarize_parameter_updates(model, snapshot)
        credit = validate_training_credit(
            {
                "segment_token_counts": torch.tensor([0, 2, 1, 1, 0]),
                "segment_signal_grad_rms": torch.tensor(0.2),
                "segment_signal_grad_rms_by_segment": torch.tensor(
                    [0.0, 0.1, 0.2, 0.3, 0.0]
                ),
                "stages": [
                    {
                        "step": 0,
                        "state_grad_rms": torch.tensor(0.4),
                        "relative_to_final": torch.tensor(1.0),
                        "state_grad_rms_by_segment": torch.tensor(
                            [0.0, 0.1, 0.2, 0.3, 0.0]
                        ),
                        "control_grad_rms": None,
                    }
                ],
            }
        )
        record = {
            "step": 1,
            "training_loss": 1.25,
            "training_exact_accuracy": 0.25,
            "diagnostic_task_cross_entropy": 1.0,
            "diagnostic_exact_accuracy": 0.5,
            "model": model_summary,
            "optimization": {
                "gradient_before_clipping": gradients,
                "gradient_after_clipping": gradients,
                "gradient_clip_threshold": None,
                "gradient_clip_scale": 1.0,
                "optimizer_parameter_groups": [
                    {"group": 0, "learning_rate": 0.1, "weight_decay": 0.0}
                ],
                "optimizer_update": updates,
            },
            "training_credit": credit,
        }

        output = _format_training_model_diagnostic(seed=74, record=record)

        self.assertIn("pre_update_backward_loss=1.250000", output)
        self.assertIn("post_update_same_batch_CE=1.000000", output)
        self.assertIn("MODEL_TRAIN_OPTIMIZER", output)
        self.assertIn("MODEL_TRAIN_SEGMENT", output)
        self.assertIn("MODEL_TRAIN_BRANCH", output)
        self.assertIn("MODEL_TRAIN_CREDIT", output)


if __name__ == "__main__":
    unittest.main()
