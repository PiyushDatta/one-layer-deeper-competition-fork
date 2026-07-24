from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmark import ModelSpec, OptimizerBundle, Submission
from benchmark.manifest import load_manifest
from benchmark.metrics import MetricRecorder
from data.squaring_mod import generate_squaring_mod_smoke_dataset

from benchmark.runner import (
    _evaluate,
    _depth_split_names,
    _resolve_batch_sizes,
    _run_seed,
    _scoring_split_names,
    _train,
    _with_batch_size,
)


ROOT = Path(__file__).resolve().parents[1]


class RunnerBudgetTests(unittest.TestCase):
    def test_official_scoring_split_layouts(self) -> None:
        self.assertEqual(
            _scoring_split_names(
                {
                    "train": object(),
                    "test": object(),
                    "ood": object(),
                }
            ),
            ("test", "ood"),
        )
        self.assertEqual(
            _scoring_split_names(
                {
                    "train": object(),
                    "ood_n_t": object(),
                    "ood_t": object(),
                    "test": object(),
                }
            ),
            ("test", "ood_t", "ood_n_t"),
        )
        dataloaders = {
            "train": object(),
            "test": object(),
            "depth_t_64": object(),
            "depth_t_2": object(),
            "depth_t_16": object(),
            "depth_ood_n_t_2": object(),
            "depth_ood_n_t_64": object(),
        }
        self.assertEqual(_scoring_split_names(dataloaders), ("test",))
        self.assertEqual(
            _depth_split_names(dataloaders),
            ("depth_t_2", "depth_t_16", "depth_t_64"),
        )
        self.assertEqual(
            _depth_split_names(dataloaders, "depth_ood_n_t_"),
            ("depth_ood_n_t_2", "depth_ood_n_t_64"),
        )

    def test_squaring_mod_generation_omits_and_removes_eval_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "eval.jsonl").write_text("stale\n", encoding="utf-8")
            config = generate_squaring_mod_smoke_dataset(root, seed=74)

            self.assertEqual(
                {path.name for path in root.glob("*.jsonl")},
                {"train.jsonl", "test.jsonl", "ood.jsonl"},
            )
            self.assertEqual(
                config["split_counts"],
                {"ood": 100, "test": 60, "train": 240},
            )
            self.assertNotIn("eval_fraction", config["generator_config"])

    def test_submission_can_change_training_and_evaluation_batch_size(self) -> None:
        manifest = load_manifest(ROOT / "benchmark" / "manifests" / "smoke_cpu.json")
        dataset = torch.utils.data.TensorDataset(torch.arange(40))
        dataloaders = {
            "train": torch.utils.data.DataLoader(dataset, batch_size=8),
            "test": torch.utils.data.DataLoader(dataset, batch_size=16),
        }
        resized = _with_batch_size(
            dataloaders,
            manifest,
            batch_size=4,
            eval_batch_size=10,
            seed=74,
        )
        self.assertEqual(resized["train"].batch_size, 4)
        self.assertIs(resized["train"].dataset, dataset)
        self.assertEqual(resized["test"].batch_size, 10)
        self.assertIs(resized["test"].dataset, dataset)

    def test_submission_import_time_is_charged_before_model_construction(self) -> None:
        model_built = False

        def build_model(spec):
            nonlocal model_built
            model_built = True
            return torch.nn.Linear(1, 1)

        submission = Submission(
            build_model=build_model,
            build_optimizer=lambda model, spec: None,
            batch_size=7,
        )
        manifest = load_manifest(ROOT / "benchmark" / "manifests" / "smoke_cpu.json")
        defaults = Submission(
            build_model=build_model,
            build_optimizer=lambda model, spec: None,
        )
        self.assertEqual(
            _resolve_batch_sizes(defaults, manifest),
            (manifest.data.batch_size, manifest.data.eval_batch_size),
        )
        explicit_eval = replace(defaults, eval_batch_size=11)
        self.assertEqual(
            _resolve_batch_sizes(explicit_eval, manifest),
            (manifest.data.batch_size, 11),
        )
        model_spec = ModelSpec(1, 1, 1)

        with patch("benchmark.runner.make_dataloaders", return_value={}) as make:
            with self.assertRaisesRegex(TimeoutError, "import exhausted"):
                _run_seed(
                    submission,
                    manifest,
                    model_spec,
                    torch.device("cpu"),
                    seed=74,
                    budget_seconds=0.1,
                    submission_load_seconds=1.0,
                )
        data_config = make.call_args.args[0]
        self.assertEqual(data_config.batch_size, 7)
        self.assertEqual(data_config.eval_batch_size, 7)
        self.assertFalse(model_built)

    def test_evaluation_fails_when_its_separate_budget_is_exhausted(self) -> None:
        model = torch.nn.Linear(1, 1)

        with patch("benchmark.runner.time.monotonic", return_value=10.0):
            with self.assertRaisesRegex(
                TimeoutError,
                r"evaluation exhausted its 5\.0s time budget",
            ):
                _evaluate(
                    model,
                    [object()],
                    object(),
                    torch.device("cpu"),
                    deadline=10.0,
                    budget_seconds=5.0,
                )

    def test_training_metrics_include_terminal_step_between_log_intervals(self) -> None:
        model = torch.nn.Linear(1, 1)
        bundle = OptimizerBundle(torch.optim.SGD(model.parameters(), lr=0.01))
        manifest = SimpleNamespace(
            runtime=SimpleNamespace(grad_clip=None, log_every=100),
            model_state=object(),
        )
        recorder = MetricRecorder()

        def loss_and_accuracy(*args, **kwargs):
            return model.weight.sum(), 0.5, 1, 1

        with (
            patch("benchmark.runner._loss_and_accuracy", side_effect=loss_and_accuracy),
            patch("benchmark.runner.validate_model_state"),
            patch("benchmark.runner.validate_optimizer", return_value=0),
            patch("benchmark.runner.time.monotonic", return_value=1.0),
        ):
            _, completed_steps, _, _ = _train(
                raw_model=model,
                train_model=model,
                training_loss=None,
                bundle=bundle,
                dataloader=[object()],
                manifest=manifest,
                device=torch.device("cpu"),
                started_at=0.0,
                deadline=2.0,
                budget_seconds=2.0,
                max_steps=37,
                seed=74,
                metric_recorder=recorder,
            )

        self.assertEqual(completed_steps, 37)
        self.assertEqual(
            [record["step"] for record in recorder.snapshot()],
            [1, 37],
        )

    def test_seed_receives_half_its_training_allowance_for_evaluation(self) -> None:
        manifest = load_manifest(ROOT / "benchmark" / "manifests" / "smoke_cpu.json")
        model_spec = ModelSpec(1, 1, 2)

        def build_model(spec):
            model = torch.nn.Linear(1, 1)
            model.config = SimpleNamespace(
                vocab_size=spec.vocab_size,
                max_seq_len=spec.max_seq_len,
            )
            return model

        submission = Submission(
            build_model=build_model,
            build_optimizer=lambda model, spec: OptimizerBundle(
                torch.optim.SGD(model.parameters(), lr=0.1)
            ),
            max_steps=manifest.runtime.max_steps + 1,
        )
        evaluation = {"loss": 0.0, "exact_accuracy": 1.0}
        dataset = torch.utils.data.TensorDataset(torch.arange(64))
        dataloaders = {
            "train": torch.utils.data.DataLoader(dataset, batch_size=32),
            "test": torch.utils.data.DataLoader(dataset, batch_size=64),
        }

        with (
            patch(
                "benchmark.runner._train",
                return_value=(0.0, 1, 1.0, 0),
            ) as train,
            patch("benchmark.runner._evaluate", return_value=evaluation) as evaluate,
        ):
            result = _run_seed(
                submission,
                manifest,
                model_spec,
                torch.device("cpu"),
                seed=74,
                budget_seconds=10.0,
                submission_load_seconds=0.0,
                dataloaders=dataloaders,
            )

        self.assertEqual(train.call_args.kwargs["max_steps"], manifest.runtime.max_steps)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(evaluate.call_args.kwargs["budget_seconds"], 5.0)
        self.assertEqual(result["evaluation_budget_seconds"], 5.0)
        self.assertEqual(result["training_batch_size"], manifest.data.batch_size)
        self.assertEqual(
            result["evaluation_batch_size"],
            manifest.data.eval_batch_size,
        )
        self.assertEqual(result["evaluation"], {"test": evaluation})

    def test_depth_profile_requires_a_perfect_prefix_but_evaluates_later_rungs(self) -> None:
        manifest = load_manifest(ROOT / "benchmark" / "manifests" / "smoke_cpu.json")
        model_spec = ModelSpec(1, 1, 2)

        def build_model(spec):
            model = torch.nn.Linear(1, 1)
            model.config = SimpleNamespace(
                vocab_size=spec.vocab_size,
                max_seq_len=spec.max_seq_len,
            )
            return model

        submission = Submission(
            build_model=build_model,
            build_optimizer=lambda model, spec: OptimizerBundle(
                torch.optim.SGD(model.parameters(), lr=0.1)
            ),
        )
        dataset = torch.utils.data.TensorDataset(torch.arange(4))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        dataloaders = {
            "train": loader,
            "test": loader,
            "depth_t_1": loader,
            "depth_t_2": loader,
            "depth_t_4": loader,
            "depth_t_8": loader,
            "depth_ood_n_t_1": loader,
            "depth_ood_n_t_2": loader,
            "depth_ood_n_t_4": loader,
            "depth_ood_n_t_8": loader,
        }
        primary = {
            "loss": 1.0,
            "exact_accuracy": 0.5,
            "correct_examples": 2,
            "example_count": 4,
        }
        perfect = {
            "loss": 0.0,
            "exact_accuracy": 1.0,
            "correct_examples": 4,
            "example_count": 4,
        }
        failed = {
            "loss": 0.1,
            "exact_accuracy": 0.75,
            "correct_examples": 3,
            "example_count": 4,
        }

        with (
            patch("benchmark.runner._train", return_value=(0.0, 1, 1.0, 0)),
            patch(
                "benchmark.runner._evaluate",
                side_effect=(
                    primary, perfect, perfect, failed, perfect,
                    perfect, failed, perfect, perfect,
                ),
            ) as evaluate,
        ):
            result = _run_seed(
                submission,
                manifest,
                model_spec,
                torch.device("cpu"),
                seed=74,
                budget_seconds=10.0,
                submission_load_seconds=0.0,
                dataloaders=dataloaders,
            )

        self.assertEqual(evaluate.call_count, 9)
        self.assertEqual(result["evaluation"], {"test": primary})
        self.assertEqual(result["depth_profile"]["max_certified_time_steps"], 2)
        self.assertEqual(result["depth_profile"]["depth_factor"], 2)
        self.assertEqual(
            [rung["status"] for rung in result["depth_profile"]["rungs"]],
            ["certified", "certified", "failed", "passed_uncertified"],
        )

        self.assertEqual(
            result["depth_profile"]["ood_n_max_certified_time_steps"], 1
        )
        self.assertEqual(
            [
                rung["status"]
                for rung in result["depth_profile"]["ood_n_rungs"]
            ],
            ["certified", "failed", "passed_uncertified", "passed_uncertified"],
        )

    def test_dataset_files_cannot_be_reopened_after_preload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "test.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            script = (
                "from benchmark.runner import _deny_dataset_file_access;"
                f"_deny_dataset_file_access({directory!r});"
                f"open({str(dataset)!r}, encoding='utf-8').read()"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PermissionError", result.stderr)


if __name__ == "__main__":
    unittest.main()
