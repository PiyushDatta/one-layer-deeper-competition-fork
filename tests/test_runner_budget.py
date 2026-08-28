from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
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
from benchmark.model_diagnostics import TrainingModelDiagnosticsTrajectory
from data.squaring_mod import generate_squaring_mod_smoke_dataset

from benchmark.runner import (
    _evaluate,
    _depth_split_names,
    _emit_final_metrics,
    _format_competition_progress,
    _read_experiment_number,
    _resolve_batch_sizes,
    _run_seed,
    _scoring_split_names,
    _submission_requests_act_diagnostics,
    _train,
    _with_batch_size,
    _write_debug_metrics_report,
    cli,
)


ROOT = Path(__file__).resolve().parents[1]


class RunnerBudgetTests(unittest.TestCase):
    def test_submission_debug_constant_requests_act_diagnostics(self) -> None:
        debug_submission = SimpleNamespace(
            build_model=SimpleNamespace(__globals__={"DBUG": True})
        )
        final_submission = SimpleNamespace(
            build_model=SimpleNamespace(__globals__={"DBUG": False})
        )
        submission_without_debug_constant = SimpleNamespace(
            build_model=SimpleNamespace(__globals__={})
        )

        self.assertTrue(_submission_requests_act_diagnostics(debug_submission))
        self.assertFalse(_submission_requests_act_diagnostics(final_submission))
        self.assertFalse(
            _submission_requests_act_diagnostics(
                submission_without_debug_constant
            )
        )

    def test_reads_explicit_experiment_number_without_scanning_config_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "EXPERIMENTS.md"
            path.write_text(
                "# Experiments\r\n"
                "  Next experiment number: 13  \r\n"
                "| C-A32-100 | not an ordinal |\r\n",
                encoding="utf-8",
            )
            self.assertEqual(_read_experiment_number(path), 13)

            path.write_text("# No counter\n", encoding="utf-8")
            self.assertIsNone(_read_experiment_number(path))
            self.assertIsNone(
                _read_experiment_number(Path(directory) / "missing.md")
            )

    def test_rejects_duplicate_or_invalid_experiment_number_markers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "EXPERIMENTS.md"
            path.write_text(
                "Next experiment number: 13\n"
                "Next experiment number: 14\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "more than one"):
                _read_experiment_number(path)

            path.write_text(
                "Next experiment number: thirteen\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid"):
                _read_experiment_number(path)

    def test_debug_metrics_report_uses_temp_directory_and_keeps_full_result(
        self,
    ) -> None:
        result = {
            "manifest": "debug-run",
            "score": {"mean_exact_accuracy": 0.25},
            "seeds": [
                {
                    "seed": 74,
                    "act_diagnostics": {"large": "act"},
                    "model_diagnostics": {"large": "model"},
                    "training_model_diagnostics": {"large": "training"},
                }
            ],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "benchmark.runner._format_training_model_diagnostics",
                return_value="TRAINING_SENTINEL",
            ),
            patch(
                "benchmark.runner._format_act_diagnostics",
                return_value="ACT_SENTINEL",
            ),
            patch(
                "benchmark.runner._format_model_diagnostics",
                return_value="MODEL_SENTINEL",
            ),
            patch(
                "benchmark.runner._format_competition_progress",
                return_value="COMPETITION_SENTINEL",
            ),
        ):
            submission_path = Path(directory) / "submission.py"
            experiments_path = Path(directory) / "EXPERIMENTS.md"
            submission_source = "DBUG = True\n# exact source sentinel"
            path = _write_debug_metrics_report(
                result,
                experiment_number=13,
                experiments_path=experiments_path,
                submission_path=submission_path,
                submission_source=submission_source,
                directory=directory,
            )
            report = path.read_text(encoding="utf-8")

        self.assertEqual(path.parent, Path(directory).resolve())
        self.assertEqual(report.splitlines()[0], "EXPERIMENT_NUMBER=13")
        self.assertIn(f"EXPERIMENTS_FILE={experiments_path.resolve()}", report)
        self.assertIn(f"SUBMISSION_FILE={submission_path.resolve()}", report)
        self.assertIn("[TRAINING_DIAGNOSTICS]\nTRAINING_SENTINEL", report)
        self.assertIn("[ACT_DIAGNOSTICS]\nACT_SENTINEL", report)
        self.assertIn("[MODEL_DIAGNOSTICS]\nMODEL_SENTINEL", report)
        self.assertIn(
            "[COMPETITION_PROGRESS]\nCOMPETITION_SENTINEL",
            report,
        )
        full_result_json = report.split("[RESULT_JSON]\n", 1)[1].split(
            "\n\n[SUBMISSION_SOURCE]",
            1,
        )[0]
        full_result = json.loads(full_result_json)
        self.assertIn("act_diagnostics", full_result["seeds"][0])
        self.assertIn("model_diagnostics", full_result["seeds"][0])
        self.assertIn(
            "training_model_diagnostics",
            full_result["seeds"][0],
        )
        self.assertTrue(
            report.endswith("[SUBMISSION_SOURCE]\n" + submission_source + "\n")
        )

    def test_debug_console_is_compact_and_ends_with_metrics_path(self) -> None:
        result = {
            "manifest": "debug-run",
            "score": {"mean_exact_accuracy": 0.25},
            "seeds": [
                {
                    "seed": 74,
                    "evaluation": {"test": {"exact_accuracy": 0.25}},
                    "act_diagnostics": {"large": "act"},
                    "model_diagnostics": {"large": "model"},
                    "training_model_diagnostics": {"large": "training"},
                }
            ],
        }
        expected_path = Path(tempfile.gettempdir()).resolve() / "metrics.txt"
        output = io.StringIO()
        with (
            patch(
                "benchmark.runner._write_debug_metrics_report",
                return_value=expected_path,
            ),
            patch(
                "benchmark.runner._format_competition_progress",
                return_value="COMPETITION_SENTINEL",
            ),
            redirect_stdout(output),
        ):
            returned_path = _emit_final_metrics(
                result,
                debug_metrics_to_file=True,
                experiment_number=13,
                experiments_path="EXPERIMENTS.md",
                submission_path="submission.py",
                submission_source="DBUG = True\n",
            )

        lines = [line for line in output.getvalue().splitlines() if line]
        self.assertEqual(returned_path, expected_path)
        self.assertEqual(
            lines[-1],
            f"DEBUG_METRICS_FILE | path={expected_path}",
        )
        self.assertEqual(lines.count("COMPETITION_SENTINEL"), 1)
        result_line = next(
            line for line in lines if line.startswith("RESULT_JSON=")
        )
        compact_result = json.loads(result_line.removeprefix("RESULT_JSON="))
        compact_seed = compact_result["seeds"][0]
        self.assertIn("evaluation", compact_seed)
        self.assertNotIn("act_diagnostics", compact_seed)
        self.assertNotIn("model_diagnostics", compact_seed)
        self.assertNotIn("training_model_diagnostics", compact_seed)
        self.assertEqual(
            sum(line.startswith("DEBUG_METRICS_FILE |") for line in lines),
            1,
        )

    def test_non_debug_console_preserves_full_result_and_creates_no_file(
        self,
    ) -> None:
        result = {
            "manifest": "normal-run",
            "score": {"mean_exact_accuracy": 0.25},
            "seeds": [{"seed": 74, "model_diagnostics": {"marker": 1}}],
        }
        output = io.StringIO()
        with (
            patch("benchmark.runner._write_debug_metrics_report") as writer,
            patch(
                "benchmark.runner._format_act_diagnostics",
                return_value=None,
            ),
            patch(
                "benchmark.runner._format_model_diagnostics",
                return_value=None,
            ),
            patch(
                "benchmark.runner._format_competition_progress",
                return_value=None,
            ),
            redirect_stdout(output),
        ):
            returned_path = _emit_final_metrics(
                result,
                debug_metrics_to_file=False,
            )

        writer.assert_not_called()
        self.assertIsNone(returned_path)
        result_line = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith("RESULT_JSON=")
        )
        emitted_result = json.loads(result_line.removeprefix("RESULT_JSON="))
        self.assertEqual(
            emitted_result["seeds"][0]["model_diagnostics"],
            {"marker": 1},
        )

    def test_formats_competition_progress_from_worst_seed(self) -> None:
        result = {
            "depth_profile": {
                "ladder": [1, 2, 4, 8, 16, 32, 64],
                "max_certified_time_steps": None,
                "ood_n_ladder": [1, 2, 4, 8, 16, 32, 64],
                "ood_n_max_certified_time_steps": None,
            },
            "seeds": [
                {
                    "depth_profile": {
                        "rungs": [
                            {
                                "time_steps": 1,
                                "correct_examples": 1,
                                "example_count": 38,
                                "exact_accuracy": 1 / 38,
                            }
                        ],
                        "ood_n_rungs": [
                            {
                                "time_steps": 1,
                                "correct_examples": 3,
                                "example_count": 512,
                                "exact_accuracy": 3 / 512,
                            }
                        ],
                    }
                },
                {
                    "depth_profile": {
                        "rungs": [
                            {
                                "time_steps": 1,
                                "correct_examples": 2,
                                "example_count": 38,
                                "exact_accuracy": 2 / 38,
                            }
                        ],
                        "ood_n_rungs": [
                            {
                                "time_steps": 1,
                                "correct_examples": 4,
                                "example_count": 512,
                                "exact_accuracy": 4 / 512,
                            }
                        ],
                    }
                },
            ],
        }

        self.assertEqual(
            _format_competition_progress(result),
            "COMPETITION_PROGRESS | "
            "ID: MaxT=None, Next=T=1, Acc=2.6316% (1/38) | "
            "OOD N: MaxT=None, Next=T=1, Acc=0.5859% (3/512)",
        )

    def test_formats_fully_certified_and_missing_ood_profile(self) -> None:
        result = {
            "depth_profile": {
                "ladder": [1, 2, 4, 8, 16, 32, 64],
                "max_certified_time_steps": 64,
                "ood_n_ladder": [],
                "ood_n_max_certified_time_steps": None,
            },
            "seeds": [],
        }

        self.assertEqual(
            _format_competition_progress(result),
            "COMPETITION_PROGRESS | ID: MaxT=64, Certified | OOD N: N/A",
        )

    def test_omits_competition_progress_without_a_depth_profile(self) -> None:
        self.assertIsNone(_format_competition_progress({"seeds": []}))

    def test_cli_passes_num_workers_override(self) -> None:
        argv = [
            "benchmark.runner",
            "--manifest",
            "manifest.json",
            "--submission-file",
            "submission.py",
            "--num-workers",
            "0",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("benchmark.runner.run_submission_file") as run,
        ):
            cli()

        run.assert_called_once_with(
            "submission.py",
            "manifest.json",
            include_structured_metrics=False,
            include_act_diagnostics=False,
            num_workers=0,
        )

    def test_cli_enables_local_act_diagnostics(self) -> None:
        argv = [
            "benchmark.runner",
            "--manifest",
            "manifest.json",
            "--submission-file",
            "submission.py",
            "--include-act-diagnostics",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("benchmark.runner.run_submission_file") as run,
        ):
            cli()

        run.assert_called_once_with(
            "submission.py",
            "manifest.json",
            include_structured_metrics=False,
            include_act_diagnostics=True,
            num_workers=None,
        )

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

    def test_training_model_diagnostics_capture_log_and_terminal_steps(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        model.collect_model_diagnostics = False
        model.collect_training_diagnostics = False
        credit_flags = []

        def consume_credit():
            return {
                "segment_token_counts": torch.tensor([0, 1, 1, 1, 0]),
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

        model.consume_training_grad_diagnostics = consume_credit
        bundle = OptimizerBundle(torch.optim.SGD(model.parameters(), lr=0.1))
        manifest = SimpleNamespace(
            runtime=SimpleNamespace(grad_clip=0.25, log_every=3),
            model_state=object(),
        )
        trajectory = TrainingModelDiagnosticsTrajectory()

        def loss_and_accuracy(*args, **kwargs):
            credit_flags.append(model.collect_training_diagnostics)
            return model.weight.sum(), 0.5, 1, 1

        with (
            patch("benchmark.runner._loss_and_accuracy", side_effect=loss_and_accuracy),
            patch(
                "benchmark.runner._collect_training_model_checkpoint",
                return_value=({"marker": "model"}, 0.75, 0.25),
            ),
            patch(
                "benchmark.runner._format_training_model_diagnostic",
                return_value="MODEL_TRAIN_TEST",
            ) as format_diagnostic,
            patch("benchmark.runner.validate_model_state"),
            patch("benchmark.runner.validate_optimizer", return_value=0),
            patch("benchmark.runner.time.monotonic", return_value=1.0),
        ):
            _train(
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
                max_steps=4,
                seed=74,
                model_diagnostics_trajectory=trajectory,
            )

        format_diagnostic.assert_not_called()
        summary = trajectory.summary()
        assert summary is not None
        self.assertEqual(
            [record["step"] for record in summary["records"]],
            [1, 3, 4],
        )
        self.assertEqual(credit_flags, [True, True, True, True])
        terminal = summary["records"][-1]
        self.assertEqual(terminal["training_credit"]["segment_token_counts"], [0, 1, 1, 1, 0])
        gradients = terminal["optimization"]
        self.assertGreater(
            gradients["gradient_before_clipping"]["gradient_norm"],
            gradients["gradient_after_clipping"]["gradient_norm"],
        )
        self.assertGreater(
            gradients["optimizer_update"]["update_norm"],
            0.0,
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

    def test_act_diagnostics_run_after_scoring_on_a_separate_pass(self) -> None:
        manifest = load_manifest(ROOT / "benchmark" / "manifests" / "smoke_cpu.json")
        model_spec = ModelSpec(1, 1, 2)

        def build_model(spec):
            model = torch.nn.Linear(1, 1)
            model.config = SimpleNamespace(
                vocab_size=spec.vocab_size,
                max_seq_len=spec.max_seq_len,
            )
            model.collect_act_diagnostics = False
            model.collect_model_diagnostics = False
            return model

        submission = Submission(
            build_model=build_model,
            build_optimizer=lambda model, spec: OptimizerBundle(
                torch.optim.SGD(model.parameters(), lr=0.1)
            ),
        )
        dataloaders = {
            "train": object(),
            "test": object(),
            "depth_t_1": object(),
            "depth_ood_n_t_1": object(),
        }
        scoring_metrics = {
            "loss": 1.0,
            "exact_accuracy": 0.5,
            "correct_examples": 2,
            "example_count": 4,
        }

        def failed_profile():
            return {
                "ladder": [1],
                "max_certified_time_steps": None,
                "rungs": [
                    {
                        "time_steps": 1,
                        "status": "failed",
                        "correct_examples": 0,
                        "example_count": 4,
                        "exact_accuracy": 0.0,
                    }
                ],
            }

        diagnostics = [
            {
                "act_diagnostics": {"marker": "test"},
                "model_diagnostics": {"marker": "model/test"},
            },
            {
                "act_diagnostics": {"marker": "seen_n/T=1"},
                "model_diagnostics": {"marker": "model/seen_n/T=1"},
            },
            {
                "act_diagnostics": {"marker": "ood_n/T=1"},
                "model_diagnostics": {"marker": "model/ood_n/T=1"},
            },
        ]
        with (
            patch("benchmark.runner._train", return_value=(0.0, 1, 1.0, 0)),
            patch(
                "benchmark.runner._evaluate_depth_profile",
                side_effect=(failed_profile(), failed_profile()),
            ),
            patch(
                "benchmark.runner._evaluate",
                side_effect=(scoring_metrics, *diagnostics),
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
                include_act_diagnostics=True,
            )

        self.assertEqual(evaluate.call_count, 4)
        self.assertFalse(
            evaluate.call_args_list[0].kwargs.get(
                "include_act_diagnostics", False
            )
        )
        for call in evaluate.call_args_list[1:]:
            self.assertTrue(call.kwargs["include_act_diagnostics"])
            self.assertTrue(call.kwargs["include_model_diagnostics"])
            self.assertEqual(call.kwargs["deadline"], float("inf"))
        self.assertEqual(
            result["act_diagnostics"]["scoring_splits"]["test"]["marker"],
            "test",
        )
        self.assertEqual(
            result["act_diagnostics"]["first_uncertified_depth_rungs"][
                "seen_n"
            ]["diagnostics"]["marker"],
            "seen_n/T=1",
        )
        self.assertEqual(
            result["act_diagnostics"]["first_uncertified_depth_rungs"][
                "ood_n"
            ]["diagnostics"]["marker"],
            "ood_n/T=1",
        )
        self.assertIn("debug_diagnostics_seconds", result)
        self.assertNotIn("act_diagnostics_seconds", result)
        self.assertEqual(
            result["model_diagnostics"]["scoring_splits"]["test"]["marker"],
            "model/test",
        )
        self.assertEqual(
            result["model_diagnostics"]["first_uncertified_depth_rungs"][
                "seen_n"
            ]["diagnostics"]["marker"],
            "model/seen_n/T=1",
        )
        self.assertNotIn("model_diagnostics_seconds", result)

    def test_multi_pass_updates_and_bounded_batch_reuse(self) -> None:
        class CountingSGD(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=0.0)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = CountingSGD(model.parameters())
        scheduler_steps = []
        between_calls = []
        reuse_calls = []
        seen_batches = []

        def between_backward_passes(context):
            self.assertFalse(torch.is_grad_enabled())
            between_calls.append(context)

        def should_reuse_batch(context):
            self.assertFalse(torch.is_grad_enabled())
            reuse_calls.append(context)
            return True

        bundle = OptimizerBundle(
            optimizer,
            scheduler=SimpleNamespace(step=lambda: scheduler_steps.append(True)),
            backward_passes_per_step=2,
            between_backward_passes=between_backward_passes,
            should_reuse_batch=should_reuse_batch,
        )
        manifest = SimpleNamespace(
            runtime=SimpleNamespace(grad_clip=None, log_every=100),
            model_state=object(),
        )

        def loss_and_accuracy(model, batch, *args, **kwargs):
            seen_batches.append(float(batch.item()))
            return model.weight.sum() * batch, float(batch.item()), 1, 1

        with (
            patch("benchmark.runner._loss_and_accuracy", side_effect=loss_and_accuracy),
            patch("benchmark.runner.validate_model_state"),
            patch("benchmark.runner.time.monotonic", return_value=1.0),
            patch("benchmark.runner.MAX_OPTIMIZER_STEPS_PER_BATCH", 3),
        ):
            _, completed_steps, _, _ = _train(
                raw_model=model,
                train_model=model,
                training_loss=None,
                bundle=bundle,
                dataloader=[torch.tensor(1.0), torch.tensor(2.0)],
                manifest=manifest,
                device=torch.device("cpu"),
                started_at=0.0,
                deadline=2.0,
                budget_seconds=2.0,
                max_steps=5,
                seed=74,
            )

        self.assertEqual(completed_steps, 5)
        self.assertEqual(optimizer.step_calls, 5)
        self.assertEqual(len(scheduler_steps), 5)
        self.assertEqual(seen_batches, [1.0] * 6 + [2.0] * 4)
        self.assertEqual(
            [context.completed_steps for context in between_calls],
            [0, 1, 2, 3, 4],
        )
        self.assertTrue(
            all(
                context.pass_index == 1 and context.total_passes == 2
                for context in between_calls
            )
        )
        self.assertEqual(
            [context.current_batch_uses for context in reuse_calls],
            [1, 2, 1, 2],
        )
        self.assertEqual(
            [context.completed_steps for context in reuse_calls],
            [1, 2, 4, 5],
        )
        self.assertEqual(
            [context.loss for context in reuse_calls], [1.0, 1.0, 2.0, 2.0]
        )
        self.assertTrue(all(type(context.loss) is float for context in reuse_calls))


    def test_multi_pass_and_reuse_callback_validation_at_runtime(self) -> None:
        model = torch.nn.Linear(1, 1)
        manifest = SimpleNamespace(
            runtime=SimpleNamespace(grad_clip=None, log_every=1),
            model_state=object(),
        )
        too_many = OptimizerBundle(
            torch.optim.SGD(model.parameters(), lr=0.0),
            backward_passes_per_step=9,
        )
        with self.assertRaisesRegex(ValueError, "maximum of 8"):
            _train(
                raw_model=model,
                train_model=model,
                training_loss=None,
                bundle=too_many,
                dataloader=[object()],
                manifest=manifest,
                device=torch.device("cpu"),
                started_at=0.0,
                deadline=2.0,
                budget_seconds=2.0,
                max_steps=1,
                seed=74,
            )

        invalid_reuse = OptimizerBundle(
            torch.optim.SGD(model.parameters(), lr=0.0),
            should_reuse_batch=lambda context: 1,
        )

        def loss_and_accuracy(*args, **kwargs):
            return model.weight.sum(), 0.0, 1, 1

        with (
            patch("benchmark.runner._loss_and_accuracy", side_effect=loss_and_accuracy),
            patch("benchmark.runner.validate_model_state"),
            patch("benchmark.runner.time.monotonic", return_value=1.0),
            self.assertRaisesRegex(TypeError, "must return bool"),
        ):
            _train(
                raw_model=model,
                train_model=model,
                training_loss=None,
                bundle=invalid_reuse,
                dataloader=[object()],
                manifest=manifest,
                device=torch.device("cpu"),
                started_at=0.0,
                deadline=2.0,
                budget_seconds=2.0,
                max_steps=1,
                seed=74,
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
