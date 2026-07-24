from __future__ import annotations

from pathlib import Path
import unittest

from benchmark.manifest import load_manifest
from service.tiers import (
    TIERS,
    resolve_tier_dataset,
    submission_manifest_timeouts,
    tier_public_payload,
)


ROOT = Path(__file__).resolve().parents[1]


class TierCatalogTests(unittest.TestCase):
    def test_catalog_has_expected_budgets_quotas_and_datasets(self) -> None:
        self.assertEqual([tier.id for tier in TIERS], ["easy", "medium", "hard"])
        self.assertEqual(
            [tier.training_seconds for tier in TIERS],
            [60, 600, 3600],
        )
        self.assertEqual(
            [tier.evaluation_seconds for tier in TIERS],
            [30, 300, 1800],
        )
        self.assertEqual(
            [tier.evaluator_timeout_seconds for tier in TIERS],
            [390, 1200, 6120],
        )
        self.assertEqual([tier.daily_attempts for tier in TIERS], [60, 6, 1])
        self.assertEqual([len(tier.datasets) for tier in TIERS], [5, 5, 1])

    def test_remote_allowlist_is_derived_from_the_catalog(self) -> None:
        expected = {
            dataset.manifest_filename: tier.evaluator_timeout_seconds
            for tier in TIERS
            for dataset in tier.datasets
        }
        self.assertEqual(submission_manifest_timeouts(), expected)
        self.assertEqual(len(expected), 11)

    def test_modal_image_generation_uses_synced_environment_and_no_network(self) -> None:
        source = (ROOT / "modal_runner.py").read_text(encoding="utf-8")
        self.assertIn(
            "uv run --no-sync bash scripts/generate_datasets.sh",
            source,
        )
        self.assertIn("block_network=True", source)
        self.assertIn('"--include-structured-metrics"', source)
        self.assertIn('.add_local_python_source("modal_runner")', source)
        self.assertNotIn('.add_local_dir("model"', source)
        self.assertNotIn('.add_local_dir("optim"', source)

    def test_every_public_dataset_is_generated_for_bidirectional_prompts(self) -> None:
        generation_script = (ROOT / "scripts" / "generate_datasets.sh").read_text(
            encoding="utf-8"
        )
        active_script = "\n".join(
            line
            for line in generation_script.splitlines()
            if not line.lstrip().startswith("#")
        )
        generation_commands = active_script.split("python -m data.")[1:]
        self.assertEqual(len(generation_commands), 10)
        for command in generation_commands:
            with self.subTest(command=command.splitlines()[0]):
                self.assertIn("--separate_input_output true", command)
        self.assertNotIn("_causal_", generation_script)
        self.assertEqual(
            list((ROOT / "benchmark" / "manifests").glob("h100_hard_*.json")),
            [],
        )

    def test_modal_deploy_job_is_smoke_gated(self) -> None:
        dockerfile = (ROOT / "Dockerfile.modal-deploy").read_text(encoding="utf-8")
        deploy_script = (ROOT / "scripts/deploy_modal.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("submissions/baseline_adamw/submission.py", dockerfile)
        self.assertNotIn("COPY model ", dockerfile)
        self.assertNotIn("COPY optim ", dockerfile)
        self.assertIn('CMD ["sh", "scripts/deploy_modal.sh"]', dockerfile)
        self.assertIn("modal deploy modal_runner.py", deploy_script)
        self.assertIn("python modal_runner.py smoke", deploy_script)

    def test_dataset_selection_is_tier_scoped(self) -> None:
        tier, dataset = resolve_tier_dataset("hard", None)
        self.assertEqual((tier.id, dataset.id), ("hard", "h1"))
        with self.assertRaisesRegex(ValueError, "dataset for Easy"):
            resolve_tier_dataset("easy", "m1")
        with self.assertRaisesRegex(ValueError, "unknown tier"):
            resolve_tier_dataset("extreme", "e1")

    def test_hard_public_label_does_not_disclose_dataset_contents(self) -> None:
        hard = next(tier for tier in TIERS if tier.id == "hard")
        payload = tier_public_payload(hard)
        self.assertEqual(
            payload["datasets"],
            [{"id": "h1", "label": "H1 · Hidden evaluation"}],
        )

    def test_all_catalog_manifests_match_tier_runtime(self) -> None:
        for tier in TIERS:
            if tier.id == "hard":
                continue
            for dataset in tier.datasets:
                with self.subTest(tier=tier.id, dataset=dataset.id):
                    manifest = load_manifest(
                        ROOT / "benchmark" / "manifests" / dataset.manifest_filename
                    )
                    self.assertEqual(
                        manifest.runtime.total_training_time_seconds,
                        tier.training_seconds,
                    )
                    self.assertEqual(manifest.runtime.seeds, (74,))
                    self.assertEqual(manifest.data.batch_size, 512)
                    self.assertEqual(manifest.data.eval_batch_size, 512)
                    self.assertEqual(manifest.model_state.maximum_elements, 500_000_000)
                    self.assertIsNotNone(manifest.data.data_root)
                    self.assertIn("bidirectional", manifest.data.data_root)


if __name__ == "__main__":
    unittest.main()
