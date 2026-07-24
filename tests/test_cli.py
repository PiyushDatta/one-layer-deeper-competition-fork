from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from client import cli
from client.cli import build_parser, command_jobs, command_metrics


class CliTests(unittest.TestCase):
    def test_validate_accepts_a_standalone_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            path.write_text("SUBMISSION = object()\n", encoding="utf-8")
            args = build_parser().parse_args(["validate", str(path)])
            output = io.StringIO()
            with redirect_stdout(output):
                result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertIn("valid:", output.getvalue())

    def test_saved_key_is_scoped_by_server_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with patch.object(cli, "CONFIG_PATH", config_path):
                cli._save_api_key("https://one-layer.example/", "old_test")
                loaded = cli._load_saved_api_key("https://one-layer.example")

            self.assertEqual(loaded, "old_test")
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)

    def test_submit_validates_tier_and_posts_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            path.write_text("SUBMISSION = object()\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "submit",
                    str(path),
                    "--tier",
                    "easy",
                    "--dataset",
                    "e1",
                    "--api-key",
                    "old_test",
                    "--server",
                    "https://one-layer.example/",
                ]
            )
            response = Mock(is_error=False)
            response.json.return_value = {
                "submission_id": str(uuid4()),
                "tier": "easy",
                "dataset_id": "e1",
                "remaining_attempts_today": 59,
                "html_url": "/submissions/example",
            }
            client = Mock()
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=False)
            client.post.return_value = response

            with patch("client.cli.httpx.Client", return_value=client):
                with redirect_stdout(io.StringIO()):
                    result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertEqual(client.post.call_args.args[0], "https://one-layer.example/api/submissions")
        self.assertEqual(client.post.call_args.kwargs["data"], {"tier": "easy", "dataset": "e1"})
        self.assertEqual(
            client.post.call_args.kwargs["headers"],
            {"Authorization": "Bearer old_test"},
        )

    def test_jobs_defaults_to_active_and_prints_submission_id(self) -> None:
        submission_id = str(uuid4())
        response = Mock()
        response.json.return_value = [
            {
                "id": submission_id,
                "filename": "submission.py",
                "status": "running",
                "tier": "easy",
                "dataset_label": "E1",
            }
        ]
        args = build_parser().parse_args(
            [
                "jobs",
                "--server",
                "https://one-layer.example/",
                "--api-key",
                "old_test",
            ]
        )

        output = io.StringIO()
        with (
            patch("client.cli.httpx.get", return_value=response) as get,
            redirect_stdout(output),
        ):
            result = args.handler(args)

        self.assertIs(args.handler, command_jobs)
        self.assertEqual(result, 0)
        self.assertIn(submission_id, output.getvalue())
        get.assert_called_once_with(
            "https://one-layer.example/api/submissions",
            params={"active": True},
            headers={"Authorization": "Bearer old_test"},
            timeout=30.0,
        )

    def test_jobs_all_json_requests_history(self) -> None:
        rows = [{"id": str(uuid4()), "status": "succeeded"}]
        response = Mock()
        response.json.return_value = rows
        args = build_parser().parse_args(
            ["jobs", "--all", "--json", "--api-key", "old_test"]
        )

        output = io.StringIO()
        with patch("client.cli.httpx.get", return_value=response) as get:
            with redirect_stdout(output):
                result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), rows)
        self.assertEqual(get.call_args.kwargs["params"], {"active": False})

    def test_metrics_downloads_authenticated_artifact(self) -> None:
        submission_id = str(uuid4())
        response = Mock(content=b'{"type":"summary"}\n')

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.jsonl"
            args = build_parser().parse_args(
                [
                    "metrics",
                    submission_id,
                    "--output",
                    str(output_path),
                    "--server",
                    "https://one-layer.example/",
                    "--api-key",
                    "old_test",
                ]
            )
            output = io.StringIO()
            with (
                patch("client.cli.httpx.get", return_value=response) as get,
                redirect_stdout(output),
            ):
                result = args.handler(args)

            self.assertEqual(output_path.read_bytes(), response.content)

        self.assertIs(args.handler, command_metrics)
        self.assertEqual(result, 0)
        self.assertIn("saved metrics", output.getvalue())
        get.assert_called_once_with(
            f"https://one-layer.example/api/submissions/{submission_id}/metrics",
            headers={"Authorization": "Bearer old_test"},
            timeout=30.0,
        )
        response.raise_for_status.assert_called_once_with()

    def test_metrics_existing_output_requires_force(self) -> None:
        submission_id = str(uuid4())
        args = build_parser().parse_args(
            ["metrics", submission_id, "--api-key", "old_test"]
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.jsonl"
            output_path.write_text("existing\n", encoding="utf-8")
            args.output = str(output_path)
            with patch("client.cli.httpx.get") as get:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    args.handler(args)

        get.assert_not_called()

    def test_metrics_force_overwrites_existing_output(self) -> None:
        submission_id = str(uuid4())
        response = Mock(content=b'{"type":"summary"}\n')

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.jsonl"
            output_path.write_text("existing\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "metrics",
                    submission_id,
                    "--output",
                    str(output_path),
                    "--force",
                    "--api-key",
                    "old_test",
                ]
            )
            with (
                patch("client.cli.httpx.get", return_value=response),
                redirect_stdout(io.StringIO()),
            ):
                result = args.handler(args)

            self.assertEqual(result, 0)
            self.assertEqual(output_path.read_bytes(), response.content)

    def test_leaderboard_does_not_require_authentication(self) -> None:
        response = Mock()
        response.json.return_value = [
            {
                "max_certified_time_steps": 32,
                "ood_n_max_certified_time_steps": 16,
                "submitter": "Ada",
                "filename": "submission.py",
            }
        ]
        args = build_parser().parse_args(
            ["leaderboard", "--server", "https://one-layer.example"]
        )
        output = io.StringIO()

        with (
            patch("client.cli.httpx.get", return_value=response) as get,
            redirect_stdout(output),
        ):
            result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertIn("32", output.getvalue())
        self.assertIn("16", output.getvalue())
        get.assert_called_once_with(
            "https://one-layer.example/api/leaderboard",
            timeout=30.0,
        )


    def test_status_requests_the_owned_submission(self) -> None:
        submission_id = str(uuid4())
        response = Mock()
        response.json.return_value = {
            "id": submission_id,
            "filename": "submission.py",
            "status": "succeeded",
            "score": 0.75,
            "tier": "hard",
            "dataset_label": "H1",
            "manifest_name": "hidden-hard-h1",
            "run_id": str(uuid4()),
        }
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.get.return_value = response
        args = build_parser().parse_args(
            [
                "status",
                submission_id,
                "--api-key",
                "old_test",
                "--server",
                "https://one-layer.example/",
            ]
        )
        output = io.StringIO()

        with (
            patch("client.cli.httpx.Client", return_value=client),
            redirect_stdout(output),
        ):
            result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertIn("75.00%", output.getvalue())
        client.get.assert_called_once_with(
            f"https://one-layer.example/api/submissions/{submission_id}",
            headers={"Authorization": "Bearer old_test"},
        )

    def test_whoami_prints_the_authenticated_identity(self) -> None:
        response = Mock()
        response.json.return_value = {
            "display_name": "Ada Lovelace",
            "github_login": "ada",
            "email": "ada@example.com",
            "api_key_prefix": "old_example",
            "status": "active",
        }
        args = build_parser().parse_args(
            [
                "whoami",
                "--api-key",
                "old_test",
                "--server",
                "https://one-layer.example/",
            ]
        )
        output = io.StringIO()

        with (
            patch("client.cli.httpx.get", return_value=response) as get,
            redirect_stdout(output),
        ):
            result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertIn("Ada Lovelace", output.getvalue())
        get.assert_called_once_with(
            "https://one-layer.example/api/me",
            headers={"Authorization": "Bearer old_test"},
            timeout=30.0,
        )


if __name__ == "__main__":
    unittest.main()
