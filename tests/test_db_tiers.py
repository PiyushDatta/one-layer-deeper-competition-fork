from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest.mock import patch
from uuid import uuid4

from service.db import DailyQuotaExceeded, Database


class _Result:
    def __init__(self, row=None, rows=None) -> None:
        self.row = row
        self.rows = rows or []

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(
        self,
        attempts: int = 0,
        rate_limit_exempt: bool = False,
        has_active_run: bool = False,
    ) -> None:
        self.attempts = attempts
        self.rate_limit_exempt = rate_limit_exempt
        self.has_active_run = has_active_run
        self.queries: list[tuple[str, tuple | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query: str, params=None):
        normalized = " ".join(query.split())
        self.queries.append((normalized, params))
        if normalized.startswith("SELECT status, rate_limit_exempt FROM users"):
            return _Result(
                {
                    "status": "active",
                    "rate_limit_exempt": self.rate_limit_exempt,
                }
            )
        if "COUNT(*)::integer AS attempts" in normalized:
            return _Result(
                {
                    "attempts": self.attempts,
                    "reset_at": datetime(2026, 7, 16, tzinfo=timezone.utc),
                }
            )
        if normalized.startswith("SELECT 1 FROM submissions"):
            return _Result({"?column?": 1} if self.has_active_run else None)
        if normalized.startswith("WITH ranked AS"):
            return _Result(rows=[])
        return _Result()


class TierDatabaseTests(unittest.TestCase):
    def test_accepted_submission_records_tier_and_returns_remaining(self) -> None:
        connection = _Connection(attempts=5)
        database = Database("unused")
        with patch.object(database, "_connect", return_value=connection):
            _, _, remaining = database.create_submission(
                user_id=uuid4(),
                filename="submission.py",
                source="SUBMISSION = object()",
                manifest_name="h100_medium_m2.json",
                tier="medium",
                dataset_id="m2",
                dataset_label="M2",
                daily_attempt_limit=6,
            )
        self.assertEqual(remaining, 0)
        run_insert = next(
            params
            for query, params in connection.queries
            if query.startswith("INSERT INTO runs")
        )
        self.assertEqual(run_insert[3:6], ("medium", "m2", "M2"))

    def test_quota_rejection_creates_no_submission(self) -> None:
        connection = _Connection(attempts=1)
        database = Database("unused")
        with patch.object(database, "_connect", return_value=connection):
            with self.assertRaises(DailyQuotaExceeded):
                database.create_submission(
                    user_id=uuid4(),
                    filename="submission.py",
                    source="SUBMISSION = object()",
                    manifest_name="h100_hard_h1.json",
                    tier="hard",
                    dataset_id="h1",
                    dataset_label="H1",
                    daily_attempt_limit=1,
                )
        self.assertFalse(
            any(query.startswith("INSERT INTO") for query, _ in connection.queries)
        )

    def test_rate_limit_exempt_user_skips_daily_and_active_run_limits(self) -> None:
        connection = _Connection(
            attempts=100,
            rate_limit_exempt=True,
            has_active_run=True,
        )
        database = Database("unused")
        with patch.object(database, "_connect", return_value=connection):
            _, _, remaining = database.create_submission(
                user_id=uuid4(),
                filename="submission.py",
                source="SUBMISSION = object()",
                manifest_name="h100_hard_h1.json",
                tier="hard",
                dataset_id="h1",
                dataset_label="H1",
                daily_attempt_limit=1,
            )
        self.assertIsNone(remaining)
        self.assertFalse(
            any(
                "COUNT(*)::integer AS attempts" in query
                for query, _ in connection.queries
            )
        )
        self.assertFalse(
            any(
                query.startswith("SELECT 1 FROM submissions")
                for query, _ in connection.queries
            )
        )

    def test_regular_user_with_active_run_is_rejected(self) -> None:
        connection = _Connection(has_active_run=True)
        database = Database("unused")
        with patch.object(database, "_connect", return_value=connection):
            with self.assertRaisesRegex(
                PermissionError,
                "one submission is already queued or running",
            ):
                database.create_submission(
                    user_id=uuid4(),
                    filename="submission.py",
                    source="SUBMISSION = object()",
                    manifest_name="h100_easy_e1.json",
                    tier="easy",
                    dataset_id="e1",
                    dataset_label="E1",
                    daily_attempt_limit=60,
                )
        self.assertFalse(
            any(query.startswith("INSERT INTO") for query, _ in connection.queries)
        )

    def test_leaderboard_is_best_successful_hard_run_per_user(self) -> None:
        connection = _Connection()
        database = Database("unused")
        with patch.object(database, "_connect", return_value=connection):
            self.assertEqual(database.leaderboard(), [])
        query = connection.queries[0][0]
        self.assertIn("r.tier = 'hard'", query)
        self.assertIn("r.status = 'succeeded'", query)
        self.assertIn("PARTITION BY u.id", query)
        self.assertIn("participant_rank = 1", query)
        self.assertIn("max_certified_time_steps", query)
        self.assertIn("ood_n_max_certified_time_steps", query)
        self.assertIn("COALESCE(max_certified_time_steps, 0) DESC", query)
        self.assertIn("COALESCE(ood_n_max_certified_time_steps, 0) DESC", query)


if __name__ == "__main__":
    unittest.main()
