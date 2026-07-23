from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException, UploadFile

from service import app as service_app
from service.competition import (
    SUBMISSION_DEADLINE,
    SUBMISSIONS_CLOSED_MESSAGE,
    SubmissionsClosed,
    submissions_are_closed,
)
from service.views import submit_page


class SubmissionDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def test_deadline_is_the_published_pacific_time_instant(self) -> None:
        self.assertEqual(
            SUBMISSION_DEADLINE.astimezone(timezone.utc),
            datetime(2026, 9, 1, 5, tzinfo=timezone.utc),
        )

    def test_cutoff_closes_submissions_at_the_deadline(self) -> None:
        self.assertFalse(
            submissions_are_closed(
                now=SUBMISSION_DEADLINE - timedelta(microseconds=1)
            )
        )
        self.assertTrue(submissions_are_closed(now=SUBMISSION_DEADLINE))
        self.assertTrue(
            submissions_are_closed(
                now=SUBMISSION_DEADLINE + timedelta(microseconds=1)
            )
        )

    def test_closed_upload_page_disables_submission_controls(self) -> None:
        page = submit_page(submissions_are_closed=True)
        self.assertIn(SUBMISSIONS_CLOSED_MESSAGE, page)
        self.assertIn("Submissions closed</button>", page)
        self.assertIn('class="upload-card submissions-closed"', page)
        self.assertIn('name="api_key" type="password"', page)
        self.assertIn("required disabled", page)

    async def test_browser_submission_returns_gone_after_deadline(self) -> None:
        upload = UploadFile(
            filename="submission.py",
            file=BytesIO(b"SUBMISSION = None"),
        )
        with patch.object(
            service_app,
            "require_submissions_open",
            side_effect=SubmissionsClosed,
        ):
            response = await service_app.submit(
                BackgroundTasks(),
                upload,
                "old_example",
                "easy",
                "e1",
            )

        self.assertEqual(response.status_code, 410)
        self.assertIn(SUBMISSIONS_CLOSED_MESSAGE, response.body.decode())

    async def test_api_submission_returns_gone_after_deadline(self) -> None:
        upload = UploadFile(
            filename="submission.py",
            file=BytesIO(b"SUBMISSION = None"),
        )
        with patch.object(
            service_app,
            "require_submissions_open",
            side_effect=SubmissionsClosed,
        ):
            with self.assertRaises(HTTPException) as raised:
                await service_app.submit_api(
                    BackgroundTasks(),
                    upload,
                    "easy",
                    "e1",
                    "Bearer old_example",
                )

        self.assertEqual(raised.exception.status_code, 410)
        self.assertEqual(raised.exception.detail, SUBMISSIONS_CLOSED_MESSAGE)

    def test_queue_rechecks_deadline_before_writing(self) -> None:
        with (
            patch.object(
                service_app,
                "require_submissions_open",
                side_effect=SubmissionsClosed,
            ),
            patch.object(service_app.database, "create_submission") as create,
        ):
            with self.assertRaises(SubmissionsClosed):
                service_app._queue_submission(
                    BackgroundTasks(),
                    {"id": "participant"},
                    "submission.py",
                    "SUBMISSION = None",
                    "easy",
                    "e1",
                )

        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
