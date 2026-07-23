"""Competition-wide dates and submission availability."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


SUBMISSION_DEADLINE_LABEL = "August 31 at 10:00 PM PT"
SUBMISSION_DEADLINE = datetime(
    2026,
    8,
    31,
    22,
    tzinfo=timezone(timedelta(hours=-7), name="PT"),
)
SUBMISSIONS_CLOSED_MESSAGE = (
    f"The submission deadline was {SUBMISSION_DEADLINE_LABEL}. "
    "Submissions are now closed."
)


class SubmissionsClosed(RuntimeError):
    """Raised when a submission is attempted at or after the deadline."""

    def __init__(self) -> None:
        super().__init__(SUBMISSIONS_CLOSED_MESSAGE)


def submissions_are_closed(*, now: datetime | None = None) -> bool:
    current_time = now if now is not None else datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current_time >= SUBMISSION_DEADLINE


def require_submissions_open(*, now: datetime | None = None) -> None:
    if submissions_are_closed(now=now):
        raise SubmissionsClosed
