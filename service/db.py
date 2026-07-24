"""Synchronous Postgres access for participants, submissions, and runs."""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from .auth import api_key_prefix, generate_api_key, hash_api_key
from .github_oauth import GitHubIdentity


def failure_code_for_error(error: object) -> str:
    message = str(error or "").lower()
    if "timeout" in message or "timed_out=true" in message:
        return "TIMEOUT"
    if "out of memory" in message or "cuda oom" in message:
        return "OUT_OF_MEMORY"
    if "non-finite" in message:
        return "NON_FINITE_LOSS"
    return "EVALUATION_FAILED"


class DailyQuotaExceeded(PermissionError):
    def __init__(self, tier: str, limit: int, reset_at) -> None:
        super().__init__(
            f"{tier.title()} daily attempt limit reached ({limit}); "
            f"resets at {reset_at.isoformat()}"
        )
        self.tier = tier
        self.limit = limit
        self.reset_at = reset_at


class Database:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    github_user_id BIGINT UNIQUE,
                    github_login TEXT,
                    api_key_hash TEXT NOT NULL UNIQUE,
                    api_key_prefix TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'banned')),
                    rate_limit_exempt BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    banned_at TIMESTAMPTZ,
                    ban_reason TEXT
                )
                """
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_idx "
                "ON users ((lower(email)))"
            )
            connection.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_user_id BIGINT"
            )
            connection.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_login TEXT"
            )
            connection.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                "rate_limit_exempt BOOLEAN NOT NULL DEFAULT FALSE"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS users_github_user_id_idx "
                "ON users (github_user_id) WHERE github_user_id IS NOT NULL"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_states (
                    state_hash TEXT PRIMARY KEY,
                    cli_port INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '10 minutes'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id),
                    filename TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id UUID PRIMARY KEY,
                    submission_id UUID NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
                    manifest_name TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
                    modal_call_id TEXT,
                    result JSONB,
                    metrics JSONB,
                    log_tail TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    error_code TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    deadline_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ
                )
                """
            )

            connection.execute(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS tier TEXT"
            )
            connection.execute(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS dataset_id TEXT"
            )
            connection.execute(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS dataset_label TEXT"
            )
            connection.execute(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS deadline_at TIMESTAMPTZ"
            )
            connection.execute(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS metrics JSONB"
            )
            connection.execute(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS error_code TEXT"
            )
            connection.execute(
                """
                UPDATE runs
                SET error_code = CASE
                    WHEN lower(COALESCE(error, '')) LIKE '%timeout%'
                      OR lower(COALESCE(error, '')) LIKE '%timed_out=true%'
                        THEN 'TIMEOUT'
                    WHEN lower(COALESCE(error, '')) LIKE '%out of memory%'
                      OR lower(COALESCE(error, '')) LIKE '%cuda oom%'
                        THEN 'OUT_OF_MEMORY'
                    WHEN lower(COALESCE(error, '')) LIKE '%non-finite%'
                        THEN 'NON_FINITE_LOSS'
                    ELSE 'EVALUATION_FAILED'
                END
                WHERE status = 'failed' AND error_code IS NULL
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS submissions_user_id_idx "
                "ON submissions(user_id)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS runs_submission_id_idx "
                "ON runs(submission_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_tier_created_at_idx "
                "ON runs(tier, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_status_deadline_at_idx "
                "ON runs(status, deadline_at) WHERE deadline_at IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_log_retention_idx "
                "ON runs(finished_at) "
                "WHERE log_tail <> '' OR error IS NOT NULL"
            )

    def ping(self) -> None:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()

    def create_oauth_state(self, cli_port: int | None) -> str:
        raw_state = generate_api_key()
        with self._connect() as connection:
            connection.execute("DELETE FROM oauth_states WHERE expires_at < NOW()")
            connection.execute(
                "INSERT INTO oauth_states (state_hash, cli_port) VALUES (%s, %s)",
                (hash_api_key(raw_state), cli_port),
            )
        return raw_state

    def consume_oauth_state(self, raw_state: str) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                DELETE FROM oauth_states
                WHERE state_hash = %s AND expires_at >= NOW()
                RETURNING cli_port
                """,
                (hash_api_key(raw_state),),
            ).fetchone()
        if row is None:
            raise ValueError("invalid or expired GitHub login state")
        return row["cli_port"]

    def login_github_user(self, identity: GitHubIdentity) -> tuple[dict, str]:
        raw_key = generate_api_key()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, status, ban_reason FROM users
                WHERE github_user_id = %s OR lower(email) = lower(%s)
                ORDER BY (github_user_id = %s) DESC
                LIMIT 1
                FOR UPDATE
                """,
                (identity.user_id, identity.email, identity.user_id),
            ).fetchone()
            if existing is not None and existing["status"] == "banned":
                reason = existing["ban_reason"] or "operator ban"
                raise PermissionError(f"GitHub account is banned: {reason}")
            if existing is None:
                row = connection.execute(
                    """
                    INSERT INTO users
                        (id, display_name, email, github_user_id, github_login,
                         api_key_hash, api_key_prefix)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, display_name, email, github_user_id, github_login,
                              api_key_prefix, status, rate_limit_exempt, created_at
                    """,
                    (
                        uuid4(),
                        identity.display_name,
                        identity.email,
                        identity.user_id,
                        identity.login,
                        hash_api_key(raw_key),
                        api_key_prefix(raw_key),
                    ),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    UPDATE users
                    SET display_name = %s, email = %s, github_user_id = %s,
                        github_login = %s, api_key_hash = %s, api_key_prefix = %s
                    WHERE id = %s
                    RETURNING id, display_name, email, github_user_id, github_login,
                              api_key_prefix, status, rate_limit_exempt, created_at
                    """,
                    (
                        identity.display_name,
                        identity.email,
                        identity.user_id,
                        identity.login,
                        hash_api_key(raw_key),
                        api_key_prefix(raw_key),
                        existing["id"],
                    ),
                ).fetchone()
        assert row is not None
        return row, raw_key

    def authenticate_api_key(self, raw_key: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, display_name, email, github_user_id, github_login,
                       api_key_prefix, status, rate_limit_exempt,
                       created_at, banned_at, ban_reason
                FROM users
                WHERE api_key_hash = %s
                """,
                (hash_api_key(raw_key),),
            ).fetchone()
        if row is None:
            raise ValueError("invalid API key")
        if row["status"] == "banned":
            raise PermissionError(
                f"API key is banned: {row['ban_reason'] or 'operator ban'}"
            )
        return row

    def list_users(self) -> list[dict]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT id, display_name, email, github_user_id, github_login,
                           api_key_prefix, status, rate_limit_exempt,
                           created_at, banned_at, ban_reason
                    FROM users
                    ORDER BY created_at
                    """
                ).fetchall()
            )

    def set_user_banned(
        self,
        participant: str,
        *,
        banned: bool,
        reason: str | None,
    ) -> dict | None:
        status = "banned" if banned else "active"
        with self._connect() as connection:
            return connection.execute(
                """
                UPDATE users
                SET status = %s,
                    banned_at = CASE WHEN %s THEN NOW() ELSE NULL END,
                    ban_reason = CASE WHEN %s THEN %s ELSE NULL END
                WHERE lower(email) = lower(%s) OR api_key_prefix = %s
                      OR lower(github_login) = lower(%s)
                RETURNING id, display_name, email, github_user_id, github_login,
                          api_key_prefix, status, rate_limit_exempt,
                          created_at, banned_at, ban_reason
                """,
                (
                    status,
                    banned,
                    banned,
                    reason,
                    participant,
                    participant,
                    participant.removeprefix("@"),
                ),
            ).fetchone()

    def set_user_rate_limit_exempt(
        self,
        participant: str,
        *,
        exempt: bool,
    ) -> dict | None:
        with self._connect() as connection:
            return connection.execute(
                """
                UPDATE users
                SET rate_limit_exempt = %s
                WHERE lower(email) = lower(%s) OR api_key_prefix = %s
                      OR lower(github_login) = lower(%s)
                RETURNING id, display_name, email, github_user_id, github_login,
                          api_key_prefix, status, rate_limit_exempt,
                          created_at, banned_at, ban_reason
                """,
                (
                    exempt,
                    participant,
                    participant,
                    participant.removeprefix("@"),
                ),
            ).fetchone()

    def create_submission(
        self,
        *,
        user_id: UUID,
        filename: str,
        source: str,
        manifest_name: str,
        tier: str,
        dataset_id: str,
        dataset_label: str,
        daily_attempt_limit: int,
    ) -> tuple[UUID, UUID, int | None]:
        submission_id = uuid4()
        run_id = uuid4()
        with self._connect() as connection:
            user = connection.execute(
                """
                SELECT status, rate_limit_exempt
                FROM users
                WHERE id = %s
                FOR UPDATE
                """,
                (user_id,),
            ).fetchone()
            if user is None:
                raise ValueError("unknown participant")
            if user["status"] == "banned":
                raise PermissionError("API key is banned")
            quota = None
            if not user["rate_limit_exempt"]:
                quota = connection.execute(
                    """
                    SELECT
                        COUNT(*)::integer AS attempts,
                        (
                            date_trunc('day', NOW() AT TIME ZONE 'UTC')
                            + interval '1 day'
                        ) AT TIME ZONE 'UTC' AS reset_at
                    FROM submissions s
                    JOIN runs r ON r.submission_id = s.id
                    WHERE s.user_id = %s
                      AND r.tier = %s
                      AND r.created_at >= (
                          date_trunc('day', NOW() AT TIME ZONE 'UTC')
                          AT TIME ZONE 'UTC'
                      )
                    """,
                    (user_id, tier),
                ).fetchone()
                assert quota is not None
                if quota["attempts"] >= daily_attempt_limit:
                    raise DailyQuotaExceeded(
                        tier, daily_attempt_limit, quota["reset_at"]
                    )
            if not user["rate_limit_exempt"]:
                active = connection.execute(
                    """
                    SELECT 1
                    FROM submissions s
                    JOIN runs r ON r.submission_id = s.id
                    WHERE s.user_id = %s AND r.status IN ('queued', 'running')
                    LIMIT 1
                    """,
                    (user_id,),
                ).fetchone()
                if active is not None:
                    raise PermissionError(
                        "one submission is already queued or running for this key"
                    )
            connection.execute(
                """
                INSERT INTO submissions (id, user_id, filename, source)
                VALUES (%s, %s, %s, %s)
                """,
                (submission_id, user_id, filename, source),
            )
            connection.execute(
                """
                INSERT INTO runs
                    (id, submission_id, manifest_name, tier, dataset_id,
                     dataset_label, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'queued')
                """,
                (
                    run_id,
                    submission_id,
                    manifest_name,
                    tier,
                    dataset_id,
                    dataset_label,
                ),
            )
        remaining = (
            None
            if quota is None
            else daily_attempt_limit - quota["attempts"] - 1
        )
        return submission_id, run_id, remaining

    def mark_running(
        self,
        run_id: UUID,
        call_id: str,
        deadline_seconds: int,
    ) -> None:
        if deadline_seconds < 1:
            raise ValueError("deadline_seconds must be positive")
        with self._connect() as connection:
            row = connection.execute(
                """
                UPDATE runs
                SET status = 'running', modal_call_id = %s, started_at = NOW(),
                    deadline_at = NOW() + (%s * INTERVAL '1 second')
                WHERE id = %s
                RETURNING id
                """,
                (call_id, deadline_seconds, run_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")

    def backfill_run_deadlines(
        self,
        deadline_seconds_by_tier: dict[str, int],
    ) -> int:
        for deadline_seconds in deadline_seconds_by_tier.values():
            if deadline_seconds < 1:
                raise ValueError("deadline seconds must be positive")

        updated = 0
        with self._connect() as connection:
            for tier, deadline_seconds in deadline_seconds_by_tier.items():
                result = connection.execute(
                    """
                    UPDATE runs
                    SET deadline_at = started_at + (%s * INTERVAL '1 second')
                    WHERE status = 'running'
                      AND deadline_at IS NULL
                      AND started_at IS NOT NULL
                      AND tier = %s
                    """,
                    (deadline_seconds, tier),
                )
                updated += result.rowcount
        return updated

    def fail_expired_runs(self) -> list[dict]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'failed',
                        error = 'TimeoutError: evaluation exceeded its service deadline before a terminal result was recorded.',
                        error_code = 'TIMEOUT',
                        finished_at = NOW()
                    WHERE status = 'running'
                      AND deadline_at IS NOT NULL
                      AND deadline_at <= NOW()
                    RETURNING id, submission_id, modal_call_id, deadline_at
                    """
                ).fetchall()
            )

    def mark_succeeded(
        self,
        run_id: UUID,
        result: dict,
        log_tail: str,
        metrics: list[dict] | None,
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                UPDATE runs
                SET status = 'succeeded', result = %s::jsonb,
                    metrics = %s::jsonb, log_tail = %s, error = NULL,
                    error_code = NULL, finished_at = NOW()
                WHERE id = %s
                RETURNING id
                """,
                (
                    json.dumps(result),
                    json.dumps(metrics) if metrics is not None else None,
                    log_tail,
                    run_id,
                ),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")

    def mark_failed(self, run_id: UUID, error: str, log_tail: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs SET status = 'failed', error = %s, error_code = %s,
                    log_tail = %s, finished_at = NOW()
                WHERE id = %s
                """,
                (
                    error[:4000],
                    failure_code_for_error(error),
                    log_tail[-40000:],
                    run_id,
                ),
            )

    def purge_expired_run_logs(self, retention_seconds: int) -> int:
        if retention_seconds < 1:
            raise ValueError("retention seconds must be positive")
        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE runs SET log_tail = '', error = NULL
                WHERE finished_at < NOW() - (%s * INTERVAL '1 second')
                  AND (log_tail <> '' OR error IS NOT NULL)
                """,
                (retention_seconds,),
            )
        return result.rowcount

    def leaderboard(self) -> list[dict]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    WITH ranked AS (
                        SELECT
                            s.id, s.filename, r.status, s.created_at,
                            r.manifest_name, r.tier, r.dataset_id,
                            r.dataset_label, r.finished_at,
                            u.id AS user_id, u.display_name AS submitter,
                            (r.result -> 'depth_profile' ->> 'max_certified_time_steps')::integer AS max_certified_time_steps,
                            (r.result -> 'depth_profile' ->> 'ood_n_max_certified_time_steps')::integer AS ood_n_max_certified_time_steps,
                            COALESCE((r.result -> 'depth_profile' ->> 'ood_n_profile_available')::boolean, false) AS ood_n_profile_available,
                            ROW_NUMBER() OVER (
                                PARTITION BY u.id
                                ORDER BY
                                    COALESCE((r.result -> 'depth_profile' ->> 'max_certified_time_steps')::integer, 0) DESC,
                                    COALESCE((r.result -> 'depth_profile' ->> 'ood_n_max_certified_time_steps')::integer, 0) DESC,
                                    s.created_at,
                                    s.id
                            ) AS participant_rank
                        FROM submissions s
                        JOIN runs r ON r.submission_id = s.id
                        JOIN users u ON u.id = s.user_id
                        WHERE r.tier = 'hard' AND r.status = 'succeeded'
                    )
                    SELECT id, filename, status, created_at, manifest_name,
                           tier, dataset_id, dataset_label, finished_at,
                           submitter, max_certified_time_steps,
                           ood_n_max_certified_time_steps, ood_n_profile_available
                    FROM ranked
                    WHERE participant_rank = 1
                    ORDER BY COALESCE(max_certified_time_steps, 0) DESC,
                             COALESCE(ood_n_max_certified_time_steps, 0) DESC,
                             created_at, id
                    """
                ).fetchall()
            )

    def list_submissions(
        self, user_id: UUID, *, active_only: bool = True
    ) -> list[dict]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT
                        s.id, s.filename, s.created_at,
                        r.id AS run_id, r.manifest_name, r.tier,
                        r.dataset_id, r.dataset_label, r.status,
                        r.started_at, r.deadline_at, r.finished_at,
                        (r.result -> 'score' ->> 'mean_exact_accuracy')::double precision AS score,
                        (r.result -> 'depth_profile' ->> 'depth_factor')::integer AS depth_factor,
                        (r.result -> 'depth_profile' ->> 'max_certified_time_steps')::integer AS max_certified_time_steps,
                        (r.result -> 'depth_profile' ->> 'ood_n_max_certified_time_steps')::integer AS ood_n_max_certified_time_steps,
                        COALESCE((r.result -> 'depth_profile' ->> 'ood_n_profile_available')::boolean, false) AS ood_n_profile_available
                    FROM submissions s
                    JOIN runs r ON r.submission_id = s.id
                    WHERE s.user_id = %s
                      AND (NOT %s OR r.status IN ('queued', 'running'))
                    ORDER BY s.created_at DESC, s.id DESC
                    """,
                    (user_id, active_only),
                ).fetchall()
            )

    def get_public_submission(self, submission_id: UUID) -> dict | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    s.id, s.filename, r.status, s.created_at,
                    r.manifest_name, r.tier, r.dataset_id, r.dataset_label,
                    r.finished_at,
                    u.display_name AS submitter,
                    CASE WHEN r.status = 'succeeded'
                        THEN (r.result -> 'depth_profile' ->> 'max_certified_time_steps')::integer
                    END AS max_certified_time_steps,
                    CASE WHEN r.status = 'succeeded'
                        THEN (r.result -> 'depth_profile' ->> 'ood_n_max_certified_time_steps')::integer
                    END AS ood_n_max_certified_time_steps,
                    CASE WHEN r.status = 'succeeded'
                        THEN COALESCE((r.result -> 'depth_profile' ->> 'ood_n_profile_available')::boolean, false)
                    END AS ood_n_profile_available
                FROM submissions s
                JOIN runs r ON r.submission_id = s.id
                JOIN users u ON u.id = s.user_id
                WHERE s.id = %s
                """,
                (submission_id,),
            ).fetchone()

    def get_submission_metrics(
        self,
        submission_id: UUID,
        user_id: UUID,
    ) -> dict | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT r.status, r.metrics
                FROM submissions s
                JOIN runs r ON r.submission_id = s.id
                WHERE s.id = %s AND s.user_id = %s
                """,
                (submission_id, user_id),
            ).fetchone()

    def get_submission(self, submission_id: UUID) -> dict | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    s.id, s.user_id, s.filename, s.created_at,
                    u.display_name AS submitter,
                    r.id AS run_id, r.manifest_name, r.tier, r.dataset_id,
                    r.dataset_label, r.status,
                    r.modal_call_id, r.result, r.log_tail, r.error, r.error_code,
                    (r.metrics IS NOT NULL) AS metrics_available,
                    r.created_at AS run_created_at, r.started_at,
                    r.deadline_at, r.finished_at
                FROM submissions s
                JOIN runs r ON r.submission_id = s.id
                JOIN users u ON u.id = s.user_id
                WHERE s.id = %s
                """,
                (submission_id,),
            ).fetchone()
