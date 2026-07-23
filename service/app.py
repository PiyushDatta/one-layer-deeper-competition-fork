"""FastAPI upload service and leaderboard."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import logging
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from .auth import bearer_api_key
from .competition import (
    SubmissionsClosed,
    require_submissions_open,
    submissions_are_closed,
)
from .config import Settings
from .db import DailyQuotaExceeded, Database, failure_code_for_error
from .evaluator import evaluate_run
from .github_oauth import exchange_github_identity
from .metrics import metrics_to_jsonl
from .tiers import TIERS, resolve_tier_dataset, tier_public_payload
from submission_validation import validate_submission_source
from .views import (
    leaderboard_page,
    problem_page,
    register_page,
    registered_page,
    submission_page,
    submit_page,
)


settings = Settings.from_env()
database = Database(settings.database_url)
service_root = Path(__file__).resolve().parent
repo_root = service_root.parent
logger = logging.getLogger(__name__)
RUN_DEADLINE_WATCHDOG_INTERVAL_SECONDS = 60
RUN_LOG_RETENTION_SECONDS = 24 * 60 * 60


def _expire_overdue_runs() -> None:
    expired = database.fail_expired_runs()
    if expired:
        logger.warning("marked %d overdue evaluation run(s) failed", len(expired))
    purged = database.purge_expired_run_logs(RUN_LOG_RETENTION_SECONDS)
    if purged:
        logger.info("purged logs from %d evaluation run(s)", purged)


async def _run_deadline_watchdog() -> None:
    while True:
        await asyncio.sleep(RUN_DEADLINE_WATCHDOG_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(_expire_overdue_runs)
        except Exception:
            logger.exception("run deadline watchdog failed")



@asynccontextmanager
async def lifespan(app: FastAPI):
    database.initialize()
    database.backfill_run_deadlines(
        {tier.id: tier.run_deadline_seconds for tier in TIERS}
    )
    _expire_overdue_runs()
    watchdog = asyncio.create_task(_run_deadline_watchdog())
    try:
        yield
    finally:
        watchdog.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog


app = FastAPI(title="One Layer Deeper", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=service_root / "static"), name="static")


def _api_user(authorization: str | None) -> dict:
    try:
        raw_key = bearer_api_key(authorization)
        return database.authenticate_api_key(raw_key)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _github_redirect_uri() -> str:
    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(status_code=503, detail="GitHub login is not configured")
    return f"{settings.public_url}/auth/github/callback"


def _participant_submission_payload(row: dict) -> dict:
    fields = (
        "id",
        "filename",
        "created_at",
        "submitter",
        "run_id",
        "manifest_name",
        "tier",
        "dataset_id",
        "dataset_label",
        "status",
        "modal_call_id",
        "result",
        "run_created_at",
        "started_at",
        "deadline_at",
        "finished_at",
        "metrics_available",
    )
    payload = {field: row.get(field) for field in fields}
    if row.get("status") == "failed":
        payload["error_code"] = row.get("error_code") or failure_code_for_error(
            row.get("error")
        )
    if row.get("metrics_available"):
        payload["metrics_url"] = f"/api/submissions/{row['id']}/metrics"
    return payload


async def _read_submission(file: UploadFile) -> tuple[str, str]:
    raw = await file.read(settings.max_submission_bytes + 1)
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("submission.py must be UTF-8") from exc
    filename = validate_submission_source(
        file.filename or "submission.py",
        source,
        settings.max_submission_bytes,
    )
    return filename, source


def _queue_submission(
    background_tasks: BackgroundTasks,
    user: dict,
    filename: str,
    source: str,
    tier_id: str,
    dataset_id: str | None,
) -> tuple[UUID, UUID, int | None]:
    require_submissions_open()
    tier, dataset = resolve_tier_dataset(tier_id, dataset_id)
    submission_id, run_id, remaining_attempts = database.create_submission(
        user_id=user["id"],
        filename=filename,
        source=source,
        manifest_name=dataset.manifest_filename,
        tier=tier.id,
        dataset_id=dataset.id,
        dataset_label=dataset.label,
        daily_attempt_limit=tier.daily_attempts,
    )
    background_tasks.add_task(
        evaluate_run,
        database=database,
        settings=settings,
        run_id=run_id,
        source=source,
        tier=tier,
        dataset=dataset,
    )
    return submission_id, run_id, remaining_attempts


@app.get("/", response_class=HTMLResponse)
def leaderboard() -> str:
    return leaderboard_page(database.leaderboard())


@app.get("/problem", response_class=HTMLResponse)
def problem() -> str:
    return problem_page()


@app.get("/submit", response_class=HTMLResponse)
def submit_form() -> str:
    closed = submissions_are_closed()
    return submit_page(submissions_are_closed=closed)


@app.get("/register", response_class=HTMLResponse)
def register_form() -> str:
    return register_page()


@app.get("/auth/github")
def github_login(cli_port: int | None = None):
    redirect_uri = _github_redirect_uri()
    if cli_port is not None and not 1024 <= cli_port <= 65535:
        raise HTTPException(status_code=400, detail="invalid CLI callback port")
    state = database.create_oauth_state(cli_port)
    query = urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": redirect_uri,
            "scope": "user:email",
            "state": state,
            "allow_signup": "true",
        }
    )
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{query}")


@app.get("/auth/github/callback", response_class=HTMLResponse)
def github_callback(code: str, state: str):
    redirect_uri = _github_redirect_uri()
    try:
        cli_port = database.consume_oauth_state(state)
        identity = exchange_github_identity(
            code=code,
            client_id=settings.github_client_id,
            client_secret=settings.github_client_secret,
            redirect_uri=redirect_uri,
        )
        user, api_key = database.login_github_user(identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=400, detail=f"GitHub login failed: {exc}") from exc

    if cli_port is not None:
        query = urlencode(
            {
                "api_key": api_key,
                "display_name": user["display_name"],
                "github_login": user["github_login"],
            }
        )
        return RedirectResponse(f"http://127.0.0.1:{cli_port}/callback?{query}")
    return registered_page(user, api_key)


@app.post("/submit")
async def submit(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    tier: str = Form(...),
    dataset: str | None = Form(default=None),
):
    try:
        require_submissions_open()
    except SubmissionsClosed as exc:
        return HTMLResponse(
            submit_page(
                str(exc),
                selected_tier=tier,
                selected_dataset=dataset,
                submissions_are_closed=True,
            ),
            status_code=410,
        )
    try:
        user = database.authenticate_api_key(api_key)
    except (ValueError, PermissionError) as exc:
        return HTMLResponse(
            submit_page(str(exc), selected_tier=tier, selected_dataset=dataset),
            status_code=403,
        )
    try:
        filename, source = await _read_submission(file)
    except ValueError as exc:
        return HTMLResponse(
            submit_page(str(exc), selected_tier=tier, selected_dataset=dataset),
            status_code=400,
        )

    try:
        submission_id, _, _ = _queue_submission(
            background_tasks,
            user,
            filename,
            source,
            tier,
            dataset,
        )
    except SubmissionsClosed as exc:
        return HTMLResponse(
            submit_page(
                str(exc),
                selected_tier=tier,
                selected_dataset=dataset,
                submissions_are_closed=True,
            ),
            status_code=410,
        )
    except DailyQuotaExceeded as exc:
        return HTMLResponse(
            submit_page(str(exc), selected_tier=tier, selected_dataset=dataset),
            status_code=429,
        )
    except ValueError as exc:
        return HTMLResponse(
            submit_page(str(exc), selected_tier=tier, selected_dataset=dataset),
            status_code=400,
        )
    except PermissionError as exc:
        return HTMLResponse(
            submit_page(str(exc), selected_tier=tier, selected_dataset=dataset),
            status_code=409,
        )
    return RedirectResponse(f"/submissions/{submission_id}", status_code=303)


@app.post("/api/submissions", status_code=202)
async def submit_api(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tier: str = Form(...),
    dataset: str | None = Form(default=None),
    authorization: str | None = Header(default=None),
):
    try:
        require_submissions_open()
    except SubmissionsClosed as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    user = _api_user(authorization)
    try:
        filename, source = await _read_submission(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        submission_id, run_id, remaining_attempts = _queue_submission(
            background_tasks,
            user,
            filename,
            source,
            tier,
            dataset,
        )
    except SubmissionsClosed as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except DailyQuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    selected_tier, selected_dataset = resolve_tier_dataset(tier, dataset)
    return {
        "submission_id": str(submission_id),
        "run_id": str(run_id),
        "status": "queued",
        "tier": selected_tier.id,
        "dataset_id": selected_dataset.id,
        "remaining_attempts_today": remaining_attempts,
        "status_url": f"/api/submissions/{submission_id}",
        "html_url": f"/submissions/{submission_id}",
    }


@app.get("/api/submissions")
def submissions_api(
    active: bool = True,
    authorization: str | None = Header(default=None),
):
    user = _api_user(authorization)
    rows = database.list_submissions(user["id"], active_only=active)
    return JSONResponse(jsonable_encoder(rows))

@app.get("/api/tiers")
def tiers_api():
    return {"tiers": [tier_public_payload(tier) for tier in TIERS]}


@app.get("/api/me")
def me(authorization: str | None = Header(default=None)):
    user = _api_user(authorization)
    return JSONResponse(jsonable_encoder(user))


@app.get("/submissions/{submission_id}", response_class=HTMLResponse)
def submission_detail(submission_id: UUID) -> str:
    row = database.get_public_submission(submission_id)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    return submission_page(row)


@app.get("/samples/submission.py")
def sample_submission():
    path = repo_root / "submissions" / "baseline_adamw" / "submission.py"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="sample not installed")
    return FileResponse(path, media_type="text/x-python", filename="submission.py")


@app.get("/api/leaderboard")
def leaderboard_api():
    return JSONResponse(jsonable_encoder(database.leaderboard()))


@app.get("/api/submissions/{submission_id}")
def submission_api(
    submission_id: UUID,
    authorization: str | None = Header(default=None),
):
    user = _api_user(authorization)
    row = database.get_submission(submission_id)
    if row is None or row.get("user_id") != user["id"]:
        raise HTTPException(status_code=404, detail="submission not found")
    return JSONResponse(jsonable_encoder(_participant_submission_payload(row)))


@app.get("/api/submissions/{submission_id}/metrics")
def submission_metrics_api(
    submission_id: UUID,
    authorization: str | None = Header(default=None),
):
    user = _api_user(authorization)
    row = database.get_submission_metrics(submission_id, user["id"])
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    if row.get("status") != "succeeded":
        raise HTTPException(status_code=409, detail="metrics are not ready")
    metrics = row.get("metrics")
    if not isinstance(metrics, list):
        raise HTTPException(status_code=404, detail="metrics are unavailable")
    return Response(
        metrics_to_jsonl(metrics),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{submission_id}-metrics.jsonl"'
            )
        },
    )


@app.get("/healthz")
def health() -> dict[str, str]:
    database.ping()
    return {"status": "ok"}
