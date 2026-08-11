"""Small command-line client for One Layer Deeper submissions."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import UUID
import webbrowser

import httpx

from submission_validation import validate_submission_source
from service.tiers import resolve_tier_dataset


HOSTED_SERVER = "https://onelayerdeeper.ai"
DEFAULT_SERVER = os.environ.get("ONE_LAYER_URL", HOSTED_SERVER)
CONFIG_PATH = Path(
    os.environ.get(
        "ONE_LAYER_CONFIG_PATH",
        str(Path.home() / ".config" / "one-layer" / "config.json"),
    )
)
TERMINAL_STATUSES = {"succeeded", "failed"}
DEPTH_LADDER = (1, 2, 4, 8, 16, 32, 64)


def _server(value: str) -> str:
    return value.rstrip("/")


def _score(row: dict) -> str:
    value = row.get("score")
    if value is None:
        value = ((row.get("result") or {}).get("score") or {}).get(
            "mean_exact_accuracy"
        )
    return "—" if value is None else f"{100 * float(value):.2f}%"



def _max_t(row: dict, key: str) -> str:
    profile = (row.get("result") or {}).get("depth_profile") or {}
    if key == "ood_n_max_certified_time_steps":
        available = row.get("ood_n_profile_available")
        if available is None:
            available = profile.get("ood_n_profile_available")
        if available is False:
            return "N/A"
    value = row.get(key)
    if value is None:
        value = profile.get(key)
    if value is None:
        return "None certified" if row.get("status") == "succeeded" else "—"
    return str(int(value))


def _leaderboard_score(
    row: dict,
    key: str,
    accuracy_key: str,
    *,
    available: bool = True,
) -> str:
    if not available:
        return "N/A; Not evaluated"
    value = row.get(key)
    certified = 0 if value is None else int(value)
    next_rung = next((rung for rung in DEPTH_LADDER if rung > certified), None)
    if next_rung is None:
        return f"T={DEPTH_LADDER[-1]}; Certified"
    accuracy_percent = row.get(accuracy_key)
    accuracy = (
        f"Acc {float(accuracy_percent):.4f}%"
        if accuracy_percent is not None
        else "Acc unavailable"
    )
    return f"T={next_rung}; {accuracy}"


def _load_saved_api_key(server: str) -> str | None:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    entry = (config.get("servers") or {}).get(_server(server)) or {}
    value = entry.get("api_key")
    return value if isinstance(value, str) and value else None


def _save_api_key(server: str, api_key: str) -> None:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        config = {}
    servers = config.setdefault("servers", {})
    servers[_server(server)] = {"api_key": api_key}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    CONFIG_PATH.chmod(0o600)


def _api_key(args) -> str:
    value = (
        getattr(args, "api_key", None)
        or os.environ.get("ONE_LAYER_API_KEY")
        or _load_saved_api_key(args.server)
    )
    if not value:
        raise ValueError(
            "no API key; run `one-layer login`, set ONE_LAYER_API_KEY, or pass --api-key"
        )
    return value


def _auth_headers(args) -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key(args)}"}


def _print_status(row: dict) -> None:
    print(f"submission  {row['id']}")
    print(f"file        {row['filename']}")
    print(f"status      {row['status']}")
    print(f"score       {_score(row)}")
    print(f"max T       {_max_t(row, 'max_certified_time_steps')}")
    print(f"OOD N max T {_max_t(row, 'ood_n_max_certified_time_steps')}")
    print(f"tier        {row.get('tier') or 'legacy'}")
    print(f"dataset     {row.get('dataset_label') or row['manifest_name']}")
    print(f"suite       {row['manifest_name']}")
    print(f"run         {row['run_id']}")
    print(f"modal call  {row.get('modal_call_id') or 'not started'}")
    if row.get("error_code"):
        print(f"error       {row['error_code']}")


def _get_status(
    client: httpx.Client,
    server: str,
    submission_id: str,
    headers: dict[str, str],
) -> dict:
    response = client.get(
        f"{server}/api/submissions/{submission_id}",
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


def _watch(
    client: httpx.Client,
    server: str,
    submission_id: str,
    interval: float,
    headers: dict[str, str],
) -> dict:
    previous = None
    while True:
        row = _get_status(client, server, submission_id, headers)
        if row["status"] != previous:
            print(f"[{row['status']}] {row['filename']}", flush=True)
            previous = row["status"]
        if row["status"] in TERMINAL_STATUSES:
            return row
        time.sleep(interval)


def command_validate(args) -> int:
    path = Path(args.file)
    try:
        source = path.read_text(encoding="utf-8")
        validate_submission_source(path.name, source, 256 * 1024)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2
    print(f"valid: {path} ({len(source.encode('utf-8'))} bytes)")
    return 0


def command_submit(args) -> int:
    selected_tier, selected_dataset = resolve_tier_dataset(
        args.tier, args.dataset
    )
    path = Path(args.file)
    validation = command_validate(argparse.Namespace(file=str(path)))
    if validation:
        return validation
    server = _server(args.server)
    headers = _auth_headers(args)
    with httpx.Client(timeout=args.request_timeout) as client:
        with path.open("rb") as file:
            response = client.post(
                f"{server}/api/submissions",
                files={"file": (path.name, file, "text/x-python")},
                data={
                    "tier": selected_tier.id,
                    "dataset": selected_dataset.id,
                },
                headers=headers,
            )
        if response.is_error:
            print(f"submission rejected: {response.text}", file=sys.stderr)
            return 1
        accepted = response.json()
        submission_id = accepted["submission_id"]
        print(f"queued: {submission_id}")
        print(f"tier:   {accepted['tier']}")
        print(f"data:   {accepted['dataset_id']}")
        remaining = accepted["remaining_attempts_today"]
        print(
            f"left:   {remaining} today"
            if remaining is not None
            else "left:   unlimited"
        )
        print(f"view:   {server}{accepted['html_url']}")
        if not args.wait:
            return 0
        row = _watch(client, server, submission_id, args.poll_interval, headers)
        _print_status(row)
        return 0 if row["status"] == "succeeded" else 1


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        api_key = (query.get("api_key") or [None])[0]
        if parsed.path != "/callback" or not api_key:
            self.send_response(400)
            message = "GitHub login did not return an API key."
        else:
            self.server.oauth_result = {  # type: ignore[attr-defined]
                "api_key": api_key,
                "display_name": (query.get("display_name") or ["participant"])[0],
                "github_login": (query.get("github_login") or [""])[0],
            }
            self.send_response(200)
            message = "One Layer Deeper login complete. You can close this tab."
        body = f"<!doctype html><title>One Layer Deeper</title><h1>{message}</h1>".encode()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        del format, args


def command_login(args) -> int:
    server = _server(args.server)
    callback = HTTPServer(("127.0.0.1", 0), _OAuthCallbackHandler)
    callback.timeout = args.login_timeout
    callback.oauth_result = None  # type: ignore[attr-defined]
    port = callback.server_address[1]
    login_url = f"{server}/auth/github?{urlencode({'cli_port': port})}"
    print(f"GitHub login: {login_url}")
    if not args.no_open and not webbrowser.open(login_url):
        print("Open the URL above in a browser.")
    try:
        callback.handle_request()
        registration = callback.oauth_result  # type: ignore[attr-defined]
    finally:
        callback.server_close()
    if registration is None:
        print("login timed out before GitHub authorization completed", file=sys.stderr)
        return 1
    api_key = registration["api_key"]
    github_login = registration["github_login"]
    print(f"authenticated  {registration['display_name']} (@{github_login})")
    if not args.no_save:
        _save_api_key(server, api_key)
        print(f"saved          {CONFIG_PATH}")
    else:
        print("save this key now; the service cannot recover it")
    return 0


def command_whoami(args) -> int:
    server = _server(args.server)
    response = httpx.get(
        f"{server}/api/me",
        headers=_auth_headers(args),
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    user = response.json()
    print(f"participant  {user['display_name']}")
    print(f"github       @{user['github_login']}")
    print(f"email        {user['email']}")
    print(f"key prefix   {user['api_key_prefix']}")
    print(f"status       {user['status']}")
    return 0


def command_status(args) -> int:
    UUID(args.submission_id)
    server = _server(args.server)
    headers = _auth_headers(args)
    with httpx.Client(timeout=args.request_timeout) as client:
        row = (
            _watch(
                client,
                server,
                args.submission_id,
                args.poll_interval,
                headers,
            )
            if args.watch
            else _get_status(client, server, args.submission_id, headers)
        )
    _print_status(row)
    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    return 1 if row["status"] == "failed" else 0


def command_jobs(args) -> int:
    server = _server(args.server)
    response = httpx.get(
        f"{server}/api/submissions",
        params={"active": not args.all},
        headers=_auth_headers(args),
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    rows = response.json()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("No submissions." if args.all else "No active jobs.")
        return 0
    print(f"{'status':<10}  {'tier':<6}  {'dataset':<10}  {'submission':<36}  file")
    print(f"{'—' * 10}  {'—' * 6}  {'—' * 10}  {'—' * 36}  {'—' * 20}")
    for row in rows:
        dataset = row.get("dataset_label") or row.get("dataset_id") or "legacy"
        print(
            f"{row['status']:<10}  {(row.get('tier') or 'legacy'):<6}  "
            f"{dataset:<10}  {str(row['id']):<36}  {row['filename']}"
        )
    return 0


def command_metrics(args) -> int:
    submission_id = str(UUID(args.submission_id))
    output = Path(args.output or f"{submission_id}-metrics.jsonl")
    if output.exists() and not args.force:
        raise ValueError(f"{output} already exists; pass --force to overwrite it")
    server = _server(args.server)
    response = httpx.get(
        f"{server}/api/submissions/{submission_id}/metrics",
        headers=_auth_headers(args),
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    if args.force:
        output.write_bytes(response.content)
    else:
        try:
            with output.open("xb") as file:
                file.write(response.content)
        except FileExistsError as exc:
            raise ValueError(
                f"{output} already exists; pass --force to overwrite it"
            ) from exc
    print(f"saved metrics to {output}")
    return 0


def command_leaderboard(args) -> int:
    server = _server(args.server)
    response = httpx.get(f"{server}/api/leaderboard", timeout=args.request_timeout)
    response.raise_for_status()
    rows = response.json()[: args.limit]
    if not rows:
        print("No submissions yet.")
        return 0
    print(
        "Hard leaderboard · ranked by Max T, OOD N Max T, "
        "then unrounded next-rung accuracy"
    )
    print(f"{'#':>3}  participant")
    print(f"{'—' * 3}  {'—' * 32}")
    for rank, row in enumerate(rows, start=1):
        print(f"{rank:>3}  {row['submitter']}")
        print(
            "     In Distribution N progress: "
            + _leaderboard_score(
                row,
                "max_certified_time_steps",
                "seen_tiebreak_accuracy_percent",
            )
        )
        print(
            "     Out of Distribution N progress: "
            + _leaderboard_score(
                row,
                "ood_n_max_certified_time_steps",
                "ood_n_tiebreak_accuracy_percent",
                available=row.get("ood_n_profile_available") is not False,
            )
        )
    return 0

def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server", default=DEFAULT_SERVER, help="leaderboard service URL"
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)


def _add_auth_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key",
        help="participant API key (otherwise ONE_LAYER_API_KEY or saved config)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="one-layer", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="check a submission before upload"
    )
    validate.add_argument("file")
    validate.set_defaults(handler=command_validate)

    login = subparsers.add_parser(
        "login", help="authenticate with GitHub and save an API key"
    )
    login.add_argument(
        "--no-open", action="store_true", help="print the GitHub URL without opening it"
    )
    login.add_argument(
        "--no-save", action="store_true", help="authenticate but do not save the key"
    )
    login.add_argument("--login-timeout", type=float, default=300.0)
    _add_connection_args(login)
    login.set_defaults(handler=command_login)

    submit = subparsers.add_parser("submit", help="upload one Python submission")
    submit.add_argument("file")
    submit.add_argument(
        "--tier",
        required=True,
        choices=("easy", "medium", "hard"),
        help="compute tier",
    )
    submit.add_argument(
        "--dataset",
        help="dataset ID (required for Easy/Medium; Hard defaults to h1)",
    )
    submit.add_argument(
        "--wait", action="store_true", help="wait and print the final score"
    )
    submit.add_argument("--poll-interval", type=float, default=5.0)
    _add_connection_args(submit)
    _add_auth_arg(submit)
    submit.set_defaults(handler=command_submit)

    whoami = subparsers.add_parser("whoami", help="show the participant for an API key")
    _add_connection_args(whoami)
    _add_auth_arg(whoami)
    whoami.set_defaults(handler=command_whoami)

    status = subparsers.add_parser("status", help="show one submission run")
    status.add_argument("submission_id")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--json", action="store_true")
    status.add_argument("--poll-interval", type=float, default=5.0)
    _add_connection_args(status)
    _add_auth_arg(status)
    status.set_defaults(handler=command_status)

    metrics = subparsers.add_parser(
        "metrics",
        help="download structured metrics for one completed submission",
    )
    metrics.add_argument("submission_id")
    metrics.add_argument("--output")
    metrics.add_argument("--force", action="store_true")
    _add_connection_args(metrics)
    _add_auth_arg(metrics)
    metrics.set_defaults(handler=command_metrics)

    jobs = subparsers.add_parser("jobs", help="list your submission jobs")
    jobs.add_argument(
        "--all", action="store_true", help="include completed and failed jobs"
    )
    jobs.add_argument("--json", action="store_true")
    _add_connection_args(jobs)
    _add_auth_arg(jobs)
    jobs.set_defaults(handler=command_jobs)

    board = subparsers.add_parser("leaderboard", help="print the current leaderboard")
    board.add_argument("--limit", type=int, default=20)
    _add_connection_args(board)
    board.set_defaults(handler=command_leaderboard)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except (httpx.HTTPError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
