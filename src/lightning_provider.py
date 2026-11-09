#!/usr/bin/env python3
"""Bounded Lightning AI Job adapter for the free-only GPU pool."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("lightning-provider")

EXIT_COMPLETE = 0
EXIT_JOB_FAILED = 2
EXIT_CANCELLED = 3
EXIT_TIMEOUT = 4
EXIT_CONFIG = 5
EXIT_LAUNCH_BLOCKED = 6

REPO_URL = "https://github.com/d-oit/tiny-cats-model.git"
REPO_DIR = "/tmp/tiny-cats-model"
DATA_DIR = "/tmp/tiny-cats-data"
CHECKPOINT_DIR = "/tmp/tiny-cats-checkpoints"
JOB_IMAGE = "python:3.12"
MAX_RUNTIME_SECONDS = 19_800
HARD_TIMEOUT_SECONDS = 19_800
WAIT_INTERVAL_SECONDS = 30
WAIT_TIMEOUT_SECONDS = 20_400
STOP_CONFIRM_ATTEMPTS = 5
STOP_CONFIRM_INTERVAL_SECONDS = 2
REFUSAL_HTTP_STATUSES = frozenset({400, 401, 403, 404, 422, 429})
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one Lightning slice using the Kaggle provider's CLI contract."""
    parser = argparse.ArgumentParser(
        description="Launch one bounded GPU-pool slice as a Lightning Job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=int, required=True, help="Global step target")
    parser.add_argument("--hub-repo", default="d4oit/tiny-cats-model")
    parser.add_argument("--experiment-id", default="dit-breed-conditioned-v4")
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--lr", default="5e-5")
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--save-interval", type=int, default=2_500)
    parser.add_argument("--hub-push-interval", type=int, default=5_000)
    parser.add_argument("--min-lr", default=None)
    parser.add_argument("--schedule-steps", type=int, default=None)
    parser.add_argument("--hub-resume", action="store_true")
    parser.add_argument("--no-hub-push", action="store_true")
    parser.add_argument("--allow-experiment-mismatch", action="store_true")
    parser.add_argument(
        "--teamspace",
        default=os.environ.get("LIGHTNING_TEAMSPACE"),
        help="Explicit Lightning teamspace (defaults to LIGHTNING_TEAMSPACE)",
    )
    parser.add_argument(
        "--git-ref",
        default=None,
        help="Full published commit SHA (defaults to GITHUB_SHA, else local HEAD)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", default="lightning-training.log")
    return parser.parse_args(argv)


def _train_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--data-dir",
        DATA_DIR,
        "--checkpoint-dir",
        CHECKPOINT_DIR,
        "--steps",
        str(args.target),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--warmup-steps",
        str(args.warmup_steps),
        "--save-interval",
        str(args.save_interval),
        "--hub-push-interval",
        str(args.hub_push_interval),
        "--hub-repo",
        str(args.hub_repo),
        "--experiment-id",
        str(args.experiment_id),
    ]
    if args.min_lr is not None:
        values.extend(("--min-lr", str(args.min_lr)))
    if args.schedule_steps is not None:
        values.extend(("--schedule-steps", str(args.schedule_steps)))
    if args.hub_resume:
        values.append("--hub-resume")
    if args.no_hub_push:
        values.append("--no-hub-push")
    if args.allow_experiment_mismatch:
        values.append("--allow-experiment-mismatch")
    values.extend(("--force-download", "--allow-legacy-dataset-hash"))
    return values


def build_job_command(args: argparse.Namespace, git_ref: str) -> str:
    """Build a shell-safe command pinned to an immutable published commit."""
    if not COMMIT_RE.fullmatch(git_ref):
        raise ValueError("git-ref must be a full 40-hex commit SHA")
    train_command = shlex.join(
        ["python", "scripts/train_lightning.py", *_train_args(args)]
    )
    payload = "\n".join(
        (
            "set -euo pipefail",
            f"rm -rf {shlex.quote(REPO_DIR)}",
            f"git clone {shlex.quote(REPO_URL)} {shlex.quote(REPO_DIR)}",
            f"cd {shlex.quote(REPO_DIR)}",
            f"git checkout --detach {shlex.quote(git_ref)}",
            "python -m pip install --no-cache-dir -r requirements-modal.txt",
            train_command,
        )
    )
    return (
        f"timeout --signal=TERM --kill-after=120s {HARD_TIMEOUT_SECONDS}s "
        f"bash -lc {shlex.quote(payload)}"
    )


def _resolve_git_ref(explicit: str | None) -> str:
    candidate = (explicit or os.environ.get("GITHUB_SHA") or "").strip()
    if not candidate:
        repo_root = Path(__file__).resolve().parent.parent
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("could not resolve GITHUB_SHA or local HEAD") from exc
        candidate = result.stdout.strip()
    if not COMMIT_RE.fullmatch(candidate):
        raise ValueError("git-ref must be a full 40-hex published commit SHA")
    return candidate.lower()


def _missing_prerequisites(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    if not os.environ.get("HF_TOKEN"):
        missing.append("HF_TOKEN (Hub access secret)")
    if not os.environ.get("LIGHTNING_API_KEY"):
        missing.append("LIGHTNING_API_KEY (Lightning authentication secret)")
    if not (args.teamspace or os.environ.get("LIGHTNING_TEAMSPACE")):
        missing.append("--teamspace or LIGHTNING_TEAMSPACE (explicit teamspace)")
    verified = os.environ.get("LIGHTNING_FREE_ONLY_VERIFIED", "").strip().lower()
    if verified != "true":
        missing.append(
            "LIGHTNING_FREE_ONLY_VERIFIED=true (operator attestation of promotional-credit origin, disabled auto-reload/purchased-credit/BYOC billing, and enforceable no-paid-overage)"
        )
    return missing


def _load_sdk() -> tuple[Any, Any, Any]:
    """Import the SDK only after local and free-only preflight succeeds."""
    from lightning_sdk import Job, Machine, Status

    return Job, Machine, Status


def _http_status(exc: BaseException) -> int | None:
    """Read only structured HTTP status fields, never infer from exception prose."""
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        try:
            if candidate is not None and 100 <= int(candidate) <= 599:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def classify_submission_error(exc: BaseException) -> int:
    """Classify definite HTTP refusal separately from ambiguous submission."""
    if _http_status(exc) in REFUSAL_HTTP_STATUSES:
        return EXIT_LAUNCH_BLOCKED
    return EXIT_CONFIG


def _safe_reason(exc: BaseException) -> str:
    """Return only a short status/reason, never an arbitrary response body."""
    status = _http_status(exc)
    reason = getattr(exc, "reason", None)
    if not isinstance(reason, str) or not reason.strip():
        reason = type(exc).__name__
    text = " ".join(reason.split())
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "LIGHTNING_API_KEY"):
        secret = os.environ.get(key)
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", text)
    text = re.sub(r"\b(?:hf|lg)_[A-Za-z0-9_-]{8,}\b", "[redacted]", text)
    if status is not None:
        return f"HTTP {status}: {text[:160]}"
    return text[:160] or type(exc).__name__


def _status_name(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value).rsplit(".", 1)[-1].strip()


def _job_status(job: Any) -> str:
    return _status_name(job.status)


def _wait_for_job(job: Any) -> None:
    """Wait once; the SDK raises TimeoutError and stops when deadline expires."""
    job.wait(
        interval=WAIT_INTERVAL_SECONDS,
        timeout=WAIT_TIMEOUT_SECONDS,
        stop_on_timeout=True,
    )


def _stop_and_confirm(job: Any, status_type: Any) -> None:
    try:
        job.stop()
    except Exception as exc:
        logger.warning("job stop request failed (%s)", _safe_reason(exc))
    terminal = {
        _status_name(status_type.Completed),
        _status_name(status_type.Stopped),
        _status_name(status_type.Failed),
    }
    for attempt in range(STOP_CONFIRM_ATTEMPTS):
        try:
            status = _job_status(job)
        except Exception as exc:
            logger.warning(
                "could not confirm stopped job state (%s)", _safe_reason(exc)
            )
            return
        if status in terminal:
            logger.info("job terminal state after stop: %s", status)
            return
        if attempt + 1 < STOP_CONFIRM_ATTEMPTS:
            time.sleep(STOP_CONFIRM_INTERVAL_SECONDS)
    logger.warning("could not confirm a terminal Lightning job state")


def _collect_logs(job: Any, log_file: str | Path) -> None:
    try:
        text = str(job.logs)
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.warning("could not collect Lightning job logs (%s)", _safe_reason(exc))


def _exit_for_status(status: str) -> int:
    if status == "Completed":
        return EXIT_COMPLETE
    if status == "Failed":
        return EXIT_JOB_FAILED
    if status in {"Stopped", "Stopping"}:
        return EXIT_CANCELLED
    logger.error("job returned unexpected nonterminal status %s", status or "unknown")
    return EXIT_CONFIG


def _watch_job(job: Any, status_type: Any, log_file: str | Path) -> tuple[int, str]:
    """Wait and always collect logs; stop allocated jobs on interruptions/errors."""
    previous_handlers: dict[int, Any] = {}
    interrupted = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        logger.warning("received signal %s; stopping Lightning job", signum)
        with suppress(Exception):
            job.stop()
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(ValueError):  # Non-main threads cannot install handlers.
            previous_handlers[sig] = signal.signal(sig, handle_signal)

    code = EXIT_CONFIG
    status = "unknown"
    try:
        try:
            _wait_for_job(job)
            status = _job_status(job)
            code = _exit_for_status(status)
            if code == EXIT_CONFIG:
                _stop_and_confirm(job, status_type)
                try:
                    status = _job_status(job)
                except Exception:
                    status = "unknown"
        except TimeoutError as exc:
            logger.error("Lightning job poll timeout: %s", _safe_reason(exc))
            _stop_and_confirm(job, status_type)
            try:
                status = _job_status(job)
            except Exception:
                status = "unknown"
            code = EXIT_TIMEOUT
        except KeyboardInterrupt:
            logger.warning("Lightning job cancelled by signal or keyboard interrupt")
            _stop_and_confirm(job, status_type)
            try:
                status = _job_status(job)
            except Exception:
                status = "unknown"
            code = EXIT_CANCELLED
        except Exception as exc:
            logger.error("Lightning job monitoring failed (%s)", _safe_reason(exc))
            _stop_and_confirm(job, status_type)
            try:
                status = _job_status(job)
            except Exception:
                status = "unknown"
            code = EXIT_CONFIG
    finally:
        for sig, previous in previous_handlers.items():
            with suppress(ValueError):
                signal.signal(sig, previous)
        _collect_logs(job, log_file)
    if interrupted:
        code = EXIT_CANCELLED
    return code, status


def _job_attr(job: Any, name: str) -> str | None:
    try:
        value = getattr(job, name, None)
    except Exception:
        return None
    return str(value) if value is not None else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    if args.target <= 0:
        logger.error("invalid local configuration: --target must be positive")
        return EXIT_CONFIG
    try:
        git_ref = _resolve_git_ref(args.git_ref)
        command = build_job_command(args, git_ref)
    except ValueError as exc:
        logger.error("invalid local configuration: %s", _safe_reason(exc))
        return EXIT_CONFIG

    if args.dry_run:
        print(
            "LIGHTNING_DRY_RUN="
            + json.dumps(
                {"git_ref": git_ref, "teamspace": args.teamspace, "command": command},
                sort_keys=True,
            )
        )
        return EXIT_COMPLETE

    missing = _missing_prerequisites(args)
    if missing:
        logger.error("Lightning free-only launch unavailable; missing prerequisite(s):")
        for item in missing:
            logger.error("  %s", item)
        return EXIT_LAUNCH_BLOCKED

    teamspace = args.teamspace or os.environ.get("LIGHTNING_TEAMSPACE")
    try:
        Job, Machine, Status = _load_sdk()
    except Exception as exc:
        logger.error("could not load lightning_sdk (%s)", _safe_reason(exc))
        return EXIT_CONFIG

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"tiny-cats-{args.target}-{timestamp}"
    try:
        job = Job.run(
            name=name,
            machine=Machine.T4,
            image=JOB_IMAGE,
            teamspace=teamspace,
            command=command,
            env={"HF_TOKEN": os.environ["HF_TOKEN"]},
            max_runtime=MAX_RUNTIME_SECONDS,
            max_run_attempts=1,
            keep_minutes=0,
        )
    except Exception as exc:
        code = classify_submission_error(exc)
        status_code = _http_status(exc)
        if code == EXIT_LAUNCH_BLOCKED:
            logger.error(
                "Lightning explicitly refused job submission (status=%s, reason=%s)",
                status_code,
                _safe_reason(exc),
            )
        else:
            logger.error(
                "Lightning submission outcome is ambiguous (status=%s, reason=%s); not retrying",
                status_code,
                _safe_reason(exc),
            )
        return code

    job_id = _job_attr(job, "id")
    link = _job_attr(job, "link")
    logger.info(
        "Lightning job submitted: id=%s link=%s",
        job_id or "unknown",
        link or "unknown",
    )
    code, status = _watch_job(job, Status, args.log_file)
    summary = {
        "id": job_id,
        "link": link,
        "name": name,
        "status": status,
        "exit_code": code,
        "log": str(args.log_file),
    }
    print("LIGHTNING_SLICE_JSON=" + json.dumps(summary, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
