"""Tests for the bounded Lightning provider adapter."""

from __future__ import annotations

import enum
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import lightning_provider as lp

FakeStatus = enum.Enum(
    "FakeStatus",
    {
        "NotCreated": "NotCreated",
        "Pending": "Pending",
        "Running": "Running",
        "Stopping": "Stopping",
        "Stopped": "Stopped",
        "Completed": "Completed",
        "Failed": "Failed",
    },
)


class FakeMachine:
    T4 = "T4"


class FakeJobInstance:
    id = "job-123"
    link = "https://lightning.ai/example/jobs/job-123"

    def __init__(
        self,
        *,
        status: str,
        wait_error: Exception | None,
        logs: str,
        logs_error: bool,
        on_wait=None,
    ) -> None:
        self._status = status
        self.wait_error = wait_error
        self._logs = logs
        self.logs_error = logs_error
        self.on_wait = on_wait
        self.wait_calls: list[dict[str, object]] = []
        self.stop_calls = 0

    @property
    def status(self):
        return FakeStatus(self._status)

    @property
    def logs(self):
        if self.logs_error:
            raise RuntimeError("logs unavailable")
        return self._logs

    def wait(self, *, interval: int, timeout: int, stop_on_timeout: bool) -> None:
        self.wait_calls.append(
            {
                "interval": interval,
                "timeout": timeout,
                "stop_on_timeout": stop_on_timeout,
            }
        )
        if self.on_wait:
            self.on_wait()
        if self.wait_error:
            raise self.wait_error

    def stop(self) -> None:
        self.stop_calls += 1
        self._status = "Stopped"


class FakeJob:
    run_calls: ClassVar[list[dict[str, object]]] = []
    instances: ClassVar[list[FakeJobInstance]] = []
    run_error: Exception | None = None
    status = "Completed"
    wait_error: Exception | None = None
    logs = "step 100 loss 0.42\n"
    logs_error = False
    on_wait = None

    @classmethod
    def run(cls, **kwargs):
        cls.run_calls.append(kwargs)
        if cls.run_error is not None:
            raise cls.run_error
        job = FakeJobInstance(
            status=cls.status,
            wait_error=cls.wait_error,
            logs=cls.logs,
            logs_error=cls.logs_error,
            on_wait=cls.on_wait,
        )
        cls.instances.append(job)
        return job


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    for name in (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "LIGHTNING_API_KEY",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_FREE_ONLY_VERIFIED",
        "GITHUB_SHA",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def reset_fake_job():
    FakeJob.run_calls = []
    FakeJob.instances = []
    FakeJob.run_error = None
    FakeJob.status = "Completed"
    FakeJob.wait_error = None
    FakeJob.logs = "step 100 loss 0.42\n"
    FakeJob.logs_error = False
    FakeJob.on_wait = None


@pytest.fixture
def sdk_seam(monkeypatch):
    calls = SimpleNamespace(loads=0)

    def load_sdk():
        calls.loads += 1
        return FakeJob, FakeMachine, FakeStatus

    monkeypatch.setattr(lp, "_load_sdk", load_sdk)
    return calls


def _grant_free_only(monkeypatch) -> str:
    token = "hf_test_secret_value_012345"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("LIGHTNING_API_KEY", "lightning_test_key_012345")
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "owner/free-space")
    monkeypatch.setenv("LIGHTNING_FREE_ONLY_VERIFIED", "true")
    return token


def test_dry_run_needs_no_credentials_or_sdk(monkeypatch, sdk_seam, capsys, tmp_path):
    code = lp.main(
        [
            "--target",
            "615000",
            "--dry-run",
            "--git-ref",
            "a" * 40,
            "--log-file",
            str(tmp_path / "lightning.log"),
        ]
    )
    output = capsys.readouterr().out
    assert code == lp.EXIT_COMPLETE
    assert "timeout --signal=TERM --kill-after=120s 19800s bash -lc" in output
    assert "git checkout --detach " + "a" * 40 in output
    assert sdk_seam.loads == 0
    assert FakeJob.run_calls == []


@pytest.mark.parametrize(
    ("missing", "value"),
    [
        ("HF_TOKEN", None),
        ("LIGHTNING_API_KEY", None),
        ("LIGHTNING_TEAMSPACE", None),
        ("LIGHTNING_FREE_ONLY_VERIFIED", "false"),
    ],
)
def test_free_only_gate_blocks_before_sdk_load(
    monkeypatch, sdk_seam, missing, value, caplog
):
    _grant_free_only(monkeypatch)
    if value is None:
        monkeypatch.delenv(missing, raising=False)
    else:
        monkeypatch.setenv(missing, value)
    code = lp.main(["--target", "615000", "--git-ref", "b" * 40])
    assert code == lp.EXIT_LAUNCH_BLOCKED
    assert sdk_seam.loads == 0
    assert FakeJob.run_calls == []
    assert missing in caplog.text or "LIGHTNING_FREE_ONLY_VERIFIED=true" in caplog.text


def test_success_uses_bounded_t4_job_and_collects_logs(
    monkeypatch, sdk_seam, tmp_path, capsys
):
    token = _grant_free_only(monkeypatch)
    log_file = tmp_path / "logs" / "lightning.log"
    code = lp.main(
        [
            "--target",
            "615000",
            "--teamspace",
            "owner/free-space",
            "--git-ref",
            "c" * 40,
            "--log-file",
            str(log_file),
        ]
    )
    assert code == lp.EXIT_COMPLETE
    assert sdk_seam.loads == 1
    assert len(FakeJob.run_calls) == 1
    kwargs = FakeJob.run_calls[0]
    assert kwargs["machine"] == FakeMachine.T4
    assert kwargs["image"] == "python:3.12"
    assert kwargs["teamspace"] == "owner/free-space"
    assert kwargs["max_runtime"] == 19_800
    assert kwargs["max_run_attempts"] == 1
    assert kwargs["keep_minutes"] == 0
    assert kwargs["env"] == {"HF_TOKEN": token}
    assert token not in kwargs["command"]
    assert FakeJob.instances[0].wait_calls == [
        {"interval": 30, "timeout": 20_400, "stop_on_timeout": True}
    ]
    assert log_file.read_text() == "step 100 loss 0.42\n"
    output = capsys.readouterr().out
    assert "LIGHTNING_SLICE_JSON=" in output
    assert "https://lightning.ai/example/jobs/job-123" in output
    assert token not in output


@pytest.mark.parametrize(
    ("status", "expected"),
    [("Failed", lp.EXIT_JOB_FAILED), ("Stopped", lp.EXIT_CANCELLED)],
)
def test_terminal_failure_and_cancellation_codes(
    monkeypatch, sdk_seam, status, expected, tmp_path
):
    _grant_free_only(monkeypatch)
    FakeJob.status = status
    code = lp.main(
        [
            "--target",
            "615000",
            "--git-ref",
            "d" * 40,
            "--log-file",
            str(tmp_path / "log"),
        ]
    )
    assert code == expected
    assert len(FakeJob.run_calls) == 1
    assert (tmp_path / "log").read_text() == FakeJob.logs


def test_wait_timeout_stops_job_and_returns_timeout(monkeypatch, sdk_seam, tmp_path):
    _grant_free_only(monkeypatch)
    FakeJob.wait_error = TimeoutError("deadline")
    code = lp.main(
        [
            "--target",
            "615000",
            "--git-ref",
            "e" * 40,
            "--log-file",
            str(tmp_path / "log"),
        ]
    )
    assert code == lp.EXIT_TIMEOUT
    assert FakeJob.instances[0].stop_calls >= 1
    assert FakeJob.instances[0].status == FakeStatus.Stopped
    assert (tmp_path / "log").read_text() == FakeJob.logs


def test_wait_exception_stops_job_without_fallback_code(
    monkeypatch, sdk_seam, tmp_path
):
    _grant_free_only(monkeypatch)
    FakeJob.wait_error = RuntimeError("poll failed")
    code = lp.main(
        [
            "--target",
            "615000",
            "--git-ref",
            "f" * 40,
            "--log-file",
            str(tmp_path / "log"),
        ]
    )
    assert code == lp.EXIT_CONFIG
    assert FakeJob.instances[0].stop_calls >= 1


def test_sigterm_stops_job_and_restores_handler(monkeypatch, sdk_seam, tmp_path):
    _grant_free_only(monkeypatch)
    old_handler = signal.getsignal(signal.SIGTERM)
    FakeJob.on_wait = lambda: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    code = lp.main(
        [
            "--target",
            "615000",
            "--git-ref",
            "1" * 40,
            "--log-file",
            str(tmp_path / "log"),
        ]
    )
    assert code == lp.EXIT_CANCELLED
    assert FakeJob.instances[0].stop_calls >= 1
    assert signal.getsignal(signal.SIGTERM) == old_handler


def test_log_fetch_error_does_not_mask_success(monkeypatch, sdk_seam, tmp_path):
    _grant_free_only(monkeypatch)
    FakeJob.logs_error = True
    code = lp.main(
        [
            "--target",
            "615000",
            "--git-ref",
            "2" * 40,
            "--log-file",
            str(tmp_path / "missing.log"),
        ]
    )
    assert code == lp.EXIT_COMPLETE
    assert not (tmp_path / "missing.log").exists()


class HttpError(Exception):
    def __init__(self, status_code: int, message: str = "provider refused") -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429])
def test_explicit_submission_refusal_is_definite_and_not_retried(
    monkeypatch, sdk_seam, status, caplog
):
    token = _grant_free_only(monkeypatch)
    FakeJob.run_error = HttpError(status, f"provider response included {token}")
    code = lp.main(["--target", "615000", "--git-ref", "3" * 40])
    assert code == lp.EXIT_LAUNCH_BLOCKED
    assert len(FakeJob.run_calls) == 1
    assert token not in caplog.text


@pytest.mark.parametrize(
    "error",
    [TimeoutError("network timeout"), HttpError(408), HttpError(503)],
)
def test_ambiguous_submission_failure_is_not_retried(monkeypatch, sdk_seam, error):
    _grant_free_only(monkeypatch)
    FakeJob.run_error = error
    code = lp.main(["--target", "615000", "--git-ref", "4" * 40])
    assert code == lp.EXIT_CONFIG
    assert len(FakeJob.run_calls) == 1


def test_invalid_commit_and_target_are_local_configuration_errors(sdk_seam):
    assert (
        lp.main(["--target", "1", "--dry-run", "--git-ref", "main"]) == lp.EXIT_CONFIG
    )
    assert (
        lp.main(["--target", "0", "--dry-run", "--git-ref", "5" * 40]) == lp.EXIT_CONFIG
    )
    assert sdk_seam.loads == 0


def test_command_pins_commit_wraps_all_setup_and_quotes_values():
    args = lp.parse_args(
        [
            "--target",
            "615000",
            "--experiment-id",
            "experiment with 'quotes' ; echo unsafe",
            "--lr",
            "5e-5; echo unsafe",
            "--hub-resume",
            "--no-hub-push",
            "--allow-experiment-mismatch",
            "--min-lr",
            "1e-5",
            "--schedule-steps",
            "615000",
        ]
    )
    sha = "abcdef0123456789abcdef0123456789abcdef01"
    command = lp.build_job_command(args, sha)
    outer = shlex.split(command)
    assert outer[:6] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=120s",
        "19800s",
        "bash",
        "-lc",
    ]
    assert len(outer) == 7
    payload = outer[6]
    assert f"git checkout --detach {sha}" in payload
    assert (
        "git clone https://github.com/d-oit/tiny-cats-model.git /tmp/tiny-cats-model"
        in payload
    )
    assert "requirements-modal.txt" in payload
    assert "python scripts/train_lightning.py" in payload
    assert "--force-download" in payload
    assert "--allow-legacy-dataset-hash" in payload
    assert "/tmp/tiny-cats-data" in payload
    assert "/tmp/tiny-cats-checkpoints" in payload
    assert "--hub-resume" in payload
    assert "--no-hub-push" in payload
    assert "--allow-experiment-mismatch" in payload
    train_tokens = shlex.split(payload.splitlines()[-1])
    assert (
        train_tokens[train_tokens.index("--experiment-id") + 1]
        == "experiment with 'quotes' ; echo unsafe"
    )
    assert train_tokens[train_tokens.index("--lr") + 1] == "5e-5; echo unsafe"
    assert train_tokens[train_tokens.index("--min-lr") + 1] == "1e-5"
    assert train_tokens[train_tokens.index("--schedule-steps") + 1] == "615000"
    assert "--hub-token" not in train_tokens
