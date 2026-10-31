"""tests/test_kaggle_provider.py

Unit coverage for the Kaggle slice adapter (ADR-067):

- kernel workspace packaging is self-contained and refuses alien directories
- kernel metadata satisfies the fields kaggle-api validates on push
- the slice arguments mirror the Modal launch contract (global steps, Hub
  resume, experiment identity, LR floor/horizon, fingerprint-parity download)
- polling only trusts a terminal status once the *pushed* version is live
  (a kernel reports its previous version's status until then)
- server-side push rejections and log retrieval behave as documented
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import kaggle_provider
from kaggle_provider import (
    ENTRY_FILE,
    EXIT_COMPLETE,
    EXIT_CONFIG,
    EXIT_KERNEL_ERROR,
    EXIT_TIMEOUT,
    KERNEL_METADATA_FILE,
    KaggleLaunchError,
    build_kernel_workspace,
    build_slice_args,
    fetch_kernel_logs,
    main,
    push_kernel,
    resolve_kernel_id,
    status_name,
    wait_for_kernel,
)


def _fake_repo(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "train_dit.py").write_text("# trainer\n")
    (root / "src" / "dataset.py").write_text("# dataset\n")
    (root / "data").mkdir()
    (root / "data" / "download.py").write_text("# downloader\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "train_kaggle.py").write_text("# kaggle trainer\n")
    return root


class FakeApi:
    """Scripted Kaggle API double."""

    def __init__(
        self,
        *,
        statuses: list[str] | None = None,
        version: int | None = 7,
        push_error: str | None = None,
        invalid_sources: list[str] | None = None,
        log_text: str | None = "step 10 loss 0.5\n",
    ) -> None:
        self.statuses = list(statuses or ["RUNNING", "COMPLETE"])
        self.version = version
        self.push_error = push_error
        self.invalid_sources = invalid_sources or []
        self.log_text = log_text
        self.push_calls: list[tuple[str, str | None]] = []
        self.output_calls: list[str] = []

    # ── push ────────────────────────────────────────────────────────────
    def kernels_push(self, folder: str, timeout: str | None = None) -> object:
        self.push_calls.append((folder, timeout))
        return SimpleNamespace(
            ref="d-oit/tiny-cats-dit-slice",
            kernel_id=123,
            version_number=self.version,
            url="https://kaggle.com/code/d-oit/tiny-cats-dit-slice",
            error=self.push_error,
            invalid_dataset_sources=list(self.invalid_sources),
        )

    # ── status / list ───────────────────────────────────────────────────
    def kernels_status(self, kernel: str) -> object:
        name = self.statuses.pop(0) if self.statuses else "COMPLETE"
        return SimpleNamespace(
            status=name, failure_message="boom" if name == "ERROR" else None
        )

    def kernels_list(self, **_kwargs: object) -> list[object]:
        return [
            SimpleNamespace(
                ref="d-oit/tiny-cats-dit-slice",
                slug="tiny-cats-dit-slice",
                current_version_number=self.version,
            )
        ]

    # ── outputs ─────────────────────────────────────────────────────────
    def kernels_output(
        self,
        kernel: str,
        path: str,
        file_pattern: str | None = None,
        force: bool = False,
    ) -> tuple[list[str], str]:
        self.output_calls.append(path)
        if self.log_text is not None:
            target = Path(path) / "tiny-cats-dit-slice.log"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.log_text)
        return ([], "")


class TestSliceArgs:
    def test_carries_the_campaign_contract(self) -> None:
        args = build_slice_args(
            steps=540_000,
            batch_size="32",
            lr="5e-5",
            warmup_steps=2_000,
            save_interval=2_500,
            hub_push_interval=5_000,
            hub_repo="d-oit/tiny-cats-model",
            experiment_id="dit-breed-conditioned-v4",
            min_lr="1e-5",
            schedule_steps=600_000,
            hub_resume=True,
            force_download=True,
        )
        joined = " ".join(args)
        assert "--steps 540000" in joined
        assert "--experiment-id dit-breed-conditioned-v4" in joined
        assert "--min-lr 1e-5" in joined
        assert "--schedule-steps 600000" in joined
        assert "--hub-push-interval 5000" in joined
        assert "--hub-resume" in args
        assert "--force-download" in args

    def test_optional_knobs_are_opt_in(self) -> None:
        args = build_slice_args(
            steps=15_000,
            batch_size="32",
            lr="5e-5",
            warmup_steps=2_000,
            save_interval=500,
            hub_push_interval=5_000,
            hub_repo="d-oit/tiny-cats-model",
            experiment_id="exp",
            force_download=False,
        )
        joined = " ".join(args)
        assert "--min-lr" not in joined
        assert "--schedule-steps" not in joined
        assert "--hub-resume" not in args
        assert "--force-download" not in args


class TestKernelWorkspace:
    def test_packages_a_self_contained_kernel(self, tmp_path: Path) -> None:
        repo = _fake_repo(tmp_path / "repo")
        workspace = build_kernel_workspace(
            repo,
            tmp_path / "kernel",
            kernel_id="d-oit/tiny-cats-dit-slice",
            slice_args=["--steps", "15000", "--hub-resume"],
            hf_token_dataset="d-oit/tiny-cats-hf-token",
        )

        assert (workspace / "src" / "train_dit.py").is_file()
        assert (workspace / "scripts" / "train_kaggle.py").is_file()
        assert (workspace / "data" / "download.py").is_file()

        entry = (workspace / ENTRY_FILE).read_text()
        assert "'--steps'" in entry and "15000" in entry
        assert "d-oit/tiny-cats-hf-token" in entry
        assert "runpy.run_path" in entry

        metadata = json.loads((workspace / KERNEL_METADATA_FILE).read_text())
        assert metadata["id"] == "d-oit/tiny-cats-dit-slice"
        assert metadata["code_file"] == ENTRY_FILE
        assert metadata["language"] == "python"
        assert metadata["kernel_type"] == "script"
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is True
        assert metadata["enable_internet"] is True
        assert metadata["dataset_sources"] == ["d-oit/tiny-cats-hf-token"]
        # Title must satisfy kaggle-api's >=5 chars rule and slugify to the id.
        assert metadata["title"] == "tiny cats dit slice"

    def test_tokens_are_not_embedded_in_the_kernel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_sentinel_value_12345")
        monkeypatch.setenv("KAGGLE_API_TOKEN", "kaggle-sentinel-value")
        repo = _fake_repo(tmp_path / "repo")
        workspace = build_kernel_workspace(
            repo,
            tmp_path / "kernel",
            kernel_id="d-oit/slice",
            slice_args=["--steps", "1"],
        )
        blob = "\n".join(p.read_text() for p in workspace.rglob("*") if p.is_file())
        # Credentials arrive at run time (environment, private dataset mount or
        # Kaggle Secrets); nothing is baked into the pushed kernel.
        assert "hf_sentinel_value_12345" not in blob
        assert "kaggle-sentinel-value" not in blob

    def test_rebuilds_its_own_workspace(self, tmp_path: Path) -> None:
        repo = _fake_repo(tmp_path / "repo")
        workspace = build_kernel_workspace(
            repo,
            tmp_path / "kernel",
            kernel_id="d-oit/slice",
            slice_args=["--steps", "1"],
        )
        stale = workspace / "stale.txt"
        stale.write_text("old")
        build_kernel_workspace(
            repo,
            tmp_path / "kernel",
            kernel_id="d-oit/slice",
            slice_args=["--steps", "2"],
        )
        assert not stale.exists()
        assert "--steps" in (workspace / ENTRY_FILE).read_text()

    def test_refuses_to_clear_an_unrelated_directory(self, tmp_path: Path) -> None:
        repo = _fake_repo(tmp_path / "repo")
        victim = tmp_path / "important"
        victim.mkdir()
        (victim / "keep.txt").write_text("precious")
        with pytest.raises(ValueError, match="refusing to clear"):
            build_kernel_workspace(
                repo, victim, kernel_id="d-oit/slice", slice_args=["--steps", "1"]
            )
        assert (victim / "keep.txt").read_text() == "precious"

    def test_kernel_id_must_be_owner_slash_slug(self, tmp_path: Path) -> None:
        repo = _fake_repo(tmp_path / "repo")
        with pytest.raises(ValueError, match="owner/slug"):
            build_kernel_workspace(
                repo, tmp_path / "kernel", kernel_id="no-slash", slice_args=[]
            )

    def test_missing_repo_file_is_reported(self, tmp_path: Path) -> None:
        repo = _fake_repo(tmp_path / "repo")
        (repo / "data" / "download.py").unlink()
        with pytest.raises(ValueError, match="missing kernel file"):
            build_kernel_workspace(
                repo, tmp_path / "kernel", kernel_id="d-oit/slice", slice_args=[]
            )


class TestStatusHandling:
    def test_status_name_normalises_enum_and_string(self) -> None:
        class Status:
            name = "COMPLETE"

        assert status_name(SimpleNamespace(status=Status())) == "COMPLETE"
        assert status_name("KernelWorkerStatus.RUNNING") == "RUNNING"
        assert status_name(SimpleNamespace(status="Error")) == "ERROR"

    def test_wait_returns_on_terminal_status(self) -> None:
        api = FakeApi(statuses=["QUEUED", "RUNNING", "COMPLETE"])
        outcome = wait_for_kernel(
            api,
            "d-oit/slice",
            pushed_version=7,
            clock=lambda: 0.0,
            sleep=lambda _s: None,
        )
        assert outcome.status == "COMPLETE"
        assert outcome.version == 7

    def test_stale_terminal_status_is_ignored_until_version_is_live(self) -> None:
        api = FakeApi(statuses=["COMPLETE", "QUEUED", "RUNNING", "COMPLETE"])
        api.version = 6  # the kernel still reports the previous version
        polls: list[str] = []
        outcome = wait_for_kernel(
            api,
            "d-oit/slice",
            pushed_version=7,
            clock=lambda: 0.0,
            sleep=lambda _s: None,
            on_poll=lambda name, _elapsed: polls.append(name),
        )
        assert outcome.status == "COMPLETE"
        # The first COMPLETE was stale; polling continued through the real run.
        assert polls[:2] == ["COMPLETE", "QUEUED"]

    def test_timeout_is_reported(self) -> None:
        api = FakeApi(statuses=["RUNNING"] * 10)
        ticks = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
        outcome = wait_for_kernel(
            api,
            "d-oit/slice",
            pushed_version=7,
            timeout_seconds=25,
            clock=lambda: next(ticks),
            sleep=lambda _s: None,
        )
        assert outcome.status == "TIMEOUT"

    def test_failure_message_is_surfaced(self) -> None:
        api = FakeApi(statuses=["ERROR"])
        outcome = wait_for_kernel(
            api,
            "d-oit/slice",
            pushed_version=7,
            clock=lambda: 0.0,
            sleep=lambda _s: None,
        )
        assert outcome.status == "ERROR"
        assert outcome.failure_message == "boom"


class TestPush:
    def test_push_returns_version_and_ref(self, tmp_path: Path) -> None:
        api = FakeApi()
        info = push_kernel(api, tmp_path, session_timeout_seconds=21_600)
        assert info["version"] == 7
        assert info["ref"] == "d-oit/tiny-cats-dit-slice"
        assert api.push_calls == [(str(tmp_path), "21600")]

    def test_server_error_is_loud(self, tmp_path: Path) -> None:
        api = FakeApi(push_error="invalid kernel")
        with pytest.raises(KaggleLaunchError, match="invalid kernel"):
            push_kernel(api, tmp_path)

    def test_invalid_sources_are_loud(self, tmp_path: Path) -> None:
        api = FakeApi(invalid_sources=["d-oit/missing-dataset"])
        with pytest.raises(KaggleLaunchError, match="invalid sources"):
            push_kernel(api, tmp_path)

    def test_missing_version_is_rejected(self, tmp_path: Path) -> None:
        api = FakeApi(version=None)
        with pytest.raises(KaggleLaunchError, match="no version number"):
            push_kernel(api, tmp_path)


class TestLogs:
    def test_logs_are_mirrored_to_the_workflow_path(self, tmp_path: Path) -> None:
        api = FakeApi(log_text="line one\nline two\n")
        log_file = tmp_path / "kaggle-training.log"
        result = fetch_kernel_logs(
            api,
            "d-oit/slice",
            download_dir=tmp_path / "out",
            log_file=log_file,
        )
        assert result == log_file
        assert "line two" in log_file.read_text()

    def test_missing_log_is_not_fatal(self, tmp_path: Path) -> None:
        api = FakeApi(log_text=None)
        result = fetch_kernel_logs(
            api,
            "d-oit/slice",
            download_dir=tmp_path / "out",
            log_file=tmp_path / "l.log",
        )
        assert result is None


class TestKernelIdResolution:
    """ADR-067: flag → env → authenticated account, never a hard failure."""

    def test_explicit_id_wins_over_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAGGLE_KERNEL_ID", "env/slice")
        assert resolve_kernel_id(FakeApi(), "d-oit/slice") == "d-oit/slice"

    def test_environment_id_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAGGLE_KERNEL_ID", "env/slice")
        assert resolve_kernel_id(FakeApi(), None) == "env/slice"

    def test_owner_comes_from_the_authenticated_username(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)
        api = SimpleNamespace(username="d-oit")
        assert resolve_kernel_id(api, None) == "d-oit/tiny-cats-dit-slice"

    def test_owner_comes_from_config_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)
        api = SimpleNamespace(config_values={"username": "config-owner"})
        assert resolve_kernel_id(api, None) == "config-owner/tiny-cats-dit-slice"

    def test_owner_comes_from_the_config_getter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)

        class Api:
            def get_config_value(self, key: str) -> str | None:
                return "getter-owner" if key == "username" else None

        assert resolve_kernel_id(Api(), None) == "getter-owner/tiny-cats-dit-slice"

    def test_unresolvable_owner_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)
        with pytest.raises(KaggleLaunchError, match="cannot resolve"):
            resolve_kernel_id(SimpleNamespace(), None)

    def test_malformed_environment_id_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAGGLE_KERNEL_ID", "not-a-slug")
        with pytest.raises(ValueError, match="owner/slug"):
            resolve_kernel_id(FakeApi(), None)


class TestCli:
    def test_dry_run_builds_the_workspace(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = main(
            [
                "--target",
                "15000",
                "--kernel-id",
                "d-oit/tiny-cats-dit-slice",
                "--workspace",
                str(tmp_path / "kernel"),
                "--dry-run",
            ]
        )
        assert code == EXIT_COMPLETE
        out = capsys.readouterr().out
        assert "KAGGLE_DRY_RUN" in out
        assert (tmp_path / "kernel" / KERNEL_METADATA_FILE).is_file()

    def test_bad_kernel_id_exits_config(self, tmp_path: Path) -> None:
        code = main(
            [
                "--target",
                "15000",
                "--kernel-id",
                "not-a-slug",
                "--workspace",
                str(tmp_path / "kernel"),
            ]
        )
        assert code == EXIT_CONFIG

    def test_dry_run_without_kernel_id_exits_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)
        code = main(
            [
                "--target",
                "15000",
                "--workspace",
                str(tmp_path / "kernel"),
                "--dry-run",
            ]
        )
        assert code == EXIT_CONFIG

    def test_dry_run_accepts_the_environment_kernel_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAGGLE_KERNEL_ID", "env-owner/slice")
        code = main(
            [
                "--target",
                "15000",
                "--workspace",
                str(tmp_path / "kernel"),
                "--dry-run",
            ]
        )
        assert code == EXIT_COMPLETE
        metadata = json.loads((tmp_path / "kernel" / KERNEL_METADATA_FILE).read_text())
        assert metadata["id"] == "env-owner/slice"

    def test_auth_failure_exits_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> object:
            raise RuntimeError("no credentials")

        monkeypatch.setattr(kaggle_provider, "_load_api", boom)
        code = main(
            [
                "--target",
                "15000",
                "--kernel-id",
                "d-oit/slice",
                "--workspace",
                str(tmp_path / "kernel"),
            ]
        )
        assert code == EXIT_CONFIG

    def test_terminal_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            kaggle_provider,
            "_load_api",
            lambda: FakeApi(statuses=["RUNNING", "COMPLETE"]),
        )
        code = main(
            [
                "--target",
                "15000",
                "--kernel-id",
                "d-oit/slice",
                "--workspace",
                str(tmp_path / "kernel"),
                # Interval 0 keeps the real poll loop instant (the CLI path
                # uses wait_for_kernel's bound defaults, so patching
                # `time.sleep` here would not take effect).
                "--poll-interval-seconds",
                "0",
                "--download-dir",
                str(tmp_path / "out"),
                "--log-file",
                str(tmp_path / "kaggle-training.log"),
            ]
        )
        assert code == EXIT_COMPLETE

    def test_error_status_maps_to_kernel_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            kaggle_provider, "_load_api", lambda: FakeApi(statuses=["ERROR"])
        )
        code = main(
            [
                "--target",
                "15000",
                "--kernel-id",
                "d-oit/slice",
                "--workspace",
                str(tmp_path / "kernel"),
                "--download-dir",
                str(tmp_path / "out"),
                "--log-file",
                str(tmp_path / "kaggle-training.log"),
            ]
        )
        assert code == EXIT_KERNEL_ERROR

    def test_timeout_maps_to_timeout_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = FakeApi(statuses=["RUNNING"] * 5)
        monkeypatch.setattr(kaggle_provider, "_load_api", lambda: api)
        code = main(
            [
                "--target",
                "15000",
                "--kernel-id",
                "d-oit/slice",
                "--workspace",
                str(tmp_path / "kernel"),
                # Zero deadline: the first non-terminal poll is already past
                # it, so the TIMEOUT mapping is exercised without sleeping.
                "--poll-timeout-seconds",
                "0",
                "--poll-interval-seconds",
                "0",
                "--download-dir",
                str(tmp_path / "out"),
                "--log-file",
                str(tmp_path / "kaggle-training.log"),
            ]
        )
        assert code == EXIT_TIMEOUT
