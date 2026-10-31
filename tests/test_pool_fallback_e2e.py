"""tests/test_pool_fallback_e2e.py

End-to-end control-plane coverage for ADR-067: a Modal workspace blocked by a
launch precondition must actually launch the Kaggle adapter (not merely skip
it), and that adapter must package a real kernel for the resolved account.

The only fakes are the Kaggle client and the subprocess boundary: the commands
are the real ones produced by ``provider_auto``/``providers``, the kernel
workspace is really built, and the resolved kernel id is asserted from the
generated ``kernel-metadata.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import kaggle_provider
import provider_auto


class FakeKaggleApi:
    """Minimal Kaggle client: push, poll, list and log."""

    username = "slice-owner"

    def __init__(self) -> None:
        self.push_calls: list[tuple[str, str | None]] = []
        self.statuses = ["RUNNING", "COMPLETE"]

    def kernels_push(self, folder: str, timeout: str | None = None) -> object:
        self.push_calls.append((folder, timeout))
        return SimpleNamespace(
            ref="slice-owner/tiny-cats-dit-slice",
            kernel_id=1,
            version_number=3,
            url="https://kaggle.com/code/slice-owner/tiny-cats-dit-slice",
            error=None,
            invalid_dataset_sources=[],
        )

    def kernels_status(self, kernel: str) -> object:
        name = self.statuses.pop(0) if self.statuses else "COMPLETE"
        return SimpleNamespace(status=name, failure_message=None)

    def kernels_list(self, **_kwargs: Any) -> list[object]:
        return [
            SimpleNamespace(
                ref="slice-owner/tiny-cats-dit-slice",
                slug="tiny-cats-dit-slice",
                current_version_number=3,
            )
        ]

    def kernels_output(
        self,
        kernel: str,
        path: str,
        file_pattern: str | None = None,
        force: bool = False,
    ) -> tuple[list[str], str]:
        target = Path(path) / "tiny-cats-dit-slice.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("step 540000 loss 0.12\n")
        return ([], "")


def test_blocked_modal_launches_kaggle_end_to_end(tmp_path: Path, monkeypatch) -> None:
    api = FakeKaggleApi()
    used = tmp_path / "provider-used"

    # Keep every adapter write inside the test directory: the real command does
    # not pass --workspace/--log-file, so redirect those defaults.
    real_build = kaggle_provider.build_kernel_workspace
    monkeypatch.setattr(
        kaggle_provider,
        "build_kernel_workspace",
        lambda repo, workspace, **kwargs: real_build(
            repo, tmp_path / "kernel", **kwargs
        ),
    )
    real_fetch = kaggle_provider.fetch_kernel_logs
    monkeypatch.setattr(
        kaggle_provider,
        "fetch_kernel_logs",
        lambda api_arg, kernel_id, **kwargs: real_fetch(
            api_arg,
            kernel_id,
            download_dir=tmp_path / "kaggle-output",
            log_file=tmp_path / "kaggle-training.log",
        ),
    )
    monkeypatch.setattr(kaggle_provider, "_load_api", lambda: api)
    # Polling sleeps 60s between checks in production.
    monkeypatch.setattr(kaggle_provider.time, "sleep", lambda _seconds: None)
    monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)

    commands: list[list[str]] = []

    def runner(command: list[str]) -> tuple[int, bool]:
        commands.append(command)
        if command[:2] == ["modal", "run"]:
            print("Workspace x has exceeded its spend limit")
            return 1, True
        if command[:2] == ["python", "src/kaggle_provider.py"]:
            return kaggle_provider.main(command[2:]), False
        raise AssertionError(f"unexpected command: {command}")

    code = provider_auto.main(
        [
            "--target",
            "540000",
            "--hub-resume",
            "--schedule-steps",
            "600000",
            "--provider-used-file",
            str(used),
        ],
        runner=runner,
    )

    assert code == 0
    assert len(commands) == 2
    assert commands[0][:2] == ["modal", "run"]
    assert commands[1][:2] == ["python", "src/kaggle_provider.py"]

    # The owner is resolved from the authenticated account; no repo variable
    # was set, and the launch command did not carry --kernel-id.
    assert "--kernel-id" not in commands[1]
    metadata = (tmp_path / "kernel" / kaggle_provider.KERNEL_METADATA_FILE).read_text()
    assert '"id": "slice-owner/tiny-cats-dit-slice"' in metadata

    # The slice contract survives the fallback (global target + LR horizon).
    entry = (tmp_path / "kernel" / kaggle_provider.ENTRY_FILE).read_text()
    assert "'--steps', '540000'" in entry
    assert "'--schedule-steps', '600000'" in entry
    assert "'--hub-resume'" in entry

    assert len(api.push_calls) == 1
    assert used.read_text().strip() == "kaggle"
    assert (tmp_path / "kaggle-training.log").read_text().strip()


def test_workflow_surfaces_launch_failures() -> None:
    """The annotation must run *after* a failure (a bare ``if`` does not)."""
    workflow = (
        Path(__file__).parent.parent / ".github" / "workflows" / "train-pool.yml"
    ).read_text()
    assert "if: always() && steps.train.outcome == 'failure'" in workflow
