"""tests/test_pool_fallback_e2e.py

End-to-end control-plane coverage for the free Kaggle provider and the
container import contract. The Kaggle client is fake; the real adapter builds
the self-contained kernel workspace and submits/polls it through the fake API.
"""

from __future__ import annotations

import os
import re
import subprocess
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


def test_kaggle_provider_completes_end_to_end(tmp_path: Path, monkeypatch) -> None:
    api = FakeKaggleApi()
    used = tmp_path / "provider-used"
    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-kaggle-token")
    monkeypatch.setenv("KAGGLE_HF_TOKEN_DATASET", "slice-owner/tiny-cats-hf-token")

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
    real_wait = kaggle_provider.wait_for_kernel
    monkeypatch.setattr(
        kaggle_provider,
        "wait_for_kernel",
        lambda api_arg, kernel_id, **kwargs: real_wait(
            api_arg, kernel_id, sleep=lambda _seconds: None, **kwargs
        ),
    )
    monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)

    commands: list[list[str]] = []

    def runner(command: list[str]) -> tuple[int, bool]:
        commands.append(command)
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
    assert len(commands) == 1
    assert commands[0][:2] == ["python", "src/kaggle_provider.py"]
    assert "--kernel-id" not in commands[0]
    metadata = (tmp_path / "kernel" / kaggle_provider.KERNEL_METADATA_FILE).read_text()
    assert '"id": "slice-owner/tiny-cats-dit-slice"' in metadata

    # The generated kernel pins the global target and campaign LR horizon.
    entry = (tmp_path / "kernel" / kaggle_provider.ENTRY_FILE).read_text()
    assert "'--steps', '540000'" in entry
    assert "'--schedule-steps', '600000'" in entry
    assert "'--hub-resume'" in entry

    assert len(api.push_calls) == 1
    assert used.read_text().strip() == "kaggle"
    assert (tmp_path / "kaggle-training.log").read_text().strip()


def test_train_dit_imports_without_the_modal_sdk() -> None:
    """Kaggle has no Modal SDK, yet the kernel imports train_dit.

    ``sys.modules['modal'] = None`` makes ``import modal`` raise ImportError,
    exactly like the Kaggle image, without uninstalling it from the runner.
    """
    root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['modal'] = None; import train_dit; print('ok')",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_modal_image_mounts_every_local_module_train_dit_imports() -> None:
    """A missing mount fails inside the container, after the slice launched.

    The 2026-10-31 pool run burned two 6h Modal slices on
    ``ModuleNotFoundError: No module named 'safe_checkpoint'`` because the
    image mounted train_dit's siblings by hand and this one was absent.
    """
    root = Path(__file__).parent.parent
    source = (root / "src" / "train_dit.py").read_text()
    imported = set(re.findall(r"^from ([a-z_]+) import", source, re.MULTILINE))
    local = {name for name in imported if (root / "src" / f"{name}.py").exists()}
    mounted = set(re.findall(r'add_local_file\("src/([a-z_]+)\.py"', source))
    # ADR-068: a whole-package mount covers every local module by construction;
    # a per-file whitelist drifted twice and broke dispatched slices.
    whole_package = re.search(r'add_local_dir\(\s*"src"\s*,\s*"/app"', source)
    missing = [] if whole_package else sorted(local - mounted)
    assert not missing, f"Modal image is missing local modules: {missing}"
