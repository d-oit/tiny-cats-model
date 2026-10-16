"""tests/test_provider_auto.py

The modal→kaggle fallback dispatcher (ADR-067): fall back only on launch
preconditions, never mask a training failure, and record which provider ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from provider_auto import EXIT_CONFIG, main

BASE_ARGS = ["--target", "15000", "--hub-repo", "d-oit/tiny-cats-model"]
KAGGLE_ARGS = [*BASE_ARGS, "--kernel-id", "d-oit/tiny-cats-dit-slice"]


class Runner:
    """Records commands and replays scripted ``(exit_code, blocked)`` results."""

    def __init__(self, results: list[tuple[int, bool]]) -> None:
        self.results = list(results)
        self.commands: list[list[str]] = []
        self.used_at_call: list[str | None] = []
        self.used_file: Path | None = None

    def __call__(self, command: list[str]) -> tuple[int, bool]:
        self.commands.append(command)
        if self.used_file is not None:
            self.used_at_call.append(
                self.used_file.read_text().strip() if self.used_file.exists() else None
            )
        return self.results.pop(0) if self.results else (0, False)


def test_falls_back_when_modal_is_blocked(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(1, True), (0, False)])

    code = main(
        [*KAGGLE_ARGS, "--provider-used-file", str(used)],
        runner=runner,
    )

    assert code == 0
    assert len(runner.commands) == 2
    assert runner.commands[0][:2] == ["modal", "run"]
    assert "src/kaggle_provider.py" in runner.commands[1]
    assert used.read_text().strip() == "kaggle"


def test_no_fallback_after_a_training_failure(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(1, False)])

    code = main([*KAGGLE_ARGS, "--provider-used-file", str(used)], runner=runner)

    assert code == 1
    assert len(runner.commands) == 1  # the kaggle fallback never ran
    assert used.read_text().strip() == "modal"


def test_blocked_with_no_provider_left_returns_the_exit_code(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(1, True)])

    code = main(
        [*BASE_ARGS, "--providers", "modal", "--provider-used-file", str(used)],
        runner=runner,
    )

    assert code == 1
    assert len(runner.commands) == 1


def test_unconfigured_provider_is_skipped_for_the_next_one(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(0, False)])

    code = main(
        [*BASE_ARGS, "--providers", "kaggle,modal", "--provider-used-file", str(used)],
        runner=runner,
    )

    # kaggle needs --kernel-id; unavailable providers are skipped, not fatal.
    assert code == 0
    assert len(runner.commands) == 1
    assert runner.commands[0][:2] == ["modal", "run"]
    assert used.read_text().strip() == "modal"


def test_all_providers_unavailable_is_a_config_error(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([])

    code = main(
        [*BASE_ARGS, "--providers", "kaggle", "--provider-used-file", str(used)],
        runner=runner,
    )

    assert code == EXIT_CONFIG
    assert runner.commands == []


def test_provider_is_recorded_before_it_runs(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(1, True), (0, False)])
    runner.used_file = used

    code = main([*KAGGLE_ARGS, "--provider-used-file", str(used)], runner=runner)

    assert code == 0
    # The artifact steps read this file even when the run fails, so it must be
    # written before the provider starts, not after it exits.
    assert runner.used_at_call == ["modal", "kaggle"]


def test_campaign_flags_reach_the_provider_commands(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(1, True), (0, False)])

    code = main(
        [
            *KAGGLE_ARGS,
            "--min-lr",
            "1e-5",
            "--schedule-steps",
            "600000",
            "--hub-resume",
            "--provider-used-file",
            str(used),
        ],
        runner=runner,
    )

    assert code == 0
    modal_command = " ".join(runner.commands[0])
    assert "--min-lr 1e-5" in modal_command
    assert "--schedule-steps 600000" in modal_command
    assert "--hub-resume" in modal_command
    kaggle_command = " ".join(runner.commands[1])
    assert "--kernel-id d-oit/tiny-cats-dit-slice" in kaggle_command
    assert "--min-lr 1e-5" in kaggle_command
