"""tests/test_provider_auto.py

The free-only default is Kaggle → Lightning. Explicit provider lists remain
available for deliberate manual runs; fallback is only for a confirmed
prelaunch block, never to mask a training or ambiguous submission failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from provider_auto import EXIT_CONFIG, EXIT_LAUNCH_BLOCKED, main, run_provider

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
        [
            *KAGGLE_ARGS,
            "--providers",
            "modal,kaggle",
            "--provider-used-file",
            str(used),
        ],
        runner=runner,
    )

    assert code == 0
    assert len(runner.commands) == 2
    assert runner.commands[0][:2] == ["modal", "run"]
    assert "src/kaggle_provider.py" in runner.commands[1]
    assert used.read_text().strip() == "kaggle"


def test_no_fallback_after_a_kaggle_training_failure(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(2, False)])

    code = main([*KAGGLE_ARGS, "--provider-used-file", str(used)], runner=runner)

    assert code == 2
    assert len(runner.commands) == 1
    assert runner.commands[0][:2] == ["python", "src/kaggle_provider.py"]
    assert used.read_text().strip() == "kaggle"


def test_blocked_with_no_provider_left_returns_the_exit_code(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(EXIT_LAUNCH_BLOCKED, True)])

    code = main(
        [
            *BASE_ARGS,
            "--providers",
            "kaggle",
            "--provider-used-file",
            str(used),
        ],
        runner=runner,
    )

    assert code == EXIT_LAUNCH_BLOCKED
    assert len(runner.commands) == 1


def test_kaggle_is_launchable_without_a_configured_kernel_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    used = tmp_path / "used"
    runner = Runner([(0, False)])
    monkeypatch.delenv("KAGGLE_KERNEL_ID", raising=False)

    code = main(
        [
            *BASE_ARGS,
            "--providers",
            "kaggle,lightning",
            "--provider-used-file",
            str(used),
        ],
        runner=runner,
    )

    assert code == 0
    assert len(runner.commands) == 1
    assert runner.commands[0][:2] == ["python", "src/kaggle_provider.py"]
    assert "--kernel-id" not in runner.commands[0]
    assert used.read_text().strip() == "kaggle"


def test_all_providers_unavailable_is_a_config_error(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([])

    code = main(
        [*BASE_ARGS, "--providers", "colab", "--provider-used-file", str(used)],
        runner=runner,
    )

    assert code == EXIT_CONFIG
    assert runner.commands == []


def test_provider_is_recorded_before_it_runs(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(1, True), (0, False)])
    runner.used_file = used

    code = main(
        [
            *KAGGLE_ARGS,
            "--providers",
            "modal,kaggle",
            "--provider-used-file",
            str(used),
        ],
        runner=runner,
    )

    assert code == 0
    assert runner.used_at_call == ["modal", "kaggle"]


def test_default_free_chain_falls_back_from_kaggle_refusal_to_lightning(
    tmp_path: Path,
) -> None:
    used = tmp_path / "used"
    runner = Runner([(EXIT_LAUNCH_BLOCKED, True), (0, False)])

    code = main([*BASE_ARGS, "--provider-used-file", str(used)], runner=runner)

    assert code == 0
    assert [command[1] for command in runner.commands] == [
        "src/kaggle_provider.py",
        "src/lightning_provider.py",
    ]
    assert used.read_text().strip() == "lightning"


def test_both_free_providers_block_without_modal_fallback(tmp_path: Path) -> None:
    runner = Runner([(EXIT_LAUNCH_BLOCKED, True), (EXIT_LAUNCH_BLOCKED, True)])

    code = main([*BASE_ARGS], runner=runner)

    assert code == EXIT_LAUNCH_BLOCKED
    assert [command[0] for command in runner.commands] == ["python", "python"]
    assert [command[1] for command in runner.commands] == [
        "src/kaggle_provider.py",
        "src/lightning_provider.py",
    ]


def test_ambiguous_kaggle_failure_does_not_fall_back(tmp_path: Path) -> None:
    runner = Runner([(5, False)])

    code = main([*BASE_ARGS], runner=runner)

    assert code == 5
    assert len(runner.commands) == 1
    assert runner.commands[0][1] == "src/kaggle_provider.py"


def test_modal_spend_text_is_not_a_kaggle_or_lightning_block(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import provider_auto

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = iter(["Workspace has exceeded its spend limit\n"])

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(
        provider_auto.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    assert run_provider(["python", "src/kaggle_provider.py"]) == (1, False)
    assert run_provider(["python", "src/lightning_provider.py"]) == (1, False)
    assert run_provider(["modal", "run", "src/train_dit.py"]) == (1, True)
    capsys.readouterr()


def test_exit_six_is_a_definite_prelaunch_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import provider_auto

    class FakeProcess:
        stdout = iter([])

        def wait(self) -> int:
            return EXIT_LAUNCH_BLOCKED

    monkeypatch.setattr(
        provider_auto.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    assert run_provider(["python", "src/lightning_provider.py"]) == (
        EXIT_LAUNCH_BLOCKED,
        True,
    )


def test_campaign_flags_reach_the_provider_commands(tmp_path: Path) -> None:
    used = tmp_path / "used"
    runner = Runner([(1, True), (0, False)])

    code = main(
        [
            *KAGGLE_ARGS,
            "--providers",
            "modal,kaggle",
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
