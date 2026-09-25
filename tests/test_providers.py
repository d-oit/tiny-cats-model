"""tests/test_providers.py

Issue #163 WP5 + WP6: GitHub Actions control plane — provider adapters that
fail clearly for unsupported providers, bounded/resumable slice planning,
launch-command construction, and machine-readable provider reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from providers import (
    DEFAULT_EXPERIMENT_ID,
    EXIT_REASONS,
    VERIFY_INVALID,
    VERIFY_OK,
    VERIFY_PARTIAL,
    ProviderReport,
    UnsupportedProviderError,
    build_launch_command,
    fetch_remote_completed,
    get_adapter,
    main,
    parse_slice_targets,
    resolve_exit_reason,
    verify_checkpoint,
)


class TestSlicePlanning:
    """WP6: bounded slices toward one global target (issue example layout)."""

    def test_issue_example_grid(self) -> None:
        assert parse_slice_targets(400_000, slice_size=60_000) == [
            60_000,
            120_000,
            180_000,
            240_000,
            300_000,
            360_000,
            400_000,
        ]

    def test_non_divisible_target_ends_exactly_at_global_target(self) -> None:
        assert parse_slice_targets(100, slice_size=30) == [30, 60, 90, 100]

    def test_completed_skips_finished_slices(self) -> None:
        plan = parse_slice_targets(400_000, slice_size=60_000, completed=130_000)
        assert plan[0] == 180_000
        assert plan[-1] == 400_000

    def test_already_complete_returns_single_noop_slice(self) -> None:
        # A no-op session still runs (and reports) rather than vanishing.
        assert parse_slice_targets(400_000, slice_size=60_000, completed=400_000) == [
            400_000
        ]

    def test_explicit_targets_kept_and_global_target_appended(self) -> None:
        plan = parse_slice_targets(400_000, targets=[60_000, 120_000, 300_000])
        assert plan == [60_000, 120_000, 300_000, 400_000]

    def test_explicit_targets_must_strictly_increase(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            parse_slice_targets(400_000, targets=[60_000, 60_000])

    def test_explicit_targets_cannot_exceed_global_target(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            parse_slice_targets(400_000, targets=[60_000, 500_000])

    @pytest.mark.parametrize(
        ("steps", "slice_size"),
        [(0, 100), (-5, 100), (100, 0), (100, -1)],
    )
    def test_invalid_inputs_raise(self, steps: int, slice_size: int) -> None:
        with pytest.raises(ValueError):
            parse_slice_targets(steps, slice_size=slice_size)


class TestProviderGate:
    """WP5: unsupported providers must fail clearly, never on CPU."""

    def test_modal_is_supported_with_gpu_info(self) -> None:
        adapter = get_adapter("modal")
        assert adapter.supported is True
        assert adapter.gpu_model == "T4"
        assert adapter.vram_gb == 16
        assert adapter.session_limit_minutes > 0

    @pytest.mark.parametrize(
        "provider", ["lightning", "colab", "kaggle", "hf_spaces", "local"]
    )
    def test_unsupported_providers_raise_loudly(self, provider: str) -> None:
        with pytest.raises(UnsupportedProviderError) as excinfo:
            get_adapter(provider)
        message = str(excinfo.value)
        assert provider in message
        # The failure must name the CPU-simulation refusal and the fallback.
        assert "CPU" in message

    def test_unknown_provider_lists_valid_options(self) -> None:
        with pytest.raises(UnsupportedProviderError, match="Valid providers"):
            get_adapter("some-tpu-farm")

    def test_gate_cli_fails_with_exit_code_2(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["gate", "--provider", "lightning", "--strict"]) == 2
        captured = capsys.readouterr()
        assert "not supported" in captured.err

    def test_gate_cli_all_reports_unsupported_without_failing(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["gate", "--provider", "all"]) == 0
        out = capsys.readouterr().out
        assert "supported: modal" in out
        assert "unsupported" in out
        assert "no CPU fallback" in out

    def test_gate_cli_modal_succeeds(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["gate", "--provider", "modal"]) == 0
        assert "supported: modal" in capsys.readouterr().out


class TestLaunchCommand:
    def test_modal_launch_builds_bounded_slice_session(self) -> None:
        command = build_launch_command(
            "modal",
            60_000,
            hub_resume=True,
            no_hub_push=True,
            experiment_id="exp-v9",
        )
        joined = " ".join(command)
        assert command[:2] == ["modal", "run"]
        assert "--steps 60000" in joined
        assert "--output /outputs/checkpoints/pool/dit_model.pt" in joined
        assert "--experiment-id exp-v9" in joined
        assert "--hub-resume" in command
        assert "--no-hub-push" in command

    def test_unsupported_provider_cannot_launch(self) -> None:
        with pytest.raises(UnsupportedProviderError):
            build_launch_command("kaggle", 60_000)

    def test_invalid_target_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_launch_command("modal", 0)

    def test_launch_cli_prints_shell_command(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert main(["launch", "--provider", "modal", "--target", "60000"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("modal run src/train_dit.py")
        assert "--steps 60000" in out

    def test_optional_training_knobs_only_when_requested(self) -> None:
        baseline = build_launch_command("modal", 60_000)
        assert "--warmup-steps" not in baseline
        assert "--gradient-accumulation-steps" not in baseline
        assert "--early-stopping-patience" not in baseline

        tuned = build_launch_command(
            "modal",
            60_000,
            warmup_steps="2000",
            gradient_accumulation_steps="2",
            early_stopping_patience="15",
        )
        joined = " ".join(tuned)
        assert "--warmup-steps 2000" in joined
        assert "--gradient-accumulation-steps 2" in joined
        assert "--early-stopping-patience 15" in joined

    def test_launch_cli_forwards_training_knobs(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        code = main(
            [
                "launch",
                "--provider",
                "modal",
                "--target",
                "60000",
                "--warmup-steps",
                "2000",
                "--early-stopping-patience",
                "15",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "--warmup-steps 2000" in out
        assert "--early-stopping-patience 15" in out

    def test_allow_experiment_mismatch_is_opt_in(self) -> None:
        # The migration override must never be emitted by a default launch.
        assert "--allow-experiment-mismatch" not in build_launch_command(
            "modal", 60_000
        )
        migrated = build_launch_command("modal", 60_000, allow_experiment_mismatch=True)
        assert "--allow-experiment-mismatch" in migrated

    def test_launch_cli_forwards_allow_experiment_mismatch(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        code = main(
            [
                "launch",
                "--provider",
                "modal",
                "--target",
                "60000",
                "--allow-experiment-mismatch",
            ]
        )
        assert code == 0
        assert "--allow-experiment-mismatch" in capsys.readouterr().out


class TestCheckpointVerification:
    """WP7: publication is gated on verified provider artifacts."""

    def _state(self, tmp_path: Path, completed: int, target: int) -> Path:
        state_file = tmp_path / "training_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "experiment_id": "dit-breed-conditioned-v4",
                    "completed_steps": completed,
                    "target_steps": target,
                }
            )
        )
        return state_file

    def _checkpoint(self, tmp_path: Path, *, valid: bool = True) -> Path:
        import torch

        path = tmp_path / "checkpoints" / "pool" / "dit_model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if valid:
            # A real torch checkpoint is itself a zip archive, which is exactly
            # what providers.verify_checkpoint probes with zipfile.is_zipfile.
            torch.save({"model_state_dict": {}}, path)
        else:
            path.write_bytes(b"not-a-zip")
        return path

    def test_target_reached_is_ok(self, tmp_path: Path) -> None:
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 60_000, 60_000)),
            checkpoint=str(self._checkpoint(tmp_path)),
            target=60_000,
        )
        assert result.reached_target
        assert result.checkpoint_valid
        assert result.completed_steps == 60_000
        assert result.experiment_id == "dit-breed-conditioned-v4"

    def test_short_of_target_is_not_reached(self, tmp_path: Path) -> None:
        # 45k checkpoint against a 60k target: valid artifact, target missed.
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 45_000, 60_000)),
            checkpoint=str(self._checkpoint(tmp_path)),
            target=60_000,
        )
        assert result.checkpoint_valid
        assert not result.reached_target

    def test_corrupt_checkpoint_is_invalid(self, tmp_path: Path) -> None:
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 60_000, 60_000)),
            checkpoint=str(self._checkpoint(tmp_path, valid=False)),
            target=60_000,
        )
        assert not result.checkpoint_valid
        assert not result.reached_target

    def test_missing_checkpoint_is_invalid(self, tmp_path: Path) -> None:
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 60_000, 60_000)),
            checkpoint=str(tmp_path / "nope.pt"),
            target=60_000,
        )
        assert not result.checkpoint_exists
        assert not result.checkpoint_valid

    def test_target_falls_back_to_state(self, tmp_path: Path) -> None:
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 60_000, 60_000)),
            checkpoint=str(self._checkpoint(tmp_path)),
        )
        assert result.target_steps == 60_000
        assert result.reached_target

    def test_verify_cli_exit_codes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        state = str(self._state(tmp_path, 60_000, 60_000))
        checkpoint = str(self._checkpoint(tmp_path))
        assert (
            main(
                [
                    "verify",
                    "--state-file",
                    state,
                    "--checkpoint",
                    checkpoint,
                    "--target",
                    "60000",
                ]
            )
            == VERIFY_OK
        )
        assert (
            main(
                [
                    "verify",
                    "--state-file",
                    state,
                    "--checkpoint",
                    str(tmp_path / "missing.pt"),
                    "--target",
                    "60000",
                ]
            )
            == VERIFY_INVALID
        )
        short = self._state(tmp_path, 10, 20)
        assert (
            main(
                [
                    "verify",
                    "--state-file",
                    str(short),
                    "--checkpoint",
                    checkpoint,
                    "--target",
                    "20",
                ]
            )
            == VERIFY_PARTIAL
        )
        assert "CHECKPOINT_VERIFY_JSON=" in capsys.readouterr().out

    def test_verify_cli_appends_github_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        github_output = tmp_path / "gh_output"
        code = main(
            [
                "verify",
                "--state-file",
                str(self._state(tmp_path, 45_000, 60_000)),
                "--checkpoint",
                str(self._checkpoint(tmp_path)),
                "--target",
                "60000",
                "--github-output",
                str(github_output),
            ]
        )
        assert code == VERIFY_PARTIAL
        text = github_output.read_text()
        assert "reached=false" in text
        assert "checkpoint_valid=true" in text
        assert "completed_steps=45000" in text
        assert capsys.readouterr().out

    def test_omitted_checkpoint_is_not_verified(self, tmp_path: Path) -> None:
        # A readable state at the target must not pass the artifact gate when
        # no checkpoint was supplied: the verifier validates artifacts.
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 60_000, 60_000)),
            target=60_000,
        )
        assert not result.checkpoint_valid
        assert not result.reached_target
        assert "no checkpoint supplied" in result.reason

    def test_arbitrary_zip_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        import zipfile

        path = tmp_path / "not-torch.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("readme.txt", "not a torch payload")
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 60_000, 60_000)),
            checkpoint=str(path),
            target=60_000,
        )
        assert not result.checkpoint_valid
        assert not result.reached_target

    def test_malformed_target_in_state_is_invalid_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "training_state.json"
        state_file.write_text(json.dumps({"completed_steps": 5, "target_steps": "bad"}))
        result = verify_checkpoint(
            state_file=str(state_file),
            checkpoint=str(self._checkpoint(tmp_path)),
        )
        assert not result.reached_target
        assert not result.state_valid
        assert "malformed target_steps" in result.reason

        # The CLI must report the documented invalid result, not raise.
        assert (
            main(
                [
                    "verify",
                    "--state-file",
                    str(state_file),
                    "--checkpoint",
                    str(self._checkpoint(tmp_path)),
                ]
            )
            == VERIFY_INVALID
        )

    def test_malformed_completed_steps_marks_state_invalid(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "training_state.json"
        state_file.write_text(json.dumps({"completed_steps": "nope"}))
        result = verify_checkpoint(
            state_file=str(state_file),
            checkpoint=str(self._checkpoint(tmp_path)),
            target=60_000,
        )
        assert not result.state_valid
        assert not result.reached_target
        assert "malformed completed_steps" in result.reason

    def test_non_positive_target_is_rejected(self, tmp_path: Path) -> None:
        checkpoint = str(self._checkpoint(tmp_path))
        result = verify_checkpoint(
            state_file=str(self._state(tmp_path, 0, 60_000)),
            checkpoint=checkpoint,
            target=-1,
        )
        assert not result.reached_target
        assert "target must be positive" in result.reason

        # The CLI must not report OK for a nonsensical global target.
        assert (
            main(
                [
                    "verify",
                    "--state-file",
                    str(self._state(tmp_path, 0, 60_000)),
                    "--checkpoint",
                    checkpoint,
                    "--target",
                    "-1",
                ]
            )
            == VERIFY_INVALID
        )


class TestProviderReport:
    """WP5: every session reports the 9 required fields, machine-readably."""

    def _report(self, **overrides: Any) -> ProviderReport:
        values: dict[str, Any] = {
            "provider": "modal",
            "gpu_model": "T4",
            "vram_gb": 16,
            "job_id": "1234/train-slice-60000",
            "started_at": "2026-09-23T10:00:00Z",
            "ended_at": "2026-09-23T15:00:00Z",
            "exit_reason": "completed",
            "completed_steps": 60_000,
            "target_steps": 60_000,
            "checkpoint_uri": "hf://repo/checkpoints/pool/exp/latest",
            "experiment_id": "exp-v9",
        }
        values.update(overrides)
        return ProviderReport(**values)

    def test_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        report = self._report()
        path = report.write(tmp_path / "nested" / "provider_report.json")
        loaded = ProviderReport.read(path)
        assert loaded == report
        document = json.loads(path.read_text())
        for field_name in (
            "provider",
            "gpu_model",
            "vram_gb",
            "job_id",
            "started_at",
            "ended_at",
            "completed_steps",
            "checkpoint_uri",
            "exit_reason",
        ):
            assert field_name in document, field_name

    def test_invalid_exit_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="exit_reason"):
            self._report(exit_reason="kinda-worked")

    @pytest.mark.parametrize(
        "reason", ["completed", "partial", "interrupted", "failed", "unsupported"]
    )
    def test_exit_reasons_match_contract(self, reason: str) -> None:
        assert reason in EXIT_REASONS

    @pytest.mark.parametrize(
        ("outcome", "completed", "target", "expected"),
        [
            ("success", 60_000, 60_000, "completed"),
            ("success", 400_000, 400_000, "completed"),  # no-op resume
            ("success", 45_000, 60_000, "partial"),
            ("success", None, 60_000, "partial"),
            ("cancelled", 45_000, 60_000, "interrupted"),
            ("failure", None, None, "failed"),
            ("skipped", None, None, "failed"),
        ],
    )
    def test_resolve_exit_reason(
        self,
        outcome: str,
        completed: int | None,
        target: int | None,
        expected: str,
    ) -> None:
        assert resolve_exit_reason(outcome, completed, target) == expected

    def test_report_cli_writes_file_and_status_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        state = {
            "experiment_id": "exp-v9",
            "target_steps": 60_000,
            "completed_steps": 60_000,
        }
        state_file = tmp_path / "training_state.json"
        state_file.write_text(json.dumps(state))
        out = tmp_path / "artifacts/training/provider_report.json"

        code = main(
            [
                "report",
                "--provider",
                "modal",
                "--job-id",
                "run-1-slice-60000",
                "--started-at",
                "2026-09-23T10:00:00Z",
                "--target",
                "60000",
                "--state-file",
                str(state_file),
                "--checkpoint-uri",
                "hf://repo/pool/exp/latest",
                "--outcome",
                "success",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        document = json.loads(out.read_text())
        assert document["exit_reason"] == "completed"
        assert document["completed_steps"] == 60_000
        assert document["experiment_id"] == "exp-v9"
        assert "PROVIDER_REPORT_JSON=" in capsys.readouterr().out

    def test_report_cli_failed_session_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        out = tmp_path / "provider_report.json"
        code = main(
            [
                "report",
                "--provider",
                "modal",
                "--gpu-model",
                "L4",
                "--job-id",
                "run-2",
                "--started-at",
                "2026-09-23T10:00:00Z",
                "--target",
                "120000",
                "--outcome",
                "failure",
                "--out",
                str(out),
            ]
        )
        assert code == 1
        document = json.loads(out.read_text())
        assert document["exit_reason"] == "failed"
        assert document["completed_steps"] is None
        assert document["vram_gb"] == 24  # L4


class TestPlanCLI:
    def test_plan_prints_gh_matrix_json(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["plan", "--steps", "100", "--slice-size", "30"]) == 0
        assert json.loads(capsys.readouterr().out) == {"target": [30, 60, 90, 100]}

    def test_plan_filters_by_completed(self, capsys: pytest.CaptureFixture) -> None:
        code = main(
            [
                "plan",
                "--steps",
                "400000",
                "--slice-size",
                "60000",
                "--completed",
                "61000",
            ]
        )
        assert code == 0
        assert json.loads(capsys.readouterr().out)["target"][0] == 120_000

    def test_plan_unknown_remote_degrades_with_warning(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No network in CI: --remote-completed must warn, not crash.
        def boom(*_args, **_kwargs):
            raise RuntimeError("no network")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", boom, raising=False)
        assert (
            main(["plan", "--steps", "100", "--slice-size", "30", "--remote-completed"])
            == 0
        )
        captured = capsys.readouterr()
        assert "warning" in captured.err
        assert json.loads(captured.out) == {"target": [30, 60, 90, 100]}


class TestRemotePointer:
    def test_fetch_remote_completed_reads_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pointer = {"completed_steps": 120_000, "experiment_id": DEFAULT_EXPERIMENT_ID}
        target = tmp_path / "manifest.json"
        target.write_text(json.dumps(pointer))

        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "hf_hub_download", lambda **_kwargs: str(target)
        )
        assert fetch_remote_completed() == 120_000

    def test_fetch_remote_completed_missing_pointer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        def missing(**_kwargs):
            raise FileNotFoundError("EntryNotFound")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
        with pytest.raises(FileNotFoundError):
            fetch_remote_completed()
