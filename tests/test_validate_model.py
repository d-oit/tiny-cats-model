"""tests/test_validate_model.py

Regression coverage for the production validation gate.

Pins the three defects found on 2026-09-20:

1. ``max_final_loss`` was formatted into a message and never enforced, so a
   generator at 1.28 loss against a 0.5 threshold reported "Training Metrics:
   passed".
2. The generative "Sample Quality" check required ``num_classes`` in the
   checkpoint config, which ``save_checkpoint`` never wrote — so it always
   reported "Not a generative model" and never validated a generator.
3. A missing optional dependency (onnxruntime) or artifact (.onnx) marked the
   whole report failed, making a green local gate unreachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import validate_model as vm
from dit import TinyDiT


def _tiny_dit() -> TinyDiT:
    """Smallest TinyDiT that still exercises every code path."""
    return TinyDiT(
        image_size=8,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=2,
        num_classes=3,
    )


@pytest.fixture
def tiny_dit_checkpoint(tmp_path: Path) -> Path:
    """Training-format generator checkpoint (model_state_dict + config)."""
    model = _tiny_dit()
    checkpoint = {
        "step": 5,
        "model_state_dict": model.state_dict(),
        "ema_shadow_params": model.state_dict(),
        "loss": 0.1,
        "config": {
            "image_size": 8,
            "patch_size": 4,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 2,
            "num_classes": 3,
        },
    }
    path = tmp_path / "tiny_dit.pt"
    torch.save(checkpoint, path)
    return path


class TestTrainingMetricThresholds:
    """Every configured threshold must actually be enforced."""

    def test_max_final_loss_above_threshold_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        torch.save({"step": 100, "loss": 1.2776, "config": {"depth": 12}}, path)

        result = vm.check_training_metrics(path, max_final_loss=0.5)

        assert result.passed is False
        assert result.name == "Final Loss"
        assert "1.2776" in result.message

    def test_max_final_loss_within_threshold_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        torch.save({"step": 100, "loss": 0.25, "config": {"depth": 12}}, path)

        result = vm.check_training_metrics(path, max_final_loss=0.5)

        assert result.passed is True
        assert "loss=0.2500 <= 0.5" in str(result.value)

    def test_min_val_accuracy_below_threshold_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        torch.save({"step": 3, "val_acc": 0.42}, path)

        result = vm.check_training_metrics(path, min_val_accuracy=0.85)

        assert result.passed is False
        assert result.critical is True


class TestGenerativeDetection:
    """Generator checkpoints must be recognised across layouts."""

    def test_detects_training_format_config(self) -> None:
        assert vm.is_generative_checkpoint({"config": {"depth": 12}}) is True

    def test_detects_ema_params_layout(self) -> None:
        assert vm.is_generative_checkpoint({"ema_params": {"x": 1}}) is True

    def test_detects_state_dict_layout(self) -> None:
        checkpoint = {
            "model_state_dict": {"patch_embed.proj.weight": torch.zeros(1, 1, 1, 1)}
        }
        assert vm.is_generative_checkpoint(checkpoint) is True

    def test_classifier_checkpoint_is_not_generative(self) -> None:
        checkpoint = {
            "model_state_dict": {"backbone.fc.weight": torch.zeros(2, 2)},
            "val_acc": 0.99,
        }
        assert vm.is_generative_checkpoint(checkpoint) is False

    def test_missing_num_classes_still_detected(self) -> None:
        # The exact case that made the gate silently skip every generator:
        # save_checkpoint wrote depth but never num_classes.
        assert vm.is_generative_checkpoint({"config": {"depth": 12, "image_size": 128}})

    def test_extract_state_dict_strips_module_prefix(self) -> None:
        state = vm.extract_state_dict({"model": {"module.fc.weight": torch.zeros(2)}})
        assert set(state) == {"fc.weight"}


class TestSkippedChecks:
    """Optional deps/artifacts must not fail the report."""

    def test_missing_onnxruntime_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vm, "HAS_ONNX", False)

        result = vm.validate_onnx_export(tmp_path / "model.pt")

        assert result.skipped is True
        assert result.passed is True

    def test_missing_onnx_artifact_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vm, "HAS_ONNX", True)

        result = vm.validate_onnx_export(tmp_path / "model.pt")

        assert result.skipped is True
        assert "No ONNX artifact" in result.message

    def test_skipped_result_does_not_fail_report(self) -> None:
        report = vm.ValidationReport(model_path="m.pt")
        report.add_result(vm.ValidationResult(name="ONNX", passed=True, skipped=True))

        assert report.passed is True
        assert report.warnings == []
        assert report.to_dict()["skipped_checks"] == 1
        assert report.to_dict()["failed_checks"] == 0

    def test_failing_result_still_fails_report(self) -> None:
        report = vm.ValidationReport(model_path="m.pt")
        report.add_result(vm.ValidationResult(name="Size", passed=False, message="big"))

        assert report.passed is False
        assert report.warnings == ["Size: big"]


class TestSampleQualityGate:
    """The generative check must run for generators and skip for classifiers."""

    def test_skips_non_generative_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "classifier.pt"
        torch.save(
            {
                "model_state_dict": {"backbone.fc.weight": torch.zeros(2, 2)},
                "val_acc": 0.99,
            },
            path,
        )

        result = vm.generate_sample_and_check_quality(path)

        assert result.skipped is True
        assert result.passed is True

    def test_generative_checkpoint_is_sampled(self, tiny_dit_checkpoint: Path) -> None:
        result = vm.generate_sample_and_check_quality(tiny_dit_checkpoint)

        assert result.skipped is False
        assert result.name == "Sample Quality"
        assert "NaN: False" in result.message


class TestValidateModelReport:
    """End-to-end report behaviour."""

    def test_generator_report_passes_with_optional_checks_skipped(
        self, tiny_dit_checkpoint: Path
    ) -> None:
        report = vm.validate_model(
            tiny_dit_checkpoint,
            thresholds=vm.ValidationThresholds(max_model_size_mb=100.0),
            check_all=True,
        )

        assert report.passed is True
        names = {result.name for result in report.results}
        assert "Sample Quality" in names
        assert sum(1 for r in report.results if r.skipped) >= 1

    def test_report_fails_when_final_loss_exceeds_threshold(
        self, tiny_dit_checkpoint: Path
    ) -> None:
        report = vm.validate_model(
            tiny_dit_checkpoint,
            thresholds=vm.ValidationThresholds(
                max_model_size_mb=100.0, max_final_loss=0.01
            ),
            check_all=True,
        )

        assert report.passed is False
        assert any("Final Loss" in warning for warning in report.warnings)
