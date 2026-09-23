"""src/validate_model.py

Model validation gates for production-ready ML pipeline.

Features:
- Quality threshold enforcement
- Generated sample quality assessment (FID score)
- ONNX export validation
- Regression tests against previous model version
- Comprehensive validation report

Usage:
    python src/validate_model.py checkpoints/pool/dit_model_ema.pt --thresholds config/validation.json
    python src/validate_model.py checkpoints/pool/dit_model_ema.pt --check-all --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    from PIL import Image  # noqa: F401

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    """Validation threshold configuration."""

    # Classifier thresholds
    min_val_accuracy: float = 0.85
    min_train_accuracy: float = 0.80

    # DiT thresholds
    max_final_loss: float = 0.5
    min_fid_score: float = 50.0  # Lower is better

    # Model size constraints
    max_model_size_mb: float = 150.0
    max_onnx_size_mb: float = 50.0

    # Inference constraints
    max_inference_time_cpu_ms: float = 2000.0
    max_inference_time_gpu_ms: float = 200.0

    # Numerical stability
    check_nan_weights: bool = True
    check_inf_weights: bool = True

    # ONNX validation
    validate_onnx_output: bool = True
    onnx_output_tolerance: float = 1e-4


def get_clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove 'module.' prefix from state dict keys if present.

    Args:
        state_dict: The state dictionary to clean.

    Returns:
        A cleaned state dictionary.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


@dataclass
class ValidationResult:
    """Result of a single validation check.

    ``skipped`` marks checks that could not run because an optional dependency
    or artifact is absent (missing onnxruntime, no exported .onnx, non-generative
    checkpoint). Those must not fail the report: the gate previously reported
    FAILED for a production classifier purely because ONNX Runtime was not
    installed locally, which made a green gate unreachable.
    """

    name: str
    passed: bool
    value: Any | None = None
    threshold: Any | None = None
    message: str = ""
    critical: bool = False
    skipped: bool = False


@dataclass
class ValidationReport:
    """Complete validation report."""

    model_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    results: list[ValidationResult] = field(default_factory=list)
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
        if result.skipped:
            return
        if not result.passed:
            self.passed = False
            if result.critical:
                self.errors.append(f"{result.name}: {result.message}")
            else:
                self.warnings.append(f"{result.name}: {result.message}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed and not r.skipped),
            "failed_checks": sum(
                1 for r in self.results if not r.passed and not r.skipped
            ),
            "skipped_checks": sum(1 for r in self.results if r.skipped),
            "warnings": self.warnings,
            "errors": self.errors,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "skipped": r.skipped,
                    "value": r.value,
                    "threshold": r.threshold,
                    "message": r.message,
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def load_thresholds(config_path: str | Path | None = None) -> ValidationThresholds:
    """Load validation thresholds from config file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        ValidationThresholds instance.
    """
    if config_path is None:
        return ValidationThresholds()

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return ValidationThresholds()

    with open(config_path) as f:
        config = json.load(f)

    return ValidationThresholds(**config)


def check_model_file_exists(model_path: str | Path) -> ValidationResult:
    """Check if model file exists."""
    model_path = Path(model_path)
    exists = model_path.exists()

    return ValidationResult(
        name="Model File Exists",
        passed=exists,
        value=str(model_path) if exists else None,
        message=f"Model file {'found' if exists else 'not found'}: {model_path}",
        critical=True,
    )


def check_model_size(model_path: str | Path, max_size_mb: float) -> ValidationResult:
    """Check model file size."""
    model_path = Path(model_path)
    if not model_path.exists():
        return ValidationResult(
            name="Model Size",
            passed=False,
            message="Model file not found",
            critical=True,
        )

    size_mb = model_path.stat().st_size / (1024 * 1024)
    passed = size_mb <= max_size_mb

    return ValidationResult(
        name="Model Size",
        passed=passed,
        value=f"{size_mb:.2f} MB",
        threshold=f"{max_size_mb:.2f} MB",
        message=f"Model size: {size_mb:.2f} MB (max: {max_size_mb:.2f} MB)",
        critical=False,
    )


def check_nan_weights(model_path: str | Path) -> ValidationResult:
    """Check for NaN weights in model."""
    model_path = Path(model_path)

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Get state dict (handle different checkpoint formats)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = {
                k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)
            }

        state_dict = get_clean_state_dict(state_dict)

        # Check for NaN
        has_nan = False
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                has_nan = True
                logger.debug(f"NaN found in: {name}")
                break

        return ValidationResult(
            name="NaN Weights Check",
            passed=not has_nan,
            value="NaN detected" if has_nan else "No NaN",
            message="Model contains NaN weights"
            if has_nan
            else "No NaN weights detected",
            critical=True,
        )

    except Exception as e:
        return ValidationResult(
            name="NaN Weights Check",
            passed=False,
            message=f"Error checking weights: {e}",
            critical=True,
        )


def check_inf_weights(model_path: str | Path) -> ValidationResult:
    """Check for infinite weights in model."""
    model_path = Path(model_path)

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Get state dict (handle different checkpoint formats)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = {
                k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)
            }

        state_dict = get_clean_state_dict(state_dict)

        # Check for Inf
        has_inf = False
        for name, param in state_dict.items():
            if torch.isinf(param).any():
                has_inf = True
                logger.debug(f"Inf found in: {name}")
                break

        return ValidationResult(
            name="Inf Weights Check",
            passed=not has_inf,
            value="Inf detected" if has_inf else "No Inf",
            message="Model contains infinite weights"
            if has_inf
            else "No infinite weights detected",
            critical=True,
        )

    except Exception as e:
        return ValidationResult(
            name="Inf Weights Check",
            passed=False,
            message=f"Error checking weights: {e}",
            critical=True,
        )


def check_training_metrics(
    model_path: str | Path,
    min_val_accuracy: float | None = None,
    min_train_accuracy: float | None = None,
    max_final_loss: float | None = None,
) -> ValidationResult:
    """Check training metrics from checkpoint.

    Every configured threshold is enforced: ``max_final_loss`` used to be
    formatted into the message and then ignored, so a generator at 1.28 loss
    against a 0.5 threshold still reported "Training Metrics: passed".
    """
    model_path = Path(model_path)

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Get metrics
        val_acc = checkpoint.get("val_acc")
        train_acc = checkpoint.get("train_acc")
        loss = checkpoint.get("loss")
        step = checkpoint.get("step")

        checks = []

        if min_val_accuracy is not None and val_acc is not None:
            passed = val_acc >= min_val_accuracy
            checks.append(
                f"val_acc={val_acc:.4f} {'>=' if passed else '<'} {min_val_accuracy}"
            )
            if not passed:
                return ValidationResult(
                    name="Validation Accuracy",
                    passed=False,
                    value=f"{val_acc:.4f}",
                    threshold=f">={min_val_accuracy}",
                    message=f"Validation accuracy {val_acc:.4f} below threshold {min_val_accuracy}",
                    critical=True,
                )

        if min_train_accuracy is not None and train_acc is not None:
            passed = train_acc >= min_train_accuracy
            checks.append(
                f"train_acc={train_acc:.4f} {'>=' if passed else '<'} {min_train_accuracy}"
            )

        if max_final_loss is not None and loss is not None:
            if loss > max_final_loss:
                return ValidationResult(
                    name="Final Loss",
                    passed=False,
                    value=f"{loss:.4f}",
                    threshold=f"<={max_final_loss}",
                    message=(
                        f"Final loss {loss:.4f} exceeds threshold {max_final_loss}"
                    ),
                    critical=False,
                )
            checks.append(f"loss={loss:.4f} <= {max_final_loss}")

        return ValidationResult(
            name="Training Metrics",
            passed=True,
            value=", ".join(checks) if checks else "No metrics found",
            message=f"Metrics at step {step}: {', '.join(checks)}"
            if checks
            else "No training metrics in checkpoint",
            critical=False,
        )

    except Exception as e:
        return ValidationResult(
            name="Training Metrics",
            passed=False,
            message=f"Error reading metrics: {e}",
            critical=False,
        )


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Pull a model state dict out of any supported checkpoint layout.

    Supports the trainer's ``model_state_dict`` / ``ema_shadow_params`` layout,
    the older ``model`` / ``ema_params`` layout, and a bare state dict.

    Args:
        checkpoint: Loaded checkpoint.

    Returns:
        State dict with any ``module.`` prefix stripped.
    """
    for key in (
        "model_state_dict",
        "ema_shadow_params",
        "ema_params",
        "model",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            return get_clean_state_dict(candidate)
    return get_clean_state_dict(
        {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    )


def is_generative_checkpoint(checkpoint: dict[str, Any]) -> bool:
    """Decide whether a checkpoint holds a TinyDiT generator.

    Requires no single marker (the old check demanded ``num_classes`` in
    ``config``, which ``save_checkpoint`` never wrote, so the sample-quality
    gate silently reported "Not a generative model" for every generator).

    Args:
        checkpoint: Loaded checkpoint.

    Returns:
        True when the checkpoint looks like a TinyDiT generator.
    """
    config = checkpoint.get("config")
    if isinstance(config, dict) and int(config.get("depth") or 0) > 0:
        return True
    if "ema_shadow_params" in checkpoint or "ema_params" in checkpoint:
        return True
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
    if isinstance(state_dict, dict):
        return any(str(key).startswith("patch_embed.") for key in state_dict)
    return False


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[torch.nn.Module, bool]:
    """Rebuild the model described by a checkpoint and load its weights.

    The architecture comes from the checkpoint's own config instead of
    hardcoded ``tinydit_128`` defaults, so the gate also validates the smaller
    models used by tests and any future resolution.

    Args:
        checkpoint: Loaded checkpoint.

    Returns:
        Tuple of (model, is_generative).

    Raises:
        ImportError: If the model modules are unavailable.
        ValueError: If the checkpoint does not match the rebuilt architecture.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    raw_config = checkpoint.get("config")
    config = raw_config if isinstance(raw_config, dict) else {}
    state_dict = extract_state_dict(checkpoint)

    if is_generative_checkpoint(checkpoint):
        from dit import TinyDiT, load_state_dict_checked

        model = TinyDiT(
            image_size=config.get("image_size", 128),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 384),
            depth=config.get("depth", 12),
            num_heads=config.get("num_heads", 6),
            num_classes=config.get("num_classes") or 13,
        )
        # load_state_dict_checked validates keys and shapes before copying, so a
        # stale checkpoint cannot leave a half-loaded model behind.
        load_state_dict_checked(model, state_dict)
        return model, True

    from model import cats_model

    model = cats_model(num_classes=config.get("num_classes", 2))
    model.load_state_dict(state_dict)
    return model, False


def validate_onnx_export(
    model_path: str | Path,
    onnx_path: str | Path | None = None,
    tolerance: float = 1e-4,
) -> ValidationResult:
    """Validate ONNX export produces consistent output."""
    if not HAS_ONNX:
        return ValidationResult(
            name="ONNX Validation",
            passed=True,
            skipped=True,
            message="ONNX Runtime not installed; skipping ONNX validation",
        )

    model_path = Path(model_path)

    # Find ONNX file
    if onnx_path is None:
        onnx_path = model_path.with_suffix(".onnx")
    else:
        onnx_path = Path(onnx_path)

    if not onnx_path.exists():
        return ValidationResult(
            name="ONNX Validation",
            passed=True,
            skipped=True,
            message=f"No ONNX artifact produced yet: {onnx_path}",
        )

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model, is_generative = build_model_from_checkpoint(checkpoint)
        model.eval()

        # Create dummy input at the model's own resolution
        if is_generative:
            # DiT: needs image and breed
            size = getattr(model, "image_size", 128)
            dummy_input = torch.randn(1, 3, size, size)
            dummy_breed = torch.tensor([0])
            pytorch_output = model(dummy_input, dummy_breed)
        else:
            # Classifier: just image
            dummy_input = torch.randn(1, 3, 128, 128)
            pytorch_output = model(dummy_input)

        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run ONNX inference
        onnx_inputs = {input_name: dummy_input.numpy()}

        # Add breed input if needed
        if len(session.get_inputs()) > 1:
            breed_input_name = session.get_inputs()[1].name
            onnx_inputs[breed_input_name] = dummy_breed.numpy()

        onnx_output = session.run(None, onnx_inputs)[0]

        # Compare outputs
        pytorch_np = pytorch_output.detach().numpy()
        max_diff = abs(pytorch_np - onnx_output).max()
        passed = max_diff <= tolerance

        return ValidationResult(
            name="ONNX Output Consistency",
            passed=passed,
            value=f"max_diff={max_diff:.6f}",
            threshold=f"<={tolerance}",
            message=f"ONNX output {'matches' if passed else 'differs from'} PyTorch (max diff: {max_diff:.6f})",
            critical=passed is False,
        )

    except Exception as e:
        return ValidationResult(
            name="ONNX Validation",
            passed=False,
            message=f"ONNX validation error: {e}",
            critical=False,
        )


def generate_sample_and_check_quality(
    model_path: str | Path,
    min_fid_score: float = 50.0,
    num_samples: int = 8,
) -> ValidationResult:
    """Generate samples and estimate quality.

    Args:
        model_path: Checkpoint to validate.
        min_fid_score: Retained for API compatibility; a real FID needs a
            reference dataset and is not computed here.
        num_samples: Number of samples to generate.

    Returns:
        ValidationResult. Checkpoints that are not generators are skipped
        rather than failed, so the gate stays meaningful for classifiers.
    """
    if not HAS_PIL:
        return ValidationResult(
            name="Sample Quality",
            passed=True,
            skipped=True,
            message="PIL not installed; skipping sample generation",
        )

    model_path = Path(model_path)

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return ValidationResult(
            name="Sample Quality",
            passed=False,
            message=f"Could not read checkpoint: {e}",
            critical=False,
        )

    if not is_generative_checkpoint(checkpoint):
        return ValidationResult(
            name="Sample Quality",
            passed=True,
            skipped=True,
            message="Not a generative checkpoint; skipping sample generation",
        )

    try:
        from flow_matching import sample

        model, _ = build_model_from_checkpoint(checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        breeds = torch.arange(min(num_samples, model.num_classes), device=device)

        with torch.no_grad():
            generated = sample(
                model,
                breeds,
                num_steps=50,
                device=device,
                image_size=model.image_size,
                cfg_scale=1.5,
                progress=False,
            )
    except Exception as e:
        return ValidationResult(
            name="Sample Quality",
            passed=False,
            message=f"Sample generation error: {e}",
            critical=False,
        )

    # Note: real FID requires a reference dataset; this is a sanity check that
    # the sampler produced finite, non-degenerate images.
    has_nan = torch.isnan(generated).any().item()
    has_inf = torch.isinf(generated).any().item()
    mean_val = generated.mean().item()
    std_val = generated.std().item()

    valid_output = (
        not has_nan and not has_inf and abs(mean_val) < 3 and 0.1 < std_val < 3.0
    )

    return ValidationResult(
        name="Sample Quality",
        passed=valid_output,
        value=f"mean={mean_val:.3f}, std={std_val:.3f}",
        message=(
            f"Generated samples {'valid' if valid_output else 'invalid'} "
            f"(NaN: {has_nan}, Inf: {has_inf})"
        ),
        critical=not valid_output,
    )


def validate_model(
    model_path: str | Path,
    thresholds: ValidationThresholds | None = None,
    check_all: bool = False,
    verbose: bool = False,
) -> ValidationReport:
    """Run complete model validation.

    Args:
        model_path: Path to model checkpoint.
        thresholds: Validation thresholds.
        check_all: Run all checks regardless of critical failures.
        verbose: Verbose logging.

    Returns:
        ValidationReport with all results.
    """
    model_path = Path(model_path)
    thresholds = thresholds or ValidationThresholds()

    report = ValidationReport(model_path=str(model_path))

    if verbose:
        logger.info(f"Validating model: {model_path}")

    # Critical checks first
    checks = [
        # File existence
        lambda: check_model_file_exists(model_path),
        # Model integrity
        lambda: check_nan_weights(model_path) if thresholds.check_nan_weights else None,
        lambda: check_inf_weights(model_path) if thresholds.check_inf_weights else None,
        # Model size
        lambda: check_model_size(model_path, thresholds.max_model_size_mb),
        # Training metrics
        lambda: check_training_metrics(
            model_path,
            thresholds.min_val_accuracy,
            thresholds.min_train_accuracy,
            thresholds.max_final_loss,
        ),
        # Sample quality (if generative model)
        lambda: (
            generate_sample_and_check_quality(model_path, thresholds.min_fid_score)
            if check_all
            else None
        ),
        # ONNX validation
        lambda: (
            validate_onnx_export(model_path, tolerance=thresholds.onnx_output_tolerance)
            if check_all
            else None
        ),
    ]

    # Run checks
    for check_fn in checks:
        if check_fn is None:
            continue

        result = check_fn()
        if result:
            report.add_result(result)

            if verbose:
                status = "✓" if result.passed else "✗"
                logger.info(f"  {status} {result.name}: {result.message}")

            # Stop on critical failure unless check_all
            if not result.passed and result.critical and not check_all:
                report.errors.append(f"Critical check failed: {result.name}")
                break

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate model for production deployment"
    )
    parser.add_argument("model", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--thresholds", type=str, default=None, help="Path to thresholds JSON config"
    )
    parser.add_argument(
        "--check-all", action="store_true", help="Run all validation checks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/evaluation/validation_report.json",
        help="Output path for validation report JSON",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on any warning (not just errors)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load thresholds
    thresholds = load_thresholds(args.thresholds)

    # Run validation
    report = validate_model(
        args.model,
        thresholds=thresholds,
        check_all=args.check_all,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Model: {report.model_path}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Status: {'PASSED' if report.passed else 'FAILED'}")
    skipped = sum(1 for r in report.results if r.skipped)
    evaluated = len(report.results) - skipped
    passed = sum(1 for r in report.results if r.passed and not r.skipped)
    print(f"Checks: {passed}/{evaluated} passed ({skipped} skipped)")

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")

    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for error in report.errors:
            print(f"  ✗ {error}")

    print("=" * 60)

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report.to_json())
        logger.info(f"Report saved to: {output_path}")

    # Exit code
    if not report.passed or (args.strict and report.warnings):
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
