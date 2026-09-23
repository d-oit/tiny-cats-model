"""src/artifacts.py

Canonical checkpoint + final artifact layout (issue #163 WP2).

Layout, relative to the outputs root (``/outputs`` on Modal, repo root
locally):

    checkpoints/pool/                 # live checkpoint (canonical)
        dit_model.pt
        dit_model_ema.pt
        training_state.json
        training.log
    export/                           # ONNX export/quantization staging
        model.onnx
        generator_quantized.onnx
    artifacts/
        generator/
            model.pt
            model_ema.pt
            model.onnx
            model_quantized.onnx
        training/
            final_checkpoint.pt
            training_state.json
            training.log
        evaluation/
            evaluation_report.json
            benchmark_report.json
            validation_report.json
            samples/
        manifest.json                 # sha256 manifest of the package

Legacy locations (``checkpoints/dit/current``,
``checkpoints/dit/breed-conditioned-v4``, ``tinydit_final.pt``) are read-only
migration inputs: resume can pick them up, but nothing writes them anymore.

The final package is only built when a run's global target is actually
reached, and :func:`package_final_artifacts` refuses to publish a package
whose training checkpoint is missing, unreadable, or short of the required
completed step count (issue #163: "Never publish a final model unless the
target checkpoint exists and passes validation").
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Live checkpoint directory (canonical, WP2).
POOL_DIRNAME = "checkpoints/pool"

#: Legacy live-checkpoint directories, newest/most specific first. Resume
#: migrates from these when the canonical directory is empty (WP2: no run may
#: silently restart just because the layout changed).
LEGACY_POOL_DIRNAMES: tuple[str, ...] = (
    "checkpoints/dit/current",
    "checkpoints/dit/breed-conditioned-v4",
    "checkpoints/dit",
)

#: Root of the final artifact package (WP2).
ARTIFACTS_DIRNAME = "artifacts"


class ArtifactPackageError(RuntimeError):
    """The final artifact package could not be built or validated."""


def pool_dir(outputs_root: str | Path) -> Path:
    """Canonical live-checkpoint directory under ``outputs_root``."""
    return Path(outputs_root) / POOL_DIRNAME


def export_paths(outputs_root: str | Path) -> dict[str, Path]:
    """ONNX export/quantization staging paths under ``outputs_root`` (WP2).

    Exports stage here and are *copied* into ``artifacts/generator`` by
    :func:`package_final_artifacts` — sources must live outside the package
    so the copy never becomes a same-file no-op.

    The quantized name follows ``optimize_onnx``'s model-type convention
    (``generator`` → ``generator_quantized.onnx``).
    """
    export = Path(outputs_root) / "export"
    return {
        "export_dir": export,
        "onnx": export / "model.onnx",
        "quantized": export / "generator_quantized.onnx",
    }


def pool_paths(outputs_root: str | Path) -> dict[str, Path]:
    """Canonical live-checkpoint file paths under ``outputs_root``."""
    pool = pool_dir(outputs_root)
    return {
        "model": pool / "dit_model.pt",
        "model_ema": pool / "dit_model_ema.pt",
        "training_state": pool / "training_state.json",
        "training_log": pool / "training.log",
    }


def find_live_checkpoint(outputs_root: str | Path) -> Path | None:
    """Newest *valid* live checkpoint: canonical first, then legacy dirs.

    Only structurally valid (zip-format) checkpoints are returned, so a
    stale partial file in the canonical directory cannot shadow a resumable
    legacy checkpoint during layout migration.
    """
    root = Path(outputs_root)
    candidates = [pool_dir(root), *(root / name for name in LEGACY_POOL_DIRNAMES)]
    for directory in candidates:
        candidate = directory / "dit_model.pt"
        if candidate.exists() and zipfile.is_zipfile(candidate):
            return candidate
        if candidate.exists():
            logger.warning(
                "Skipping non-zip checkpoint candidate %s (stale partial write)",
                candidate,
            )
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_into(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # SameFileError guard: when the source already lives at the destination
    # (e.g. ONNX exported straight into the package), hashing it in place is
    # the correct "copy" — found by the live Modal smoke test (issue #163).
    try:
        same_file = destination.exists() and os.path.samefile(source, destination)
    except OSError:
        same_file = False
    if not same_file:
        shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
    }


def package_final_artifacts(
    outputs_root: str | Path,
    *,
    raw_checkpoint: str | Path,
    ema_checkpoint: str | Path | None = None,
    training_log: str | Path | None = None,
    samples_dir: str | Path | None = None,
    onnx: str | Path | None = None,
    quantized: str | Path | None = None,
    evaluation_report: str | Path | None = None,
    benchmark_report: str | Path | None = None,
    require_completed: int | None = None,
) -> dict[str, Any]:
    """Build and validate the canonical final artifact package (WP2).

    Args:
        outputs_root: Root that holds ``checkpoints/`` and will hold
            ``artifacts/`` (``/outputs`` on Modal).
        raw_checkpoint: Training checkpoint (model + optimizer + manifest).
        ema_checkpoint: EMA checkpoint, when available.
        training_log: Training log file, when available.
        samples_dir: Directory of generated samples, copied verbatim into
            ``artifacts/evaluation/samples``.
        onnx: Exported ONNX generator.
        quantized: Quantized ONNX generator.
        evaluation_report: Evaluation JSON report, when already generated.
        benchmark_report: Benchmark JSON report, when already generated.
        require_completed: Minimum ``completed_steps`` the training state
            must record — typically the run's global target. The package is
            refused when the target checkpoint does not exist, is not a
            valid torch checkpoint, or has not reached this step count.

    Returns:
        The artifact manifest (also written to ``artifacts/manifest.json``).

    Raises:
        ArtifactPackageError: If the package would be incomplete or invalid.
    """
    root = Path(outputs_root)
    raw = Path(raw_checkpoint)
    artifacts = root / ARTIFACTS_DIRNAME

    if not raw.exists():
        raise ArtifactPackageError(f"Training checkpoint missing: {raw}")
    if not zipfile.is_zipfile(raw):
        raise ArtifactPackageError(
            f"Training checkpoint is not a valid torch checkpoint (not a zip): {raw}"
        )

    # Validate progress against the training state beside the checkpoint.
    from training_state import TRAINING_STATE_FILENAME, read_training_state

    state_path = raw.parent / TRAINING_STATE_FILENAME
    state = read_training_state(state_path)
    if require_completed is not None:
        if state is None:
            raise ArtifactPackageError(
                f"Cannot verify training progress: {state_path} missing"
            )
        completed = int(state.get("completed_steps", -1))
        if completed < int(require_completed):
            raise ArtifactPackageError(
                f"Refusing to publish: completed_steps={completed} < "
                f"required {require_completed}"
            )

    files: list[dict[str, Any]] = []

    # artifacts/generator/
    generator_dir = artifacts / "generator"
    files.append(_copy_into(raw, generator_dir / "model.pt"))
    ema = Path(ema_checkpoint) if ema_checkpoint else None
    if ema is not None and ema.exists():
        files.append(_copy_into(ema, generator_dir / "model_ema.pt"))
    onnx_path = Path(onnx) if onnx else None
    if onnx_path is not None and onnx_path.exists():
        files.append(_copy_into(onnx_path, generator_dir / "model.onnx"))
    quantized_path = Path(quantized) if quantized else None
    if quantized_path is not None and quantized_path.exists():
        files.append(_copy_into(quantized_path, generator_dir / "model_quantized.onnx"))

    # artifacts/training/
    training_dir = artifacts / "training"
    files.append(_copy_into(raw, training_dir / "final_checkpoint.pt"))
    if state is not None:
        files.append(_copy_into(state_path, training_dir / "training_state.json"))
    log = Path(training_log) if training_log else None
    if log is not None and log.exists():
        files.append(_copy_into(log, training_dir / "training.log"))

    # artifacts/evaluation/
    evaluation_dir = artifacts / "evaluation"
    samples = Path(samples_dir) if samples_dir else None
    if samples is not None and samples.is_dir():
        destination = evaluation_dir / "samples"
        shutil.copytree(samples, destination, dirs_exist_ok=True)
        files.extend(
            {
                "path": str(item),
                "sha256": _sha256(item),
                "bytes": item.stat().st_size,
            }
            for item in sorted(destination.rglob("*"))
            if item.is_file()
        )
    for report in (evaluation_report, benchmark_report):
        report_path = Path(report) if report else None
        if report_path is not None and report_path.exists():
            files.append(_copy_into(report_path, evaluation_dir / report_path.name))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": (state or {}).get("experiment_id"),
        "completed_steps": (state or {}).get("completed_steps"),
        "target_steps": (state or {}).get("target_steps"),
        "git_sha": (state or {}).get("git_sha"),
        "files": files,
    }
    manifest_path = artifacts / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, manifest_path)

    logger.info(
        "Final artifact package written: %s (%d files)",
        artifacts,
        len(files),
    )
    return manifest
