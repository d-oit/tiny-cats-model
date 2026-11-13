#!/usr/bin/env python3
"""Build and optionally publish a verified package from one pinned pool snapshot."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from release_pool_checkpoint_helpers import (  # noqa: E402
    assert_frontend_promotion,
    read_json_object,
    resolve_source_revision,
    verify_classifier,
    verify_graph,
    verify_package_manifest,
    verify_published_pointer,
    write_evidence,
)

from artifacts import export_paths, package_final_artifacts, pool_paths  # noqa: E402
from export_dit_onnx import export_generator_onnx, load_model  # noqa: E402
from model_release import (  # noqa: E402
    ReleaseError,
    immutable_commit_sha,
)
from safe_checkpoint import load_torch_checkpoint, sha256_file  # noqa: E402
from training_state import read_training_state  # noqa: E402

REPO_ID = "d4oit/tiny-cats-model"
DEFAULT_EXPERIMENT_ID = "dit-breed-conditioned-v4"
POOL_BASE = "checkpoints/pool"
REQUIRED_POINTER_FILES = {"dit_model.pt", "training_state.json"}


def _require_equal(
    document: Mapping[str, Any], field: str, expected: Any, label: str
) -> None:
    if field not in document:
        raise ValueError(f"{label}.{field} is missing; expected {expected!r}")
    actual = document[field]
    if isinstance(expected, int):
        valid_type = isinstance(actual, int) and not isinstance(actual, bool)
    else:
        valid_type = isinstance(actual, type(expected))
    if not valid_type or actual != expected:
        raise ValueError(f"{label}.{field}={actual!r}; expected {expected!r}")


def _validate_pointer(
    pointer: Mapping[str, Any], target: int, experiment_id: str
) -> str:
    _require_equal(pointer, "experiment_id", experiment_id, "pointer")
    _require_equal(pointer, "completed_steps", target, "pointer")
    step_dir = pointer.get("step_dir")
    if not isinstance(step_dir, str) or re.fullmatch(r"step-[0-9]+", step_dir) is None:
        raise ValueError(f"pointer.step_dir={step_dir!r}; expected step-{target}")
    step_number = int(step_dir.removeprefix("step-"))
    if step_number != target:
        raise ValueError(
            f"pointer.step_dir={step_dir!r} has step {step_number}; expected {target}"
        )
    files = pointer.get("files")
    if not isinstance(files, list) or any(not isinstance(name, str) for name in files):
        raise ValueError("pointer.files must be a list of filenames")
    missing = sorted(REQUIRED_POINTER_FILES - set(files))
    if missing:
        raise ValueError(f"pointer.files is missing required names: {missing}")
    return step_dir


def validate_snapshot(
    pointer: dict[str, Any],
    state: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    target: int,
    experiment_id: str,
) -> None:
    """Refuse a pool snapshot unless pointer, sidecar, and checkpoint agree."""
    if not isinstance(pointer, dict):
        raise ValueError("pointer document must be a JSON object")
    if not isinstance(state, dict):
        raise ValueError("training_state.json must be a JSON object")
    if not isinstance(checkpoint, dict):
        raise ValueError("dit_model.pt must contain a checkpoint object")
    _validate_pointer(pointer, target, experiment_id)
    _require_equal(state, "experiment_id", experiment_id, "training_state")
    _require_equal(state, "completed_steps", target, "training_state")
    _require_equal(checkpoint, "step", target, "checkpoint")
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint.manifest must be an object")
    _require_equal(manifest, "experiment_id", experiment_id, "checkpoint.manifest")


def _download(
    hf_hub_download: Any,
    *,
    filename: str,
    revision: str,
    token: str,
    outputs: Path,
) -> Path:
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="model",
                revision=revision,
                token=token,
                local_dir=str(outputs),
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Cannot download {filename} from {REPO_ID}@{revision}: {error}"
        ) from error
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise ValueError(f"Downloaded source file is missing or empty: {downloaded}")
    return downloaded


def _copy_download(downloaded: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, destination)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise ValueError(f"Staged source file is missing or empty: {destination}")
    return destination


def _load_source(
    outputs: Path,
    *,
    source_revision: str,
    target: int,
    experiment_id: str,
    token: str,
) -> tuple[dict[str, Path], dict[str, Any], str]:
    from huggingface_hub import hf_hub_download

    pointer_remote = f"{POOL_BASE}/{experiment_id}/latest/manifest.json"
    pointer_download = _download(
        hf_hub_download,
        filename=pointer_remote,
        revision=source_revision,
        token=token,
        outputs=outputs,
    )
    pointer = read_json_object(pointer_download, "pool pointer")
    pointer_path = _copy_download(pointer_download, outputs / "source-pointer.json")
    step_dir = _validate_pointer(pointer, target, experiment_id)
    pool = pool_paths(outputs)
    remote_base = f"{POOL_BASE}/{experiment_id}/{step_dir}"
    raw_download = _download(
        hf_hub_download,
        filename=f"{remote_base}/dit_model.pt",
        revision=source_revision,
        token=token,
        outputs=outputs,
    )
    state_download = _download(
        hf_hub_download,
        filename=f"{remote_base}/training_state.json",
        revision=source_revision,
        token=token,
        outputs=outputs,
    )
    raw_path = _copy_download(raw_download, pool["model"])
    state_path = _copy_download(state_download, pool["training_state"])
    staged = {"model": raw_path, "training_state": state_path, "pointer": pointer_path}
    if "dit_model_ema.pt" in pointer["files"]:
        ema_download = _download(
            hf_hub_download,
            filename=f"{remote_base}/dit_model_ema.pt",
            revision=source_revision,
            token=token,
            outputs=outputs,
        )
        staged["model_ema"] = _copy_download(ema_download, pool["model_ema"])

    state = read_training_state(state_path)
    if state is None:
        raise ValueError(f"training_state.json at {state_path} is missing or invalid")
    checkpoint = load_torch_checkpoint(raw_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"dit_model.pt at {raw_path} is not a checkpoint object")
    try:
        validate_snapshot(
            pointer,
            state,
            checkpoint,
            target=target,
            experiment_id=experiment_id,
        )
    except ValueError as error:
        raise ValueError(f"Invalid source snapshot at {raw_path}: {error}") from error

    ema_path = staged.get("model_ema")
    if ema_path is not None:
        ema_checkpoint = load_torch_checkpoint(ema_path, map_location="cpu")
        if not isinstance(ema_checkpoint, dict):
            raise ValueError(f"EMA checkpoint at {ema_path} is not an object")
        try:
            validate_snapshot(
                pointer,
                state,
                ema_checkpoint,
                target=target,
                experiment_id=experiment_id,
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid EMA source checkpoint at {ema_path}: {error}"
            ) from error
    return staged, pointer, sha256_file(raw_path)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _build_release(
    outputs: Path,
    *,
    target: int,
    experiment_id: str,
    source_revision: str,
    checkpoint_digest: str,
    publish: bool,
    token: str,
) -> str | None:
    pool = pool_paths(outputs)
    exports = export_paths(outputs)
    artifacts = outputs / "artifacts"
    from optimize_onnx import optimize_onnx

    if not pool["model"].is_file():
        raise ValueError(
            f"Required raw checkpoint missing before export: {pool['model']}"
        )

    classifier_path = ROOT / "frontend/public/models/cats_quantized.onnx"
    classifier_digest, _ = verify_classifier(classifier_path)
    model = load_model(pool["model"], device=torch.device("cpu"))
    export_generator_onnx(
        model,
        output_path=exports["onnx"],
        opset_version=17,
        use_cfg=False,
        cfg_scale=1.0,
    )
    optimize_onnx(
        model_path=exports["onnx"],
        output_dir=exports["export_dir"],
        method="dynamic",
        model_type="generator",
    )
    if not exports["quantized"].is_file():
        raise ValueError(
            f"Quantized generator missing after optimization: {exports['quantized']}"
        )
    fp32_outputs = verify_graph(exports["onnx"], model=model, label="FP32 ONNX")
    quantized_outputs = verify_graph(
        exports["quantized"], model=None, label="Quantized ONNX"
    )
    for breed_index, fp32, quantized in zip(
        (0, 12), fp32_outputs, quantized_outputs, strict=True
    ):
        difference = np.abs(fp32 - quantized)
        print(
            f"Quantized-vs-FP32 breed {breed_index}: "
            f"max_abs={float(difference.max()):.8g}, "
            f"mean_abs={float(difference.mean()):.8g}"
        )

    evaluation_dir = outputs / "evaluation"
    samples_dir = evaluation_dir / "samples"
    evaluation_report = evaluation_dir / "evaluation_report.json"
    benchmark_report = evaluation_dir / "benchmark_report.json"
    _run(
        [
            sys.executable,
            "src/evaluate_full.py",
            "--checkpoint",
            str(pool["model"]),
            "--generate-samples",
            "--num-samples",
            "16",
            "--all-breeds",
            "--device",
            "cpu",
            "--output-dir",
            str(samples_dir),
            "--report-path",
            str(evaluation_report),
        ],
        env={key: value for key, value in os.environ.items() if key != "HF_TOKEN"},
    )
    _run(
        [
            sys.executable,
            "src/benchmark_inference.py",
            "--model",
            str(pool["model"]),
            "--device",
            "cpu",
            "--num-warmup",
            "5",
            "--num-runs",
            "20",
            "--benchmark-throughput",
            "--batch-sizes",
            "1,4,8",
            "--report-path",
            str(benchmark_report),
        ],
        env={key: value for key, value in os.environ.items() if key != "HF_TOKEN"},
    )
    package_manifest = package_final_artifacts(
        outputs,
        raw_checkpoint=pool["model"],
        ema_checkpoint=pool.get("model_ema"),
        onnx=exports["onnx"],
        quantized=exports["quantized"],
        classifier_quantized=classifier_path,
        samples_dir=samples_dir,
        evaluation_report=evaluation_report,
        benchmark_report=benchmark_report,
        require_completed=target,
    )
    verify_checkpoint = ROOT / "src/verify_checkpoint.py"
    _run(
        [
            sys.executable,
            str(verify_checkpoint),
            "--checkpoint",
            str(artifacts / "generator/model.pt"),
            "--onnx",
            str(artifacts / "generator/model.onnx"),
            "--require-onnx",
        ]
    )
    if package_manifest.get("completed_steps") != target:
        raise ValueError(
            "Package manifest completed_steps does not equal the requested "
            f"target {target}: {package_manifest.get('completed_steps')!r}"
        )
    if package_manifest.get("experiment_id") != experiment_id:
        raise ValueError(
            "Package manifest experiment_id does not match the requested "
            f"identity {experiment_id!r}"
        )
    packaged_classifier = artifacts / "classifier/model_quantized.onnx"
    if sha256_file(packaged_classifier) != classifier_digest:
        raise ValueError("Packaged classifier bytes differ from the existing bundle")
    package_manifest_digest = verify_package_manifest(artifacts, package_manifest)
    evidence_path = outputs / "release-evidence.json"
    evidence: dict[str, Any] = {
        "source_revision": source_revision,
        "revision": None,
        "experiment_id": experiment_id,
        "completed_steps": target,
        "checkpoint_sha256": checkpoint_digest,
        "package_manifest_sha256": package_manifest_digest,
    }
    write_evidence(evidence_path, evidence)
    print(f"Release evidence: {evidence_path}")

    if not publish:
        return None

    from huggingface_hub import HfApi

    result = HfApi(token=token).upload_folder(
        folder_path=str(artifacts),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Release {target}-step DiT package",
    )
    try:
        revision = immutable_commit_sha(result)
    except ReleaseError as error:
        raise ValueError(
            f"Hub upload returned no immutable revision: {error}"
        ) from error
    evidence["revision"] = revision
    write_evidence(evidence_path, evidence)
    print(f"Published immutable Hub revision: {revision}")

    verify_published_pointer(
        outputs / "source-pointer.json",
        revision=revision,
        repo_id=REPO_ID,
        remote_path=f"{POOL_BASE}/{experiment_id}/latest/manifest.json",
        token=token,
    )

    frontend_models = ROOT / "frontend/public/models"
    shutil.copy2(
        artifacts / "generator/model_quantized.onnx",
        frontend_models / "generator_quantized.onnx",
    )
    shutil.copy2(artifacts / "classifier/model_quantized.onnx", classifier_path)
    _run(
        [
            sys.executable,
            "scripts/promote_frontend_release.py",
            "--revision",
            revision,
            "--package-manifest",
            str(artifacts / "manifest.json"),
            "--artifact-dir",
            str(frontend_models),
        ]
    )
    _run([sys.executable, "scripts/generate_frontend_metadata.py", "--check"])
    assert_frontend_promotion(
        revision, target, classifier_digest, package_manifest_digest
    )
    return revision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=600000)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target <= 0:
        raise ValueError(f"--target must be positive; got {args.target}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be set in the environment")
    outputs = args.outputs.resolve()
    if outputs.exists() and not outputs.is_dir():
        raise ValueError(f"--outputs must be a directory: {outputs}")
    outputs.mkdir(parents=True, exist_ok=True)
    if any(outputs.iterdir()):
        raise ValueError(
            f"--outputs must be new or empty; refusing to mix files in {outputs}"
        )

    source_revision = resolve_source_revision(args.source_revision, token, REPO_ID)
    staged, _, checkpoint_digest = _load_source(
        outputs,
        source_revision=source_revision,
        target=args.target,
        experiment_id=args.experiment_id,
        token=token,
    )
    if not staged["model"].is_file():
        raise ValueError(f"Required checkpoint is unavailable: {staged['model']}")
    revision = _build_release(
        outputs,
        target=args.target,
        experiment_id=args.experiment_id,
        source_revision=source_revision,
        checkpoint_digest=checkpoint_digest,
        publish=args.publish,
        token=token,
    )
    if revision is None:
        print(f"Built and verified {args.target}-step package without publishing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
