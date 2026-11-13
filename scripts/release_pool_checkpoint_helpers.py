"""Integrity checks shared by the pool checkpoint release command."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safe_checkpoint import sha256_file

ROOT = Path(__file__).resolve().parent.parent


def resolve_source_revision(
    source_revision: str | None, token: str, repo_id: str
) -> str:
    from huggingface_hub import HfApi

    from model_release import ReleaseError, validate_revision

    if source_revision is not None:
        try:
            return validate_revision(source_revision)
        except ReleaseError as error:
            raise ValueError(
                f"Invalid source revision {source_revision!r}: {error}"
            ) from error
    info = HfApi(token=token).repo_info(repo_id, revision="main")
    sha = getattr(info, "sha", None)
    if not sha:
        raise ValueError(f"Hub returned no source commit SHA for {repo_id}@main")
    try:
        return validate_revision(str(sha))
    except ReleaseError as error:
        raise ValueError(f"Hub returned invalid source SHA {sha!r}: {error}") from error


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} at {path} must be a JSON object")
    return value


def verify_graph(
    path: Path, *, model: torch.nn.Module | None, label: str
) -> list[np.ndarray]:
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((1, 3, 128, 128)).astype(np.float32)
    timestep = np.array([0.5], dtype=np.float32)
    outputs: list[np.ndarray] = []
    for breed_index in (0, 12):
        breed = np.array([breed_index], dtype=np.int64)
        output = session.run(
            ["velocity"], {"noise": noise, "timestep": timestep, "breed": breed}
        )[0]
        if output.shape != (1, 3, 128, 128):
            raise ValueError(
                f"{label} output shape {output.shape} for breed {breed_index}"
            )
        if not np.isfinite(output).all():
            raise ValueError(
                f"{label} output contains non-finite values for breed {breed_index}"
            )
        if model is not None:
            with torch.inference_mode():
                expected = (
                    model(
                        torch.from_numpy(noise),
                        torch.from_numpy(timestep),
                        torch.from_numpy(breed),
                    )
                    .cpu()
                    .numpy()
                )
            np.testing.assert_allclose(output, expected, rtol=1e-3, atol=1e-4)
        outputs.append(output)
    return outputs


def verify_classifier(classifier_path: Path) -> tuple[str, int]:
    manifest_path = ROOT / "frontend/src/generated/modelManifest.json"
    manifest = read_json_object(manifest_path, "frontend model manifest")
    models = manifest.get("models")
    classifier = models.get("classifier") if isinstance(models, dict) else None
    if not isinstance(classifier, dict):
        raise ValueError(f"Classifier record missing from {manifest_path}")
    digest = sha256_file(classifier_path)
    size = classifier_path.stat().st_size
    if digest != classifier.get("sha256") or size != classifier.get("size"):
        raise ValueError(
            f"Bundled classifier {classifier_path} does not match {manifest_path}: "
            f"sha256={digest}, bytes={size}; expected "
            f"sha256={classifier.get('sha256')}, bytes={classifier.get('size')}"
        )
    return digest, size


def verify_package_manifest(artifacts: Path, manifest: dict[str, Any]) -> str:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Package manifest has no file records")
    artifact_root = artifacts.resolve()
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"Invalid package file record: {record!r}")
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_relative_to(artifact_root):
            raise ValueError(
                f"Package manifest path escapes artifacts directory: {path}"
            )
        if not path.is_file():
            raise ValueError(f"Package manifest file is missing: {path}")
        actual_digest = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_digest != record.get("sha256") or actual_size != record.get("bytes"):
            raise ValueError(
                f"Package record mismatch for {path}: sha256={actual_digest}, "
                f"bytes={actual_size}; expected sha256={record.get('sha256')}, "
                f"bytes={record.get('bytes')}"
            )
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_frontend_promotion(
    revision: str,
    target: int,
    classifier_digest: str,
    package_manifest_digest: str,
) -> None:
    generated = ROOT / "frontend/src/generated"
    manifest = read_json_object(generated / "modelManifest.json", "model manifest")
    hub = manifest.get("hub")
    if (
        not isinstance(hub, dict)
        or hub.get("revision") != revision
        or hub.get("immutable") is not True
    ):
        raise ValueError(
            f"Frontend manifest does not pin immutable revision {revision}"
        )
    models = manifest.get("models")
    classifier = models.get("classifier") if isinstance(models, dict) else None
    if (
        not isinstance(classifier, dict)
        or classifier.get("sha256") != classifier_digest
    ):
        raise ValueError("Frontend classifier digest changed during release promotion")
    history = read_json_object(generated / "modelReleases.json", "release history")
    releases = history.get("releases")
    if not isinstance(releases, list):
        raise ValueError("Frontend release history has no releases list")
    entry = next(
        (
            item
            for item in releases
            if isinstance(item, dict) and item.get("revision") == revision
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"Frontend release history does not contain {revision}")
    package = entry.get("package_manifest")
    if (
        not isinstance(package, dict)
        or package.get("completed_steps") != target
        or package.get("sha256") != package_manifest_digest
    ):
        raise ValueError(
            "Frontend release history package manifest differs from evidence"
        )
