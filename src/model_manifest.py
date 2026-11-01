"""src/model_manifest.py

Build the browser's model release manifest from verified artifacts (issue #204).

A manifest asset records the bundled filename, the uploaded Hub path, the
bit-exact SHA-256 and size, the versioned model contract and (when ``onnx`` is
installed) the ONNX graph inputs/outputs and opset. Digests can be sourced from
a verified artifact package manifest so a stale bundle can never be promoted.

The Hub revision must be a full 40-character commit SHA and is verified by
:func:`model_release.verify_hub_artifact` before it reaches the manifest.

This module has no heavy imports (no torch/onnx at import time) so tests and
CI can use it cheaply.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from model_contract import (
    CLASSIFIER_IMAGE_SIZE,
    CLASSIFIER_LABELS,
    CONTRACT_VERSION,
    GENERATOR_CONDITIONS,
    classifier_metadata,
    generator_metadata,
)
from model_release import (
    ReleaseError,
    expected_models,
    find_package_file,
    hub_url,
    read_package_manifest,
    sha256_file,
    validate_revision,
    verify_package_file,
)

#: Repository root, used only for the default bundled-artifact directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "frontend" / "public" / "models"

HUB_REPO_ID = "d4oit/tiny-cats-model"
HUB_BASE_URL = f"https://huggingface.co/{HUB_REPO_ID}/resolve/"

#: Frontend artifact key -> (bundled filename, uploaded Hub path).
#: The Hub path mirrors ``src/upload_to_huggingface.py`` so a pinned revision
#: resolves to the bytes that were actually uploaded.
MODEL_ARTIFACTS: dict[str, tuple[str, str]] = {
    "classifier": ("cats_quantized.onnx", "classifier/model_quantized.onnx"),
    "generator": ("generator_quantized.onnx", "generator/model_quantized.onnx"),
}

#: Verifies one immutable Hub artifact: ``(url, sha256, size) -> digest``.
HubVerifier = Callable[..., str]


def _onnx_io(path: Path) -> dict[str, Any] | None:
    """Best-effort ONNX graph summary; None when onnx is unavailable."""
    try:
        import onnx
    except ImportError:
        return None
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # pragma: no cover - corrupt artifact
        raise SystemExit(f"Cannot read ONNX graph {path}: {exc}") from exc

    def describe(value: Any) -> dict[str, Any]:
        tensor_type = value.type.tensor_type
        dims: list[Any] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(int(dim.dim_value))
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        return {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower(),
            "shape": dims,
        }

    opset = None
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            opset = int(entry.version)
    return {
        "inputs": [describe(value) for value in model.graph.input],
        "outputs": [describe(value) for value in model.graph.output],
        "opset": opset,
    }


def _asset(
    artifact_dir: Path,
    filename: str,
    *,
    contract: Mapping[str, Any],
    revision: str | None,
    hub_path: str,
    expected: Mapping[str, Any] | None = None,
    hub_verifier: HubVerifier | None = None,
    base_url: str = HUB_BASE_URL,
) -> dict[str, Any]:
    """Describe one immutable asset, verified against its release record."""
    path = artifact_dir / filename
    if not path.exists():
        raise SystemExit(f"Model artifact missing: {path}")
    if expected is None:
        digest = sha256_file(path)
        size = path.stat().st_size
    else:
        try:
            digest = verify_package_file(
                {
                    "path": filename,
                    "sha256": expected["sha256"],
                    "bytes": expected["size"],
                },
                path,
                label=f"Bundled {filename}",
            )
        except ReleaseError as error:
            raise SystemExit(f"Refusing stale artifact: {error}") from error
        size = int(expected["size"])
    if revision is not None and hub_verifier is not None:
        try:
            hub_verifier(
                hub_url(base_url, revision, hub_path),
                sha256=digest,
                size=size,
                label=f"Hub {filename}",
            )
        except ReleaseError as error:
            raise SystemExit(f"Refusing unverified Hub revision: {error}") from error

    io = _onnx_io(path)
    asset: dict[str, Any] = {
        "artifact": filename,
        "localPath": f"models/{filename}",
        "hubPath": hub_path,
        "hubRevision": revision if revision else None,
        "sha256": digest,
        "size": size,
        "contractHash": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "contract": contract,
    }
    if io is not None:
        asset["onnx"] = io
    return asset


#: Candidate package-manifest paths for each frontend artifact.
PACKAGE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "classifier": (
        "classifier/model_quantized.onnx",
        "cats_quantized.onnx",
    ),
    "generator": (
        "generator/model_quantized.onnx",
        "generator_quantized.onnx",
    ),
}


def package_expectations(
    package: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve every frontend artifact to a verified-package file record."""
    expected: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for model_key, suffixes in PACKAGE_SUFFIXES.items():
        record = None
        for suffix in suffixes:
            try:
                record = find_package_file(package, *suffix.split("/"))
                break
            except ReleaseError:
                continue
        if record is None:
            missing.append(f"{model_key} ({' or '.join(suffixes)})")
            continue
        expected[model_key] = {
            "sha256": str(record["sha256"]).lower(),
            "size": int(record["bytes"]),
        }
    if missing:
        raise SystemExit(
            "Artifact package is missing the frontend release files: "
            + ", ".join(missing)
        )
    return expected


def build_manifest(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    revision: str | None = None,
    *,
    package_manifest_path: str | Path | None = None,
    release_entry: Mapping[str, Any] | None = None,
    hub_verifier: HubVerifier | None = None,
    base_url: str = HUB_BASE_URL,
    repo_id: str = HUB_REPO_ID,
) -> dict[str, Any]:
    """Build the frontend manifest from verified artifacts and a release.

    Args:
        artifact_dir: Directory holding the bundled ONNX artifacts.
        revision: Immutable Hub commit SHA, or None for a local-only bundle.
        package_manifest_path: Verified artifact package manifest to source
            digests/sizes from (stale bundles are rejected).
        release_entry: Recorded release to re-pin (rollback); takes precedence
            over ``package_manifest_path`` and forces its revision.
        hub_verifier: Downloads and verifies the Hub bytes for ``revision``.
        base_url: Hub ``resolve/`` base URL.
        repo_id: Hugging Face repository id recorded in the manifest.
    """
    package: dict[str, Any] | None = None
    if release_entry is not None:
        revision = str(release_entry["revision"])
        expected = {
            key: {"sha256": record["sha256"], "size": record["size"]}
            for key, record in expected_models(release_entry).items()
        }
    elif package_manifest_path is not None:
        package = read_package_manifest(package_manifest_path)
        expected = package_expectations(package)
    else:
        expected = {}

    if revision is not None:
        try:
            revision = validate_revision(revision)
        except ReleaseError as error:
            raise SystemExit(str(error)) from error

    classifier = classifier_metadata()
    generator = generator_metadata()
    models = {
        key: _asset(
            Path(artifact_dir),
            filename,
            contract=classifier if key == "classifier" else generator,
            revision=revision,
            hub_path=hub_path,
            expected=expected.get(key),
            hub_verifier=hub_verifier,
            base_url=base_url,
        )
        for key, (filename, hub_path) in MODEL_ARTIFACTS.items()
    }
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "contractVersion": CONTRACT_VERSION,
        "hub": {
            "repoId": repo_id,
            "baseUrl": base_url,
            "revision": revision,
            "immutable": revision is not None,
        },
        "policy": {
            "requireSha256": True,
            "requireSize": True,
            "allowMutableRevision": False,
            "acceptLocalFallbackOnlyOnDigestMatch": True,
        },
        "models": models,
        "runtime": {
            "name": "onnxruntime-web",
            "preferredProviders": ["webgpu", "wasm"],
            "fallbackProvider": "wasm",
            "imageSize": CLASSIFIER_IMAGE_SIZE,
            "classifierLabels": list(CLASSIFIER_LABELS),
            "generatorConditions": list(GENERATOR_CONDITIONS),
        },
    }
    if package is not None:
        manifest["source"] = {
            "packageManifest": str(package_manifest_path),
            "completedSteps": package.get("completed_steps"),
            "experimentId": package.get("experiment_id"),
            "gitSha": package.get("git_sha"),
        }
    return manifest


def contract_document() -> dict[str, Any]:
    """The versioned contract mirrored into the browser bundle."""
    return {
        "contract_version": CONTRACT_VERSION,
        "classifier": classifier_metadata(),
        "generator": generator_metadata(),
    }


def verify_bundled_artifacts(
    manifest: Mapping[str, Any], artifact_dir: Path = DEFAULT_ARTIFACT_DIR
) -> None:
    """Fail when a bundled artifact no longer matches the recorded digest."""
    for key, asset in manifest.get("models", {}).items():
        path = Path(artifact_dir) / str(asset["artifact"])
        try:
            verify_package_file(
                {
                    "path": asset["artifact"],
                    "sha256": asset["sha256"],
                    "bytes": asset["size"],
                },
                path,
                label=f"Bundled {key} artifact {asset['artifact']}",
            )
        except ReleaseError as error:
            raise SystemExit(f"STALE: {error}") from error
