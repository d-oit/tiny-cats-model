"""scripts/generate_frontend_metadata.py

Generate the browser's model contract and release manifest (issues #197/#204).

The frontend must never build a mutable ``resolve/main`` URL. This script
writes two generated JSON artifacts:

* ``frontend/src/generated/modelContract.json`` — the same versioned
  labels/preprocessing contract embedded in checkpoints and ONNX sidecars,
* ``frontend/src/generated/modelManifest.json`` — per-artifact SHA-256, size,
  architecture/labels/preprocessing, opset and an **immutable** Hub revision.

A release promotes artifacts by passing ``--revision <40-hex Hub commit>``.
Without a revision the manifest is local-bundle only; the browser then never
contacts the Hub and verifies the bundled bytes against the recorded digest.

Usage:
    python scripts/generate_frontend_metadata.py
    python scripts/generate_frontend_metadata.py --revision <sha> --check
    python scripts/generate_frontend_metadata.py \
        --package-manifest artifacts/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_contract import (  # noqa: E402
    CLASSIFIER_IMAGE_SIZE,
    CLASSIFIER_LABELS,
    CONTRACT_VERSION,
    GENERATOR_CONDITIONS,
    classifier_metadata,
    generator_metadata,
)

GENERATED_DIR = ROOT / "frontend" / "src" / "generated"
CONTRACT_PATH = GENERATED_DIR / "modelContract.json"
MANIFEST_PATH = GENERATED_DIR / "modelManifest.json"
DEFAULT_ARTIFACT_DIR = ROOT / "frontend" / "public" / "models"

_HUB_REVISION = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_package_entries(path: Path) -> set[tuple[str, int]]:
    """``(sha256, bytes)`` pairs from a verified artifact package manifest.

    Issue #204 requires the release manifest to be generated from verified
    package output rather than an arbitrary directory: a file whose bytes do
    not appear in ``artifacts/manifest.json`` is refused instead of being
    hashed and shipped.
    """
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Cannot read artifact package manifest {path}: {exc}"
        ) from exc
    files = document.get("files") if isinstance(document, dict) else None
    if not isinstance(files, list) or not files:
        raise SystemExit(f"Artifact package manifest {path} lists no files")
    entries: set[tuple[str, int]] = set()
    for item in files:
        if not isinstance(item, dict):
            continue
        digest = item.get("sha256")
        size = item.get("bytes")
        if isinstance(digest, str) and isinstance(size, int):
            entries.add((digest.lower(), size))
    if not entries:
        raise SystemExit(
            f"Artifact package manifest {path} has no sha256/bytes entries"
        )
    return entries


def _asset(
    artifact_dir: Path,
    filename: str,
    *,
    contract: dict[str, Any],
    revision: str | None,
    hub_path: str,
    package_entries: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    path = artifact_dir / filename
    if not path.exists():
        raise SystemExit(f"Model artifact missing: {path}")
    digest = sha256_file(path)
    size = path.stat().st_size
    if package_entries is not None and (digest, size) not in package_entries:
        raise SystemExit(
            f"{path} does not match the verified artifact package: "
            f"sha256={digest} bytes={size} is not listed in the package manifest"
        )
    io = _onnx_io(path)
    asset: dict[str, Any] = {
        "artifact": filename,
        "localPath": f"models/{filename}",
        "hubPath": hub_path,
        "hubRevision": revision if revision and _HUB_REVISION.match(revision) else None,
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


def build_manifest(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    revision: str | None = None,
    package_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build the frontend manifest from local verified artifacts."""
    if revision is not None and not _HUB_REVISION.match(revision):
        raise SystemExit(
            f"--revision must be an immutable 40-hex Hub commit SHA, got {revision!r}"
        )
    package_entries = (
        _load_package_entries(package_manifest) if package_manifest else None
    )
    classifier = classifier_metadata()
    generator = generator_metadata()
    return {
        "schemaVersion": 1,
        "contractVersion": CONTRACT_VERSION,
        "hub": {
            "repoId": "d4oit/tiny-cats-model",
            "baseUrl": "https://huggingface.co/d4oit/tiny-cats-model/resolve/",
            "revision": revision,
            "immutable": revision is not None,
        },
        "policy": {
            "requireSha256": True,
            "requireSize": True,
            "allowMutableRevision": False,
            "acceptLocalFallbackOnlyOnDigestMatch": True,
        },
        "models": {
            "classifier": _asset(
                artifact_dir,
                "cats_quantized.onnx",
                contract=classifier,
                revision=revision,
                hub_path="cats_quantized.onnx",
                package_entries=package_entries,
            ),
            "generator": _asset(
                artifact_dir,
                "generator_quantized.onnx",
                contract=generator,
                revision=revision,
                hub_path="generator_quantized.onnx",
                package_entries=package_entries,
            ),
        },
        "runtime": {
            "name": "onnxruntime-web",
            "preferredProviders": ["webgpu", "wasm"],
            "fallbackProvider": "wasm",
            "imageSize": CLASSIFIER_IMAGE_SIZE,
            "classifierLabels": list(CLASSIFIER_LABELS),
            "generatorConditions": list(GENERATOR_CONDITIONS),
        },
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--revision", default=None, help="Immutable Hub commit SHA")
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=None,
        help=(
            "artifacts/manifest.json from the verified artifact package; every "
            "bundled model must match one of its sha256/bytes entries"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when generated files are stale",
    )
    args = parser.parse_args(argv)

    contract = {
        "contract_version": CONTRACT_VERSION,
        "classifier": classifier_metadata(),
        "generator": generator_metadata(),
    }
    manifest = build_manifest(args.artifact_dir, args.revision, args.package_manifest)

    if args.check:
        for path, expected in ((CONTRACT_PATH, contract), (MANIFEST_PATH, manifest)):
            if not path.exists():
                print(f"STALE: {path} is missing")
                return 1
            actual = json.loads(path.read_text())
            if actual != expected:
                print(f"STALE: {path} does not match the current artifacts")
                return 1
        print("Frontend model metadata is up to date")
        return 0

    _write_json(CONTRACT_PATH, contract)
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {CONTRACT_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    for name, asset in manifest["models"].items():
        print(f"  {name}: {asset['sha256']} ({asset['size']} bytes)")
        if asset["hubRevision"] is None:
            print("    local-bundle only (no immutable Hub revision pinned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
