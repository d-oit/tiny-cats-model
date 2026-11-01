"""scripts/generate_frontend_metadata.py

Generate the browser's model contract, release manifest and release history
(issues #197/#204).

The frontend must never build a mutable ``resolve/main`` URL and must verify
SHA-256 before creating an ONNX session. This script writes:

* ``frontend/src/generated/modelContract.json`` — the versioned
  labels/preprocessing contract embedded in checkpoints and ONNX sidecars,
* ``frontend/src/generated/modelManifest.json`` — per-artifact SHA-256, size,
  architecture/labels/preprocessing, opset and an **immutable** Hub revision,
* ``frontend/src/generated/modelReleases.json`` — append-only release history
  (revision, digests, package manifest digest) used for traceability and
  rollback.

Manifest digests can be sourced from a verified artifact package manifest
(``artifacts/manifest.json`` from :mod:`artifacts`) with
``--package-manifest``. A release promotes artifacts by passing an immutable,
``--verify-hub``-checked revision:

    python scripts/generate_frontend_metadata.py \\
        --package-manifest artifacts/manifest.json \\
        --revision <40-hex Hub commit> --verify-hub
    python scripts/generate_frontend_metadata.py --check
    python scripts/generate_frontend_metadata.py --rollback <40-hex commit>

Without a revision the manifest is local-bundle only; the browser then never
contacts the Hub and verifies the bundled bytes against the recorded digest.
Release history is append-only: rollback re-pins a recorded revision and
requires the bundled bytes to match that release exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
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
from model_release import (  # noqa: E402
    ReleaseError,
    build_release_entry,
    expected_models,
    fetch_hub_bytes,
    find_package_file,
    hub_url,
    read_package_manifest,
    read_release_history,
    record_release,
    resolve_release,
    sha256_file,
    validate_revision,
    verify_hub_artifact,
    verify_package_file,
    write_release_history,
)

GENERATED_DIR = ROOT / "frontend" / "src" / "generated"
CONTRACT_PATH = GENERATED_DIR / "modelContract.json"
MANIFEST_PATH = GENERATED_DIR / "modelManifest.json"
HISTORY_PATH = GENERATED_DIR / "modelReleases.json"
DEFAULT_ARTIFACT_DIR = ROOT / "frontend" / "public" / "models"

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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def check_generated(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    manifest_path: Path | None = None,
    contract_path: Path | None = None,
) -> int:
    """Verify committed generated metadata still matches the artifacts."""
    manifest_path = manifest_path or MANIFEST_PATH
    contract_path = contract_path or CONTRACT_PATH
    for path in (contract_path, manifest_path):
        if not path.exists():
            print(f"STALE: {path} is missing")
            return 1
    actual_contract = _read_json(contract_path)
    if actual_contract != contract_document():
        print(f"STALE: {contract_path} does not match model_contract.py")
        return 1

    current = _read_json(manifest_path)
    revision = current.get("hub", {}).get("revision")
    try:
        rebuilt = build_manifest(
            artifact_dir,
            revision,
            release_entry={"revision": revision, "models": current["models"]}
            if revision
            else None,
            base_url=str(current.get("hub", {}).get("baseUrl", HUB_BASE_URL)),
            repo_id=str(current.get("hub", {}).get("repoId", HUB_REPO_ID)),
        )
    except SystemExit as error:
        print(f"STALE: {error}")
        return 1
    # ``build_manifest`` re-hashes the real files, so a stale bundle or a
    # hand-edited digest fails there. Provenance recorded by a promotion is
    # environment-specific, so it is compared as text rather than regenerated.
    if "source" in current:
        rebuilt["source"] = current["source"]
    if rebuilt != current:
        print(f"STALE: {manifest_path} does not match the current artifacts")
        return 1
    print("Frontend model metadata is up to date")
    return 0


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _hub_verifier(token: str | None) -> HubVerifier:
    """Return a verifier that downloads the pinned Hub bytes."""

    def verify(url: str, *, sha256: str, size: int, label: str) -> str:
        return verify_hub_artifact(
            url,
            sha256=sha256,
            size=size,
            label=label,
            fetch=lambda target: fetch_hub_bytes(target, token=token),
        )

    return verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable 40-hex Hub commit SHA for the promoted release",
    )
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=None,
        help="Verified artifact package manifest (artifacts/manifest.json)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=HISTORY_PATH,
        help="Append-only release history JSON",
    )
    parser.add_argument(
        "--rollback",
        nargs="?",
        const="latest",
        default=None,
        metavar="REVISION",
        help="Re-pin a recorded release (default: the latest recorded one)",
    )
    parser.add_argument(
        "--verify-hub",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Download and verify the pinned revision before writing (default: on)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when generated files are stale",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check_generated(args.artifact_dir)
    if args.rollback and args.revision:
        raise SystemExit(
            "--rollback re-pins a recorded revision; do not pass --revision"
        )

    history = read_release_history(args.history)
    release_entry = None
    package = None
    if args.rollback:
        selection = None if args.rollback == "latest" else args.rollback
        try:
            release_entry = resolve_release(history, selection)
        except ReleaseError as error:
            raise SystemExit(str(error)) from error
        revision = str(release_entry["revision"])
    else:
        revision = args.revision

    if args.package_manifest is not None:
        package = read_package_manifest(args.package_manifest)

    verify_hub = args.verify_hub
    if verify_hub is None:
        verify_hub = revision is not None and release_entry is None
    verifier = _hub_verifier(os.environ.get("HF_TOKEN")) if verify_hub else None

    try:
        manifest = build_manifest(
            args.artifact_dir,
            revision,
            package_manifest_path=args.package_manifest,
            release_entry=release_entry,
            hub_verifier=verifier,
        )
    except ReleaseError as error:
        raise SystemExit(str(error)) from error

    contract = contract_document()
    _write_json(CONTRACT_PATH, contract)
    _write_json(MANIFEST_PATH, manifest)

    if revision is not None and release_entry is None:
        entry = build_release_entry(
            revision,
            manifest["models"],
            package_manifest=package,
            package_manifest_path=args.package_manifest,
        )
        history = record_release(history, entry)
    if not args.history.exists() or (revision is not None and release_entry is None):
        write_release_history(args.history, history)

    print(f"Wrote {CONTRACT_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    for name, asset in manifest["models"].items():
        print(f"  {name}: {asset['sha256']} ({asset['size']} bytes)")
        if asset["hubRevision"] is None:
            print("    local-bundle only (no immutable Hub revision pinned)")
        else:
            print(f"    pinned Hub revision {asset['hubRevision']}")
    if args.rollback:
        print(f"Rolled back to release {revision} from {args.history}")
    elif revision is not None:
        print(f"Recorded release {revision} in {args.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
