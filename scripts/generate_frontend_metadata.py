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
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_manifest import (  # noqa: E402
    HUB_BASE_URL,
    HUB_REPO_ID,
    HubVerifier,
    build_manifest,
    contract_document,
)
from model_release import (  # noqa: E402
    ReleaseError,
    build_release_entry,
    fetch_hub_bytes,
    read_package_manifest,
    read_release_history,
    record_release,
    resolve_release,
    verify_hub_artifact,
    write_release_history,
)

GENERATED_DIR = ROOT / "frontend" / "src" / "generated"
CONTRACT_PATH = GENERATED_DIR / "modelContract.json"
MANIFEST_PATH = GENERATED_DIR / "modelManifest.json"
HISTORY_PATH = GENERATED_DIR / "modelReleases.json"
DEFAULT_ARTIFACT_DIR = ROOT / "frontend" / "public" / "models"


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
    try:
        import onnx  # noqa: F401
    except ImportError:
        print(
            "STALE: onnx is required to verify the generated ONNX metadata "
            "(pip install onnx)"
        )
        return 1
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
