"""scripts/promote_frontend_release.py

Pin the browser to the immutable Hub revision that was just uploaded (#204).

Resolves the repository's current commit SHA through ``huggingface_hub`` (or
accepts an explicit ``--revision``), then delegates to
``scripts/generate_frontend_metadata.py`` with ``--revision`` and Hub byte
verification. The generated manifest is only written after every uploaded
artifact's SHA-256 and size has been checked, and the promotion is appended to
the release history.

Usage:
    HF_TOKEN=... python scripts/promote_frontend_release.py \\
        --package-manifest artifacts/manifest.json
    python scripts/promote_frontend_release.py --revision <40-hex> --no-verify-hub
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_frontend_metadata as generator  # noqa: E402

from model_release import ReleaseError, validate_revision  # noqa: E402

DEFAULT_REPO_ID = "d4oit/tiny-cats-model"


def resolve_hub_revision(repo_id: str, token: str | None = None) -> str:
    """Return the repository's current main commit as an immutable revision."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - dependency error
        raise ReleaseError(
            "huggingface_hub is required to resolve the uploaded revision"
        ) from error
    info = HfApi(token=token).repo_info(repo_id, revision="main")
    sha = getattr(info, "sha", None)
    if not sha:
        raise ReleaseError(f"Hub reported no commit SHA for {repo_id}")
    return validate_revision(str(sha))


def build_arguments(args: argparse.Namespace, revision: str) -> list[str]:
    """Translate promotion options into generator arguments."""
    command = [
        "--artifact-dir",
        str(args.artifact_dir),
        "--revision",
        revision,
    ]
    if args.package_manifest is not None:
        command += ["--package-manifest", str(args.package_manifest)]
    if args.history is not None:
        command += ["--history", str(args.history)]
    if args.no_verify_hub:
        command.append("--no-verify-hub")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable commit SHA to pin (default: the Hub's current main commit)",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=generator.DEFAULT_ARTIFACT_DIR
    )
    parser.add_argument("--package-manifest", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument(
        "--no-verify-hub",
        action="store_true",
        help="Skip downloading the pinned revision (offline escape hatch)",
    )
    args = parser.parse_args(argv)

    try:
        if args.revision is not None:
            revision = validate_revision(args.revision)
        else:
            revision = resolve_hub_revision(args.repo_id, os.environ.get("HF_TOKEN"))
    except ReleaseError as error:
        raise SystemExit(str(error)) from error

    print(f"Pinning the browser to immutable Hub revision {revision}")
    return generator.main(build_arguments(args, revision))


if __name__ == "__main__":
    raise SystemExit(main())
