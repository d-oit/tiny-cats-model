#!/usr/bin/env python3
"""Validate an Oxford-style cat/not-cat dataset and optionally write its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_manifest import (  # noqa: E402
    DATASET_ID,
    IMAGE_EXTENSIONS,
    build_dataset_manifest,
    compute_manifest_hash,
    write_dataset_manifest,
)
from model_contract import CAT_BREEDS  # noqa: E402


def validate_dataset(
    root: Path,
    *,
    expected_breeds: int | None = None,
    manifest_path: Path | None = None,
) -> list[str]:
    """Return actionable validation errors for the dataset at ``root``."""
    errors: list[str] = []
    if not root.is_dir():
        return [f"Dataset root does not exist or is not a directory: {root}"]

    class_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    required = {"cat", "other"}
    if not required <= class_dirs:
        errors.append(
            f"Required class directories missing: {', '.join(sorted(required - class_dirs))}"
        )
    unexpected = class_dirs - required
    if unexpected:
        errors.append(f"Unexpected class directories: {', '.join(sorted(unexpected))}")

    breed_counts: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    valid_samples = 0
    for folder in ("cat", "other"):
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if path.parent != directory:
                errors.append(
                    f"Samples must be directly inside class directories: {relative}"
                )
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                errors.append(f"Non-image file in dataset class directory: {relative}")
                continue
            try:
                size = path.stat().st_size
            except OSError as exc:
                errors.append(f"Cannot stat {relative}: {exc}")
                continue
            if size <= 0:
                errors.append(f"Empty image file: {relative}")
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                errors.append(f"Undecodable image {relative}: {exc}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                errors.append(
                    f"Duplicate image content: {relative} matches {hashes[digest]}"
                )
            else:
                hashes[digest] = relative
            valid_samples += 1
            if folder == "cat":
                prefix = path.stem.rsplit("_", 1)[0] if "_" in path.stem else ""
                if prefix not in CAT_BREEDS:
                    errors.append(
                        f"Unknown cat-breed filename prefix in {relative}; expected one of: "
                        + ", ".join(CAT_BREEDS)
                    )
                else:
                    breed_counts[prefix] += 1

    missing_breeds = sorted(set(CAT_BREEDS) - set(breed_counts))
    if missing_breeds:
        errors.append(f"Cat breeds have no valid image: {', '.join(missing_breeds)}")
    if expected_breeds is not None and expected_breeds != len(CAT_BREEDS):
        errors.append(
            f"Expected {expected_breeds} breeds, but the dataset contract defines {len(CAT_BREEDS)}"
        )
    if valid_samples == 0:
        errors.append("Dataset contains no decodable image samples")

    if manifest_path is not None:
        try:
            supplied = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read manifest {manifest_path}: {exc}")
        else:
            current = build_dataset_manifest(root, use_cache=False, persist_cache=False)
            saved_hash = (
                supplied.get("manifest_hash") if isinstance(supplied, dict) else None
            )
            if not isinstance(saved_hash, str) or saved_hash != compute_manifest_hash(
                current
            ):
                errors.append(
                    f"Manifest does not match current dataset contents: {manifest_path}"
                )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, help="Manifest JSON to verify against the dataset"
    )
    parser.add_argument("--expected-breeds", type=int)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--version")
    args = parser.parse_args(argv)

    errors = validate_dataset(
        args.root,
        expected_breeds=args.expected_breeds,
        manifest_path=args.manifest,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.write_manifest:
        manifest = build_dataset_manifest(
            args.root,
            dataset_id=DATASET_ID,
            source=args.source,
            version=args.version,
        )
        write_dataset_manifest(args.root, manifest)
    print(f"Dataset validation passed: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
