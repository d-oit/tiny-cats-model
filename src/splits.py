"""src/splits.py

Deterministic, stratified train/validation/test manifests (issue #200).

``cats_dataloader`` used ``torch.utils.data.random_split`` at every call site:
the split was not stratified, evaluation recreated it from scratch, and the
"test" set was really a validation fold touched during checkpoint selection.
This module persists one split manifest per dataset version so training,
evaluation and quantization calibration can all prove which samples they saw.

Guarantees:
* deterministic for a given ``(dataset, seed, val_split, test_split, version)``
  and independent of filesystem traversal order,
* stratified by class (each class keeps its proportion in every split),
* partitions are disjoint and their union is exactly the manifest's entries,
* generation/evaluation never silently falls back to another split.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from dataset_manifest import (
    DATASET_ID,
    build_dataset_manifest,
    canonical_json,
)
from model_contract import CLASSIFIER_LABELS

SPLIT_SCHEMA_VERSION = 1
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


class SplitManifestError(ValueError):
    """A split manifest is missing, inconsistent, or does not cover the data."""


def _rng(seed: int, group: str, version: str) -> random.Random:
    """Stable RNG independent of Python's per-process hash randomisation."""
    material = f"{seed}:{group}:{version}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _allocate(
    group_entries: list[dict[str, Any]], val_split: float, test_split: float
) -> dict[str, list[dict[str, Any]]]:
    """Split one stratum deterministically; never overlap or drop entries."""
    n = len(group_entries)
    test_n = round(n * max(test_split, 0.0))
    val_n = round(n * max(val_split, 0.0))
    if val_n + test_n > n:
        # Tiny strata: prefer a non-empty train split over exact ratios.
        overflow = val_n + test_n - n
        val_n = max(0, val_n - overflow)
    train_n = n - val_n - test_n
    return {
        "train": group_entries[:train_n],
        "val": group_entries[train_n : train_n + val_n],
        "test": group_entries[train_n + val_n :],
    }


def split_entries(
    entries: list[dict[str, Any]],
    *,
    seed: int = 42,
    val_split: float = 0.2,
    test_split: float = 0.0,
    version: str = "v1",
    stratify_by: str = "label",
) -> list[dict[str, Any]]:
    """Assign a ``split`` to each entry, stratified and deterministically.

    Args:
        entries: Dataset-manifest entries with ``path`` and ``label``/``breed``.
        seed: Split seed.
        val_split: Fraction per stratum held out for validation.
        test_split: Fraction per stratum held out for test.
        version: Split-policy version string; changing it changes the split.
        stratify_by: ``"label"`` (classifier) or ``"breed"`` (generator).

    Returns:
        New entries with a ``split`` field, sorted by path.

    Raises:
        SplitManifestError: On invalid fractions or unknown strate.
    """
    if not (0.0 <= val_split < 1.0) or not (0.0 <= test_split < 1.0):
        raise SplitManifestError("val_split and test_split must be in [0, 1)")
    if val_split + test_split >= 1.0:
        raise SplitManifestError("val_split + test_split must be below 1.0")
    if stratify_by not in {"label", "breed"}:
        raise SplitManifestError(f"unknown stratify_by: {stratify_by!r}")

    strata: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        key = entry.get(stratify_by)
        key = "__unlabeled__" if key is None else str(key)
        strata.setdefault(key, []).append(dict(entry))

    assigned: list[dict[str, Any]] = []
    for group in sorted(strata):
        group_entries = sorted(strata[group], key=lambda item: item["path"])
        _rng(seed, group, version).shuffle(group_entries)
        allocated = _allocate(group_entries, val_split, test_split)
        for split in SPLIT_NAMES:
            for entry in allocated[split]:
                entry["split"] = split
                assigned.append(entry)

    assigned.sort(key=lambda item: item["path"])
    return assigned


def build_split_manifest(
    root: str | Path,
    *,
    seed: int = 42,
    val_split: float = 0.2,
    test_split: float = 0.0,
    version: str = "v1",
    stratify_by: str = "label",
    dataset_id: str = DATASET_ID,
) -> dict[str, Any]:
    """Build a split manifest from the dataset on disk."""
    dataset = build_dataset_manifest(root, dataset_id=dataset_id)
    if dataset.get("status") == "missing":
        raise SplitManifestError(f"Dataset root does not exist: {root}")
    entries = split_entries(
        dataset["entries"],
        seed=seed,
        val_split=val_split,
        test_split=test_split,
        version=version,
        stratify_by=stratify_by,
    )
    counts = {
        split: sum(1 for entry in entries if entry["split"] == split)
        for split in SPLIT_NAMES
    }
    # Classifier folder "other" is labelled "not_cat" in reports; compare
    # canonical dataset labels instead of the public label spelling.
    absent = sorted(
        folder
        for folder in ("cat", "other")
        if folder not in {str(entry.get("label")) for entry in entries}
    )
    manifest: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_hash": dataset["manifest_hash"],
        "dataset_hash_version": dataset.get("dataset_hash_version"),
        "seed": int(seed),
        "val_split": float(val_split),
        "test_split": float(test_split),
        "version": version,
        "stratify_by": stratify_by,
        "counts": counts,
        "absent_classes": absent,
        "entries": entries,
    }
    manifest["split_hash"] = compute_split_hash(manifest)
    return manifest


def compute_split_hash(manifest: dict[str, Any]) -> str:
    """Canonical hash of a split assignment."""
    identity = {
        "schema_version": manifest.get("schema_version"),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_hash": manifest.get("dataset_hash"),
        "seed": manifest.get("seed"),
        "val_split": manifest.get("val_split"),
        "test_split": manifest.get("test_split"),
        "version": manifest.get("version"),
        "stratify_by": manifest.get("stratify_by"),
        "entries": [
            {"path": entry.get("path"), "split": entry.get("split")}
            for entry in manifest.get("entries", [])
        ],
    }
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def validate_split_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate schema, disjointness, coverage and recorded counts.

    Raises:
        SplitManifestError: On any inconsistency.
    """
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise SplitManifestError(
            f"unsupported split manifest schema: {manifest.get('schema_version')!r}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SplitManifestError("split manifest has no entries")
    seen: set[str] = set()
    counts = dict.fromkeys(SPLIT_NAMES, 0)
    for entry in entries:
        path = entry.get("path")
        split = entry.get("split")
        if not isinstance(path, str) or split not in SPLIT_NAMES:
            raise SplitManifestError(f"invalid split entry: {entry!r}")
        if path in seen:
            raise SplitManifestError(f"sample appears in more than one split: {path}")
        seen.add(path)
        counts[split] += 1
    recorded = manifest.get("counts")
    if recorded != counts:
        raise SplitManifestError(
            f"recorded split counts {recorded} disagree with entries {counts}"
        )
    expected_hash = compute_split_hash(manifest)
    if manifest.get("split_hash") != expected_hash:
        raise SplitManifestError("split_hash does not match the manifest contents")
    return manifest


def write_split_manifest(root: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write ``split_manifest.json`` in the dataset root."""
    validate_split_manifest(manifest)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / SPLIT_MANIFEST_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def read_split_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a split manifest file."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SplitManifestError(f"Split manifest not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitManifestError(f"Unreadable split manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SplitManifestError(f"Split manifest {path} is not an object")
    return validate_split_manifest(manifest)


def split_paths(manifest: dict[str, Any], split: str) -> list[str]:
    """Relative paths assigned to ``split`` (empty when absent)."""
    if split not in SPLIT_NAMES:
        raise SplitManifestError(f"unknown split {split!r}")
    return [
        str(entry["path"])
        for entry in manifest["entries"]
        if entry.get("split") == split
    ]


class ManifestDataset:
    """Dataset over one persisted split, without relying on ImageFolder order.

    Kept dependency-light (torchvision imported lazily) so the download
    validator and split tests can import this module without torch.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: dict[str, Any],
        split: str,
        transform: Any | None = None,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise SplitManifestError(f"unknown split {split!r}")
        from preprocessing import load_classifier_image

        self.root = Path(root)
        self.loader = load_classifier_image
        self.transform = transform
        self.classes = list(CLASSIFIER_LABELS)
        self.samples: list[tuple[Path, int]] = []
        for entry in manifest["entries"]:
            if entry.get("split") != split:
                continue
            label_index = entry.get("label_index")
            if label_index is None:
                continue
            self.samples.append((self.root / str(entry["path"]), int(label_index)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def split_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Provenance summary embedded in reports/checkpoints."""
    return {
        "schema_version": manifest.get("schema_version"),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_hash": manifest.get("dataset_hash"),
        "split_version": manifest.get("version"),
        "seed": manifest.get("seed"),
        "val_split": manifest.get("val_split"),
        "test_split": manifest.get("test_split"),
        "stratify_by": manifest.get("stratify_by"),
        "counts": manifest.get("counts"),
        "absent_classes": manifest.get("absent_classes"),
        "split_hash": manifest.get("split_hash"),
    }


__all__ = [
    "SPLIT_MANIFEST_FILENAME",
    "SPLIT_NAMES",
    "SPLIT_SCHEMA_VERSION",
    "ManifestDataset",
    "SplitManifestError",
    "build_split_manifest",
    "compute_split_hash",
    "read_split_manifest",
    "split_entries",
    "split_manifest_summary",
    "split_paths",
    "validate_split_manifest",
    "write_split_manifest",
]
