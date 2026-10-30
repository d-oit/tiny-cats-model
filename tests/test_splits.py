"""Coverage for deterministic split manifests and their evaluation integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import cats_eval_loader
from splits import (
    SplitManifestError,
    build_split_manifest,
    split_entries,
)


def _dataset(root: Path, cat_count: int = 10, other_count: int = 10) -> None:
    for label, count in (("cat", cat_count), ("other", other_count)):
        directory = root / label
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (8, 8), color=(index, 0, 0)).save(
                directory / f"{index}.jpg"
            )


def test_split_assignment_is_deterministic_disjoint_and_complete() -> None:
    entries = [
        {"path": f"{label}/{index}.jpg", "label": label}
        for label in ("cat", "other")
        for index in range(20)
    ]
    first = split_entries(entries, seed=7, val_split=0.2, test_split=0.1)
    second = split_entries(entries, seed=7, val_split=0.2, test_split=0.1)
    changed = split_entries(entries, seed=8, val_split=0.2, test_split=0.1)
    assert first == second
    assert {item["path"]: item["split"] for item in first} != {
        item["path"]: item["split"] for item in changed
    }
    assigned: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    for item in first:
        assigned[item["split"]].add(item["path"])
    assert not (assigned["train"] & assigned["val"])
    assert not (assigned["train"] & assigned["test"])
    assert not (assigned["val"] & assigned["test"])
    assert set.union(*assigned.values()) == {item["path"] for item in entries}
    for label in ("cat", "other"):
        stratum = [item for item in first if item["label"] == label]
        fraction = sum(item["split"] == "val" for item in stratum) / len(stratum)
        assert fraction == pytest.approx(0.2, abs=0.05)


def test_split_entries_validates_fractions_and_records_absent_classes(
    tmp_path: Path,
) -> None:
    with pytest.raises(SplitManifestError, match=r"below 1\.0"):
        split_entries(
            [{"path": "cat/a", "label": "cat"}], val_split=0.6, test_split=0.4
        )
    with pytest.raises(SplitManifestError, match=r"\[0, 1\)"):
        split_entries([{"path": "cat/a", "label": "cat"}], val_split=-0.1)
    root = tmp_path / "dataset"
    _dataset(root, cat_count=3, other_count=0)
    manifest = build_split_manifest(root, val_split=0.2)
    assert manifest["absent_classes"] == ["other"]


def test_eval_rejects_dataset_hash_mismatch_and_requires_test_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    _dataset(root)
    manifest = build_split_manifest(root, val_split=0.2, test_split=0.2)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest))
    other_root = tmp_path / "different"
    _dataset(other_root)
    Image.new("RGB", (8, 8)).save(other_root / "cat" / "extra.jpg")
    with pytest.raises(ValueError, match="fingerprint"):
        cats_eval_loader(other_root, split_manifest=manifest_path)
    with pytest.raises(ValueError, match="requires"):
        cats_eval_loader(root, split="test")
    loader, samples, names, provenance = cats_eval_loader(
        root, split="test", split_manifest=manifest_path, num_workers=0
    )
    assert len(loader.dataset) == len(samples) > 0
    assert names == ["cat", "not_cat"]
    assert provenance["source"] == "test"
