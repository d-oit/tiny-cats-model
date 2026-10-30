"""Tests for content-hash dataset identity and cache behavior."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dataset_manifest as dm


def test_same_size_replacement_changes_fingerprint(tmp_path: Path) -> None:
    (tmp_path / "cat").mkdir()
    sample = tmp_path / "cat" / "Abyssinian_1.jpg"
    sample.write_bytes(b"bytes-one")
    before = dm.dataset_fingerprint(tmp_path, use_cache=False)
    sample.write_bytes(b"bytes-two")
    assert dm.dataset_fingerprint(tmp_path, use_cache=False) != before


def test_traversal_and_order_are_stable(tmp_path: Path, monkeypatch) -> None:
    for relative in ("cat/Bengal_1.jpg", "other/dog.jpg", "cat/Abyssinian_1.jpg"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    expected = dm.dataset_fingerprint(tmp_path, use_cache=False)
    original = dm.iter_dataset_files
    monkeypatch.setattr(
        dm, "iter_dataset_files", lambda root: list(reversed(original(root)))
    )
    assert dm.dataset_fingerprint(tmp_path, use_cache=False) == expected


def test_hidden_transient_and_non_image_files_are_excluded(tmp_path: Path) -> None:
    (tmp_path / "cat").mkdir()
    sample = tmp_path / "cat" / "Abyssinian_1.jpg"
    sample.write_bytes(b"sample")
    before = dm.dataset_fingerprint(tmp_path, use_cache=False)
    (tmp_path / "cat" / ".hidden.jpg").write_bytes(b"hidden")
    (tmp_path / "cat" / "notes.txt").write_text("ignored")
    (tmp_path / dm.HASH_CACHE_FILENAME).write_text("cache")
    (tmp_path / dm.MANIFEST_FILENAME).write_text("manifest")
    assert dm.dataset_fingerprint(tmp_path, use_cache=False) == before


def test_cache_reuse_and_invalidation(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "cat" / "Abyssinian_1.jpg"
    sample.parent.mkdir()
    sample.write_bytes(b"same-size")
    calls = 0
    original = dm._sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(dm, "_sha256_file", counted)
    dm.build_dataset_manifest(tmp_path)
    assert calls == 1
    dm.build_dataset_manifest(tmp_path)
    assert calls == 1
    stat = sample.stat()
    sample.write_bytes(b"diff-size")
    dm.build_dataset_manifest(tmp_path)
    assert calls == 2
    sample.write_bytes(b"other-size")
    os_utime = __import__("os").utime
    os_utime(sample, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    dm.build_dataset_manifest(tmp_path)
    assert calls == 3


def test_missing_root_and_legacy_identity(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert dm.dataset_fingerprint(missing) == "missing"
    root = tmp_path / "legacy"
    (root / "cat").mkdir(parents=True)
    (root / "cat" / "Abyssinian_1.jpg").write_bytes(b"a")
    (root / "other.jpg").write_bytes(b"b")
    entries = [
        f"{path.relative_to(root).as_posix()}:{path.stat().st_size}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    old_hash = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    assert dm.legacy_path_size_fingerprint(root) == old_hash
    assert dm.is_legacy_identity(old_hash, root)
    assert not dm.is_legacy_identity("not-the-hash", root)


def test_manifest_labels_and_canonical_hash(tmp_path: Path) -> None:
    (tmp_path / "cat").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "cat" / "Abyssinian_1.jpg").write_bytes(b"cat")
    (tmp_path / "other" / "x.jpg").write_bytes(b"dog")
    manifest = dm.build_dataset_manifest(tmp_path, use_cache=False, persist_cache=False)
    by_path = {entry["path"]: entry for entry in manifest["entries"]}
    assert by_path["cat/Abyssinian_1.jpg"]["label_index"] == 0
    assert by_path["cat/Abyssinian_1.jpg"]["breed"] == "Abyssinian"
    assert by_path["other/x.jpg"]["label_index"] == 1
    assert by_path["other/x.jpg"]["breed"] is None
    assert dm.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert manifest["manifest_hash"] == dm.compute_manifest_hash(manifest)
    assert json.loads(dm.canonical_json(manifest)) == manifest
