"""src/dataset_manifest.py

Content-addressed dataset identity (issues #202, #206).

The old fingerprint hashed ``relpath:size``. Replacing an image with different
bytes but the same byte length therefore kept the same experiment identity and
let a resume continue on a silently different dataset. This module builds a
canonical manifest with per-file SHA-256 hashes, derives the fingerprint from
that manifest, and caches per-file hashes keyed by ``(size, mtime_ns)`` so a
resumed provider slice does not re-read every image.

The manifest is also what ``data/download.sh`` publishes after a successful
staging validation, giving download, training and resume one identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_contract import CAT_BREEDS, CLASSIFIER_FOLDER_TO_INDEX, OTHER_CLASS_INDEX

#: Dataset identity schemes. ``path-size-v1`` is the legacy fingerprint and is
#: only ever accepted through an explicit migration path.
DATASET_HASH_VERSION = "sha256-content-v2"
LEGACY_DATASET_HASH_VERSION = "path-size-v1"

DATASET_ID = "oxford-iiit-pet-cats"
MANIFEST_FILENAME = "dataset_manifest.json"
MANIFEST_SCHEMA_VERSION = 2
HASH_CACHE_FILENAME = ".dataset_hash_cache.json"

#: Exact dataset file allowlist. Everything else (reports, README, manifests,
#: caches, hidden files) is excluded so metadata churn cannot change identity.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
)

#: Files in the dataset root that are never dataset samples.
_TRANSIENT_NAMES: frozenset[str] = frozenset(
    {
        MANIFEST_FILENAME,
        HASH_CACHE_FILENAME,
        "download.lock",
        "README.md",
        ".DS_Store",
    }
)


def canonical_json(value: Any) -> str:
    """Canonical JSON for hashes (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cache(root: Path) -> dict[str, dict[str, Any]]:
    cache_path = root / HASH_CACHE_FILENAME
    try:
        document = json.loads(cache_path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return {}
    files = document.get("files")
    return files if isinstance(files, dict) else {}


def _write_cache(root: Path, files: dict[str, dict[str, Any]]) -> None:
    cache_path = root / HASH_CACHE_FILENAME
    document = {"schema_version": 1, "files": files}
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, cache_path)
    except OSError:
        # A read-only dataset volume is fine: hashing still works, just slower.
        tmp.unlink(missing_ok=True)


def iter_dataset_files(root: str | Path) -> list[Path]:
    """Return allowlisted dataset files, sorted by relative POSIX path.

    Hidden directories/files, manifests, caches and non-image extensions are
    excluded. Sorting makes the identity independent of filesystem traversal
    order.
    """
    root = Path(root)
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:  # pragma: no cover - rglob always stays below root
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        if relative.as_posix() in _TRANSIENT_NAMES:
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _label_for(relative: Path) -> tuple[str | None, str | None]:
    """Derive (label folder, breed) from a relative path."""
    if len(relative.parts) < 2:
        return None, None
    folder = relative.parts[0]
    breed: str | None = None
    if folder == "cat":
        for candidate in CAT_BREEDS:
            if relative.name.startswith(f"{candidate}_"):
                breed = candidate
                break
    return folder, breed


def _label_index(folder: str | None) -> int | None:
    if folder is None:
        return None
    if folder in CLASSIFIER_FOLDER_TO_INDEX:
        return CLASSIFIER_FOLDER_TO_INDEX[folder]
    if folder == "other":
        return OTHER_CLASS_INDEX
    return None


def build_dataset_manifest(
    root: str | Path,
    *,
    dataset_id: str = DATASET_ID,
    source: str | None = None,
    version: str | None = None,
    use_cache: bool = True,
    persist_cache: bool = True,
) -> dict[str, Any]:
    """Build a normalized dataset manifest with per-file content hashes.

    Args:
        root: Dataset root (``data/cats``).
        dataset_id: Stable logical dataset identifier.
        source: Download source URL/release, when known.
        version: Human-readable dataset release version.
        use_cache: Reuse hashes from ``.dataset_hash_cache.json`` when
            ``(size, mtime_ns)`` still match.
        persist_cache: Write the refreshed cache back to the dataset root.

    Returns:
        Manifest dictionary; also includes ``manifest_hash`` and
        ``dataset_hash`` (equal, kept for readability).
    """
    root = Path(root)
    if not root.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "dataset_hash_version": DATASET_HASH_VERSION,
            "status": "missing",
            "entries": [],
            "manifest_hash": "missing",
        }

    cache = _load_cache(root) if use_cache else {}
    refreshed: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []

    for path in iter_dataset_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
        except OSError:
            continue
        cached = cache.get(relative)
        if (
            isinstance(cached, dict)
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            digest = cached["sha256"]
        else:
            try:
                digest = _sha256_file(path)
            except OSError:
                continue
        refreshed[relative] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
        folder, breed = _label_for(Path(relative))
        entries.append(
            {
                "path": relative,
                "label": folder,
                "label_index": _label_index(folder),
                "breed": breed,
                "size": stat.st_size,
                "sha256": digest,
            }
        )

    if persist_cache and use_cache:
        _write_cache(root, refreshed)

    entries.sort(key=lambda entry: entry["path"])
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_hash_version": DATASET_HASH_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "version": version,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "label_counts": _label_counts(entries),
        "breed_counts": _breed_counts(entries),
        "entries": entries,
    }
    manifest_hash = compute_manifest_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    manifest["dataset_hash"] = manifest_hash
    return manifest


def _label_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        label = entry.get("label")
        if label is None:
            continue
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _breed_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        breed = entry.get("breed")
        if breed is None:
            continue
        counts[breed] = counts.get(breed, 0) + 1
    return dict(sorted(counts.items()))


def compute_manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash the identity-bearing subset of a manifest.

    Timestamps, download provenance and cache metadata are excluded: the
    identity is the exact file set, labels and bytes.
    """
    identity = {
        "schema_version": manifest.get("schema_version"),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_hash_version": manifest.get("dataset_hash_version"),
        "entries": [
            {
                "path": entry.get("path"),
                "label_index": entry.get("label_index"),
                "size": entry.get("size"),
                "sha256": entry.get("sha256"),
            }
            for entry in manifest.get("entries", [])
        ],
    }
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def write_dataset_manifest(root: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write ``dataset_manifest.json`` into the dataset root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def read_dataset_manifest(root: str | Path) -> dict[str, Any] | None:
    """Read a persisted manifest; ``None`` when absent or corrupt."""
    path = Path(root) / MANIFEST_FILENAME
    try:
        document = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def dataset_fingerprint(data_dir: str | Path, *, use_cache: bool = True) -> str:
    """Content-based dataset fingerprint (``sha256-content-v2``)."""
    root = Path(data_dir)
    if not root.exists():
        return "missing"
    manifest = build_dataset_manifest(root, use_cache=use_cache, persist_cache=False)
    return str(manifest["manifest_hash"])


def legacy_path_size_fingerprint(data_dir: str | Path) -> str:
    """The pre-#206 ``relpath:size`` fingerprint, kept for explicit migration.

    Never use this for new checkpoints. It exists so an operator can prove an
    existing resumable checkpoint was trained on the current dataset before
    accepting it with ``--allow-legacy-dataset-hash``.
    """
    root = Path(data_dir)
    if not root.exists():
        return "missing"
    entries: list[str] = []
    try:
        for item in sorted(root.rglob("*")):
            if item.is_file():
                try:
                    entries.append(
                        f"{item.relative_to(root).as_posix()}:{item.stat().st_size}"
                    )
                except OSError:
                    continue
    except OSError:
        return "unreadable"
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def is_legacy_identity(saved_hash: str, data_dir: str | Path) -> bool:
    """Whether ``saved_hash`` is the legacy path:size fingerprint of ``data_dir``."""
    return bool(saved_hash) and saved_hash == legacy_path_size_fingerprint(data_dir)


__all__ = [
    "DATASET_HASH_VERSION",
    "DATASET_ID",
    "HASH_CACHE_FILENAME",
    "IMAGE_EXTENSIONS",
    "LEGACY_DATASET_HASH_VERSION",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "build_dataset_manifest",
    "canonical_json",
    "compute_manifest_hash",
    "dataset_fingerprint",
    "is_legacy_identity",
    "iter_dataset_files",
    "legacy_path_size_fingerprint",
    "read_dataset_manifest",
    "write_dataset_manifest",
]
