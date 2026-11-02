"""src/model_release.py

Immutable model release records for the browser bundle (issue #204).

The frontend must never run a model it cannot trace to a release. This module
provides the Python half of that contract:

* :func:`validate_revision` accepts only a full 40-character lowercase Hub
  commit SHA — never ``main``, a tag, or a short SHA.
* :func:`read_package_manifest` / :func:`find_package_file` read the verified
  artifact package manifest written by :mod:`artifacts`, so the frontend
  manifest is generated from package output instead of re-hashed guesses.
* :func:`verify_package_file` rejects a stale bundled artifact whose bytes no
  longer match the release record.
* :func:`verify_hub_artifact` downloads a pinned Hub revision and checks its
  SHA-256 and size before the revision is published to the frontend.
* :func:`record_release` / :func:`find_release` keep an append-only release
  history (``frontend/src/generated/modelReleases.json``) so a model update is
  traceable independently of a frontend release and can be rolled back.

Everything here is stdlib-only so the generators, CI and tests can import it
without torch/onnx/huggingface_hub installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: A Hub revision must be a full 40-character lowercase commit SHA. Branch
#: names, tags, short SHAs and uppercase variants are mutable or ambiguous.
HUB_REVISION = re.compile(r"^[0-9a-f]{40}$")

#: SHA-256 digest as recorded in every manifest (lowercase hex).
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

#: Schema version of ``frontend/src/generated/modelReleases.json``.
RELEASE_HISTORY_SCHEMA_VERSION = 1

#: Receives a URL and returns the response body.
FetchBytes = Callable[[str], bytes]


class ReleaseError(RuntimeError):
    """A model release is missing, stale or does not match its manifest."""


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed so large artifacts stay in check."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory artifact."""
    return hashlib.sha256(data).hexdigest()


def validate_revision(revision: str) -> str:
    """Return ``revision`` when it is an immutable commit SHA.

    Raises:
        ReleaseError: for branch names (``main``), tags, short SHAs, or any
            value that is not 40 lowercase hex characters.
    """
    if not isinstance(revision, str) or not HUB_REVISION.fullmatch(revision):
        raise ReleaseError(
            "Hub revision must be a full 40-character lowercase commit SHA "
            f"(got {revision!r}); never use a branch name, tag or short SHA"
        )
    return revision


def _require_digest(sha256: Any, label: str) -> str:
    if not isinstance(sha256, str) or not SHA256_HEX.fullmatch(sha256.lower()):
        raise ReleaseError(f"{label} has no valid SHA-256 digest: {sha256!r}")
    return sha256.lower()


def _require_size(size: Any, label: str) -> int:
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ReleaseError(f"{label} has no valid byte size: {size!r}")
    return size


def read_package_manifest(path: str | Path) -> dict[str, Any]:
    """Read a verified artifact package manifest (``artifacts/manifest.json``).

    Raises:
        ReleaseError: when the document is missing, unreadable, or any file
            record lacks a path, SHA-256 or byte size.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ReleaseError(f"Artifact package manifest missing: {manifest_path}")
    try:
        document = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(
            f"Cannot read artifact package manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(document, dict) or not isinstance(document.get("files"), list):
        raise ReleaseError(
            f"Artifact package manifest has no 'files' list: {manifest_path}"
        )
    for entry in document["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ReleaseError(
                f"Artifact package manifest has an entry without a path: {entry!r}"
            )
        _require_digest(entry.get("sha256"), f"Package file {entry['path']}")
        _require_size(entry.get("bytes"), f"Package file {entry['path']}")
    return document


def package_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Map POSIX-normalized package paths to their file records."""
    index: dict[str, Mapping[str, Any]] = {}
    for entry in manifest.get("files", []):
        key = str(entry["path"]).replace("\\", "/")
        index[key] = entry
    return index


def find_package_file(manifest: Mapping[str, Any], *parts: str) -> Mapping[str, Any]:
    """Find one package file by exact path or path suffix.

    ``find_package_file(manifest, "generator", "model_quantized.onnx")`` matches
    both ``artifacts/generator/model_quantized.onnx`` and a package produced with
    a different staging root, while still refusing ambiguous matches.
    """
    suffix = "/".join(part.strip("/") for part in parts if part)
    if not suffix:
        raise ReleaseError("Package file lookup needs a path or suffix")
    matches = [
        entry
        for key, entry in package_index(manifest).items()
        if key == suffix or key.endswith(f"/{suffix}")
    ]
    if not matches:
        raise ReleaseError(f"Artifact package has no file matching {suffix!r}")
    if len(matches) > 1:
        paths = sorted(str(entry["path"]) for entry in matches)
        raise ReleaseError(
            f"Artifact package has {len(matches)} files matching {suffix!r}: "
            + ", ".join(paths)
        )
    return matches[0]


def verify_artifact_bytes(
    data: bytes,
    *,
    sha256: str,
    size: int,
    label: str = "model artifact",
) -> str:
    """Verify fetched bytes against an expected digest and size.

    Returns:
        The verified lowercase SHA-256.

    Raises:
        ReleaseError: on a size or digest mismatch (a stale or tampered
            artifact is always a hard error, never a silent fallback).
    """
    expected_digest = _require_digest(sha256, label)
    expected_size = _require_size(size, label)
    if len(data) != expected_size:
        raise ReleaseError(
            f"{label} size mismatch: expected {expected_size}, got {len(data)}"
        )
    actual = sha256_bytes(data)
    if actual != expected_digest:
        raise ReleaseError(
            f"{label} digest mismatch: expected {expected_digest}, got {actual}"
        )
    return actual


def verify_package_file(
    record: Mapping[str, Any],
    path: str | Path,
    *,
    label: str | None = None,
) -> str:
    """Verify a bundled artifact against its verified-package record."""
    artifact = Path(path)
    name = label or f"Package file {record.get('path', artifact.name)}"
    if not artifact.exists():
        raise ReleaseError(f"{name} is missing: {artifact}")
    return verify_artifact_bytes(
        artifact.read_bytes(),
        sha256=str(record.get("sha256", "")),
        size=record.get("bytes", 0),  # type: ignore[arg-type]
        label=name,
    )


def hub_url(base_url: str, revision: str, hub_path: str) -> str:
    """Build an immutable ``resolve/<sha>`` URL; never ``resolve/main``."""
    validate_revision(revision)
    base = base_url if base_url.endswith("/") else f"{base_url}/"
    path = hub_path.lstrip("/")
    return f"{base}{revision}/{path}"


def fetch_hub_bytes(
    url: str, token: str | None = None, timeout: float = 120.0
) -> bytes:
    """Download a Hub artifact, authenticating when a token is available."""
    headers = {"User-Agent": "tiny-cats-model-release"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _revision_from_url(url: str) -> str:
    marker = "/resolve/"
    if marker not in url:
        raise ReleaseError(f"Not an immutable Hub resolve URL: {url}")
    remainder = url.split(marker, 1)[1]
    return remainder.split("/", 1)[0]


def verify_hub_artifact(
    url: str,
    *,
    sha256: str,
    size: int,
    fetch: FetchBytes | None = None,
    label: str = "Hub artifact",
) -> str:
    """Download a pinned Hub artifact and verify its bytes.

    Raises:
        ReleaseError: when the URL is not revision-pinned, the download fails,
            or the bytes do not match the expected digest/size. An
            unverifiable revision can therefore never be promoted into the
            frontend manifest.
    """
    validate_revision(_revision_from_url(url))
    reader = fetch or fetch_hub_bytes
    try:
        data = reader(url)
    except Exception as error:  # urllib raises a family of errors
        raise ReleaseError(f"Cannot download {url}: {error}") from error
    return verify_artifact_bytes(data, sha256=sha256, size=size, label=f"{label} {url}")


def _normalize_model_record(model_key: str, asset: Mapping[str, Any]) -> dict[str, Any]:
    label = f"Model {model_key}"
    record: dict[str, Any] = {
        "artifact": str(asset.get("artifact", "")),
        "hubPath": str(asset.get("hubPath", "")),
        "sha256": _require_digest(asset.get("sha256"), label),
        "size": _require_size(asset.get("size"), label),
    }
    contract_hash = asset.get("contractHash")
    if contract_hash is not None:
        record["contractHash"] = _require_digest(contract_hash, label)
    return record


def build_release_entry(
    revision: str,
    models: Mapping[str, Mapping[str, Any]],
    *,
    created_at: str | None = None,
    package_manifest: Mapping[str, Any] | None = None,
    package_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an immutable release record for the frontend history."""
    if not models:
        raise ReleaseError("A release needs at least one model record")
    entry: dict[str, Any] = {
        "revision": validate_revision(revision),
        "created_at": created_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "models": {
            key: _normalize_model_record(key, asset)
            for key, asset in sorted(models.items())
        },
    }
    if package_manifest is not None:
        raw = json.dumps(
            package_manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        entry["package_manifest"] = {
            "path": str(package_manifest_path) if package_manifest_path else None,
            "sha256": sha256_bytes(raw),
            "completed_steps": package_manifest.get("completed_steps"),
        }
    return entry


def immutable_commit_sha(result: Any) -> str:
    """Return the immutable commit SHA from a Hugging Face upload result.

    ``huggingface_hub`` reports the commit as ``oid`` (newer releases) or
    ``commit_sha`` (older releases); both are full 40-character SHAs.

    Raises:
        ReleaseError: when the result carries no immutable commit SHA, so an
            upload can never be recorded as a pinnable release by accident.
    """
    for attribute in ("oid", "commit_sha"):
        value = getattr(result, attribute, None)
        if value:
            return validate_revision(str(value))
    raise ReleaseError(
        "Hub upload result carries no immutable commit SHA; cannot pin the release"
    )


def expected_models(entry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return ``modelKey -> {sha256, size}`` from a release record."""
    models = entry.get("models")
    if not isinstance(models, Mapping):
        raise ReleaseError("Release record has no 'models' mapping")
    expected: dict[str, Mapping[str, Any]] = {}
    for key, record in models.items():
        if not isinstance(record, Mapping):
            raise ReleaseError(f"Release model {key!r} is not an object")
        expected[str(key)] = {
            "sha256": _require_digest(record.get("sha256"), f"Release model {key}"),
            "size": _require_size(record.get("size"), f"Release model {key}"),
        }
    return expected


def empty_release_history() -> dict[str, Any]:
    """Return an empty, versioned release history document."""
    return {
        "schemaVersion": RELEASE_HISTORY_SCHEMA_VERSION,
        "releases": [],
    }


def read_release_history(path: str | Path) -> dict[str, Any]:
    """Read the release history; a missing file is an empty history."""
    history_path = Path(path)
    if not history_path.exists():
        return empty_release_history()
    try:
        document = json.loads(history_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(
            f"Cannot read release history {history_path}: {error}"
        ) from error
    if not isinstance(document, dict) or not isinstance(document.get("releases"), list):
        raise ReleaseError(f"Release history has no 'releases' list: {history_path}")
    for entry in document["releases"]:
        if not isinstance(entry, dict):
            raise ReleaseError(f"Release history entry is not an object: {entry!r}")
        validate_revision(str(entry.get("revision", "")))
        expected_models(entry)
    document.setdefault("schemaVersion", RELEASE_HISTORY_SCHEMA_VERSION)
    return document


def write_release_history(path: str | Path, history: Mapping[str, Any]) -> None:
    """Atomically write the release history with stable formatting."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_name(history_path.name + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, history_path)


def record_release(
    history: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a new history with ``entry`` promoted (replacing its revision)."""
    revision = validate_revision(str(entry.get("revision", "")))
    expected_models(entry)
    releases = [
        item for item in history.get("releases", []) if item.get("revision") != revision
    ]
    releases.append(dict(entry))
    releases.sort(
        key=lambda item: (str(item.get("created_at", "")), str(item["revision"]))
    )
    return {
        "schemaVersion": RELEASE_HISTORY_SCHEMA_VERSION,
        "releases": releases,
    }


def find_release(history: Mapping[str, Any], revision: str) -> dict[str, Any] | None:
    """Return the release record for ``revision``, if recorded."""
    target = validate_revision(revision)
    for entry in history.get("releases", []):
        if entry.get("revision") == target:
            return entry
    return None


def latest_release(history: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the most recently recorded release, if any."""
    releases = list(history.get("releases", []))
    if not releases:
        return None
    releases.sort(
        key=lambda item: (str(item.get("created_at", "")), str(item["revision"]))
    )
    return releases[-1]


def resolve_release(
    history: Mapping[str, Any], revision: str | None = None
) -> dict[str, Any]:
    """Return the requested release, or the latest one when unspecified.

    Raises:
        ReleaseError: when the history is empty or the revision is unknown.
    """
    if revision is not None:
        entry = find_release(history, revision)
        if entry is None:
            raise ReleaseError(
                f"Release {revision} is not recorded in the release history"
            )
        return entry
    entry = latest_release(history)
    if entry is None:
        raise ReleaseError("Release history is empty; nothing to roll back to")
    return entry
