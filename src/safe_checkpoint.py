"""src/safe_checkpoint.py

Safe checkpoint loading and artifact integrity helpers (issues #201, #204).

``torch.load(..., weights_only=False)`` executes arbitrary pickle globals.
Every inference/export/verification path now goes through
:func:`load_torch_checkpoint`, which uses ``weights_only=True`` by default.
Trusted legacy resume bundles can opt out *explicitly* by passing
``allow_unsafe_legacy=True``; there is no silent fallback.

The safe schema for new checkpoints is tensor/primitive-only: state dicts,
optimizer/scheduler dicts, JSON-compatible config/metadata and RNG state with
numpy's array converted to a plain integer list (see
``training_state.capture_rng_state``).
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import torch


class UnsafeCheckpointError(RuntimeError):
    """A checkpoint needs unrestricted pickle but no explicit opt-in was given."""


class ChecksumMismatchError(RuntimeError):
    """Downloaded bytes do not match the pinned SHA-256 digest."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file and return its lowercase SHA-256 hex digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def verify_file_sha256(path: str | Path, expected: str) -> str:
    """Verify ``path`` against an expected hex digest; return the actual digest.

    Raises:
        ChecksumMismatchError: When the digest differs. Never logs-and-continues:
            a tampered or stale artifact must stop inference/publication.
    """
    if not expected:
        raise ChecksumMismatchError(f"No expected SHA-256 supplied for {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    allow_unsafe_legacy: bool = False,
) -> Any:
    """Load a torch checkpoint without unrestricted pickle by default.

    Args:
        path: Checkpoint path.
        map_location: Torch map_location passed through.
        allow_unsafe_legacy: Explicit opt-in for a trusted pre-safety checkpoint
            that still contains non-allowlisted globals (for example a numpy RNG
            array). Only training-resume migration should use this.

    Returns:
        The deserialised checkpoint.

    Raises:
        UnsafeCheckpointError: The file needs unrestricted pickle and the
            caller did not opt in. The message names the file and the opt-in
            flag so operators can migrate deliberately.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if allow_unsafe_legacy:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise UnsafeCheckpointError(
            f"Checkpoint {path} requires unrestricted pickle deserialisation "
            f"({type(exc).__name__}: {exc}). Refusing to load it. If this is a "
            "trusted legacy training-resume bundle, pass "
            "allow_unsafe_legacy=True (CLI: --allow-unsafe-legacy-checkpoint); "
            "do not use that flag for downloaded or inference artifacts."
        ) from exc


def model_sha256(path: str | Path) -> str:
    """Digest of a checkpoint or ONNX artifact."""
    return sha256_file(path)


__all__ = [
    "ChecksumMismatchError",
    "UnsafeCheckpointError",
    "load_torch_checkpoint",
    "model_sha256",
    "sha256_bytes",
    "sha256_file",
    "verify_file_sha256",
]
