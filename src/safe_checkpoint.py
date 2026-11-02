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


def _numpy_safe_globals() -> list[Any]:
    """Globals torch needs to rebuild a numpy array inside a trusted bundle.

    ``weights_only=True`` refuses every global that is not allowlisted, which
    includes numpy's array reconstruction. These classes cannot execute code,
    so allowing them keeps the unpickler restricted while letting a pre-WP1
    RNG snapshot (a raw numpy array) resume.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        return []
    from importlib import import_module

    # One name, two module layouts: importing the attribute directly in both
    # branches makes mypy treat the second import as a redefinition.
    try:
        _multiarray = import_module("numpy._core.multiarray")
    except ImportError:  # pragma: no cover - numpy < 2
        _multiarray = import_module("numpy.core.multiarray")
    reconstruct = vars(_multiarray)["_reconstruct"]
    allowed: list[Any] = [np.ndarray, np.dtype, reconstruct]
    # numpy >= 2 pickles the dtype as e.g. numpy.dtypes.UInt32DType, which
    # weights_only also refuses; numpy 1.x only needs the dtype class above.
    dtype_module = getattr(np, "dtypes", None)
    if dtype_module is not None:
        allowed.extend(
            cls
            for name, cls in vars(dtype_module).items()
            if name.endswith("DType") and isinstance(cls, type)
        )
    allowed.extend(
        getattr(np, name)
        for name in (
            "bool_",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
        if hasattr(np, name)
    )
    return allowed


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    allow_unsafe_legacy: bool = False,
    allow_numpy_globals: bool = False,
) -> Any:
    """Load a torch checkpoint without unrestricted pickle by default.

    Args:
        path: Checkpoint path.
        map_location: Torch map_location passed through.
        allow_unsafe_legacy: Explicit opt-in for a trusted pre-safety checkpoint
            that still contains non-allowlisted globals (for example a numpy RNG
            array). Only training-resume migration should use this.
        allow_numpy_globals: Allow numpy's array-reconstruction globals while
            still loading with ``weights_only=True``. Training resume needs
            this for pre-WP1 checkpoints whose RNG state is a raw numpy array;
            it never permits arbitrary callables.

    Returns:
        The deserialised checkpoint.

    Raises:
        UnsafeCheckpointError: The file needs unrestricted pickle and the
            caller did not opt in. The message names the file and the opt-in
            flag so operators can migrate deliberately.
    """
    path = Path(path)
    safe_globals = _numpy_safe_globals() if allow_numpy_globals else []
    try:
        if safe_globals:
            with torch.serialization.safe_globals(safe_globals):
                return torch.load(path, map_location=map_location, weights_only=True)
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if allow_unsafe_legacy:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise UnsafeCheckpointError(
            f"Checkpoint {path} requires unrestricted pickle deserialisation "
            f"({type(exc).__name__}: {exc}). Refusing to load it. If this is a "
            "trusted legacy training-resume bundle, pass "
            "allow_unsafe_legacy=True (CLI: --allow-unsafe-legacy-checkpoint) "
            "or, for a numpy RNG snapshot, allow_numpy_globals=True; do not "
            "use either for downloaded inference artifacts."
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
