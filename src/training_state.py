"""src/training_state.py

Immutable experiment manifests and exact global-step resume state (issue #163).

Two artifacts:

1. **Experiment manifest** — a JSON-serialisable description of everything that
   defines *which* experiment a checkpoint belongs to: architecture, dataset
   identity, optimizer-critical hyperparameters and seed. It is embedded in
   every checkpoint (``checkpoint["manifest"]``) and mirrored into
   ``training_state.json`` beside every checkpoint. The embedded copy is
   authoritative (it survives HF Hub transport, which only moves the ``.pt``);
   the sidecar exists so tooling can reason about a run without unpickling a
   multi-hundred-MB checkpoint.

2. **``training_state.json``** — the manifest flattened to top level plus
   progress (``completed_steps`` / ``target_steps``), written atomically after
   every checkpoint save.

Resume rule (WP1): ``--steps`` is a *global target*, never "train this many
more". A resume performs ``max(0, target_steps - completed_steps)`` additional
steps. A checkpoint whose manifest does not match the current run's manifest is
rejected with :class:`IncompatibleExperimentError` unless the caller passes an
explicit override (``--allow-experiment-mismatch``).

Manifest fields that are recorded but deliberately *not* compared:

- ``target_steps`` — slices of one experiment raise the global target
  (60k -> 120k -> ... -> 400k, WP6); the contract enforced instead is
  ``completed_steps <= target_steps`` (a completed target is a no-op, not an
  error).
- ``git_sha`` — code legitimately advances between slices of one experiment.
- ``provider`` — the whole point of the manifest is that providers differ.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Filename written beside every checkpoint.
TRAINING_STATE_FILENAME = "training_state.json"

REPOSITORY_URL = "https://github.com/d-oit/tiny-cats-model"
DATASET_ID = "oxford-iiit-pet-cats"

#: Fields recorded in every manifest (issue #163 WP4 required fields).
MANIFEST_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "repository",
    "git_sha",
    "dataset_id",
    "dataset_version",
    "dataset_hash",
    "breed_mapping_hash",
    "image_size",
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "num_classes",
    "optimizer",
    "learning_rate",
    "warmup_steps",
    "scheduler",
    "batch_size",
    "gradient_accumulation_steps",
    "augmentation_level",
    "seed",
    "target_steps",
    "provider",
)

#: Manifest fields that must match for a resume to be accepted. Excludes
#: target_steps/git_sha/provider/dataset_version — see module docstring.
IMMUTABLE_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "dataset_id",
    "dataset_hash",
    "breed_mapping_hash",
    "image_size",
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "num_classes",
    "optimizer",
    "learning_rate",
    "warmup_steps",
    "scheduler",
    "batch_size",
    "gradient_accumulation_steps",
    "augmentation_level",
    "seed",
)


class IncompatibleExperimentError(RuntimeError):
    """Resume rejected: the checkpoint belongs to a different experiment.

    Raised instead of silently restarting from step 0 (issue #163: "Reject
    incompatible experiment configuration instead of silently restarting").
    """


def steps_to_run(target_steps: int, completed_steps: int) -> int:
    """Additional global steps a resume must perform (WP1 acceptance rule).

    A checkpoint at 45,000 with ``--steps 60,000`` performs exactly 15,000;
    an already-complete checkpoint performs 0 and the run is a successful
    no-op.
    """
    return max(0, int(target_steps) - int(completed_steps))


def git_sha(repo_dir: str | Path | None = None) -> str:
    """Best-effort current commit SHA; ``"unknown"`` outside a git checkout."""
    cwd = (
        str(repo_dir)
        if repo_dir is not None
        else str(Path(__file__).resolve().parent.parent)
    )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def dataset_fingerprint(data_dir: str | Path) -> str:
    """Deterministic dataset identity: sha256 over sorted ``relpath:size``.

    Uses paths and sizes only (no content reads) so it stays cheap on the
    7,390-image dataset while still detecting added, removed or replaced
    files — the signal the resume check needs.
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


def breed_mapping_hash() -> str:
    """Hash of the breed label ordering (dataset mapping continuity, WP4)."""
    try:
        from dataset import CAT_BREEDS

        breeds = [str(b) for b in CAT_BREEDS]
    except Exception:  # pragma: no cover - dataset module always present in practice
        breeds = []
    return hashlib.sha256("\n".join(breeds).encode()).hexdigest()


def _detect_provider() -> str:
    try:
        from gpu_pool import detect_provider

        return detect_provider().value
    except Exception:  # pragma: no cover - gpu_pool optional in minimal envs
        return "unknown"


def build_manifest(
    *,
    experiment_id: str,
    data_dir: str | Path,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    num_classes: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    augmentation_level: str,
    seed: int,
    target_steps: int,
    optimizer: str = "adamw",
    scheduler: str = "warmup_cosine",
    repository: str = REPOSITORY_URL,
    dataset_id: str = DATASET_ID,
    provider: str | None = None,
    git_sha_value: str | None = None,
) -> dict[str, Any]:
    """Build the immutable experiment manifest for the *current* run.

    Every field except ``target_steps``/``git_sha``/``provider`` must match a
    checkpoint's stored manifest for a resume to be accepted.
    """
    dataset_hash = dataset_fingerprint(data_dir)
    return {
        "experiment_id": experiment_id,
        "repository": repository,
        "git_sha": git_sha_value if git_sha_value is not None else git_sha(),
        "dataset_id": dataset_id,
        "dataset_version": dataset_hash[:12],
        "dataset_hash": dataset_hash,
        "breed_mapping_hash": breed_mapping_hash(),
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "embed_dim": int(embed_dim),
        "depth": int(depth),
        "num_heads": int(num_heads),
        "num_classes": int(num_classes),
        "optimizer": optimizer,
        "learning_rate": float(learning_rate),
        "warmup_steps": int(warmup_steps),
        "scheduler": scheduler,
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "augmentation_level": augmentation_level,
        "seed": int(seed),
        "target_steps": int(target_steps),
        "provider": provider if provider is not None else _detect_provider(),
    }


def manifest_mismatches(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Human-readable incompatibilities between a checkpoint and this run.

    Returns an empty list when the resume is allowed. Fields missing from the
    saved manifest are reported too — a manifest we cannot verify is a
    manifest we cannot trust.
    """
    problems: list[str] = []
    for field in IMMUTABLE_FIELDS:
        saved_value = saved.get(field, "<missing>")
        current_value = current.get(field, "<missing>")
        if saved_value != current_value:
            problems.append(
                f"{field}: checkpoint={saved_value!r} current={current_value!r}"
            )
    return problems


def manifest_of(state: dict[str, Any]) -> dict[str, Any]:
    """Extract the manifest subset from a training-state document."""
    return {field: state[field] for field in MANIFEST_FIELDS if field in state}


def write_training_state(
    state_path: str | Path,
    *,
    manifest: dict[str, Any],
    completed_steps: int,
) -> None:
    """Atomically write ``training_state.json`` beside a checkpoint."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
        "completed_steps": int(completed_steps),
    }
    tmp_path = state_path.with_name(state_path.name + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, state_path)


def read_training_state(state_path: str | Path) -> dict[str, Any] | None:
    """Read a training-state document; ``None`` when missing or corrupt."""
    state_path = Path(state_path)
    try:
        document = json.loads(state_path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable training state at %s (%s)", state_path, exc)
        return None
    if not isinstance(document, dict):
        logger.warning("Training state at %s is not a JSON object", state_path)
        return None
    return document


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that influences training (PyTorch guidance)."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore an RNG snapshot captured by :func:`capture_rng_state`.

    Must be called *after* model/optimizer construction and *before* the
    training loop starts — restore last, then never reseed (PyTorch
    reproducibility guidance).
    """
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError as exc:  # device count changed across providers
            logger.warning("Could not restore CUDA RNG state: %s", exc)
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:  # pragma: no cover - numpy is a hard dependency
        import numpy as np

        np.random.set_state(state["numpy"])
