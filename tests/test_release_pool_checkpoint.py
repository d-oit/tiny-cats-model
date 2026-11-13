"""Refusal boundaries for building a release from one pool snapshot."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import release_pool_checkpoint as release  # noqa: E402

TARGET = 600_000
EXPERIMENT_ID = "dit-breed-conditioned-v4"


@pytest.fixture
def pool_snapshot() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pointer = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "completed_steps": TARGET,
        "step_dir": "step-600000",
        "files": ["dit_model.pt", "training_state.json"],
    }
    state = {
        "experiment_id": EXPERIMENT_ID,
        "completed_steps": TARGET,
        "target_steps": 480_000,
    }
    checkpoint = {
        "step": TARGET,
        "manifest": {"experiment_id": EXPERIMENT_ID},
    }
    return pointer, state, checkpoint


def test_validate_snapshot_accepts_matching_completed_progress(
    pool_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pointer, state, checkpoint = pool_snapshot

    release.validate_snapshot(
        pointer,
        state,
        checkpoint,
        target=TARGET,
        experiment_id=EXPERIMENT_ID,
    )


def test_validate_snapshot_rejects_checkpoint_at_different_step(
    pool_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pointer, state, checkpoint = copy.deepcopy(pool_snapshot)
    checkpoint["step"] = TARGET - 1

    with pytest.raises(ValueError, match=r"checkpoint\.step"):
        release.validate_snapshot(
            pointer,
            state,
            checkpoint,
            target=TARGET,
            experiment_id=EXPERIMENT_ID,
        )


def test_validate_snapshot_rejects_sidecar_at_different_step(
    pool_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pointer, state, checkpoint = copy.deepcopy(pool_snapshot)
    state["completed_steps"] = TARGET - 1

    with pytest.raises(ValueError, match=r"training_state\.completed_steps"):
        release.validate_snapshot(
            pointer,
            state,
            checkpoint,
            target=TARGET,
            experiment_id=EXPERIMENT_ID,
        )


def test_validate_snapshot_rejects_embedded_experiment_mismatch(
    pool_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pointer, state, checkpoint = copy.deepcopy(pool_snapshot)
    checkpoint["manifest"]["experiment_id"] = "another-experiment"

    with pytest.raises(ValueError, match=r"checkpoint\.manifest\.experiment_id"):
        release.validate_snapshot(
            pointer,
            state,
            checkpoint,
            target=TARGET,
            experiment_id=EXPERIMENT_ID,
        )


def test_validate_snapshot_rejects_missing_sidecar() -> None:
    pointer = {
        "experiment_id": EXPERIMENT_ID,
        "completed_steps": TARGET,
        "step_dir": "step-600000",
        "files": ["dit_model.pt", "training_state.json"],
    }
    checkpoint = {"step": TARGET, "manifest": {"experiment_id": EXPERIMENT_ID}}

    with pytest.raises(ValueError, match=r"training_state\.json must be a JSON object"):
        release.validate_snapshot(
            pointer,
            None,  # type: ignore[arg-type]
            checkpoint,
            target=TARGET,
            experiment_id=EXPERIMENT_ID,
        )


def test_validate_snapshot_rejects_missing_required_pointer_filename(
    pool_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pointer, state, checkpoint = copy.deepcopy(pool_snapshot)
    pointer["files"].remove("training_state.json")

    with pytest.raises(ValueError, match=r"pointer\.files is missing required names"):
        release.validate_snapshot(
            pointer,
            state,
            checkpoint,
            target=TARGET,
            experiment_id=EXPERIMENT_ID,
        )


@pytest.mark.parametrize(
    "step_dir", ["../step-600000", "step-599999", "step-600000/.."]
)
def test_validate_snapshot_rejects_traversal_or_wrong_step_directory(
    pool_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    step_dir: str,
) -> None:
    pointer, state, checkpoint = copy.deepcopy(pool_snapshot)
    pointer["step_dir"] = step_dir

    with pytest.raises(ValueError, match=r"pointer\.step_dir"):
        release.validate_snapshot(
            pointer,
            state,
            checkpoint,
            target=TARGET,
            experiment_id=EXPERIMENT_ID,
        )
