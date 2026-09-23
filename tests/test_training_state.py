"""tests/test_training_state.py

Issue #163 (WP1 + WP4) unit coverage:

- exact global-step arithmetic (the 45k -> 60k = 15,000 acceptance rule)
- the immutable experiment manifest and its mismatch detection
- ``training_state.json`` atomic write/read round-trips
- RNG capture/restore for cross-provider handoff
- dataset fingerprint stability
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_state import (
    IMMUTABLE_FIELDS,
    MANIFEST_FIELDS,
    build_manifest,
    capture_rng_state,
    dataset_fingerprint,
    git_sha,
    manifest_mismatches,
    manifest_of,
    read_training_state,
    restore_rng_state,
    steps_to_run,
    write_training_state,
)


def make_manifest(**overrides: Any) -> dict[str, Any]:
    """Manifest for a fixed reference experiment, with optional overrides."""
    base: dict[str, Any] = {
        "experiment_id": "exp-a",
        "data_dir": "data/does-not-exist-in-tests",
        "image_size": 128,
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "num_classes": 13,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "warmup_steps": 2000,
        "augmentation_level": "full",
        "seed": 42,
        "target_steps": 400000,
    }
    base.update(overrides)
    return build_manifest(**base)


class TestGlobalStepAccounting:
    """WP1: ``--steps`` is a global target, never an additional count."""

    def test_fresh_run_is_zero_to_target(self) -> None:
        assert steps_to_run(400000, 0) == 400000

    def test_45k_to_60k_is_exactly_15k(self) -> None:
        """Issue acceptance: checkpoint 45,000 + --steps 60,000 -> 15,000."""
        assert steps_to_run(60000, 45000) == 15000

    def test_60k_to_400k_without_restarting(self) -> None:
        assert steps_to_run(400000, 60000) == 340000

    def test_already_complete_checkpoint_is_a_noop(self) -> None:
        assert steps_to_run(60000, 60000) == 0

    def test_past_target_clamps_to_zero(self) -> None:
        assert steps_to_run(60000, 72000) == 0


class TestExperimentManifest:
    """WP4: immutable manifest + resume compatibility rules."""

    def test_required_issue_fields_present(self) -> None:
        manifest = make_manifest()
        for field in (
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
            "optimizer",
            "learning_rate",
            "warmup_steps",
            "scheduler",
            "batch_size",
            "gradient_accumulation_steps",
            "augmentation_level",
            "seed",
            "target_steps",
        ):
            assert field in manifest, f"missing required manifest field: {field}"

    def test_manifest_fields_cover_every_manifest_key(self) -> None:
        assert set(make_manifest()) == set(MANIFEST_FIELDS)

    def test_identical_manifests_match(self) -> None:
        assert manifest_mismatches(make_manifest(), make_manifest()) == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("image_size", 256),
            ("embed_dim", 192),
            ("depth", 24),
            ("num_heads", 12),
            ("num_classes", 14),
            ("learning_rate", 1e-3),
            ("warmup_steps", 500),
            ("batch_size", 64),
            ("gradient_accumulation_steps", 4),
            ("augmentation_level", "basic"),
            ("seed", 7),
            ("experiment_id", "exp-b"),
        ],
    )
    def test_incompatible_fields_are_reported(self, field: str, value: Any) -> None:
        saved = make_manifest()
        current = make_manifest(**{field: value})
        report = manifest_mismatches(saved, current)
        assert any(field in line for line in report), report

    def test_dataset_mismatch_is_reported(self) -> None:
        saved = make_manifest()
        current = make_manifest()
        saved["dataset_hash"] = "aaa"
        saved["dataset_version"] = "aaa000"
        current["dataset_hash"] = "bbb"
        current["dataset_version"] = "bbb000"
        report = manifest_mismatches(saved, current)
        assert any("dataset_hash" in line for line in report)

    def test_target_steps_may_advance_across_slices(self) -> None:
        """WP6 slice contract: 60k -> 400k must not be a mismatch."""
        saved = make_manifest(target_steps=60000)
        current = make_manifest(target_steps=400000)
        assert manifest_mismatches(saved, current) == []

    def test_git_sha_and_provider_may_differ(self) -> None:
        saved = make_manifest(git_sha_value="aaa", provider="modal")
        current = make_manifest(git_sha_value="bbb", provider="lightning")
        assert manifest_mismatches(saved, current) == []

    def test_unverifiable_field_is_reported(self) -> None:
        saved = make_manifest()
        saved.pop("seed")
        report = manifest_mismatches(saved, make_manifest())
        assert any("seed" in line for line in report)

    def test_immutable_fields_exclude_slice_progress_fields(self) -> None:
        for field in ("target_steps", "git_sha", "provider", "dataset_version"):
            assert field not in IMMUTABLE_FIELDS


class TestTrainingStateFile:
    """``training_state.json`` round-trips atomically."""

    def test_round_trip(self, tmp_path: Path) -> None:
        manifest = make_manifest()
        state_path = tmp_path / "training_state.json"

        write_training_state(state_path, manifest=manifest, completed_steps=60000)

        state = read_training_state(state_path)
        assert state is not None
        assert state["completed_steps"] == 60000
        assert state["target_steps"] == 400000
        assert state["experiment_id"] == "exp-a"
        assert state["schema_version"] == 1
        assert manifest_of(state) == manifest

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        write_training_state(
            tmp_path / "training_state.json",
            manifest=make_manifest(),
            completed_steps=1,
        )
        assert [p.name for p in tmp_path.iterdir()] == ["training_state.json"]

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_training_state(tmp_path / "nope.json") is None

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "training_state.json"
        path.write_text("{not json at all")
        assert read_training_state(path) is None

    def test_non_object_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "training_state.json"
        path.write_text(json.dumps([1, 2, 3]))
        assert read_training_state(path) is None


class TestRngState:
    """WP1: optimizer/scheduler/EMA/RNG state survives provider handoff."""

    def test_torch_stream_continues_after_restore(self) -> None:
        state = capture_rng_state()
        expected = torch.rand(4)
        restore_rng_state(state)
        assert torch.equal(torch.rand(4), expected)

    def test_python_random_stream_continues(self) -> None:
        state = capture_rng_state()
        expected = random.random()
        restore_rng_state(state)
        assert random.random() == expected

    def test_numpy_stream_continues(self) -> None:
        np = pytest.importorskip("numpy")
        state = capture_rng_state()
        expected = np.random.rand(3).copy()
        restore_rng_state(state)
        assert np.allclose(np.random.rand(3), expected)

    def test_restore_is_repeatable(self) -> None:
        state = capture_rng_state()
        restore_rng_state(state)
        first = torch.rand(2)
        restore_rng_state(state)
        assert torch.equal(torch.rand(2), first)

    def test_captures_expected_keys(self) -> None:
        state = capture_rng_state()
        assert "torch" in state
        assert "python" in state


class TestFingerprints:
    def test_dataset_fingerprint_is_stable(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(b"1234")
        first = dataset_fingerprint(tmp_path)
        assert first == dataset_fingerprint(tmp_path)

    def test_dataset_fingerprint_detects_changes(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(b"1234")
        before = dataset_fingerprint(tmp_path)
        (tmp_path / "a.jpg").write_bytes(b"12345")
        assert dataset_fingerprint(tmp_path) != before

    def test_dataset_fingerprint_detects_additions(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(b"1234")
        before = dataset_fingerprint(tmp_path)
        (tmp_path / "b.jpg").write_bytes(b"1")
        assert dataset_fingerprint(tmp_path) != before

    def test_missing_dataset_dir(self, tmp_path: Path) -> None:
        assert dataset_fingerprint(tmp_path / "missing") == "missing"

    def test_git_sha_is_a_string(self) -> None:
        value = git_sha()
        assert isinstance(value, str)
        assert value
