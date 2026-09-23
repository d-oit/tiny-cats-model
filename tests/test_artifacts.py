"""tests/test_artifacts.py

Issue #163 WP2: canonical checkpoint/artifact layout, legacy layout
migration, and the "never publish an invalid final package" rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from artifacts import (
    ArtifactPackageError,
    export_paths,
    find_live_checkpoint,
    package_final_artifacts,
    pool_paths,
)
from training_state import write_training_state

MANIFEST = {"experiment_id": "exp-a", "target_steps": 10, "git_sha": "abc123"}


def make_checkpoint(path: Path, step: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model_state_dict": {}}, path)


class TestPoolPaths:
    def test_canonical_paths(self, tmp_path: Path) -> None:
        paths = pool_paths(tmp_path)
        assert paths["model"] == tmp_path / "checkpoints/pool/dit_model.pt"
        assert paths["model_ema"].name == "dit_model_ema.pt"
        assert paths["training_state"].name == "training_state.json"
        assert paths["training_log"].name == "training.log"


class TestFindLiveCheckpoint:
    def test_none_when_empty(self, tmp_path: Path) -> None:
        assert find_live_checkpoint(tmp_path) is None

    def test_canonical_wins(self, tmp_path: Path) -> None:
        canonical = tmp_path / "checkpoints/pool/dit_model.pt"
        make_checkpoint(canonical, 5)
        legacy = tmp_path / "checkpoints/dit/current/dit_model.pt"
        make_checkpoint(legacy, 3)

        assert find_live_checkpoint(tmp_path) == canonical

    @pytest.mark.parametrize(
        "legacy_dirname",
        [
            "checkpoints/dit/current",
            "checkpoints/dit/breed-conditioned-v4",
            "checkpoints/dit",
        ],
    )
    def test_legacy_fallback(self, tmp_path: Path, legacy_dirname: str) -> None:
        legacy = tmp_path / legacy_dirname / "dit_model.pt"
        make_checkpoint(legacy, 7)

        assert find_live_checkpoint(tmp_path) == legacy

    def test_skips_invalid_canonical_so_legacy_can_resume(self, tmp_path: Path) -> None:
        canonical = tmp_path / "checkpoints/pool/dit_model.pt"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"partial-write-garbage")
        legacy = tmp_path / "checkpoints/dit/current/dit_model.pt"
        make_checkpoint(legacy, 7)

        assert find_live_checkpoint(tmp_path) == legacy


class TestPackageFinalArtifacts:
    @pytest.fixture
    def ready_outputs(self, tmp_path: Path) -> Path:
        pool = tmp_path / "checkpoints/pool"
        make_checkpoint(pool / "dit_model.pt", 10)
        torch.save({"step": 10}, pool / "dit_model_ema.pt")
        write_training_state(
            pool / "training_state.json", manifest=MANIFEST, completed_steps=10
        )
        (pool / "training.log").write_text("step log\n")
        samples = pool / "samples"
        samples.mkdir()
        (samples / "step_10_breed_0.png").write_bytes(b"png-bytes")
        onnx = tmp_path / "export/model.onnx"
        onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.write_bytes(b"onnx-bytes")
        quant = tmp_path / "export/model_quantized.onnx"
        quant.parent.mkdir(parents=True, exist_ok=True)
        quant.write_bytes(b"quant-bytes")
        return tmp_path

    def _package(self, outputs: Path, **overrides: Any):
        kwargs: dict[str, Any] = {
            "raw_checkpoint": outputs / "checkpoints/pool/dit_model.pt",
            "ema_checkpoint": outputs / "checkpoints/pool/dit_model_ema.pt",
            "training_log": outputs / "checkpoints/pool/training.log",
            "samples_dir": outputs / "checkpoints/pool/samples",
            "onnx": outputs / "export/model.onnx",
            "quantized": outputs / "export/model_quantized.onnx",
            "require_completed": 10,
        }
        kwargs.update(overrides)
        return package_final_artifacts(outputs, **kwargs)

    def test_package_layout_and_manifest(self, ready_outputs: Path) -> None:
        manifest = self._package(ready_outputs)

        generator = ready_outputs / "artifacts/generator"
        for name in (
            "model.pt",
            "model_ema.pt",
            "model.onnx",
            "model_quantized.onnx",
        ):
            assert (generator / name).exists(), name

        training = ready_outputs / "artifacts/training"
        for name in ("final_checkpoint.pt", "training_state.json", "training.log"):
            assert (training / name).exists(), name

        assert (
            ready_outputs / "artifacts/evaluation/samples/step_10_breed_0.png"
        ).exists()

        manifest_path = ready_outputs / "artifacts/manifest.json"
        document = json.loads(manifest_path.read_text())
        assert document == manifest
        assert document["completed_steps"] == 10
        assert document["experiment_id"] == "exp-a"
        assert document["git_sha"] == "abc123"
        assert document["files"]
        assert all(
            entry["sha256"] and entry["bytes"] > 0 for entry in document["files"]
        )
        # Atomic write: no temp files left behind
        assert not list(ready_outputs.rglob("*.tmp"))

    def test_missing_checkpoint_refused(self, ready_outputs: Path) -> None:
        with pytest.raises(ArtifactPackageError, match="missing"):
            self._package(ready_outputs, raw_checkpoint=ready_outputs / "nope.pt")

    def test_non_zip_checkpoint_refused(self, ready_outputs: Path) -> None:
        bad = ready_outputs / "bad.pt"
        bad.write_bytes(b"garbage-not-a-zip")
        with pytest.raises(ArtifactPackageError, match="not a zip"):
            self._package(ready_outputs, raw_checkpoint=bad)

    def test_target_not_reached_refused(self, ready_outputs: Path) -> None:
        # completed=10 in the state but the caller requires a higher target:
        # a short-of-target checkpoint must never be published.
        with pytest.raises(ArtifactPackageError, match="Refusing to publish"):
            self._package(ready_outputs, require_completed=20)

    def test_missing_state_refused_when_progress_required(
        self, ready_outputs: Path
    ) -> None:
        (ready_outputs / "checkpoints/pool/training_state.json").unlink()
        with pytest.raises(ArtifactPackageError, match="Cannot verify"):
            self._package(ready_outputs)

    def test_state_missing_is_fine_without_requirement(
        self, ready_outputs: Path
    ) -> None:
        (ready_outputs / "checkpoints/pool/training_state.json").unlink()
        manifest = self._package(ready_outputs, require_completed=None)
        assert manifest["files"]

    def test_onnx_already_at_destination_is_not_a_crash(
        self, ready_outputs: Path
    ) -> None:
        # Live-smoke regression: exporting straight into the package made
        # copy2(source, source) raise SameFileError. An in-place "copy" must
        # hash the file rather than crash (issue #163 WP2).
        destination = ready_outputs / "artifacts/generator/model.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"onnx-already-there")
        manifest = self._package(
            ready_outputs,
            onnx=destination,  # source IS the destination
            quantized=None,
        )
        entries = [f for f in manifest["files"] if f["path"].endswith("model.onnx")]
        assert len(entries) == 1
        assert entries[0]["bytes"] == len(b"onnx-already-there")


class TestExportPaths:
    def test_staging_is_outside_the_package(self, tmp_path: Path) -> None:
        paths = export_paths(tmp_path)
        assert paths["export_dir"] == tmp_path / "export"
        assert paths["onnx"] == tmp_path / "export/model.onnx"
        # optimize_onnx names generator output "generator_quantized.onnx".
        assert paths["quantized"] == tmp_path / "export/generator_quantized.onnx"
        # Sources must never equal package destinations (SameFileError).
        generator = tmp_path / "artifacts/generator"
        assert paths["onnx"] != generator / "model.onnx"
        assert paths["quantized"] != generator / "model_quantized.onnx"
