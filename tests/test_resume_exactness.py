"""tests/test_resume_exactness.py

End-to-end CPU verification of issue #163 against the *real* training loop,
using the ``model_fn`` seam to inject a tiny DiT (WP8's CPU smoke/resume
requirement without a paid GPU):

- fresh run 0 -> N performs exactly N global steps
- a checkpoint at 10 with ``--steps 20`` performs exactly 10 more (not 9/11)
- already-complete checkpoint: 0 steps, bytes untouched, successful exit
- no-op resume into a different output path preserves the checkpoint
- corrupted checkpoint is quarantined and restarts cleanly (stale
  ``training_state.json`` sidecar does not poison the restart)
- architecture and optimizer-critical mismatches are rejected, with an
  explicit ``allow_experiment_mismatch`` override

Training forwards are counted via ``torch.is_grad_enabled()`` so the
``@torch.no_grad()`` validation evaluations never inflate the count; with
``gradient_accumulation_steps=1`` one forward == one optimizer step.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import train_dit
from dataset import CAT_BREEDS
from dit import TinyDiT
from train_dit import train_dit_local
from training_state import (
    IncompatibleExperimentError,
    build_manifest,
    read_training_state,
    write_training_state,
)


@pytest.fixture(autouse=True)
def _null_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep MLflow out of the real training loop during tests."""

    class _NullTracker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start_run(self, *args: object, **kwargs: object) -> None:
            return None

        def log_params(self, *args: object, **kwargs: object) -> None:
            return None

        def log_metrics(self, *args: object, **kwargs: object) -> None:
            return None

        def log_artifact(self, *args: object, **kwargs: object) -> None:
            return None

        def end_run(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(train_dit, "ExperimentTracker", _NullTracker)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Minimal cat/other dataset with breed-encoded filenames (8 images).

    Lives in ``tmp_path / "data"`` so it is disjoint from the checkpoint
    directory (``tmp_path / "ckpts"``) — mirroring production, where the
    dataset fingerprint must not be perturbed by checkpoint writes.
    """
    root = tmp_path / "data"
    cat_dir = root / "cat"
    other_dir = root / "other"
    cat_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(cat_dir / f"{CAT_BREEDS[0]}_1.jpg")
    Image.new("RGB", (16, 16)).save(cat_dir / f"{CAT_BREEDS[0]}_2.jpg")
    for index in range(6):
        Image.new("RGB", (16, 16)).save(other_dir / f"beagle_{index}.jpg")
    return root


class CountingModelFactory:
    """Builds a tiny DiT and counts training (grad-enabled) forwards."""

    def __init__(self, embed_dim: int = 16) -> None:
        self.embed_dim = embed_dim
        self.train_forwards = 0

    def __call__(self) -> TinyDiT:
        model = TinyDiT(
            image_size=8,
            patch_size=4,
            embed_dim=self.embed_dim,
            depth=1,
            num_heads=2,
            num_classes=13,
        )
        original_forward = model.forward

        def counting_forward(*args: object, **kwargs: object) -> object:
            if torch.is_grad_enabled():
                self.train_forwards += 1
            return original_forward(*args, **kwargs)  # type: ignore

        model.forward = counting_forward  # type: ignore
        return model


def run_train(
    dataset: Path,
    output: Path,
    factory: CountingModelFactory,
    *,
    steps: int,
    resume: Path | None = None,
    lr: float = 1e-3,
    save_interval: int = 5,
    allow_experiment_mismatch: bool = False,
    experiment_id: str = "resume-e2e",
) -> float:
    """Drive ``train_dit_local`` with tiny, fast, deterministic settings."""
    return train_dit_local(
        data_dir=str(dataset),
        steps=steps,
        batch_size=2,
        lr=lr,
        image_size=8,
        output=str(output),
        ema_output=str(output.parent / "dit_model_ema.pt"),
        num_workers=0,
        mixed_precision=False,
        val_split=0.25,
        val_batches=1,
        log_interval=10_000,
        save_interval=save_interval,
        sample_interval=10_000,
        early_stopping_patience=0,
        seed=42,
        resume=str(resume) if resume is not None else None,
        no_hub_push=True,
        experiment_id=experiment_id,
        allow_experiment_mismatch=allow_experiment_mismatch,
        model_fn=factory,
    )


def load_ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestFreshRun:
    def test_zero_to_target_is_exact(self, tiny_dataset: Path, tmp_path: Path) -> None:
        factory = CountingModelFactory()
        output = tmp_path / "ckpts" / "dit_model.pt"

        run_train(tiny_dataset, output, factory, steps=20, save_interval=10)

        assert factory.train_forwards == 20

        checkpoint = load_ckpt(output)
        assert checkpoint["step"] == 20
        assert checkpoint["target_steps"] == 20
        assert checkpoint["manifest"]["experiment_id"] == "resume-e2e"
        # RNG stream is persisted for provider handoff (WP1)
        assert checkpoint["rng_state"] is not None
        assert "torch" in checkpoint["rng_state"]

        state = read_training_state(output.parent / "training_state.json")
        assert state is not None
        assert state["completed_steps"] == 20
        assert state["target_steps"] == 20
        for field in (
            "experiment_id",
            "git_sha",
            "dataset_hash",
            "breed_mapping_hash",
            "seed",
            "optimizer",
            "scheduler",
            "gradient_accumulation_steps",
        ):
            assert field in state


class TestSliceResume:
    def test_10_to_20_performs_exactly_10(
        self, tiny_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "ckpts" / "dit_model.pt"

        first = CountingModelFactory()
        run_train(tiny_dataset, output, first, steps=10)
        assert first.train_forwards == 10
        assert load_ckpt(output)["step"] == 10

        second = CountingModelFactory()
        # No override flag passed: raising the global target is the WP6 slice
        # contract and must NOT count as an experiment mismatch.
        run_train(tiny_dataset, output, second, steps=20, resume=output)

        assert second.train_forwards == 10  # exactly — not 9, not 11

        checkpoint = load_ckpt(output)
        assert checkpoint["step"] == 20
        state = read_training_state(output.parent / "training_state.json")
        assert state is not None
        assert state["completed_steps"] == 20
        assert state["target_steps"] == 20


class TestAlreadyComplete:
    def test_noop_preserves_checkpoint_bytes(
        self, tiny_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "ckpts" / "dit_model.pt"
        first = CountingModelFactory()
        run_train(tiny_dataset, output, first, steps=10)
        assert first.train_forwards == 10

        digest_before = sha256(output)
        factory = CountingModelFactory()

        # Returns normally (successful exit) without training a step and
        # without overwriting the valid completed checkpoint.
        run_train(tiny_dataset, output, factory, steps=10, resume=output)

        assert factory.train_forwards == 0
        assert sha256(output) == digest_before

        state = read_training_state(output.parent / "training_state.json")
        assert state is not None
        assert state["completed_steps"] == 10

    def test_noop_into_different_output_path(
        self, tiny_dataset: Path, tmp_path: Path
    ) -> None:
        source = tmp_path / "ckpts" / "dit_model.pt"
        first = CountingModelFactory()
        run_train(tiny_dataset, source, first, steps=10)
        source_digest = sha256(source)

        other = tmp_path / "elsewhere" / "model.pt"
        factory = CountingModelFactory()
        run_train(tiny_dataset, other, factory, steps=10, resume=source)

        assert factory.train_forwards == 0
        assert other.exists()
        # The copy must not have been followed by a no-op checkpoint write
        assert sha256(source) == source_digest

        state = read_training_state(other.parent / "training_state.json")
        assert state is not None
        assert state["completed_steps"] == 10


class TestCorruptCheckpoint:
    def test_quarantine_and_clean_restart(
        self, tiny_dataset: Path, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        output = ckpt_dir / "dit_model.pt"
        output.write_bytes(b"garbage-not-a-torch-checkpoint")

        # A stale sidecar from a different experiment must not poison the
        # restart (the manifest gate only runs after a successful load).
        write_training_state(
            ckpt_dir / "training_state.json",
            manifest=build_manifest(
                experiment_id="stale-experiment",
                data_dir=str(tiny_dataset),
                image_size=128,
                patch_size=16,
                embed_dim=16,
                depth=1,
                num_heads=2,
                num_classes=13,
                batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                warmup_steps=10,
                augmentation_level="full",
                seed=42,
                target_steps=999,
            ),
            completed_steps=999,
        )

        factory = CountingModelFactory()
        run_train(tiny_dataset, output, factory, steps=5, resume=output)

        assert factory.train_forwards == 5
        assert (ckpt_dir / "dit_model.pt.corrupt").exists()

        state = read_training_state(ckpt_dir / "training_state.json")
        assert state is not None
        assert state["completed_steps"] == 5  # refreshed by the fresh run


class TestIncompatibleResume:
    def test_architecture_mismatch_is_rejected(
        self, tiny_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "ckpts" / "dit_model.pt"
        first = CountingModelFactory(embed_dim=16)
        run_train(tiny_dataset, output, first, steps=3)

        factory = CountingModelFactory(embed_dim=32)
        with pytest.raises(IncompatibleExperimentError, match="architecture"):
            run_train(tiny_dataset, output, factory, steps=6, resume=output)

        assert factory.train_forwards == 0  # rejected before any training

    def test_optimizer_critical_mismatch_then_override(
        self, tiny_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "ckpts" / "dit_model.pt"
        first = CountingModelFactory()
        run_train(tiny_dataset, output, first, steps=3, lr=1e-3)

        rejected = CountingModelFactory()
        with pytest.raises(IncompatibleExperimentError, match="learning_rate"):
            run_train(
                tiny_dataset,
                output,
                rejected,
                steps=6,
                resume=output,
                lr=5e-4,
            )
        assert rejected.train_forwards == 0

        overridden = CountingModelFactory()
        run_train(
            tiny_dataset,
            output,
            overridden,
            steps=6,
            resume=output,
            lr=5e-4,
            allow_experiment_mismatch=True,
        )
        assert overridden.train_forwards == 3
