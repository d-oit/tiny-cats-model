"""tests/test_train_lightning.py

The Lightning trainer's control-plane contract: forced canonical downloads
(fingerprint parity, timeout, failure), CUDA refusal, fail-closed Hub resume,
sync-only CPU behavior, and canonical-only Hub publication.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import train_lightning


def _detect(*, has_gpu: bool) -> dict:
    return {
        "is_lightning": True,
        "studio_id": "studio-test",
        "has_gpu": has_gpu,
        "gpu_name": "Tesla T4" if has_gpu else "N/A",
        "gpu_memory_gb": 16.0 if has_gpu else 0.0,
    }


def _args(tmp_path: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "--data-dir",
        str(tmp_path / "cats-data"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
        *(extra or []),
    ]


def test_force_download_discards_stale_data_before_download(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "cats-data"
    (data_dir / "cat").mkdir(parents=True)
    stale = data_dir / "cat" / "Abyssinian_1.jpg"
    stale.write_bytes(b"stale")
    calls = 0
    timeouts: list[float] = []

    def fake_run(_command: list[str], **kwargs) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        timeouts.append(kwargs["timeout"])
        assert not stale.exists(), "stale data must be gone before downloading"
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert train_lightning.download_data(str(data_dir), force_download=True) is True
    assert calls == 1
    assert timeouts == [train_lightning.LIGHTNING_DATA_PREP_TIMEOUT_SECONDS]
    assert train_lightning.LIGHTNING_DATA_PREP_TIMEOUT_SECONDS == 2_700


def test_force_download_failure_propagates(tmp_path: Path, monkeypatch) -> None:
    """A failed forced download must not be papered over as success."""
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )

    assert (
        train_lightning.download_data(str(tmp_path / "d"), force_download=True) is False
    )


def test_force_download_failure_aborts_training(tmp_path: Path, monkeypatch) -> None:
    """A failed canonical download must stop before Hub resume or training."""
    monkeypatch.setattr(
        train_lightning, "detect_lightning_environment", lambda: _detect(has_gpu=True)
    )
    monkeypatch.setattr(train_lightning, "download_data", lambda *a, **k: False)
    monkeypatch.setattr(
        train_lightning,
        "pull_resume_checkpoint",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("no pull expected")),
    )

    assert train_lightning.main(_args(tmp_path, ["--force-download"])) == 1


def test_training_without_cuda_is_refused(tmp_path: Path, monkeypatch) -> None:
    """No CPU fallback: a CPU-only runner must not download or train."""
    monkeypatch.setattr(
        train_lightning, "detect_lightning_environment", lambda: _detect(has_gpu=False)
    )
    monkeypatch.setattr(
        train_lightning,
        "download_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download expected")),
    )
    monkeypatch.setattr(
        train_lightning,
        "pull_resume_checkpoint",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("no pull expected")),
    )

    rc = train_lightning.main(_args(tmp_path))
    assert rc == 1


def test_hub_resume_requires_the_canonical_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """A missing Hub snapshot must not start fresh from a stale local file."""
    monkeypatch.setattr(
        train_lightning, "detect_lightning_environment", lambda: _detect(has_gpu=True)
    )
    monkeypatch.setattr(train_lightning, "download_data", lambda *a, **k: True)
    calls = 0

    def missing_snapshot(**kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(train_lightning, "pull_resume_checkpoint", missing_snapshot)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "dit_model_ema.pt").write_bytes(b"stale local state")

    import train_dit

    monkeypatch.setattr(
        train_dit,
        "train_dit_local",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("training must not start without the canonical snapshot")
        ),
    )

    assert train_lightning.main(_args(tmp_path, ["--hub-resume"])) == 1
    assert calls == 1


def test_training_path_does_not_publish_legacy_flat_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    """The provider must not publish a parallel legacy flat layout."""
    monkeypatch.setattr(
        train_lightning,
        "detect_lightning_environment",
        lambda: _detect(has_gpu=True),
    )
    monkeypatch.setattr(train_lightning, "download_data", lambda *a, **k: True)

    import train_dit

    monkeypatch.setattr(train_dit, "train_dit_local", lambda **kwargs: 0.5)

    import gpu_pool

    def fail_legacy_publish(**kwargs):
        raise AssertionError(
            f"legacy flat publish attempted: {kwargs.get('checkpoint_name')}"
        )

    monkeypatch.setattr(gpu_pool, "push_checkpoint_to_hub", fail_legacy_publish)

    assert train_lightning.main(_args(tmp_path)) == 0


def test_backstop_skipped_when_hub_push_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        train_lightning, "detect_lightning_environment", lambda: _detect(has_gpu=True)
    )
    monkeypatch.setattr(train_lightning, "download_data", lambda *a, **k: True)
    monkeypatch.setattr(train_lightning, "pull_resume_checkpoint", lambda **k: None)

    import train_dit

    monkeypatch.setattr(train_dit, "train_dit_local", lambda **kwargs: 0.5)
    monkeypatch.setattr(
        train_lightning,
        "push_progress_backstop",
        lambda **k: (_ for _ in ()).throw(AssertionError("backstop expected off")),
    )

    rc = train_lightning.main(_args(tmp_path, ["--no-hub-push"]))
    assert rc == 0


def test_sync_only_skips_gpu_data_and_training(tmp_path: Path, monkeypatch) -> None:
    """Sync-only remains a CPU-capable Hub lookup, never a training run."""
    monkeypatch.setattr(
        train_lightning, "detect_lightning_environment", lambda: _detect(has_gpu=False)
    )
    pulls = 0

    def pull_only(**kwargs):
        nonlocal pulls
        pulls += 1
        return None

    monkeypatch.setattr(train_lightning, "pull_resume_checkpoint", pull_only)
    monkeypatch.setattr(
        train_lightning,
        "download_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download expected")),
    )

    import train_dit

    monkeypatch.setattr(
        train_dit,
        "train_dit_local",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("no training expected")),
    )

    assert train_lightning.main(_args(tmp_path, ["--sync-only"])) == 0
    assert pulls == 1
