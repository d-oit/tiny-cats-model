"""tests/test_train_kaggle.py

The shared backstop tests cover recovery between push boundaries and safe
no-op behavior at a boundary or without training state. Kaggle-specific data
preparation (`--force-download` fingerprint parity) stays in `train_kaggle`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import train_kaggle

import slice_runtime


def _write_state(checkpoint_dir: Path, completed: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps({"completed_steps": completed, "experiment_id": "exp"})
    )
    for name in ("dit_model.pt", "dit_model_ema.pt"):
        (checkpoint_dir / name).write_bytes(b"ckpt")


def test_pushes_when_the_run_ended_between_intervals(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []

    def fake_push(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    import gpu_pool

    monkeypatch.setattr(gpu_pool, "push_checkpoint_to_hub", fake_push)
    _write_state(tmp_path, completed=531_238)

    slice_runtime.push_progress_backstop(
        checkpoint_dir=tmp_path,
        hub_repo="d-oit/tiny-cats-model",
        experiment_id="dit-breed-conditioned-v4",
        hub_push_interval=5_000,
    )

    assert len(calls) == 2  # model + EMA
    assert calls[0]["completed_steps"] == 531_238
    assert calls[0]["experiment_id"] == "dit-breed-conditioned-v4"
    assert calls[0]["training_state_path"] == tmp_path / "training_state.json"


def test_skips_when_the_run_ended_on_a_push_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []

    import gpu_pool

    monkeypatch.setattr(
        gpu_pool, "push_checkpoint_to_hub", lambda **kwargs: calls.append(kwargs)
    )
    _write_state(tmp_path, completed=530_000)

    slice_runtime.push_progress_backstop(
        checkpoint_dir=tmp_path,
        hub_repo="d-oit/tiny-cats-model",
        experiment_id="dit-breed-conditioned-v4",
        hub_push_interval=5_000,
    )

    assert calls == []  # the trainer already pushed this exact step


def test_missing_state_is_not_fatal(tmp_path: Path, monkeypatch) -> None:
    import gpu_pool

    monkeypatch.setattr(
        gpu_pool, "push_checkpoint_to_hub", lambda **kwargs: pytest_fail()
    )

    slice_runtime.push_progress_backstop(
        checkpoint_dir=tmp_path,
        hub_repo="d-oit/tiny-cats-model",
        experiment_id="dit-breed-conditioned-v4",
        hub_push_interval=5_000,
    )


def test_force_download_discards_a_stale_staged_copy(
    tmp_path: Path, monkeypatch
) -> None:
    """A leftover /kaggle/working copy must not survive --force-download.

    Kaggle persists the working directory across kernel versions; a copy from
    an earlier mounted-dataset run previously short-circuited the download and
    gave the slice a different dataset identity (step-540000, 2026-11-02).
    """
    data_dir = tmp_path / "cats-data"
    (data_dir / "cat").mkdir(parents=True)
    stale = data_dir / "cat" / "Abyssinian_1.jpg"
    stale.write_bytes(b"stale")
    calls = 0
    timeouts: list[float] = []

    def fake_run(_command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        timeouts.append(kwargs["timeout"])  # type: ignore[arg-type]
        assert not stale.exists(), "stale data must be gone before downloading"
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    # ``prepare_kaggle_data`` imports subprocess inside the function, so the
    # module attribute is the seam.
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert train_kaggle.prepare_kaggle_data(data_dir, force_download=True) is True
    assert calls == 1
    assert timeouts == [train_kaggle.KAGGLE_DATA_PREP_TIMEOUT_SECONDS]


def test_existing_data_is_reused_without_force_download(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "cats-data"
    data_dir.mkdir()
    (data_dir / "cat.jpg").write_bytes(b"x")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest_fail())

    assert train_kaggle.prepare_kaggle_data(data_dir, force_download=False) is True


def pytest_fail() -> bool:  # pragma: no cover - only reached on regression
    raise AssertionError("no push/subprocess expected in this path")
