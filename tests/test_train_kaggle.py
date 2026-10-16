"""tests/test_train_kaggle.py

The Kaggle trainer's canonical Hub backstop (`push_progress_backstop`): a run
that ends between push intervals gets one canonical safety push; a run that
ended exactly on a boundary does not re-upload the same snapshot (ADR-064/067).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import train_kaggle


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

    train_kaggle.push_progress_backstop(
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

    train_kaggle.push_progress_backstop(
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

    train_kaggle.push_progress_backstop(
        checkpoint_dir=tmp_path,
        hub_repo="d-oit/tiny-cats-model",
        experiment_id="dit-breed-conditioned-v4",
        hub_push_interval=5_000,
    )


def pytest_fail() -> bool:  # pragma: no cover - only reached on regression
    raise AssertionError("no push expected without a training state")
