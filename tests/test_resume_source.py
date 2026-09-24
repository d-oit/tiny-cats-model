"""tests/test_resume_source.py

Issue #163 regression: the ``--hub-resume`` checkpoint must outrank a *local*
checkpoint.

HF Hub is the authoritative cross-provider transport (ADR-064). The previous
ordering only pulled from the Hub when no local checkpoint existed, so a stale
file left on a Modal volume shadowed the real experiment state forever — every
scheduled ``train-pool`` slice resumed the stale file and died on the manifest
gate (``warmup_steps: checkpoint=10 current=2000``).
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_dit import resolve_resume_checkpoint


def zip_checkpoint(path: Path, marker: bytes = b"stub") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", marker)
    return path


def resolve(tmp_path: Path, **overrides: object) -> str | None:
    values: dict[str, object] = {
        "explicit": None,
        "output": tmp_path / "checkpoints" / "pool" / "dit_model.pt",
        "hub_pulled": None,
        "logger": logging.getLogger("test-resume-source"),
    }
    values.update(overrides)
    return resolve_resume_checkpoint(**values)  # type: ignore[arg-type]


def test_fresh_run_has_no_resume_source(tmp_path: Path) -> None:
    assert resolve(tmp_path) is None


def test_local_checkpoint_is_used_without_hub(tmp_path: Path) -> None:
    local = zip_checkpoint(tmp_path / "checkpoints" / "pool" / "dit_model.pt")
    assert resolve(tmp_path) == str(local)


def test_hub_checkpoint_outranks_a_stale_local_checkpoint(tmp_path: Path) -> None:
    """The regression: a local file must not shadow the authoritative Hub state."""
    zip_checkpoint(tmp_path / "checkpoints" / "pool" / "dit_model.pt")
    pulled = str(tmp_path / "pulled" / "step-0050000" / "dit_model_ema.pt")
    assert resolve(tmp_path, hub_pulled=pulled) == pulled


def test_hub_checkpoint_used_when_no_local_checkpoint(tmp_path: Path) -> None:
    pulled = str(tmp_path / "pulled" / "dit_model_ema.pt")
    assert resolve(tmp_path, hub_pulled=pulled) == pulled


def test_explicit_resume_outranks_hub_and_local(tmp_path: Path) -> None:
    zip_checkpoint(tmp_path / "checkpoints" / "pool" / "dit_model.pt")
    assert (
        resolve(
            tmp_path,
            explicit="/somewhere/else/model.pt",
            hub_pulled=str(tmp_path / "pulled" / "dit_model_ema.pt"),
        )
        == "/somewhere/else/model.pt"
    )


def test_non_zip_local_file_starts_fresh(tmp_path: Path) -> None:
    stale = tmp_path / "checkpoints" / "pool" / "dit_model.pt"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"not-a-zip")
    assert resolve(tmp_path) is None
