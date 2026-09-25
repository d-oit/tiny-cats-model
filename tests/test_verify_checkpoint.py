"""tests/test_verify_checkpoint.py

Issue #163 WP7/WP8: ``verify_checkpoint.py`` is the publication gate used by
``train.yml``. Its default contract is "ONNX problems are a warning", and
``--require-onnx`` upgrades that to a hard failure so a broken export can never
be published as a verified final model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import verify_checkpoint as vc


@pytest.fixture
def valid_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the (heavy) real checkpoint load: report a valid checkpoint."""
    monkeypatch.setattr(
        vc,
        "verify_checkpoint",
        lambda _path: {"valid": True, "path": "ckpt.pt"},
    )


def test_invalid_checkpoint_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vc, "verify_checkpoint", lambda _path: {"valid": False, "error": "corrupt"}
    )
    with pytest.raises(SystemExit) as excinfo:
        vc.main(["--checkpoint", "ckpt.pt", "--skip-onnx"])
    assert excinfo.value.code == 1


def test_onnx_failure_is_a_warning_by_default(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vc, "verify_onnx_inference", lambda _path: {"valid": False, "error": "boom"}
    )
    # No SystemExit: the checkpoint is valid and ONNX is advisory here.
    vc.main(["--checkpoint", "ckpt.pt", "--onnx", "model.onnx"])


def test_require_onnx_fails_when_onnx_invalid(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vc, "verify_onnx_inference", lambda _path: {"valid": False, "error": "boom"}
    )
    with pytest.raises(SystemExit) as excinfo:
        vc.main(["--checkpoint", "ckpt.pt", "--onnx", "model.onnx", "--require-onnx"])
    assert excinfo.value.code == 1


def test_require_onnx_passes_when_onnx_valid(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vc, "verify_onnx_inference", lambda _path: {"valid": True, "error": None}
    )
    vc.main(["--checkpoint", "ckpt.pt", "--onnx", "model.onnx", "--require-onnx"])


def test_skip_onnx_never_calls_onnx_verification(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_path: str) -> dict:
        raise AssertionError("ONNX verification must not run with --skip-onnx")

    monkeypatch.setattr(vc, "verify_onnx_inference", boom)
    vc.main(["--checkpoint", "ckpt.pt", "--skip-onnx"])
