from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safe_checkpoint import (
    ChecksumMismatchError,
    UnsafeCheckpointError,
    load_torch_checkpoint,
    verify_file_sha256,
)


class _Malicious:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (Path.write_text, (self.marker, "executed"))


def test_malicious_pickle_is_not_executed_by_default(tmp_path: Path) -> None:
    checkpoint = tmp_path / "malicious.pt"
    marker = tmp_path / "marker"
    torch.save(_Malicious(marker), checkpoint)

    with pytest.raises(UnsafeCheckpointError, match="allow_unsafe_legacy=True"):
        load_torch_checkpoint(checkpoint)
    assert not marker.exists()


def test_explicit_unsafe_opt_in_loads_trusted_legacy_pickle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    marker = tmp_path / "marker"
    torch.save(_Malicious(marker), checkpoint)

    load_torch_checkpoint(checkpoint, allow_unsafe_legacy=True)
    assert marker.read_text() == "executed"


def test_safe_dict_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "safe.pt"
    expected = {"weight": torch.arange(4), "epoch": 3, "config": {"width": 4}}
    torch.save(expected, checkpoint)
    actual = load_torch_checkpoint(checkpoint)
    assert actual["epoch"] == expected["epoch"]
    assert actual["config"] == expected["config"]
    torch.testing.assert_close(actual["weight"], expected["weight"])


def test_sha256_verification_detects_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    expected = hashlib.sha256(b"artifact").hexdigest()
    assert verify_file_sha256(artifact, expected) == expected
    with pytest.raises(ChecksumMismatchError, match="SHA-256 mismatch"):
        verify_file_sha256(artifact, hashlib.sha256(b"changed").hexdigest())
