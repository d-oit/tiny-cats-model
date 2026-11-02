"""tests/test_pool_pointer.py

The canonical pointer check: a slice that ran but never flipped
``latest/manifest.json`` must fail, while an intentional no-op slice passes
(2026-11-02: the end-of-run push published the previous step and HuggingFace
skipped it as an empty commit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import check_pool_pointer as checker


def _pointer(step: int) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "dit-breed-conditioned-v4",
        "completed_steps": step,
        "step_dir": f"step-{step:06d}",
        "files": ["dit_model.pt"],
        "updated_at": "2026-11-02T10:00:00Z",
    }


def test_record_then_advancement_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = iter([540_100, 555_100])
    monkeypatch.setattr(
        checker, "fetch_pool_pointer", lambda **_: _pointer(next(steps))
    )
    record = tmp_path / "before.json"

    assert checker.main(["--record", str(record)]) == 0
    assert json.loads(record.read_text())["completed_steps"] == 540_100
    assert (
        checker.main(
            ["--record-file", str(record), "--expect-advanced", "--target", "555100"]
        )
        == 0
    )


def test_unchanged_pointer_fails_when_the_target_required_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(checker, "fetch_pool_pointer", lambda **_: _pointer(540_100))
    record = tmp_path / "before.json"
    record.write_text(json.dumps({"completed_steps": 540_100}))

    rc = checker.main(
        ["--record-file", str(record), "--expect-advanced", "--target", "555100"]
    )

    assert rc == 1
    assert "did not advance" in capsys.readouterr().out


def test_no_op_slice_does_not_require_advancement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(checker, "fetch_pool_pointer", lambda **_: _pointer(540_100))
    record = tmp_path / "before.json"
    record.write_text(json.dumps({"completed_steps": 540_100}))

    rc = checker.main(
        ["--record-file", str(record), "--expect-advanced", "--target", "540100"]
    )

    assert rc == 0
    assert "no pointer advancement is expected" in capsys.readouterr().out


def test_missing_pointer_after_a_slice_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checker, "fetch_pool_pointer", lambda **_: None)
    record = tmp_path / "before.json"
    record.write_text(json.dumps({"completed_steps": 540_100}))

    rc = checker.main(
        ["--record-file", str(record), "--expect-advanced", "--target", "555100"]
    )

    assert rc == 1


def test_missing_record_file_is_a_clear_failure(tmp_path: Path) -> None:
    rc = checker.main(
        [
            "--record-file",
            str(tmp_path / "absent.json"),
            "--expect-advanced",
            "--target",
            "555100",
        ]
    )

    assert rc == 1


def test_record_never_fails_the_pre_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hub hiccup before training must not block the slice."""

    def boom(**_: object) -> None:
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(checker, "fetch_pool_pointer", boom)
    record = tmp_path / "before.json"

    assert checker.main(["--record", str(record)]) == 0
    # The failure is recorded so the post-slice step skips with a warning
    # instead of claiming the pointer did not advance.
    document = json.loads(record.read_text())
    assert document["recorded_failure"]
    assert (
        checker.main(
            ["--record-file", str(record), "--expect-advanced", "--target", "555100"]
        )
        == 0
    )
