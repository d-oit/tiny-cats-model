"""tests/test_app_gradio.py

The demo previously fed ``x``/``t``/``breeds`` to the ONNX graph, but the
exporter declares ``noise``/``timestep``/``breed`` — so generation against the
published artifact failed at ``session.run``. These tests pin the name mapping
that fixes it (caught by the ONNX parity check, 2026-10-02).
"""

from __future__ import annotations

import pytest

# The demo module needs the demo deps (gradio, onnxruntime); CI installs them.
pytest.importorskip("gradio", reason="gradio not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from app_gradio import _generator_feed_keys


class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def get_inputs(self) -> list[_Spec]:
        return [_Spec(name) for name in self._names]


def test_exported_convention() -> None:
    """The exporter's noise/timestep/breed names must be used as-is."""
    session = _FakeSession(["noise", "timestep", "breed"])
    assert _generator_feed_keys(session) == ("noise", "timestep", "breed")


def test_legacy_convention() -> None:
    """Older hand-made graphs used x/t/breeds."""
    session = _FakeSession(["x", "t", "breeds"])
    assert _generator_feed_keys(session) == ("x", "t", "breeds")


def test_input_order_does_not_matter() -> None:
    session = _FakeSession(["breed", "noise", "timestep"])
    assert _generator_feed_keys(session) == ("noise", "timestep", "breed")


def test_unrecognized_graph_raises() -> None:
    session = _FakeSession(["image", "time", "label"])
    with pytest.raises(ValueError, match="missing expected inputs"):
        _generator_feed_keys(session)
