"""Tests for the Gradio inference helpers."""

from __future__ import annotations

import pytest

# The demo module needs the demo deps (gradio, onnxruntime); CI installs them.
pytest.importorskip("gradio", reason="gradio not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

import numpy as np

from app_gradio import _generator_feed_keys, _preprocess_classifier, classify_cat


def test_preprocess_classifier_shape_and_dtype() -> None:
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    tensor = _preprocess_classifier(image)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


class _Spec:
    def __init__(self, name: str, shape: list[int | str | None] | None = None) -> None:
        self.name = name
        self.shape = shape or [None, 3, 224, 224]


class _FakeSession:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def get_inputs(self) -> list[_Spec]:
        return [_Spec(name) for name in self._names]


def test_classifier_maps_two_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app_gradio.get_session", lambda _model_type: _ClassifierSession(2)
    )
    result = classify_cat(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result == {
        "Cat": pytest.approx(0.2689414),
        "Not a cat": pytest.approx(0.7310586),
    }


def test_classifier_rejects_thirteen_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app_gradio.get_session", lambda _model_type: _ClassifierSession(13)
    )
    result = classify_cat(np.zeros((224, 224, 3), dtype=np.uint8))
    assert isinstance(result, str)
    assert "expected 2" in result


class _ClassifierSession(_FakeSession):
    def __init__(self, output_count: int) -> None:
        super().__init__(["classifier_input"])
        self.output_count = output_count

    def run(self, _outputs: object, _feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.arange(self.output_count, dtype=np.float32)[None, :]]


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
