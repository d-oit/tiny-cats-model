"""tests/test_optimize_onnx.py

Unit tests for src/optimize_onnx.py.

Covers:
- _build_generator_inputs shape/dtype contract (noise + timestep + breed)
- _build_generator_inputs respects the ``image_size`` kwarg
- validate_accuracy rejects unknown ``model_type`` (regression guard for the
  silent-NaN failure mode where neither classifier nor generator branch runs)

These tests guard the refactor that fixes the post-training ONNX export
warning surfaced by the DiT smoke run -- the original implementation built
a single-image feed dict inside validate_accuracy even when called with
``model_type="generator"``, which produced a feed dict that did not contain
``timestep`` and ``breed`` and therefore failed ONNXRuntime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# optimize_onnx imports onnxruntime at module load. Skip the entire file in
# environments where onnxruntime is not installed (CI image is expected to
# have it via requirements*.txt).
pytest.importorskip("onnxruntime", reason="optimize_onnx.py requires onnxruntime")


def test_build_generator_inputs_shape_and_dtype() -> None:
    """_build_generator_inputs(0) emits the documented feed dict that matches
    the shape/dtype contract used by GeneratorCalibrationDataReader.
    """
    from optimize_onnx import _build_generator_inputs

    feeds = _build_generator_inputs(seed=0)

    assert set(feeds.keys()) == {"noise", "timestep", "breed"}

    # noise: (1, 3, DEFAULT_GENERATOR_SIZE=128, DEFAULT_GENERATOR_SIZE=128) float32
    assert feeds["noise"].shape == (1, 3, 128, 128)
    assert feeds["noise"].dtype == np.float32

    # timestep: (1,) float32 in [0, 1)
    assert feeds["timestep"].shape == (1,)
    assert feeds["timestep"].dtype == np.float32
    assert 0.0 <= float(feeds["timestep"].item()) < 1.0

    # breed: (1,) int64 in [0, 13) -- 12 cat breeds + null (index 13) exists
    # in the embedder but is never sampled here because calibration data is
    # constrained to the 13 real breed classes.
    assert feeds["breed"].shape == (1,)
    assert feeds["breed"].dtype == np.int64
    assert 0 <= int(feeds["breed"].item()) < 13


def test_build_generator_inputs_custom_image_size() -> None:
    """_build_generator_inputs respects the ``image_size`` kwarg for
    non-default spatial sizes (e.g., 64 or 96) that callers may use for
    testing or downsampled calibration runs.
    """
    from optimize_onnx import _build_generator_inputs

    feeds_64 = _build_generator_inputs(seed=0, image_size=64)
    assert feeds_64["noise"].shape == (1, 3, 64, 64)
    assert feeds_64["noise"].dtype == np.float32

    feeds_96 = _build_generator_inputs(seed=0, image_size=96)
    assert feeds_96["noise"].shape == (1, 3, 96, 96)
    assert feeds_96["timestep"].shape == (1,)
    assert feeds_96["breed"].shape == (1,)


def test_build_generator_inputs_is_deterministic_per_seed() -> None:
    """Seeded calls produce byte-identical feeds. Important so that
    calibration and validation inputs do not drift apart when both call
    the helper with seeds derived from the loop index.
    """
    from optimize_onnx import _build_generator_inputs

    feeds_a = _build_generator_inputs(seed=42)
    feeds_b = _build_generator_inputs(seed=42)
    for key in feeds_a:
        assert np.array_equal(feeds_a[key], feeds_b[key]), f"{key} mismatch"


def test_validate_accuracy_rejects_unknown_model_type() -> None:
    """validate_accuracy raises ValueError for unsupported ``model_type``
    values, before attempting to load any ONNX file. Without this guard the
    function would skip the validation loop (no feeds built, no diff
    accumulated) and ``np.mean([])`` would silently return NaN.
    """
    from optimize_onnx import validate_accuracy

    # Plain invalid string
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_accuracy(
            "definitely_does_not_exist.onnx",
            "definitely_does_not_exist.onnx",
            num_samples=5,
            model_type="invalid",
        )

    # Empty string is also invalid
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_accuracy(
            "fake.onnx",
            "fake.onnx",
            num_samples=1,
            model_type="",
        )

    # Typo of a valid type
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_accuracy(
            "fake.onnx",
            "fake.onnx",
            num_samples=1,
            model_type="genrator",  # intentional typo
        )
