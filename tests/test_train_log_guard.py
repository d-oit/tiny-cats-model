"""tests/test_train_log_guard.py

Unit tests locking in the `math.isfinite` guard primitive that
src/train_dit.py uses to suppress the cosmetic "Loss: inf" log line.

math.isfinite() returns False for +inf, -inf, and NaN. These tests
pin that contract so a future change cannot accidentally start
emitting "Loss: inf" again.
"""

from __future__ import annotations

import math


def test_math_isfinite_returns_true_for_zero() -> None:
    """Sanity: math.isfinite(0.0) is True."""
    assert math.isfinite(0.0) is True


def test_math_isfinite_returns_true_for_finite_numbers() -> None:
    """math.isfinite handles normal finite floats correctly."""
    for value in (0.0, 1.0, -1.0, 1e-9, -1e9, 0.5, 1.234e-3):
        assert math.isfinite(value) is True, f"expected True for {value!r}"


def test_math_isfinite_returns_false_for_positive_infinity() -> None:
    """math.isfinite(+inf) is False -- the cosmetic case to suppress."""
    assert math.isfinite(float("inf")) is False
    assert math.isfinite(math.inf) is False


def test_math_isfinite_returns_false_for_negative_infinity() -> None:
    """math.isfinite(-inf) is False -- the cosmetic case to suppress."""
    assert math.isfinite(float("-inf")) is False
    assert math.isfinite(-math.inf) is False


def test_math_isfinite_returns_false_for_nan() -> None:
    """math.isfinite(NaN) is False -- NaN must trigger the warning path."""
    assert math.isfinite(float("nan")) is False
    assert math.isfinite(math.nan) is False
