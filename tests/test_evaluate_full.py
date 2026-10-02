"""tests/test_evaluate_full.py

Regression tests for the FID math in src/evaluate_full.py.

The previous Newton-Schulz matrix square root produced NaN whenever the
covariance was rank-deficient (fewer samples than feature dimensions) and
its call site silently swallowed failures into cov_term=0. FID must be
finite and ~0 for identical distributions in every case.
"""

from __future__ import annotations

import math

import pytest
import torch

from evaluate_full import _frechet_distance, _sqrtm_psd


def _cov_from_samples(num_samples: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randn(num_samples, dim, generator=generator)
    return torch.cov(samples.T)


def _mean(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=generator)


class TestSqrtmPsd:
    def test_square_root_of_identity_is_identity(self) -> None:
        eye = torch.eye(8)
        assert torch.allclose(_sqrtm_psd(eye), eye, atol=1e-5)

    def test_recovers_matrix_from_its_square(self) -> None:
        cov = _cov_from_samples(num_samples=16, dim=8, seed=0)
        sqrt = _sqrtm_psd(cov)
        assert torch.allclose(sqrt @ sqrt, cov, atol=1e-4)

    def test_singular_matrix_stays_finite(self) -> None:
        # Rank <= 8 < 32 dims: the old Newton-Schulz path returned NaN here.
        cov = _cov_from_samples(num_samples=8, dim=32, seed=1)
        sqrt = _sqrtm_psd(cov)
        assert torch.isfinite(sqrt).all()


class TestFrechetDistance:
    def test_identical_distributions_score_zero(self) -> None:
        mean = _mean(16, seed=2)
        cov = _cov_from_samples(num_samples=64, dim=16, seed=3)
        assert _frechet_distance(mean, cov, mean, cov) == pytest.approx(0.0, abs=1e-4)

    def test_identical_singular_distributions_score_zero(self) -> None:
        # Regression: identical sets scored NaN with n < dim (26 samples in a
        # 2048-dim Inception feature space, observed 2026-10-01).
        mean = _mean(32, seed=4)
        cov = _cov_from_samples(num_samples=8, dim=32, seed=5)
        fid = _frechet_distance(mean, cov, mean, cov)
        assert math.isfinite(fid)
        assert fid == pytest.approx(0.0, abs=1e-4)

    def test_mean_shift_matches_squared_distance(self) -> None:
        cov = torch.eye(8)
        mean1 = torch.zeros(8)
        mean2 = torch.zeros(8)
        mean2[0] = 3.0
        assert _frechet_distance(mean1, cov, mean2, cov) == pytest.approx(9.0, abs=1e-4)

    def test_symmetric(self) -> None:
        mean1, cov1 = _mean(8, seed=6), _cov_from_samples(32, 8, seed=7)
        mean2, cov2 = _mean(8, seed=8), _cov_from_samples(32, 8, seed=9)
        forward = _frechet_distance(mean1, cov1, mean2, cov2)
        backward = _frechet_distance(mean2, cov2, mean1, cov1)
        assert forward == pytest.approx(backward, abs=1e-4)

    def test_nonnegative(self) -> None:
        mean1, cov1 = _mean(8, seed=10), _cov_from_samples(32, 8, seed=11)
        mean2, cov2 = _mean(8, seed=12), _cov_from_samples(32, 8, seed=13)
        fid = _frechet_distance(mean1, cov1, mean2, cov2)
        assert fid >= -1e-6
