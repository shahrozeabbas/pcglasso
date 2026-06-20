"""Shared fixtures and covariance builders for the pcglasso test suite.

All matrices are constructed deterministically from a seeded RNG so tests are
reproducible. Expected values are derived from mathematical properties of the
PCGLASSO estimator (symmetry, positive-definiteness, scale invariance, etc.)
rather than golden reference numbers, keeping the suite robust to ULP-level
solver differences.
"""

from __future__ import annotations

import numpy as np
import pytest


def make_spd_cov(p: int, n: int, seed: int) -> np.ndarray:
    """Empirical covariance of a hub-and-spoke Gaussian (positive definite)."""
    rng = np.random.default_rng(seed)
    prec = np.eye(p)
    spokes = min(p - 1, max(3, p // 4))
    for j in range(1, 1 + spokes):
        prec[0, j] = prec[j, 0] = 0.35

    prec = (prec + prec.T) * 0.5
    evals = np.linalg.eigvalsh(prec)
    if evals.min() <= 0:
        prec += (abs(evals.min()) + 0.05) * np.eye(p)

    cov = np.linalg.inv(prec)
    cov = (cov + cov.T) * 0.5
    chol = np.linalg.cholesky(cov)
    x = rng.standard_normal((n, p)) @ chol.T
    s = (x.T @ x) / n
    return np.ascontiguousarray((s + s.T) * 0.5)


def make_rank_deficient_cov(p: int, n: int, seed: int) -> np.ndarray:
    """Empirical covariance with n < p, so S is rank deficient (k > 0)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    s = (x.T @ x) / n
    return np.ascontiguousarray((s + s.T) * 0.5)


@pytest.fixture
def spd_cov() -> np.ndarray:
    """A modest, well-conditioned SPD covariance (p=12)."""
    return make_spd_cov(p=12, n=200, seed=1234)


@pytest.fixture
def rank_deficient_cov() -> np.ndarray:
    """A rank-deficient covariance (p=20, n=12)."""
    return make_rank_deficient_cov(p=20, n=12, seed=4321)


@pytest.fixture
def data_X() -> np.ndarray:
    """A raw data matrix (n_samples, n_features) for estimator tests."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((150, 10))


@pytest.fixture(params=["primal", "dual"])
def method(request) -> str:
    """Parametrize tests across both solver methods."""
    return request.param
