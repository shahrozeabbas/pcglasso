"""Mathematical invariants of the PCGLASSO estimate (tolerance-based)."""

from __future__ import annotations

import numpy as np

from pcglasso import pcglasso

from conftest import make_spd_cov


def test_precision_symmetric(spd_cov, method):
    res = pcglasso(spd_cov, alpha=0.1, method=method)
    assert np.allclose(res.precision_, res.precision_.T, atol=1e-8)


def test_precision_positive_definite(spd_cov, method):
    res = pcglasso(spd_cov, alpha=0.1, method=method)
    evals = np.linalg.eigvalsh(res.precision_)
    assert evals.min() > 0.0


def test_partial_correlation_unit_diagonal(spd_cov, method):
    res = pcglasso(spd_cov, alpha=0.1, method=method)
    assert np.allclose(np.diag(res.partial_correlation_), 1.0, atol=1e-8)


def test_covariance_is_precision_inverse(spd_cov, method):
    res = pcglasso(spd_cov, alpha=0.1, method=method)
    identity = res.precision_ @ res.covariance_
    assert np.allclose(identity, np.eye(spd_cov.shape[0]), atol=1e-6)


def test_adjacency_consistency(spd_cov, method):
    res = pcglasso(spd_cov, alpha=0.1, method=method)
    assert res.adjacency_.dtype == bool
    assert not np.any(np.diag(res.adjacency_))
    off = ~np.eye(spd_cov.shape[0], dtype=bool)
    expected = (res.partial_correlation_ != 0.0) & off
    assert np.array_equal(res.adjacency_, expected)


def test_scale_invariance(spd_cov):
    """Scaling variables leaves the partial correlations unchanged."""
    p = spd_cov.shape[0]
    rng = np.random.default_rng(99)
    scales = rng.uniform(0.5, 5.0, size=p)
    d = np.diag(scales)
    scaled = d @ spd_cov @ d

    base = pcglasso(spd_cov, alpha=0.1, c=1.0)
    rescaled = pcglasso(scaled, alpha=0.1, c=1.0)
    assert np.allclose(
        base.partial_correlation_, rescaled.partial_correlation_, atol=1e-4
    )


def test_sparsity_increases_with_alpha():
    s = make_spd_cov(p=20, n=400, seed=2024)
    low = pcglasso(s, alpha=0.02)
    high = pcglasso(s, alpha=0.4)
    n_low = int(np.sum(low.adjacency_))
    n_high = int(np.sum(high.adjacency_))
    assert n_high <= n_low


def test_high_alpha_is_diagonal():
    """A very strong penalty drives the graph empty (diagonal precision)."""
    s = make_spd_cov(p=10, n=300, seed=55)
    res = pcglasso(s, alpha=5.0)
    assert int(np.sum(res.adjacency_)) == 0
