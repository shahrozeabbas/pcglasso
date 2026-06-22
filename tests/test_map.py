"""Tests for pcglasso_map: parity vs single-fit, thread invariance, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from pcglasso import PCGLasso, pcglasso_map


def _make_data(n_samples: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def _random_subsets(n_features: int, n_subsets: int, p_min: int, p_max: int, seed: int):
    rng = np.random.default_rng(seed)
    subsets = []
    for _ in range(n_subsets):
        p = int(rng.integers(p_min, p_max + 1))
        cols = rng.choice(n_features, size=p, replace=False)
        subsets.append(np.sort(cols))
    return subsets


def test_map_parity_vs_estimator(method):
    """Full-rank subsets: map matches PCGLasso().fit on each column subset."""
    n_samples, n_features = 80, 20
    X = _make_data(n_samples, n_features, seed=11)
    index_sets = _random_subsets(n_features, n_subsets=12, p_min=4, p_max=10, seed=22)

    mapped = pcglasso_map(
        X,
        index_sets,
        alpha=0.12,
        method=method,
        max_iter=2000,
        tol=1e-5,
        n_jobs=1,
    )

    for subset, res in zip(index_sets, mapped):
        p = len(subset)
        assert n_samples - 1 >= p
        ref = PCGLasso(
            alpha=0.12,
            method=method,
            max_iter=2000,
            tol=1e-5,
        ).fit(X[:, subset])

        assert res.covariance_ is None
        assert np.allclose(
            res.partial_correlation_,
            ref.partial_correlation_,
            atol=1e-4,
        )
        assert np.allclose(res.precision_, ref.precision_, atol=1e-4)
        assert np.array_equal(res.adjacency_, ref.adjacency_)
        assert res.converged_ == ref.converged_


def test_map_thread_invariance():
    X = _make_data(60, 16, seed=3)
    index_sets = _random_subsets(16, n_subsets=20, p_min=3, p_max=8, seed=4)

    one = pcglasso_map(X, index_sets, alpha=0.1, n_jobs=1, tol=1e-5)
    four = pcglasso_map(X, index_sets, alpha=0.1, n_jobs=4, tol=1e-5)

    assert len(one) == len(four)
    for a, b in zip(one, four):
        assert np.array_equal(a.precision_, b.precision_)
        assert a.objective_ == b.objective_
        assert a.n_iter_ == b.n_iter_
        assert a.converged_ == b.converged_


def test_map_singleton_subset():
    X = _make_data(50, 5, seed=5)
    res = pcglasso_map(X, [[2]], alpha=0.1, n_jobs=1)
    assert len(res) == 1
    assert res[0].precision_.shape == (1, 1)
    assert res[0].converged_ is False


def test_map_zero_variance_column():
    X = _make_data(40, 4, seed=6)
    X[:, 1] = 3.0
    res = pcglasso_map(X, [[0, 1, 2]], alpha=0.1, n_jobs=1)
    assert len(res) == 1
    assert res[0].converged_ is False


def test_map_rank_deficient_subset():
    """Arithmetic-c branch: n_samples < p_sub."""
    n_samples, n_features = 8, 12
    X = _make_data(n_samples, n_features, seed=7)
    subset = np.arange(10)
    res = pcglasso_map(X, [subset], alpha=0.15, n_jobs=1, max_iter=1000)
    assert len(res) == 1
    assert res[0].c_ < 1.0
    assert res[0].precision_.shape == (10, 10)
    assert np.all(np.isfinite(res[0].precision_))


def test_map_empty_index_sets():
    X = _make_data(20, 5, seed=8)
    assert pcglasso_map(X, [], alpha=0.1) == []


def test_map_invalid_index_raises():
    X = _make_data(20, 5, seed=9)
    with pytest.raises(ValueError):
        pcglasso_map(X, [[0, 99]], alpha=0.1)
