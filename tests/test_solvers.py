"""Solver behaviour: primal/dual agreement, c defaults, convergence."""

from __future__ import annotations

import numpy as np

from pcglasso import pcglasso

from conftest import make_spd_cov


def test_primal_dual_agree(spd_cov):
    primal = pcglasso(spd_cov, alpha=0.15, method="primal", tol=1e-6, max_iter=5000)
    dual = pcglasso(spd_cov, alpha=0.15, method="dual", tol=1e-6, max_iter=5000)
    assert np.allclose(
        primal.precision_, dual.precision_, atol=1e-2, rtol=1e-2
    )


def test_primal_dual_objective_close(spd_cov):
    primal = pcglasso(spd_cov, alpha=0.15, method="primal", tol=1e-6, max_iter=5000)
    dual = pcglasso(spd_cov, alpha=0.15, method="dual", tol=1e-6, max_iter=5000)
    assert abs(primal.objective_ - dual.objective_) < 1e-2


def test_c_default_pd_is_one(spd_cov):
    res = pcglasso(spd_cov, alpha=0.1, c=None)
    assert res.c_ == 1.0


def test_c_default_rank_deficient(rank_deficient_cov):
    res = pcglasso(rank_deficient_cov, alpha=0.1, c=None)
    # For a rank-deficient S the default is 0.9 * (1 - k/p) < 1.
    assert 0.0 < res.c_ < 1.0


def test_converges_within_max_iter(spd_cov, method):
    res = pcglasso(spd_cov, alpha=0.2, method=method, tol=1e-4, max_iter=5000)
    assert res.converged_
    assert res.n_iter_ <= 5000


def test_zero_penalty_runs(spd_cov, method):
    """alpha = 0 is the unpenalised problem; it must still solve."""
    res = pcglasso(spd_cov, alpha=0.0, method=method)
    assert np.all(np.isfinite(res.precision_))
    assert np.allclose(res.precision_, res.precision_.T, atol=1e-8)


def test_larger_problem_converges(method):
    s = make_spd_cov(p=40, n=120, seed=321)
    res = pcglasso(s, alpha=0.15, method=method, max_iter=5000)
    assert res.converged_
