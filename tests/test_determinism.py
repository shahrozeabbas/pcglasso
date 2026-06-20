"""The solver is deterministic: identical inputs give identical outputs."""

from __future__ import annotations

import numpy as np

from pcglasso import pcglasso


def test_repeated_solve_identical(spd_cov, method):
    a = pcglasso(spd_cov, alpha=0.12, method=method)
    b = pcglasso(spd_cov, alpha=0.12, method=method)
    assert np.array_equal(a.precision_, b.precision_)
    assert a.objective_ == b.objective_
    assert a.n_iter_ == b.n_iter_
