"""Warm starts must reach the same optimum as a cold start."""

from __future__ import annotations

import numpy as np

from pcglasso import PCGLasso, pcglasso


def test_warm_equals_cold(data_X, method):
    cold = PCGLasso(alpha=0.1, method=method, warm_start=False).fit(data_X)

    warm = PCGLasso(alpha=0.1, method=method, warm_start=True)
    warm.fit(data_X)
    warm.fit(data_X)  # second fit resumes from the first solution

    assert np.allclose(cold.precision_, warm.precision_, atol=1e-4)


def test_warm_state_roundtrip(spd_cov, method):
    """Feeding a previous result's state_ back in matches a fresh solve."""
    first = pcglasso(spd_cov, alpha=0.1, method=method)
    resumed = pcglasso(spd_cov, alpha=0.1, method=method, init=first.state_)
    assert np.allclose(first.precision_, resumed.precision_, atol=1e-4)


def test_warm_start_mismatched_dim_ignored(method):
    """A warm state of the wrong size is ignored, not fatal."""
    rng = np.random.default_rng(1)
    model = PCGLasso(alpha=0.1, method=method, warm_start=True)
    model.fit(rng.standard_normal((100, 8)))
    # Different feature count: warm state cannot apply, should still fit.
    model.fit(rng.standard_normal((100, 6)))
    assert model.precision_.shape == (6, 6)
