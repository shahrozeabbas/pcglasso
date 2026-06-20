"""Input validation: bad inputs must raise, not silently misbehave."""

from __future__ import annotations

import numpy as np
import pytest

from pcglasso import PCGLasso, pcglasso


def test_non_symmetric_raises():
    s = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValueError):
        pcglasso(s, alpha=0.1)


def test_non_square_raises():
    s = np.ones((3, 4))
    with pytest.raises(ValueError):
        pcglasso(s, alpha=0.1)


def test_non_finite_raises():
    s = np.eye(3)
    s[0, 0] = np.nan
    with pytest.raises(ValueError):
        pcglasso(s, alpha=0.1)


def test_negative_alpha_raises(spd_cov):
    with pytest.raises(ValueError):
        pcglasso(spd_cov, alpha=-0.01)


def test_missing_alpha_raises(spd_cov):
    with pytest.raises((ValueError, TypeError)):
        pcglasso(spd_cov, alpha=None)


def test_invalid_method_raises(spd_cov):
    with pytest.raises(ValueError):
        pcglasso(spd_cov, alpha=0.1, method="bogus")


def test_nonpositive_c_raises(spd_cov):
    with pytest.raises(ValueError):
        pcglasso(spd_cov, alpha=0.1, c=0.0)


def test_non_positive_definite_diagonal_raises():
    s = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        pcglasso(s, alpha=0.1)


def test_estimator_requires_2d():
    model = PCGLasso(alpha=0.1)
    with pytest.raises(ValueError):
        model.fit(np.ones(10))


def test_estimator_requires_two_samples():
    model = PCGLasso(alpha=0.1)
    with pytest.raises(ValueError):
        model.fit(np.ones((1, 5)))
