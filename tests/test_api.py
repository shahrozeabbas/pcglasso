"""API surface: return types, shapes, dtypes, attributes, sklearn params."""

from __future__ import annotations

import numpy as np
import pytest

from pcglasso import PCGLasso, PCGLassoResult, pcglasso


def test_functional_returns_result(spd_cov, method):
    res = pcglasso(spd_cov, alpha=0.1, method=method)
    assert isinstance(res, PCGLassoResult)
    p = spd_cov.shape[0]
    assert res.precision_.shape == (p, p)
    assert res.partial_correlation_.shape == (p, p)
    assert res.adjacency_.shape == (p, p)
    assert res.covariance_.shape == (p, p)
    assert isinstance(res.objective_, float)
    assert isinstance(res.n_iter_, int)
    assert isinstance(res.converged_, bool)
    assert isinstance(res.c_, float)


def test_functional_dtypes(spd_cov):
    res = pcglasso(spd_cov, alpha=0.1)
    assert res.precision_.dtype == np.float64
    assert res.partial_correlation_.dtype == np.float64
    assert res.adjacency_.dtype == bool


def test_estimator_fit_returns_self(data_X, method):
    model = PCGLasso(alpha=0.1, method=method)
    out = model.fit(data_X)
    assert out is model


def test_estimator_attributes_present(data_X):
    model = PCGLasso(alpha=0.1).fit(data_X)
    p = data_X.shape[1]
    for attr in (
        "precision_",
        "partial_correlation_",
        "adjacency_",
        "covariance_",
        "location_",
        "objective_",
        "n_iter_",
        "converged_",
        "c_",
        "n_features_in_",
    ):
        assert hasattr(model, attr), attr
    assert model.precision_.shape == (p, p)
    assert model.location_.shape == (p,)
    assert model.n_features_in_ == p


def test_get_set_params_roundtrip():
    model = PCGLasso(alpha=0.1)
    params = model.get_params()
    assert set(params) == {
        "alpha",
        "c",
        "method",
        "max_iter",
        "tol",
        "warm_start",
        "assume_centered",
    }
    model.set_params(alpha=0.2, method="dual")
    assert model.alpha == 0.2
    assert model.method == "dual"


def test_set_params_rejects_unknown():
    model = PCGLasso(alpha=0.1)
    with pytest.raises(ValueError):
        model.set_params(not_a_param=1)


def test_feature_names_from_dataframe():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(6)]
    df = pd.DataFrame(rng.standard_normal((80, 6)), columns=cols)
    model = PCGLasso(alpha=0.1).fit(df)
    assert list(model.feature_names_in_) == cols
