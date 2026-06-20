"""sklearn-style estimator for the Partial Correlation Graphical LASSO."""

from __future__ import annotations

import numpy as np

from ._base import solve_precision

_PARAM_NAMES = (
    "alpha",
    "c",
    "method",
    "max_iter",
    "tol",
    "warm_start",
    "assume_centered",
)


class PCGLasso:
    """Partial Correlation Graphical LASSO.

    Estimates a sparse Gaussian precision matrix by penalising the *partial
    correlations* (rather than the raw precision entries as in the graphical
    LASSO), which makes the estimator scale invariant.

    Parameters
    ----------
    alpha : float
        L1 penalty on the off-diagonal partial correlations.
    c : float, optional
        Diagonal parameter. ``None`` selects a data-dependent default
        (``1`` if the correlation matrix is positive definite, otherwise
        ``0.9 * (1 - k/p)``).
    method : {"primal", "dual"}, default "primal"
        Coordinate-descent variant. ``"primal"`` (``pcglassoFast``) uses the
        closed-form Theorem-5 kernel and returns partial correlations directly;
        ``"dual"`` (``pcglassoFast_Dual``) adapts the GLASSO dual.
    max_iter : int, default 1000
        Maximum number of outer (biconvex) iterations.
    tol : float, default 1e-4
        Outer convergence tolerance.
    warm_start : bool, default False
        When ``True``, reuse the solution of the previous :meth:`fit` as the
        starting point of the next one. Useful for fitting a sequence of
        similar problems (e.g. bootstrap resamples).
    assume_centered : bool, default False
        If ``True``, ``X`` is assumed already centred and the covariance is
        computed about the origin; otherwise the sample mean is removed.

    Attributes
    ----------
    precision_ : ndarray of shape (p, p)
        Estimated precision matrix.
    partial_correlation_ : ndarray of shape (p, p)
        Partial correlations (unit diagonal); off-diagonal zeros encode
        conditional independence.
    adjacency_ : ndarray of shape (p, p), dtype bool
        Conditional-dependence graph (off-diagonal nonzeros, zero diagonal).
    covariance_ : ndarray of shape (p, p)
        Model-implied covariance (inverse of ``precision_``).
    location_ : ndarray of shape (p,)
        Estimated mean (zeros if ``assume_centered``).
    objective_ : float
        Final penalised objective value (correlation scale).
    n_iter_ : int
        Number of outer iterations run.
    converged_ : bool
        Whether the tolerance was met before ``max_iter``.
    c_ : float
        The resolved diagonal parameter.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray
        Feature names, if ``X`` was a DataFrame.
    """

    def __init__(
        self,
        alpha,
        c=None,
        *,
        method="primal",
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        assume_centered=False,
    ):
        self.alpha = alpha
        self.c = c
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.assume_centered = assume_centered

    # -- sklearn-compatible parameter access ------------------------------
    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in _PARAM_NAMES:
                raise ValueError(f"Invalid parameter {key!r} for PCGLasso")
            setattr(self, key, value)
        return self

    # -- fitting ----------------------------------------------------------
    def _empirical_covariance(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        n_samples = X_arr.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to estimate a covariance")
        if self.assume_centered:
            location = np.zeros(X_arr.shape[1])
            Xc = X_arr
        else:
            location = X_arr.mean(axis=0)
            Xc = X_arr - location
        S = (Xc.T @ Xc) / n_samples
        return S, location

    def fit(self, X, y=None):
        """Fit the estimator to data ``X`` of shape (n_samples, n_features)."""
        feature_names = getattr(X, "columns", None)
        S, location = self._empirical_covariance(X)
        p = S.shape[0]

        init = None
        if self.warm_start:
            state = getattr(self, "_state", None)
            if state is not None and state[0].shape[0] == p:
                init = state

        res = solve_precision(
            S,
            self.alpha,
            c=self.c,
            method=self.method,
            max_iter=self.max_iter,
            tol=self.tol,
            init=init,
        )

        self.precision_ = res["precision"]
        self.partial_correlation_ = res["partial_correlation"]
        self.adjacency_ = res["adjacency"]
        self.covariance_ = res["covariance"]
        self.location_ = location
        self.objective_ = res["objective"]
        self.n_iter_ = res["n_iter"]
        self.converged_ = res["converged"]
        self.c_ = res["c"]
        self.n_features_in_ = p
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self._state = res["state"]
        return self
