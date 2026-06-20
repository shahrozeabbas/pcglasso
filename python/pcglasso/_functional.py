"""Functional API: ``pcglasso(S, alpha, ...)`` mirroring the R package's
entry point, for callers that already have a covariance matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._base import solve_precision


@dataclass
class PCGLassoResult:
    """Result of a single :func:`pcglasso` call.

    Attributes use the sklearn trailing-underscore convention. ``state_`` is the
    correlation-scale ``(R, W, d)`` warm-start triple; pass it back as ``init``
    for a subsequent, similar fit.
    """

    precision_: np.ndarray
    partial_correlation_: np.ndarray
    adjacency_: np.ndarray
    covariance_: np.ndarray
    objective_: float
    n_iter_: int
    converged_: bool
    c_: float
    state_: tuple


def pcglasso(
    S,
    alpha,
    c=None,
    *,
    method="primal",
    max_iter=1000,
    tol=1e-4,
    init=None,
):
    """Compute the PCGLASSO precision-matrix estimate from a covariance matrix.

    Parameters
    ----------
    S : (p, p) array_like
        Sample covariance (or correlation) matrix.
    alpha : float
        L1 penalty on the off-diagonal partial correlations (``rho`` in the R
        package / ``lambda`` in Bogdan et al.).
    c : float, optional
        Diagonal parameter. Defaults to ``1`` if ``S`` is positive definite,
        else ``0.9 * (1 - k/p)`` for a rank-``(p-k)`` matrix.
    method : {"primal", "dual"}, default "primal"
        Coordinate-descent variant.
    max_iter : int, default 1000
        Maximum number of outer (biconvex) iterations.
    tol : float, default 1e-4
        Outer convergence tolerance.
    init : tuple, optional
        A ``state_`` triple from a previous result, used as a warm start.

    Returns
    -------
    PCGLassoResult
    """
    res = solve_precision(
        S,
        alpha,
        c=c,
        method=method,
        max_iter=max_iter,
        tol=tol,
        init=init,
    )
    return PCGLassoResult(
        precision_=res["precision"],
        partial_correlation_=res["partial_correlation"],
        adjacency_=res["adjacency"],
        covariance_=res["covariance"],
        objective_=res["objective"],
        n_iter_=res["n_iter"],
        converged_=res["converged"],
        c_=res["c"],
        state_=res["state"],
    )
