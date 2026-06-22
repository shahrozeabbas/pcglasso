"""Functional API: ``pcglasso(S, alpha, ...)`` mirroring the R package's
entry point, for callers that already have a covariance matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _core
from ._base import _assemble_result, _pack_index_sets, _resolve_n_threads, solve_precision


@dataclass
class PCGLassoResult:
    """Result of a single :func:`pcglasso` or :func:`pcglasso_map` call.

    Attributes use the sklearn trailing-underscore convention. ``state_`` is the
    correlation-scale ``(R, W, d)`` warm-start triple; pass it back as ``init``
    for a subsequent, similar fit. ``covariance_`` may be ``None`` when the
    inverse was skipped (``pcglasso_map``).
    """

    precision_: np.ndarray
    partial_correlation_: np.ndarray
    adjacency_: np.ndarray
    covariance_: Optional[np.ndarray]
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


def pcglasso_map(
    X,
    index_sets,
    alpha,
    c=None,
    *,
    method="primal",
    max_iter=1000,
    tol=1e-4,
    n_jobs=-1,
):
    """Run PCGLASSO on many column subsets of a data matrix in parallel.

    Parameters
    ----------
    X : (n_samples, n_features) array_like
        Data matrix; each subset selects feature columns.
    index_sets : sequence of array_like
        Integer column indices per sub-problem (overlap and varying sizes OK).
    alpha : float
        L1 penalty on off-diagonal partial correlations.
    c : float, optional
        Diagonal parameter. ``None`` selects the arithmetic rank default
        (structural ``k = max(0, p - (n - 1))``) instead of an eigenvalue count.
    method : {"primal", "dual"}, default "primal"
        Coordinate-descent variant.
    max_iter : int, default 1000
        Maximum number of outer iterations per subset.
    tol : float, default 1e-4
        Outer convergence tolerance.
    n_jobs : int, default -1
        Parallel worker count. ``-1`` uses all available cores,
        ``-2`` all but one, ``1`` serial.

    Returns
    -------
    list of PCGLassoResult
        One result per ``index_sets`` entry. ``covariance_`` is ``None``
        (the model-implied covariance inverse is skipped for speed).
    """
    if alpha is None:
        raise ValueError("alpha (the L1 penalty) must be provided")
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if method not in ("primal", "dual"):
        raise ValueError("method must be 'primal' or 'dual'")
    if c is not None and float(c) <= 0.0:
        raise ValueError("c must be positive")

    x = np.ascontiguousarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
    if not np.all(np.isfinite(x)):
        raise ValueError("X must contain only finite values")

    index_sets = list(index_sets)
    if len(index_sets) == 0:
        return []

    n_features = x.shape[1]
    for subset in index_sets:
        cols = np.asarray(subset, dtype=np.int64).ravel()
        if cols.size == 0:
            continue
        if cols.min() < 0 or cols.max() >= n_features:
            raise ValueError("index_sets entries must be valid column indices")

    indptr, indices = _pack_index_sets(index_sets)
    n_threads = _resolve_n_threads(n_jobs)
    c_opt = None if c is None else float(c)

    raw = _core.pcglasso_solve_map(
        x,
        indptr,
        indices,
        alpha,
        c_opt,
        method,
        int(max_iter),
        float(tol),
        n_threads,
    )

    results = []
    for item in raw:
        R, dvec, sd, W, n_iter, converged, objective, c_val = item
        assembled = _assemble_result(
            R,
            dvec,
            sd,
            objective=objective,
            n_iter=n_iter,
            converged=converged,
            c_val=c_val,
            W=W,
            compute_covariance=False,
        )
        results.append(
            PCGLassoResult(
                precision_=assembled["precision"],
                partial_correlation_=assembled["partial_correlation"],
                adjacency_=assembled["adjacency"],
                covariance_=assembled["covariance"],
                objective_=assembled["objective"],
                n_iter_=assembled["n_iter"],
                converged_=assembled["converged"],
                c_=assembled["c"],
                state_=assembled["state"],
            )
        )
    return results
