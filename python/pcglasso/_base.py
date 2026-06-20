"""Shared plumbing: covariance -> correlation, parameter defaults, the call
into the Rust core, and rescaling back to the original variable scale.

All of the linear algebra that benefits from LAPACK (eigenvalues for the
positive-(semi)definite check, the final ``np.linalg.inv``) lives here on the
NumPy side; the Rust core is pure-Rust and works only on the correlation
matrix.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import _core

# Eigenvalues below this are treated as zero when detecting rank deficiency.
_PD_TOL = 1e-8


def _validate_covariance(S):
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square 2D matrix")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")
    if not np.allclose(S, S.T, atol=1e-8, rtol=0.0):
        raise ValueError("S must be symmetric")
    return S


def _resolve_c(corr, c, p):
    """Resolve the diagonal parameter ``c`` and count near-zero eigenvalues.

    Mirrors the R package: for a positive-definite correlation matrix the
    default is ``c = 1``; for a rank-deficient (n < p) one it is
    ``0.9 * (1 - k/p)`` where ``k`` is the number of zero eigenvalues.
    """
    evals = np.linalg.eigvalsh(corr)
    if evals.min() < -_PD_TOL:
        raise ValueError("S is not positive semidefinite")
    k = int(np.sum(evals < _PD_TOL))

    if c is None:
        c_val = 1.0 if k == 0 else 0.9 * (1.0 - k / p)
    else:
        c_val = float(c)
        if c_val <= 0.0:
            raise ValueError("c must be positive")
        if k > 0 and c_val >= 1.0 - k / p:
            warnings.warn(
                "c is too large for a rank-deficient covariance; "
                "the PCGLASSO solution may not exist.",
                stacklevel=3,
            )
    return c_val, k


def solve_precision(
    S,
    alpha,
    c=None,
    method="primal",
    max_iter=1000,
    tol=1e-4,
    init=None,
):
    """Run PCGLASSO on a covariance matrix ``S`` and return a results dict.

    The returned ``state`` is the correlation-scale ``(R, W, d)`` warm-start
    triple; it is scale-invariant and can be fed back in via ``init`` for a
    subsequent, similar problem.
    """
    if alpha is None:
        raise ValueError("alpha (the L1 penalty) must be provided")
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if method not in ("primal", "dual"):
        raise ValueError("method must be 'primal' or 'dual'")

    S = _validate_covariance(S)
    p = S.shape[0]

    sd = np.sqrt(np.diag(S))
    if np.any(sd <= 0.0):
        raise ValueError("S must have strictly positive diagonal entries")

    # Correlation matrix C = D_s^{-1} S D_s^{-1}; symmetrise to kill round-off.
    corr = S / np.outer(sd, sd)
    corr = np.ascontiguousarray((corr + corr.T) * 0.5)
    np.fill_diagonal(corr, 1.0)

    c_val, k = _resolve_c(corr, c, p)

    warm_r = warm_w = warm_d = None
    if init is not None:
        cand_r, cand_w, cand_d = init
        cand_r = np.ascontiguousarray(cand_r, dtype=np.float64)
        cand_w = np.ascontiguousarray(cand_w, dtype=np.float64)
        cand_d = np.ascontiguousarray(cand_d, dtype=np.float64)
        if cand_r.shape == (p, p) and cand_w.shape == (p, p) and cand_d.shape == (p,):
            warm_r, warm_w, warm_d = cand_r, cand_w, cand_d

    R, dvec, W, n_iter, converged, objective = _core.pcglasso_solve(
        corr,
        alpha,
        c_val,
        method,
        int(max_iter),
        float(tol),
        warm_r,
        warm_w,
        warm_d,
    )

    # The dual solver recovers R = -beta, which is only exactly symmetric at
    # convergence; symmetrise so the reported matrices are always symmetric.
    R = np.asarray(R, dtype=np.float64)
    R = (R + R.T) * 0.5
    np.fill_diagonal(R, 1.0)

    # Rescale to the original variable scale. Because PCGLASSO is scale
    # invariant, the partial correlations are just the (negated) off-diagonals
    # of R; the precision is diag(g) R diag(g) with g = d / sqrt(diag(S)).
    g = dvec / sd
    precision = R * np.outer(g, g)
    precision = (precision + precision.T) * 0.5

    partial_correlation = -R.copy()
    np.fill_diagonal(partial_correlation, 1.0)

    adjacency = R != 0.0
    np.fill_diagonal(adjacency, False)

    covariance = np.linalg.inv(precision)
    covariance = (covariance + covariance.T) * 0.5

    return {
        "precision": precision,
        "partial_correlation": partial_correlation,
        "adjacency": adjacency,
        "covariance": covariance,
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "objective": float(objective),
        "c": float(c_val),
        "n_zero_eigenvalues": k,
        "state": (
            np.ascontiguousarray(R),
            np.ascontiguousarray(W),
            np.ascontiguousarray(dvec),
        ),
    }
