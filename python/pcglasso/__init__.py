"""Partial Correlation Graphical LASSO (PCGLASSO).

A scale-invariant sparse precision-matrix estimator, with the iterative core
implemented in Rust. Two coordinate-descent solvers are available
(``method="primal"`` and ``method="dual"``).
"""

from ._functional import PCGLassoResult, pcglasso, pcglasso_map
from .estimator import PCGLasso

# sklearn-style long alias.
PartialCorrelationGraphicalLasso = PCGLasso

__all__ = [
    "PCGLasso",
    "PartialCorrelationGraphicalLasso",
    "pcglasso",
    "pcglasso_map",
    "PCGLassoResult",
]
__version__ = "0.1.0"
