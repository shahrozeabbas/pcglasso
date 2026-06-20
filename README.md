# pcglasso

[![CI](https://github.com/shahrozeabbas/pcglasso/actions/workflows/ci.yml/badge.svg)](https://github.com/shahrozeabbas/pcglasso/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pcglasso.svg)](https://pypi.org/project/pcglasso/)
[![Python](https://img.shields.io/pypi/pyversions/pcglasso.svg)](https://pypi.org/project/pcglasso/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`pcglasso` is a Python package for finding direct relationships between
variables in noisy, high-dimensional data.

Instead of asking which variables are merely correlated, PCGLASSO estimates
which variables remain connected after accounting for all the others. The result
can be read as a sparse network: variables are nodes, and selected edges are
direct conditional relationships.

The package exposes a small Python API and runs the iterative solver in Rust for
speed.

## What you can use it for

- Build an interpretable network from tabular data.
- Separate direct relationships from indirect correlations.
- Estimate sparse Gaussian graphical models.
- Work with data where variable scales differ, because PCGLASSO is scale
  invariant.
- Fit repeated, similar problems efficiently with warm starts.

Common use cases include genomics, neuroscience, finance, survey analysis, and
other settings where many variables may be related but only some relationships
are direct.

## Install

```bash
pip install pcglasso
```

To build from source, use maturin with Python 3.10 or newer:

```bash
pip install maturin
maturin develop --release
```

## Quick start

Use `PCGLasso` like a small sklearn-style estimator:

```python
import numpy as np
from pcglasso import PCGLasso

X = np.random.default_rng(0).standard_normal((200, 20))

model = PCGLasso(alpha=0.1).fit(X)

model.partial_correlation_  # strength of direct relationships
model.adjacency_            # selected network edges
model.precision_            # estimated inverse covariance matrix
```

The most readable output is often `adjacency_`, a boolean matrix showing which
variables are connected after the model has removed weaker indirect effects.

If you already have a covariance or correlation matrix, use the functional API:

```python
from pcglasso import pcglasso

res = pcglasso(S, alpha=0.1, c=None, method='dual')
res.precision_, res.partial_correlation_, res.objective_
```

## Choosing `alpha`

`alpha` controls how sparse the network is:

- Larger `alpha` values remove more edges and produce simpler networks.
- Smaller `alpha` values keep more edges and produce denser networks.

There is no universal best value. In practice, choose `alpha` by validation,
stability analysis, domain knowledge, or by fitting a sequence of values and
inspecting how the graph changes.

## Warm starts

Set `warm_start=True` to reuse the previous solution as the next fit's starting
point. This is useful for a sequence of similar problems, such as bootstrap
resamples or a path of nearby `alpha` values.

```python
model = PCGLasso(alpha=0.1, warm_start=True)
graphs = []
for X_b in resamples:
    model.fit(X_b)
    graphs.append(model.precision_.copy())
```

## Main outputs

- `partial_correlation_`: direct relationship strengths on a common scale.
- `adjacency_`: boolean conditional-dependence graph with a zero diagonal.
- `precision_`: estimated precision matrix.
- `covariance_`: model-implied covariance matrix.
- `n_iter_` and `converged_`: basic solver diagnostics.

## Advanced options

PCGLASSO estimates a sparse Gaussian precision matrix by penalising partial
correlations rather than raw precision-matrix entries. This is what makes the
estimator scale invariant and helps with hub-structured graphs.

The package includes two coordinate-descent solvers from Bogdan et al. (2026):

- `method='primal'` (default): uses the `pcglassoFast` approach and returns
  partial correlations directly. This is a good default, especially for
  hub-structured problems.
- `method='dual'`: uses the `pcglassoFast_Dual` approach, adapted from the
  GLASSO dual. This can be faster on some generic sparse problems.

Both solvers use a Rust core through PyO3 and maturin. The hot loop does not
require BLAS or LAPACK, which helps keep wheels portable.

Other parameters:

- `c`: diagonal parameter. When `None`, the package chooses a data-dependent
  default.
- `max_iter`: maximum number of outer iterations.
- `tol`: convergence tolerance.
- `assume_centered`: whether input data has already been centered.

## Status

This is an early Python implementation. The implementation follows the original
R package and the source papers; CI currently builds the package and runs a
smoke test across Linux, macOS, and Windows.

## References

- Carter, Rossell & Smith (2024). *Partial correlation graphical LASSO.* Scandinavian Journal of Statistics.
- Carter & Molinari (2025). *Existence and optimisation of the partial correlation graphical lasso.*
- Bogdan, Chojecki, Hejný, Kołodziejek & Wallin (2026). *Identifying network hubs with the partial correlation graphical LASSO.*
