# pcglasso

A fast Python implementation of the **Partial Correlation Graphical LASSO
(PCGLASSO)** with the iterative core written in Rust (PyO3 / maturin).

PCGLASSO estimates a sparse Gaussian precision matrix by penalising the
**partial correlations** rather than the raw precision entries (as the graphical
LASSO does). This makes the estimator **scale invariant** and improves recovery
of hub-structured graphs.

## Why this implementation

It implements the two fast coordinate-descent algorithms of Bogdan et al.
(2026) rather than the proximal-splitting algorithm of the original R package:

- **`method="primal"`** (default) — `pcglassoFast`: a unit-diagonal-constrained
  GLASSO solved column-by-column with the closed-form Theorem-5 element update.
  Returns partial correlations directly; strong on hub-structured problems.
- **`method="dual"`** — `pcglassoFast_Dual`: the GLASSO dual (à la `glassoFast`)
  with a free diagonal that enforces the unit-diagonal constraint. Often fastest
  on generic sparse problems.

Both share a diagonal-Newton update for the diagonal sub-problem and maintain
`W = R⁻¹` incrementally, so the hot loop is BLAS-1/2 only — **the Rust core
needs no BLAS/LAPACK**, which keeps wheels trivially portable.

## Install / build

This is a maturin project (Rust toolchain + Python ≥ 3.10 required):

```bash
pip install maturin
maturin develop --release      # build the extension into the current venv
```

## Usage

sklearn-style estimator:

```python
import numpy as np
from pcglasso import PCGLasso

X = np.random.default_rng(0).standard_normal((200, 20))

model = PCGLasso(alpha=0.1).fit(X)
model.precision_             # estimated precision matrix
model.partial_correlation_  # partial correlations (unit diagonal)
model.adjacency_            # boolean conditional-dependence graph
```

Functional form (when you already have a covariance matrix `S`):

```python
from pcglasso import pcglasso

res = pcglasso(S, alpha=0.1, c=None, method="dual")
res.precision_, res.partial_correlation_, res.objective_
```

### Warm starts

Set `warm_start=True` to reuse the previous solution as the next fit's starting
point — ideal for a sequence of similar problems (e.g. bootstrap resamples).
The warm-start state is on the (scale-invariant) correlation scale and works for
both `method` values.

```python
model = PCGLasso(alpha=0.1, warm_start=True)
graphs = []
for X_b in resamples:          # similar-but-different datasets
    model.fit(X_b)             # resumes from the previous solution
    graphs.append(model.precision_.copy())
```

## Parameters

| Name    | Meaning |
|---------|---------|
| `alpha` | L1 penalty on off-diagonal partial correlations (`rho` in the R package, `λ` in Bogdan et al.). |
| `c`     | Diagonal parameter (`c = 1 − α` of Bogdan et al.). `None` ⇒ data-dependent default. |
| `method`| `"primal"` (default) or `"dual"`. |

## Releasing

CI (`.github/workflows/ci.yml`) builds the extension and runs a smoke test on
Linux/macOS/Windows for every push and PR.

Releases are automated by `.github/workflows/release.yml`:

1. Bump the version in **both** `pyproject.toml` and `Cargo.toml`.
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.

The tag triggers wheel builds (Linux x86_64/aarch64, macOS universal2, Windows
x64 — `abi3`, so one wheel per platform covers Python ≥ 3.10), creates a GitHub
release with the artifacts, and publishes to PyPI via **trusted publishing**
(OIDC; no stored token). A manual run (Actions → *Release* → *Run workflow*) can
publish to TestPyPI for a dry run.

**One-time setup:** register the repository as a trusted publisher on PyPI (and
TestPyPI) — project → *Publishing* → add a GitHub Actions publisher pointing at
workflow `release.yml`.

## Status

Early version. The implementation follows the R package and the two source
papers; **no automated test suite is included yet** (CI does a build plus a
smoke test). Wheels are built against the CPython stable ABI (abi3, Python ≥
3.10) and pinned to pyo3 / rust-numpy 0.22.

## References

- Carter, Rossell & Smith (2024). *Partial correlation graphical LASSO.* Scandinavian Journal of Statistics.
- Carter & Molinari (2025). *Existence and optimisation of the partial correlation graphical lasso.*
- Bogdan, Chojecki, Hejný, Kołodziejek & Wallin (2026). *Identifying network hubs with the partial correlation graphical LASSO.*
