//! PyO3 bindings for the PCGLASSO Rust core.
//!
//! Exposes a single low-level entry point, `pcglasso_solve`, which the Python
//! layer calls after converting the covariance to a correlation matrix and
//! resolving the parameters. Targets pyo3 / rust-numpy 0.22.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod common;
mod d_update;
mod r_dual;
mod r_primal;
mod solver;

use solver::{solve, Method};

/// Solve PCGLASSO on a correlation matrix.
///
/// Parameters
/// ----------
/// c        : (p, p) sample correlation matrix.
/// alpha    : L1 penalty on off-diagonal partial correlations.
/// c_diag   : diagonal parameter `c`.
/// method   : "primal" or "dual".
/// max_iter : maximum number of outer iterations.
/// tol      : outer convergence tolerance.
/// warm_*   : optional `(R, W, d)` warm start (all-or-nothing).
///
/// Returns `(R, d, W, n_iter, converged, objective)` on the correlation scale.
#[pyfunction]
#[pyo3(signature = (c, alpha, c_diag, method, max_iter, tol, warm_r=None, warm_w=None, warm_d=None))]
#[allow(clippy::too_many_arguments)]
fn pcglasso_solve<'py>(
    py: Python<'py>,
    c: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    c_diag: f64,
    method: &str,
    max_iter: usize,
    tol: f64,
    warm_r: Option<PyReadonlyArray2<'py, f64>>,
    warm_w: Option<PyReadonlyArray2<'py, f64>>,
    warm_d: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    usize,
    bool,
    f64,
)> {
    let m = match method {
        "primal" => Method::Primal,
        "dual" => Method::Dual,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "method must be 'primal' or 'dual', got {other:?}"
            )))
        }
    };

    let c_owned: Array2<f64> = c.as_array().to_owned();

    let warm = match (warm_r, warm_w, warm_d) {
        (Some(r), Some(w), Some(d)) => Some((
            r.as_array().to_owned(),
            w.as_array().to_owned(),
            d.as_array().to_owned(),
        )),
        _ => None,
    };

    // Release the GIL while the (pure-Rust) solver runs.
    let result = py.allow_threads(move || solve(c_owned, alpha, c_diag, m, max_iter, tol, warm));

    Ok((
        result.r.into_pyarray_bound(py),
        result.d.into_pyarray_bound(py),
        result.w.into_pyarray_bound(py),
        result.n_iter,
        result.converged,
        result.objective,
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pcglasso_solve, m)?)?;
    Ok(())
}
