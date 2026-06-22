//! PyO3 bindings for the PCGLASSO Rust core.
//!
//! Exposes low-level entry points: ``pcglasso_solve`` for a single correlation
//! matrix and ``pcglasso_solve_map`` for parallel fits on column subsets.
//! Targets pyo3 / rust-numpy 0.22.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

mod common;
mod d_update;
mod map;
mod r_dual;
mod r_primal;
mod solver;

use map::solve_one;
use solver::{solve, Method};

fn parse_method(method: &str) -> PyResult<Method> {
    match method {
        "primal" => Ok(Method::Primal),
        "dual" => Ok(Method::Dual),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "method must be 'primal' or 'dual', got {other:?}"
        ))),
    }
}

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
    let m = parse_method(method)?;

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

/// Parallel PCGLASSO over column subsets of a data matrix.
///
/// Parameters
/// ----------
/// x        : (n_samples, n_features) data matrix.
/// indptr   : CSR row pointers into ``indices``, length n_sets + 1.
/// indices  : flattened column indices for each subset.
/// alpha    : L1 penalty.
/// c_opt    : diagonal parameter; ``None`` selects the arithmetic default.
/// method   : "primal" or "dual".
/// max_iter : maximum outer iterations per subset.
/// tol      : outer convergence tolerance.
/// n_threads: Rayon pool size; 0 uses the Rayon default.
///
/// Returns a list of per-subset tuples
/// ``(R, d, sd, W, n_iter, converged, objective, c_diag)``.
#[pyfunction]
#[pyo3(signature = (x, indptr, indices, alpha, c_opt, method, max_iter, tol, n_threads=0))]
#[allow(clippy::too_many_arguments)]
fn pcglasso_solve_map<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    alpha: f64,
    c_opt: Option<f64>,
    method: &str,
    max_iter: usize,
    tol: f64,
    n_threads: usize,
) -> PyResult<Bound<'py, PyList>> {
    let m = parse_method(method)?;
    let x_view = x.as_array();
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let n_sets = indptr_slice.len().saturating_sub(1);

    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let results = py.allow_threads(|| {
        pool.install(|| {
            (0..n_sets)
                .into_par_iter()
                .map(|t| {
                    let lo = indptr_slice[t] as usize;
                    let hi = indptr_slice[t + 1] as usize;
                    let cols = &indices_slice[lo..hi];
                    solve_one(&x_view, cols, alpha, c_opt, m, max_iter, tol)
                })
                .collect::<Vec<_>>()
        })
    });

    let out = PyList::empty_bound(py);
    for res in results {
        let tuple = (
            res.r.into_pyarray_bound(py),
            res.d.into_pyarray_bound(py),
            res.sd.into_pyarray_bound(py),
            res.w.into_pyarray_bound(py),
            res.n_iter,
            res.converged,
            res.objective,
            res.c_diag,
        );
        out.append(tuple)?;
    }
    Ok(out)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pcglasso_solve, m)?)?;
    m.add_function(wrap_pyfunction!(pcglasso_solve_map, m)?)?;
    Ok(())
}
