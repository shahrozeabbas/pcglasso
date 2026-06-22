//! Parallel map over independent PCGLASSO sub-problems on column subsets of a
//! data matrix. Each subset: gather -> center -> covariance -> cov2cor ->
//! arithmetic ``c`` -> ``solver::solve``.

use numpy::ndarray::{Array1, Array2, ArrayView2};
use crate::solver::{solve, Method, SolveResult};

/// One subset result before Python assembly.
pub struct MapOneResult {
    pub r: Array2<f64>,
    pub d: Array1<f64>,
    pub w: Array2<f64>,
    pub sd: Array1<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub objective: f64,
    pub c_diag: f64,
}

fn degenerate_result(p: usize, sd: Array1<f64>) -> MapOneResult {
    MapOneResult {
        r: Array2::eye(p),
        d: Array1::ones(p),
        w: Array2::eye(p),
        sd,
        n_iter: 0,
        converged: false,
        objective: f64::NAN,
        c_diag: 1.0,
    }
}

fn resolve_c_diag(n_samples: usize, p: usize, c_opt: Option<f64>) -> f64 {
    if let Some(c) = c_opt {
        return c;
    }
    let k = if p > n_samples.saturating_sub(1) {
        p - (n_samples.saturating_sub(1))
    } else {
        0
    };
    if k == 0 {
        1.0
    } else {
        0.9 * (1.0 - k as f64 / p as f64)
    }
}

fn cov2cor(s: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let p = s.nrows();
    let mut sd = Array1::<f64>::zeros(p);
    for i in 0..p {
        sd[i] = s[[i, i]].sqrt();
    }

    let mut corr = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            corr[[i, j]] = 0.5
                * (s[[i, j]] / (sd[i] * sd[j]) + s[[j, i]] / (sd[j] * sd[i]));
        }
        corr[[i, i]] = 1.0;
    }
    (corr, sd)
}

/// Solve PCGLASSO on a column subset of ``x`` (samples x features).
pub fn solve_one(
    x: &ArrayView2<f64>,
    col_idx: &[i64],
    alpha: f64,
    c_opt: Option<f64>,
    method: Method,
    max_iter: usize,
    tol: f64,
) -> MapOneResult {
    let p = col_idx.len();
    let n = x.nrows();

    if p < 2 {
        let sd = if p == 1 {
            let gi = col_idx[0] as usize;
            let col = x.column(gi);
            let mean = col.sum() / n as f64;
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            Array1::from_elem(1, var.sqrt())
        } else {
            Array1::<f64>::zeros(0)
        };
        return degenerate_result(p.max(1), sd);
    }

    // Gather + center columns (matches X.mean(axis=0) in Python).
    let mut xc = Array2::<f64>::zeros((n, p));
    for (j, &gi) in col_idx.iter().enumerate() {
        let col = x.column(gi as usize);
        let mean = col.sum() / n as f64;
        for i in 0..n {
            xc[[i, j]] = col[i] - mean;
        }
    }

    // S = Xc^T Xc / n
    let s = xc.t().dot(&xc) / n as f64;
    let (corr, sd) = cov2cor(&s);

    if sd.iter().any(|&v| !(v > 0.0)) {
        return degenerate_result(p, sd);
    }

    let c_diag = resolve_c_diag(n, p, c_opt);
    let SolveResult {
        r,
        w,
        d,
        n_iter,
        converged,
        objective,
    } = solve(corr, alpha, c_diag, method, max_iter, tol, None);

    MapOneResult {
        r,
        d,
        w,
        sd,
        n_iter,
        converged,
        objective,
        c_diag,
    }
}
