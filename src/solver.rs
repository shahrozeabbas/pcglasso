//! Outer biconvex alternation: repeatedly solve the `D` sub-problem (diagonal
//! Newton) and the `R` sub-problem (primal or dual coordinate descent) until
//! the partial correlations and diagonal stop changing.
//!
//! Everything here operates on the sample *correlation* matrix `C`; the Python
//! layer is responsible for `cov2cor` and the final rescaling to the original
//! variable scale.

use numpy::ndarray::{Array1, Array2};

use crate::common::{cholesky_logdet, full_objective};
use crate::d_update::d_update;
use crate::r_dual::r_update_dual;
use crate::r_primal::r_update_primal;

#[derive(Clone, Copy)]
pub enum Method {
    Primal,
    Dual,
}

pub struct SolveResult {
    pub r: Array2<f64>, // Delta (partial-correlation matrix, correlation scale)
    pub w: Array2<f64>, // Delta^{-1}
    pub d: Array1<f64>, // xi (sqrt-diagonal, correlation scale)
    pub n_iter: usize,
    pub converged: bool,
    pub objective: f64,
}

const MAX_D_ITER: usize = 100;
const MAX_R_SWEEPS: usize = 100;

fn offdiag_l1(m: &Array2<f64>) -> f64 {
    let p = m.nrows();
    let mut s = 0.0;
    for i in 0..p {
        for j in 0..p {
            if i != j {
                s += m[[i, j]].abs();
            }
        }
    }
    s
}

fn offdiag_l1_change(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let p = a.nrows();
    let mut s = 0.0;
    for i in 0..p {
        for j in 0..p {
            if i != j {
                s += (a[[i, j]] - b[[i, j]]).abs();
            }
        }
    }
    s
}

/// Solve PCGLASSO on the correlation matrix `c_mat`.
///
/// * `alpha`  - L1 penalty on the off-diagonal partial correlations.
/// * `c_diag` - diagonal parameter `c` (= `1 - alpha_paper1`).
/// * `warm`   - optional `(R, W, d)` warm start (correlation scale).
pub fn solve(
    c_mat: Array2<f64>,
    alpha: f64,
    c_diag: f64,
    method: Method,
    max_iter: usize,
    tol: f64,
    warm: Option<(Array2<f64>, Array2<f64>, Array1<f64>)>,
) -> SolveResult {
    let p = c_mat.nrows();

    let (mut r_mat, mut w_mat, mut d) = match warm {
        Some((r, w, dd)) if r.nrows() == p && w.nrows() == p && dd.len() == p => (r, w, dd),
        // Cold start: empty graph R = I (=> W = I) and unit diagonal.
        _ => (Array2::eye(p), Array2::eye(p), Array1::ones(p)),
    };

    let mut converged = false;
    let mut n_iter = 0;

    for it in 0..max_iter {
        n_iter = it + 1;
        let r_old = r_mat.clone();
        let d_old = d.clone();

        // Annealed inner tolerance (loose early, tightening with the outer loop).
        let inner_tol = (1e-3 * 0.9_f64.powi(it as i32)).max(0.1 * tol);

        // D given R, then R given D.
        d_update(&mut d, &r_mat, &c_mat, c_diag, inner_tol, MAX_D_ITER);
        match method {
            Method::Primal => {
                r_update_primal(&mut r_mat, &mut w_mat, &c_mat, &d, alpha, inner_tol, MAX_R_SWEEPS)
            }
            Method::Dual => {
                r_update_dual(&mut r_mat, &mut w_mat, &c_mat, &d, alpha, inner_tol, MAX_R_SWEEPS)
            }
        }

        // Outer convergence: relative L1 change of off-diagonal R plus d.
        let dr = offdiag_l1_change(&r_mat, &r_old);
        let r_norm = offdiag_l1(&r_old).max(1e-8);
        let dd: f64 = d
            .iter()
            .zip(d_old.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum();
        let d_norm = d_old.iter().map(|&x| x.abs()).sum::<f64>().max(1e-8);
        if dr / r_norm + dd / d_norm < tol {
            converged = true;
            break;
        }
    }

    let logdet_r = cholesky_logdet(&r_mat).unwrap_or(f64::NAN);
    let objective = full_objective(&r_mat, &d, &c_mat, alpha, c_diag, logdet_r);

    SolveResult {
        r: r_mat,
        w: w_mat,
        d,
        n_iter,
        converged,
        objective,
    }
}
