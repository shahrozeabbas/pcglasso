//! Dual coordinate descent for the `R` (= Delta) sub-problem
//! (`pcglassoFast_Dual`, Bogdan et al. Algorithm 4 / Lemma 4).
//!
//! The dual of the unit-diagonal GLASSO is
//!
//! `max_{W>0} log det W   s.t.  |W_ij - S'_ij| <= alpha for all i != j`,
//!
//! with the diagonal of `W` free. This is the classic Friedman/glassoFast block
//! coordinate ascent with two PCGLASSO-specific twists: the free diagonal is set
//! so that the recovered `R` has unit diagonal (`W_jj = 1 + beta^T w_12`), and
//! the partial-correlation column is recovered as `R_{-j,j} = -beta`.
//!
//! `beta` (the LASSO coefficients) is stored with a zero diagonal; on entry it
//! is derived from the incumbent `Delta` (`beta = -offdiag(Delta)`), which makes
//! warm starts work seamlessly.

use crate::common::soft_threshold;
use numpy::ndarray::linalg::general_mat_vec_mul;
use numpy::ndarray::{Array1, Array2};

const MAX_INNER_LASSO: usize = 100;

pub fn r_update_dual(
    r_mat: &mut Array2<f64>,
    w_mat: &mut Array2<f64>,
    c_mat: &Array2<f64>,
    d: &Array1<f64>,
    alpha: f64,
    tol: f64,
    max_sweeps: usize,
) {
    let p = r_mat.nrows();
    if p < 2 {
        return;
    }

    // beta = -offdiag(Delta), zero diagonal.
    let mut beta = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            if i != j {
                beta[[i, j]] = -r_mat[[i, j]];
            }
        }
    }

    let mut v = Array1::<f64>::zeros(p); // v = W * beta[:, j]

    for _ in 0..max_sweeps {
        let mut outer_change = 0.0_f64;

        for j in 0..p {
            // v = W * beta[:, j]  (GEMV; W is symmetric, so no transpose needed).
            general_mat_vec_mul(1.0, &*w_mat, &beta.column(j), 0.0, &mut v);

            // Inner box-constrained LASSO over beta[:, j] (i != j).
            for _ in 0..MAX_INNER_LASSO {
                let mut delta_max = 0.0_f64;
                for i in 0..p {
                    if i == j {
                        continue;
                    }
                    let w_ii = w_mat[[i, i]].max(1e-12);
                    let s_ij = d[i] * c_mat[[i, j]] * d[j];
                    let x = s_ij - v[i] + w_ii * beta[[i, j]];
                    let beta_new = soft_threshold(x, alpha) / w_ii;
                    let delta = beta_new - beta[[i, j]];
                    if delta != 0.0 {
                        beta[[i, j]] = beta_new;
                        v.scaled_add(delta, &w_mat.column(i));
                        let ad = delta.abs();
                        if ad > delta_max {
                            delta_max = ad;
                        }
                    }
                }
                if delta_max * (p as f64) < tol {
                    break;
                }
            }

            // Off-diagonal column change (for the convergence test).
            let mut col_change = 0.0;
            for i in 0..p {
                if i != j {
                    col_change += (w_mat[[i, j]] - v[i]).abs();
                }
            }

            // Write the off-diagonal column/row of W: w_12 = v.
            for i in 0..p {
                if i != j {
                    w_mat[[i, j]] = v[i];
                    w_mat[[j, i]] = v[i];
                }
            }

            // Free diagonal: W_jj = 1 + beta^T w_12  (enforces R_jj = 1).
            let mut dotp = 0.0;
            for k in 0..p {
                dotp += v[k] * beta[[k, j]];
            }
            let new_diag = 1.0 + dotp;
            let diag_change = (new_diag - w_mat[[j, j]]).abs();
            w_mat[[j, j]] = new_diag;

            let change = col_change.max(diag_change);
            if change > outer_change {
                outer_change = change;
            }
        }

        if outer_change < tol {
            break;
        }
    }

    // Recover Delta: off-diagonal = -beta, unit diagonal.
    for i in 0..p {
        for j in 0..p {
            if i == j {
                r_mat[[i, j]] = 1.0;
            } else {
                r_mat[[i, j]] = -beta[[i, j]];
            }
        }
    }
}
