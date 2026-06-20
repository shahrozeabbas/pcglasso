//! Primal coordinate descent for the `R` (= Delta) sub-problem
//! (`pcglassoFast`, Bogdan et al. Algorithms 2 & 3) plus the closed-form
//! element update of Theorem 5.
//!
//! With `d` fixed, `R` solves the unit-diagonal-constrained GLASSO
//!
//! `min_{R: R_ii=1, R>0}  -log det R + tr(S' R) + alpha * ||R||_{1,off}`,
//!
//! where `S' = D C D`. We sweep over columns; for each column the inverse of
//! the leading block `Q = R_11^{-1}` is obtained from `W = R^{-1}` by a Schur
//! complement (O(p^2)), the column is updated coordinate-wise via Theorem 5,
//! and `W` is refreshed with a rank update — O(p^3) per sweep, no
//! eigendecomposition.

use numpy::ndarray::{Array1, Array2};

const MAX_COLUMN_SWEEPS: usize = 100;

/// Theorem 5: closed-form maximiser of
///
/// `l(r) = 0.5 * ln(1 - a r^2 - 2 b r - cc) - s r - lam |r|`
///
/// over the feasible interval `{ r : 1 - a r^2 - 2 b r - cc > 0 }`.
/// Preconditions: `a > 0`, `lam >= 0`, `cc < 1 + b^2/a` (guaranteed by the
/// caller, since the incumbent column is feasible).
#[inline]
fn elem_update(a: f64, b: f64, cc: f64, s: f64, lam: f64) -> f64 {
    // Sign of the optimum (soft-threshold / feasibility rule).
    let sigma: f64;
    if cc < 1.0 {
        // r = 0 is feasible; it is optimal iff the slope is within the penalty.
        let xi = -b / (1.0 - cc) - s;
        if xi.abs() <= lam {
            return 0.0;
        }
        sigma = if xi >= 0.0 { 1.0 } else { -1.0 };
    } else {
        // r = 0 is infeasible (cc >= 1); the optimum is non-zero.
        sigma = if -b >= 0.0 { 1.0 } else { -1.0 };
    }

    let lam_s = sigma * lam;
    let zeta = s + lam_s;
    if zeta == 0.0 {
        // Quadratic coefficient vanishes; the stationary equation is linear.
        return -b / a;
    }

    // Stationary equation a_t r^2 + b_t r + c_t = 0; take the root in the
    // feasible interval (the one selected by sign(a_t)).
    let a_t = -zeta * a;
    let b_t = a - 2.0 * zeta * b;
    let c_t = zeta * (1.0 - cc) + b;
    let half = b_t / (2.0 * a_t);
    let disc = (half * half - c_t / a_t).max(0.0);
    let sign_at = if a_t >= 0.0 { 1.0 } else { -1.0 };
    -half + sign_at * disc.sqrt()
}

#[inline]
fn column_objective(c0: f64, s: &Array1<f64>, r: &Array1<f64>, lam: f64) -> f64 {
    0.5 * (1.0 - c0).max(1e-12).ln() - s.dot(r) - lam * r.iter().map(|&x| x.abs()).sum::<f64>()
}

/// Algorithm 2: element-wise coordinate descent for a single off-diagonal
/// column `r`, maintaining `B = Q r` and `c0 = r^T Q r` incrementally.
/// Returns `(B, c0)` at convergence (both needed for the `W` update).
fn update_column(
    r: &mut Array1<f64>,
    q: &Array2<f64>,
    s: &Array1<f64>,
    lam: f64,
    tol: f64,
) -> (Array1<f64>, f64) {
    let m = r.len();
    let mut b = q.dot(&*r);
    let mut c0 = r.dot(&b);
    let mut obj_old = column_objective(c0, s, r, lam);

    for _ in 0..MAX_COLUMN_SWEEPS {
        for j in 0..m {
            let a = q[[j, j]];
            let bj = b[j] - a * r[j];
            let cc = c0 - a * r[j] * r[j] - 2.0 * bj * r[j];
            let r_new = elem_update(a, bj, cc, s[j], lam);
            let delta = r_new - r[j];
            if delta != 0.0 {
                r[j] = r_new;
                // Update c0 with the *old* B_j, then update B (Algorithm 2).
                c0 += 2.0 * delta * b[j] + delta * delta * a;
                for k in 0..m {
                    b[k] += delta * q[[k, j]];
                }
            }
        }
        let obj_new = column_objective(c0, s, r, lam);
        if (obj_new - obj_old).abs() <= tol * obj_old.abs().max(1.0) {
            break;
        }
        obj_old = obj_new;
    }
    (b, c0)
}

/// Algorithm 3: primal coordinate descent for the whole `R` sub-problem.
/// Updates `r_mat` (= Delta) and keeps `w_mat` (= R^{-1}) in sync.
pub fn r_update_primal(
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
    let m = p - 1;

    // Scratch buffers reused across columns.
    let mut q = Array2::<f64>::zeros((m, m));
    let mut s = Array1::<f64>::zeros(m);
    let mut r = Array1::<f64>::zeros(m);

    for _ in 0..max_sweeps {
        let mut max_change = 0.0_f64;

        for i in 0..p {
            let w_ii = w_mat[[i, i]];

            // Build Q = R_11^{-1} = W_{-i,-i} - W_{-i,i} W_{i,-i} / W_ii,
            // s = S'_{-i,i}, and the incumbent column r (skipping index i).
            for jr in 0..m {
                let fj = if jr < i { jr } else { jr + 1 };
                s[jr] = d[fj] * c_mat[[fj, i]] * d[i];
                r[jr] = r_mat[[fj, i]];
                let wji = w_mat[[fj, i]];
                for kr in 0..m {
                    let fk = if kr < i { kr } else { kr + 1 };
                    q[[jr, kr]] = w_mat[[fj, fk]] - wji * w_mat[[i, fk]] / w_ii;
                }
            }

            let (b_vec, c0) = update_column(&mut r, &q, &s, alpha, tol);

            // Track change and write the new column back into R (symmetric).
            for jr in 0..m {
                let fj = if jr < i { jr } else { jr + 1 };
                let diff = (r[jr] - r_mat[[fj, i]]).abs();
                if diff > max_change {
                    max_change = diff;
                }
                r_mat[[fj, i]] = r[jr];
                r_mat[[i, fj]] = r[jr];
            }

            // Refresh W = R^{-1} via the block inverse:
            //   schur  = 1 - r^T Q r,   W_ii = 1/schur,
            //   W_{-i,i} = -W_ii * (Q r),
            //   W_{-i,-i} = Q + W_ii * (Q r)(Q r)^T.
            let schur = (1.0 - c0).max(1e-12);
            let w_ii_new = 1.0 / schur;
            for jr in 0..m {
                let fj = if jr < i { jr } else { jr + 1 };
                let off = -w_ii_new * b_vec[jr];
                w_mat[[fj, i]] = off;
                w_mat[[i, fj]] = off;
                for kr in 0..m {
                    let fk = if kr < i { kr } else { kr + 1 };
                    w_mat[[fj, fk]] = q[[jr, kr]] + w_ii_new * b_vec[jr] * b_vec[kr];
                }
            }
            w_mat[[i, i]] = w_ii_new;
        }

        if max_change < tol {
            break;
        }
    }
}
