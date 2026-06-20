//! Algorithm 1 (Bogdan et al.): diagonal-Hessian Newton method for the
//! `D` (= xi) sub-problem, shared by both the primal and dual solvers.
//!
//! Given the partial-correlation matrix `R` fixed, the diagonal `d` minimises
//!
//! `f(d) = 0.5 * d^T A d - sum_i log d_i`,   with `A = (R .* C) / c`,
//!
//! whose unique stationary point satisfies `D A D 1 = 1`. We use Newton's
//! method with a diagonal Hessian approximation (`h_i = A_ii + 1/d_i^2`) and a
//! backtracking Armijo line search that keeps `d` strictly positive. Each
//! iteration costs O(p^2) (one matrix-vector product), no eigendecomposition.

use numpy::ndarray::{Array1, Array2};

const MAX_LINE_SEARCH: usize = 60;

#[inline]
fn objective(a_mat: &Array2<f64>, d: &Array1<f64>) -> f64 {
    let ad = a_mat.dot(d);
    0.5 * d.dot(&ad) - d.iter().map(|&x| x.ln()).sum::<f64>()
}

pub fn d_update(
    d: &mut Array1<f64>,
    r_mat: &Array2<f64>,
    c_mat: &Array2<f64>,
    c_diag: f64,
    tol: f64,
    max_iter: usize,
) {
    let p = d.len();

    // A = (R .* C) / c_diag
    let mut a_mat = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            a_mat[[i, j]] = r_mat[[i, j]] * c_mat[[i, j]] / c_diag;
        }
    }

    let mut f_old = objective(&a_mat, d);

    for _ in 0..max_iter {
        let ad = a_mat.dot(d);
        let mut step = Array1::<f64>::zeros(p);
        let mut g_dot_step = 0.0; // g^T step = sum_i g_i^2 / h_i  (>= 0)
        for i in 0..p {
            let g = ad[i] - 1.0 / d[i];
            let h = a_mat[[i, i]] + 1.0 / (d[i] * d[i]);
            step[i] = g / h;
            g_dot_step += g * step[i];
        }

        // Cap the step so that d - eta*step stays strictly positive.
        let mut eta = 1.0_f64;
        for i in 0..p {
            if step[i] > 0.0 {
                let bound = 0.99 * d[i] / step[i];
                if bound < eta {
                    eta = bound;
                }
            }
        }

        // Backtracking Armijo line search on f(d - eta*step).
        let c1 = 1e-4;
        let mut f_new = f_old;
        let mut accepted = false;
        for _ in 0..MAX_LINE_SEARCH {
            let mut cand = d.clone();
            let mut feasible = true;
            for i in 0..p {
                cand[i] = d[i] - eta * step[i];
                if cand[i] <= 0.0 {
                    feasible = false;
                    break;
                }
            }
            if feasible {
                f_new = objective(&a_mat, &cand);
                if f_new <= f_old - c1 * eta * g_dot_step {
                    *d = cand;
                    accepted = true;
                    break;
                }
            }
            eta *= 0.5;
        }

        if !accepted {
            break;
        }
        if (f_old - f_new).abs() <= tol * f_old.abs().max(1.0) {
            break;
        }
        f_old = f_new;
    }
}
