//! Shared helpers: soft-thresholding, Cholesky log-determinant, and the
//! full PCGLASSO objective.

use numpy::ndarray::{Array1, Array2};

/// Soft-threshold operator `S(x, lam) = sign(x) * max(|x| - lam, 0)`.
#[inline]
pub fn soft_threshold(x: f64, lam: f64) -> f64 {
    if x > lam {
        x - lam
    } else if x < -lam {
        x + lam
    } else {
        0.0
    }
}

/// Log-determinant of a symmetric positive-definite matrix via a hand-rolled
/// Cholesky factorisation (no LAPACK dependency). Returns `None` if `m` is not
/// numerically positive definite.
pub fn cholesky_logdet(m: &Array2<f64>) -> Option<f64> {
    let p = m.nrows();
    let mut l = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut sum = m[[i, j]];
            for k in 0..j {
                sum -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * p + j] = sum.sqrt();
            } else {
                l[i * p + j] = sum / l[j * p + j];
            }
        }
    }
    let mut logdet = 0.0;
    for i in 0..p {
        logdet += 2.0 * l[i * p + i].ln();
    }
    Some(logdet)
}

/// Full PCGLASSO objective on the correlation scale:
///
/// `J = -log det R - 2 c * sum_i log d_i + tr(C D R D) + alpha * sum_{i!=j} |R_ij|`
///
/// where `R = Delta`, `d = xi`, `C` is the sample correlation matrix and
/// `logdet_r = log det R` (passed in so it is computed only once).
pub fn full_objective(
    r: &Array2<f64>,
    d: &Array1<f64>,
    c_mat: &Array2<f64>,
    alpha: f64,
    c_diag: f64,
    logdet_r: f64,
) -> f64 {
    let p = r.nrows();
    let mut tr = 0.0;
    let mut pen = 0.0;
    for i in 0..p {
        for j in 0..p {
            tr += c_mat[[i, j]] * d[i] * d[j] * r[[i, j]];
            if i != j {
                pen += r[[i, j]].abs();
            }
        }
    }
    let diag_term = -2.0 * c_diag * d.iter().map(|&x| x.ln()).sum::<f64>();
    -logdet_r + diag_term + tr + alpha * pen
}
