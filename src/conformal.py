"""Interval constructions (v2, revised after independent review).

Exchangeable objects are ENGINES, not windows. Every conformal variant below is
written in one 'augmented weighted distribution' form:

    Q_{1-a}( sum_i p_i * delta_{V_i} + p_test * delta_{+inf} )

where calibration point i carries weight w_i (unit-balanced: each engine has
total weight 1, i.e. the CDF-pooling estimator of Dunn, Wasserman & Ramdas,
2022) and the test engine carries weight w_test (1 when unweighted, or the
likelihood ratio for weighted CP, Tibshirani et al. 2019). The +inf atom makes
the finite-sample correction exact, including the regime where the valid
interval is infinite (e.g. k=5 calibration engines at a=0.1) -- this fixes
review issue C2.

Guarantee caveats (documented in the write-up):
  * CV+ (Barber et al., 2021) guarantees 1-2a in the worst case (~1-a in practice).
  * Unit-balanced weighting targets a test point drawn uniformly over the windows of
    a fresh engine. NASA's official test truncation is a different distribution
    (review C3), so on the official test set coverage is NOT guaranteed even in
    domain; protocol B evaluates under the matched distribution.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

CHUNK = 256


def unit_weights(units: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(units, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def _wq_aug(V, W, w_test, q):
    """Upper weighted quantile at level q of rows of V (n_t, n_c) with weights
    W ((n_c,) or (n_t, n_c)) plus an atom at +inf with weight w_test (n_t,)."""
    n_t = V.shape[0]
    W = np.broadcast_to(W, V.shape)
    out = np.empty(n_t)
    for s in range(0, n_t, CHUNK):
        v, w, wt = V[s:s + CHUNK], W[s:s + CHUNK], w_test[s:s + CHUNK]
        o = np.argsort(v, -1)
        vs, cw = np.take_along_axis(v, o, -1), np.cumsum(np.take_along_axis(w, o, -1), -1)
        tot = cw[:, -1] + wt
        cw = cw / tot[:, None]
        idx = (cw < q - 1e-12).sum(-1)
        res = np.full(len(v), np.inf)                      # quantile falls on the +inf atom
        ok = idx < v.shape[1]
        res[ok] = vs[ok, idx[ok]]
        out[s:s + CHUNK] = res
    return out


def cv_interval(L, U, W, w_test, alpha):
    """Generic CV+/split interval: lower = Q_a{L_i, -inf}, upper = Q_{1-a}{U_i, +inf}."""
    hi = _wq_aug(U, W, w_test, 1 - alpha)
    lo = -_wq_aug(-L, W, w_test, 1 - alpha)
    return lo, hi


# ---------------- Gaussian (model-based) intervals ----------------
def gauss_interval(mu, var, alpha):
    z = norm.ppf(1 - alpha / 2)
    s = np.sqrt(var)
    return mu - z * s, mu + z * s


def mixture_moments(mu_k, var_k):
    """Moments of an equally weighted Gaussian mixture over members (rows)."""
    return mu_k.mean(0), var_k.mean(0) + mu_k.var(0)


# ---------------- CV+ variants (fold-model predictions mu_k: (K, n_test)) ----------------
def cvplus_abs(mu_k, oof_mu, oof_y, fold, w, alpha, w_test=None, W=None):
    R = np.abs(oof_y - oof_mu)
    w_test = np.ones(mu_k.shape[1]) if w_test is None else w_test
    W = w if W is None else W
    return cv_interval(mu_k[fold].T - R, mu_k[fold].T + R, W, w_test, alpha)


def cvplus_norm(mu_k, sd_k, oof_mu, oof_sd, oof_y, fold, w, alpha):
    S = np.abs(oof_y - oof_mu) / oof_sd
    L = (mu_k[fold] - S[:, None] * sd_k[fold]).T
    U = (mu_k[fold] + S[:, None] * sd_k[fold]).T
    return cv_interval(L, U, w, np.ones(mu_k.shape[1]), alpha)


def cvplus_cqr(qlo_k, qhi_k, oof_qlo, oof_qhi, oof_y, fold, w, alpha):
    E = np.maximum(oof_qlo - oof_y, oof_y - oof_qhi)
    return cv_interval(qlo_k[fold].T - E, qhi_k[fold].T + E, w, np.ones(qlo_k.shape[1]), alpha)


def mondrian_cvplus_abs(mu_k, oof_mu, oof_y, fold, w, alpha, bins):
    """Horizon-conditional (Mondrian) CV+, fixed per review C4: the category of the
    pair (test x, calibration i) is the predicted-RUL bin of the SAME fold model
    k(i) on both points; weights are the global unit weights restricted to the
    category. Valid conditionally on the predicted-RUL bin; report coverage by
    PREDICTED bin, not true-RUL bin."""
    b_cal = np.digitize(oof_mu, bins)                 # bin of mu_{k(i)}(x_i)
    b_test = np.digitize(mu_k[fold].T, bins)           # (n_t, n_c): bin of mu_{k(i)}(x)
    Wm = np.where(b_test == b_cal[None, :], w[None, :], 0.0)
    return cvplus_abs(mu_k, oof_mu, oof_y, fold, w, alpha, W=Wm)


def weighted_cvplus_abs(mu_k, oof_mu, oof_y, fold, w, logit_cal, logit_test, alpha, clip=8.0):
    """Weighted CP under covariate shift (Tibshirani et al., 2019) in CV+ form.
    Likelihood ratio r(x) = exp(domain-classifier log-odds), clipped to [e^-8, e^8]
    for numerical stability. Calibration weight = unit weight x r(x_i); test
    weight = r(x) (one engine). Returns interval and the effective sample size of
    the calibration weights (unit-level Kish ESS)."""
    r_cal = np.exp(np.clip(logit_cal, -clip, clip))
    r_test = np.exp(np.clip(logit_test, -clip, clip))
    Wc = w * r_cal
    lo, hi = cvplus_abs(mu_k, oof_mu, oof_y, fold, w, alpha, w_test=r_test, W=Wc)
    ess = Wc.sum() ** 2 / (Wc ** 2).sum()
    return lo, hi, ess


# ---------------- split conformal (k-shot target recalibration) ----------------
def split_abs(mu_test, mu_cal, y_cal, units_cal, alpha):
    R = np.abs(y_cal - mu_cal)
    n = len(mu_test)
    return cv_interval(mu_test[:, None] - R[None, :], mu_test[:, None] + R[None, :],
                       unit_weights(units_cal), np.ones(n), alpha)


def split_norm(mu_test, sd_test, mu_cal, sd_cal, y_cal, units_cal, alpha):
    S = np.abs(y_cal - mu_cal) / sd_cal
    n = len(mu_test)
    return cv_interval(mu_test[:, None] - S[None, :] * sd_test[:, None],
                       mu_test[:, None] + S[None, :] * sd_test[:, None],
                       unit_weights(units_cal), np.ones(n), alpha)
