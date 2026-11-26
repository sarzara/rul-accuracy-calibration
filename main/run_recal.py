"""Exploratory deviation D7 (see DEVIATIONS.md): post-hoc recalibration of the native intervals.

Re-fits the three models exactly as run_main.py (same seed, same splits), checks that the test-set
point predictions reproduce the saved main-study outputs, then builds recalibrated intervals from the
calibration engines:
  * isotonic recalibration (Kuleshov et al., 2018): map nominal quantile levels through the
    engine-weighted empirical CDF of the calibration PIT values (all three models);
  * variance scaling (Levi et al., 2019): sigma * s with s^2 = engine-weighted mean squared z (Ridge, LSTM).
Saves results/raw_recal/{FD}_seed{s}.npz. No metric is computed here.

Usage: python run_recal.py FD001 0
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

import run_main as RM
from run_main import LEVELS, QUANTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load, regime_stats, normalise, windows  # noqa: E402
from models import handcrafted  # noqa: E402
import conformal as C  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "raw_recal"
TAUS = np.array(QUANTS)


# ---------------------------------------------------------------- helpers
def wquantile(v, w, q):
    """Weighted quantile (inverse of the weighted empirical CDF, right-continuous step)."""
    o = np.argsort(v)
    v, cw = v[o], np.cumsum(w[o]) / w.sum()
    return v[np.searchsorted(cw, q - 1e-12)]


def gbm_knots(Q):
    """Piecewise-linear quantile function through the sorted quantiles, extended linearly to tau=0 and 1.
    Q: (n, 12) sorted. Returns taus (14,) and values (n, 14), strictly increasing in tau."""
    lo = Q[:, 0] - (Q[:, 1] - Q[:, 0]) / (TAUS[1] - TAUS[0]) * TAUS[0]
    hi = Q[:, -1] + (Q[:, -1] - Q[:, -2]) / (TAUS[-1] - TAUS[-2]) * (1 - TAUS[-1])
    t = np.r_[0.0, TAUS, 1.0]
    V = np.column_stack([lo, Q, hi])
    V = V + t[None, :] * 1e-6  # break ties so the CDF is well defined
    return t, V


def gbm_pit(Q, y):
    t, V = gbm_knots(Q)
    return np.array([np.interp(yi, vi, t) for yi, vi in zip(y, V)])


def gbm_quantile(Q, tau):
    t, V = gbm_knots(Q)
    j = int(np.clip(np.searchsorted(t, tau) - 1, 0, len(t) - 2))
    f = (tau - t[j]) / (t[j + 1] - t[j])
    return V[:, j] + f * (V[:, j + 1] - V[:, j])


# ---------------------------------------------------------------- main
def main(fd, seed):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    saved = dict(np.load(ROOT / "results" / "raw" / f"{fd}_seed{seed}.npz"))
    S, T = load(fd, "train"), load(fd, "test")
    fit_u, val_u, cal_u = RM.split_engines(S.unit.values, seed)
    stats = regime_stats(S[S.unit.isin(np.r_[fit_u, val_u])])
    Xs, _ = normalise(S, stats)
    Xt, _ = normalise(T, stats)

    def sub(units):
        m = S.unit.isin(units).values
        return windows(S[m].reset_index(drop=True), Xs[m])

    (Xf, yf, _, _), (Xv, yv, _, _), (Xc, yc, _, uc) = sub(fit_u), sub(val_u), sub(cal_u)
    Xte, yte, _, ute = windows(T, Xt, last_only=True)
    Ff, Fv, Fc, Fte = map(handcrafted, (Xf, Xv, Xc, Xte))
    w = C.unit_weights(uc)

    # re-fit (identical calls to run_main.py)
    rp, rinfo = RM.fit_ridge(Ff, yf, Fv, yv)
    gp, gq, _ = RM.fit_lgbm(Ff, yf, Fv, yv, seed)
    net, _ = RM.train_lstm(Xf, yf, Xv, yv, seed=1000 + seed)

    pred = {"ridge": rp(Fte), "gbm": gp(Fte), "lstm": RM.predict_lstm(net, Xte)[0]}
    check = {m: float(np.max(np.abs(pred[m] - saved[f"{m}_point"]))) for m in pred}
    if max(check.values()) > 1e-3:
        raise RuntimeError(f"re-fit does not reproduce the main study: {check}")

    out = {"y": yte, "unit": ute}
    meta = {"subset": fd, "seed": seed, "reproduction_max_abs_diff": check, "n_cal_engines": len(cal_u)}

    # --- Gaussian models: Ridge (constant sigma) and LSTM (heteroscedastic)
    mu_lstm_c, var_lstm_c = RM.predict_lstm(net, Xc)
    mu_lstm_t, var_lstm_t = RM.predict_lstm(net, Xte)
    gauss = {"ridge": (rp(Fc), np.full(len(yc), rinfo["sigma"]), pred["ridge"], np.full(len(yte), rinfo["sigma"])),
             "lstm": (mu_lstm_c, np.sqrt(var_lstm_c), mu_lstm_t, np.sqrt(var_lstm_t))}
    for m, (mc, sc, mt, st) in gauss.items():
        z = (yc - mc) / sc
        pit = norm.cdf(z)
        s = float(np.sqrt(np.sum(w * z ** 2) / w.sum()))
        meta[f"{m}_varscale_s"] = s
        out[f"{m}_cal_pit"] = pit
        for L in LEVELS:
            tl, th = wquantile(pit, w, (1 - L) / 2), wquantile(pit, w, (1 + L) / 2)
            tl, th = np.clip([tl, th], 1e-6, 1 - 1e-6)
            out[f"{m}_iso_lo_{L}"], out[f"{m}_iso_hi_{L}"] = mt + norm.ppf(tl) * st, mt + norm.ppf(th) * st
            out[f"{m}_iso_tau_{L}"] = np.array([tl, th])
            zL = norm.ppf(.5 + L / 2)
            out[f"{m}_vs_lo_{L}"], out[f"{m}_vs_hi_{L}"] = mt - zL * s * st, mt + zL * s * st

    # --- gradient boosting: quantile-based CDF
    Qc, Qt = gq(Fc), gq(Fte)
    pit = gbm_pit(Qc, yc)
    out["gbm_cal_pit"] = pit
    for L in LEVELS:
        tl, th = wquantile(pit, w, (1 - L) / 2), wquantile(pit, w, (1 + L) / 2)
        out[f"gbm_iso_lo_{L}"], out[f"gbm_iso_hi_{L}"] = gbm_quantile(Qt, tl), gbm_quantile(Qt, th)
        out[f"gbm_iso_tau_{L}"] = np.array([tl, th])
        # sanity: at the native taus the knot interpolation must reproduce the native bounds
        if round((1 - L) / 2, 3) in QUANTS:
            nb = saved[f"gbm_native_lo_{L}"]
            meta.setdefault("gbm_native_reproduction", {})[str(L)] = float(
                np.max(np.abs(gbm_quantile(Qt, (1 - L) / 2) - nb)))

    # calibration-set PIT summary (engine-weighted): fraction of PIT in the central 90% band
    for m in ("ridge", "gbm", "lstm"):
        p = out[f"{m}_cal_pit"]
        meta[f"{m}_cal_native_cov90_engine_weighted"] = float(np.sum(w * ((p >= .05) & (p <= .95))) / w.sum())

    np.savez_compressed(OUT / f"{fd}_seed{seed}.npz", **out)
    meta["seconds"] = time.time() - t0
    (OUT / f"{fd}_seed{seed}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"[{fd} s{seed}] recal done {meta['seconds']:.0f}s repro={check}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
