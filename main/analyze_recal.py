"""Exploratory deviation D7: evaluation of post-hoc recalibration against native and conformal intervals.
Same evaluation as the main study (clipping to [0,125], seed-averaged per-engine values, engine bootstrap).
Writes results/tables/recal_*.csv and results/figures/fig5_recalibration.*   Usage: python analyze_recal.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_main as AM  # figure style, interval score
from inference import load_arrays, diff_ci, value_ci, resolved, LEVELS, HIGH, MODELS, SUBSETS, CAP, BANDS

ROOT = Path(__file__).resolve().parent
RAWR, TAB, FIG = ROOT / "results" / "raw_recal", ROOT / "results" / "tables", ROOT / "results" / "figures"
METHODS = {"native": "native", "conformal": "split conformal", "iso": "isotonic recalibration", "vs": "variance scaling"}


def load_all(fd):
    """Main-study arrays plus recalibrated ones. Adds keys (m,'cov'|'w'|'is', meth, L) for meth in iso, vs."""
    A, y = load_arrays(fd)
    main = sorted((ROOT / "results" / "raw").glob(f"{fd}_seed*.npz"))
    rec = sorted(RAWR.glob(f"{fd}_seed*.npz"))
    assert [f.name for f in main] == [f.name for f in rec], "recalibration runs missing"
    taus = []
    for fm, fr in zip(main, rec):
        z, r = dict(np.load(fm)), dict(np.load(fr))
        assert np.array_equal(z["unit"], r["unit"])
        for m in MODELS:
            p = z[f"{m}_point"]
            for meth in ("native", "conformal", "iso", "vs"):
                for L in LEVELS:
                    if meth == "native":
                        lo, hi = z[f"{m}_native_lo_{L}"], z[f"{m}_native_hi_{L}"]
                    elif meth == "conformal":
                        q = float(z[f"{m}_cq_{L}"]); lo, hi = p - q, p + q
                    else:
                        if f"{m}_{meth}_lo_{L}" not in r:
                            continue
                        lo, hi = r[f"{m}_{meth}_lo_{L}"], r[f"{m}_{meth}_hi_{L}"]
                    lc, hc = np.clip(lo, 0, CAP), np.clip(hi, 0, CAP)
                    if meth in ("iso", "vs"):
                        A.setdefault((m, "cov", meth, L), []).append(((y >= lc) & (y <= hc)).astype(float))
                        A.setdefault((m, "w", meth, L), []).append(hc - lc)
                    A.setdefault((m, "is", meth, L), []).append(AM.interval_score(lc, hc, y, 1 - L))
                if meth == "iso":
                    for L in LEVELS:
                        tl, th = r[f"{m}_iso_tau_{L}"]
                        taus.append({"subset": fd, "seed": int(fm.stem.split("seed")[1]), "model": m, "level": L,
                                     "tau_lo": float(tl), "tau_hi": float(th), "nominal_lo": (1 - L) / 2, "nominal_hi": (1 + L) / 2})
    A = {k: (np.stack(v) if isinstance(v, list) else v) for k, v in A.items()}
    return A, y, taus


def is_ci(A, n_s, n_e, m, meth, L, seed=0):
    X = A[(m, "is", meth, L)].mean(0)
    rng = np.random.default_rng(seed)
    b = X[rng.integers(0, n_e, (2000, n_e))].mean(1)
    return float(X.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def is_diff_ci(A, n_e, m, ma, mb, L, seed=0):
    d = A[(m, "is", ma, L)].mean(0) - A[(m, "is", mb, L)].mean(0)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, n_e, (2000, n_e))].mean(1)
    return float(d.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def method_diff_ci(A, n_s, n_e, m, ma, mb, what, L=None, mask=None, seed=0):
    """Paired (same model, two interval methods) engine-bootstrap difference; reuses inference.diff_ci by aliasing."""
    B = dict(A)
    for l in LEVELS:
        for kind in ("cov", "w"):
            B[("__a", kind, "x", l)] = A[(m, kind, ma, l)]
            B[("__b", kind, "x", l)] = A[(m, kind, mb, l)]
    return diff_ci(B, n_s, n_e, "__a", "__b", what, "x", L, mask=mask, seed=seed)


def run():
    vals, pairs, bands, taus = [], [], [], []
    for fd in SUBSETS:
        A, y, t = load_all(fd); taus += t
        n_s, n_e = A[("ridge", "abs")].shape
        for m in MODELS:
            meths = ["native", "conformal", "iso"] + (["vs"] if m != "gbm" else [])
            for meth in meths:
                v = value_ci(A, n_s, n_e, m, "CE", meth)
                vals.append({"subset": fd, "model": m, "method": meth, "metric": "CE", "level": np.nan,
                             "value": v[0], "lo95": v[1], "hi95": v[2]})
                for L in LEVELS:
                    for what in ("PICP", "MPIW"):
                        v = value_ci(A, n_s, n_e, m, what, meth, L)
                        vals.append({"subset": fd, "model": m, "method": meth, "metric": what, "level": L,
                                     "value": v[0], "lo95": v[1], "hi95": v[2]})
                    v = is_ci(A, n_s, n_e, m, meth, L)
                    vals.append({"subset": fd, "model": m, "method": meth, "metric": "IS", "level": L,
                                 "value": v[0], "lo95": v[1], "hi95": v[2]})
                for lo_b, hi_b in BANDS:
                    mask = (y > lo_b) & (y <= hi_b) if lo_b > 0 else (y <= hi_b)
                    cov = A[(m, "cov", meth, 0.90)][:, mask].mean()
                    bands.append({"subset": fd, "model": m, "method": meth, "band": f"{lo_b}-{hi_b}",
                                  "n": int(mask.sum()), "PICP90": float(cov)})
            comps = [("iso", "native"), ("iso", "conformal")] + ([("vs", "native"), ("vs", "conformal")] if m != "gbm" else [])
            for ma, mb in comps:
                d = method_diff_ci(A, n_s, n_e, m, ma, mb, "CE")
                pairs.append({"subset": fd, "model": m, "A": ma, "B": mb, "metric": "CE", "level": np.nan,
                              "diff": d[0], "lo95": d[1], "hi95": d[2], "resolved": resolved(*d[1:])})
                for L in HIGH:
                    for what in ("PICP", "MPIW"):
                        d = method_diff_ci(A, n_s, n_e, m, ma, mb, what, L)
                        pairs.append({"subset": fd, "model": m, "A": ma, "B": mb, "metric": what, "level": L,
                                      "diff": d[0], "lo95": d[1], "hi95": d[2], "resolved": resolved(*d[1:])})
                    d = is_diff_ci(A, n_e, m, ma, mb, L)
                    pairs.append({"subset": fd, "model": m, "A": ma, "B": mb, "metric": "IS", "level": L,
                                  "diff": d[0], "lo95": d[1], "hi95": d[2], "resolved": resolved(*d[1:])})
            # near-failure coverage at 90%, iso vs native
            mask = y <= 40
            d = method_diff_ci(A, n_s, n_e, m, "iso", "native", "PICP", 0.90, mask=mask)
            pairs.append({"subset": fd, "model": m, "A": "iso", "B": "native", "metric": "PICP_RUL<=40", "level": 0.90,
                          "diff": d[0], "lo95": d[1], "hi95": d[2], "resolved": resolved(*d[1:])})
    V, P, Bd = pd.DataFrame(vals), pd.DataFrame(pairs), pd.DataFrame(bands)
    Tt = pd.DataFrame(taus).groupby(["subset", "model", "level"])[["nominal_lo", "tau_lo", "nominal_hi", "tau_hi"]].mean().reset_index()
    V.to_csv(TAB / "recal_values.csv", index=False); P.to_csv(TAB / "recal_paired.csv", index=False)
    Bd.to_csv(TAB / "recal_bands.csv", index=False); Tt.to_csv(TAB / "recal_taus.csv", index=False)
    meta = [json.loads(f.read_text()) for f in sorted(RAWR.glob("*.json"))]
    M = pd.DataFrame([{"subset": d["subset"], "seed": d["seed"], "repro_max": max(d["reproduction_max_abs_diff"].values()),
                       "ridge_s": d["ridge_varscale_s"], "lstm_s": d["lstm_varscale_s"],
                       **{f"{m}_cal_cov90": d[f"{m}_cal_native_cov90_engine_weighted"] for m in MODELS}} for d in meta])
    M.to_csv(TAB / "recal_runs.csv", index=False)
    return V, P, Bd, Tt, M


def lstm_tail_diagnostic():
    """Post-plan diagnostic (not in the D7 plan): why variance scaling over-widens the LSTM intervals.
    Compares the maximum-likelihood scale factor with the factor that would give 90% coverage on the
    calibration windows, and the share of windows with |z| > 3 (Gaussian: 0.27%)."""
    import sys
    from scipy.stats import norm
    import run_main as RM
    sys.path.insert(0, str(ROOT.parents[0] / "src"))
    from data import load, regime_stats, normalise, windows
    import conformal as C
    rows = []
    for fd in SUBSETS:
        S = load(fd, "train")
        for f in sorted(RAWR.glob(f"{fd}_seed*.npz")):
            seed = int(f.stem.split("seed")[1]); r = np.load(f)
            fit_u, val_u, cal_u = RM.split_engines(S.unit.values, seed)
            Xs, _ = normalise(S, regime_stats(S[S.unit.isin(np.r_[fit_u, val_u])]))
            m = S.unit.isin(cal_u).values
            _, _, _, uc = windows(S[m].reset_index(drop=True), Xs[m])
            w = C.unit_weights(uc); w = w / w.sum()
            z = norm.ppf(np.clip(r["lstm_cal_pit"], 1e-12, 1 - 1e-12))
            o = np.argsort(np.abs(z))
            q90 = np.abs(z)[o][np.searchsorted(np.cumsum(w[o]), 0.9)]
            meta = json.loads(f.with_suffix(".json").read_text())
            rows.append({"subset": fd, "seed": seed, "s_mle": meta["lstm_varscale_s"],
                         "s_needed_for_90pct": float(q90 / norm.ppf(0.95)), "share_abs_z_gt_3": float(np.sum(w * (np.abs(z) > 3)))})
    D = pd.DataFrame(rows); D.to_csv(TAB / "recal_lstm_tail_diagnostic.csv", index=False)
    return D


def figure(V):
    plt = AM.plt
    style = {"native": dict(ls=":", marker="o"), "conformal": dict(ls="--", marker="s"), "iso": dict(ls="-", marker="D")}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), sharex=True, sharey=True, constrained_layout=True)
    for i, fd in enumerate(SUBSETS):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]; ax.plot([0.45, 1], [0.45, 1], color=AM.MUTED, lw=1)
            for meth, st in style.items():
                g = V[(V.subset == fd) & (V.model == m) & (V.method == meth) & (V.metric == "PICP")].sort_values("level")
                if meth == "iso":
                    ax.fill_between(g.level, g.lo95, g.hi95, color=AM.COL[m], alpha=0.15, lw=0)
                ax.plot(g.level, g.value, color=AM.COL[m] if meth == "iso" else AM.INK2, lw=2 if meth == "iso" else 1.3,
                        ms=4.5, label=METHODS[meth], **st)
            ax.set_title(f"{fd} — {AM.NAME[m]}"); ax.set_xlim(0.45, 1.0); ax.set_ylim(0.2, 1.0)
            if i == 1: ax.set_xlabel("nominal coverage")
            if j == 0: ax.set_ylabel("empirical coverage (test engines)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=AM.INK2, lw=1.3, **style["native"], ms=5, label="native"),
          Line2D([], [], color=AM.INK2, lw=1.3, **style["conformal"], ms=5, label="split conformal"),
          Line2D([], [], color=AM.INK2, lw=2, **style["iso"], ms=5, label="isotonic recalibration (band: 95% engine bootstrap)")]
    fig.legend(handles=hs, loc="outside lower center", ncol=3)
    fig.suptitle("Exploratory (D7): calibration curves after post-hoc isotonic recalibration; solid grey line = perfect calibration",
                 color=AM.INK)
    AM.save(fig, "fig5_recalibration")


if __name__ == "__main__":
    V, P, Bd, Tt, M = run()
    figure(V)
    pd.set_option("display.width", 220)
    print(lstm_tail_diagnostic().round(3).to_string(index=False))
    print(M.round(3).to_string(index=False))
    print(V[V.metric == "CE"].round(3).to_string(index=False))
    print(V[(V.metric != "CE") & (V.level == 0.9)].round(3).to_string(index=False))
    print(P.round(3).to_string(index=False))
    print(Bd.round(3).to_string(index=False))
    print(Tt[Tt.level.isin([0.8, 0.9, 0.95])].round(3).to_string(index=False))
