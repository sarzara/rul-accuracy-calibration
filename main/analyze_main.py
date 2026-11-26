"""Analysis of the main study, exactly as pre-registered (PREREGISTRATION.md).
Writes results/tables/*.csv and results/figures/*.png|pdf.  Usage: python analyze_main.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW, TAB, FIG = ROOT / "results" / "raw", ROOT / "results" / "tables", ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
HIGH = [0.80, 0.90, 0.95]
MODELS = ["ridge", "gbm", "lstm"]
NAME = {"ridge": "Ridge", "gbm": "Gradient boosting", "lstm": "LSTM"}
CAP, SUBSETS = 125.0, ["FD001", "FD004"]
BANDS = [(0, 40), (40, 80), (80, 125)]


def nasa_mean(p, y):
    d = p - y
    return float(np.mean(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def interval_score(lo, hi, y, a):
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)


def intervals(z, m, method, L):
    p = z[f"{m}_point"]
    if method == "native":
        return z[f"{m}_native_lo_{L}"], z[f"{m}_native_hi_{L}"]
    q = float(z[f"{m}_cq_{L}"])
    return p - q, p + q


def collect():
    rows, eng = [], []  # eng: per-engine records for bootstrap
    for f in sorted(RAW.glob("*.npz")):
        fd, seed = f.stem.split("_seed")[0], int(f.stem.split("_seed")[1])
        z = dict(np.load(f)); y, yraw, u = z["y"], z["y_raw"], z["unit"]
        for m in MODELS:
            p = z[f"{m}_point"]; pc = np.clip(p, 0, CAP)
            base = {"subset": fd, "seed": seed, "model": m}
            rows.append({**base, "method": "point", "level": np.nan, "MAE": float(np.mean(np.abs(pc - y))),
                         "RMSE": float(np.sqrt(np.mean((pc - y) ** 2))),
                         "RMSE_unclipped": float(np.sqrt(np.mean((p - y) ** 2))),
                         "NASA_mean_raw": nasa_mean(p, yraw)})
            for e, ae in zip(u, np.abs(pc - y)):
                eng.append({**base, "method": "point", "level": np.nan, "unit": int(e), "abs_err": float(ae), "sq_err": float(ae ** 2)})
            for method in ("native", "conformal"):
                for L in LEVELS:
                    lo, hi = intervals(z, m, method, L)
                    inf = ~np.isfinite(hi)
                    lc, hc = np.clip(lo, 0, CAP), np.clip(hi, 0, CAP)
                    cov = (y >= lc) & (y <= hc); w = hc - lc
                    isc = interval_score(lc, hc, y, 1 - L)
                    unc = y < CAP
                    r = {**base, "method": method, "level": L, "PICP": float(cov.mean()), "MPIW": float(w.mean()),
                         "IS": float(isc.mean()), "frac_infinite": float(inf.mean()),
                         "PICP_unclipped": float(((y >= lo) & (y <= hi)).mean()),
                         "MPIW_unclipped": float(np.mean(hi - lo)) if not inf.any() else np.inf,
                         "PICP_uncapped_pts": float(cov[unc].mean())}
                    for a, b in BANDS:
                        s = (y > a) & (y <= b) if a > 0 else (y <= b)
                        r[f"PICP_band_{a}-{b}"] = float(cov[s].mean()); r[f"n_band_{a}-{b}"] = int(s.sum())
                    rows.append(r)
                    for e, c, ww, ii in zip(u, cov, w, isc):
                        eng.append({**base, "method": method, "level": L, "unit": int(e), "cov": float(c),
                                    "width": float(ww), "is": float(ii)})
    return pd.DataFrame(rows), pd.DataFrame(eng)


def boot_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v); b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def summarise(D, E):
    # seed mean and s.d.
    pt = D[D.method == "point"].groupby(["subset", "model"])[["MAE", "RMSE", "RMSE_unclipped", "NASA_mean_raw"]].agg(["mean", "std"])
    pt.to_csv(TAB / "accuracy.csv")
    iv = D[D.method != "point"].groupby(["subset", "model", "method", "level"])[
        ["PICP", "MPIW", "IS", "frac_infinite", "PICP_uncapped_pts", "PICP_band_0-40", "PICP_band_40-80", "PICP_band_80-125"]].agg(["mean", "std"])
    iv.to_csv(TAB / "intervals.csv")
    # calibration error per seed, then mean/sd
    ce = (D[D.method != "point"].assign(abs_dev=lambda d: (d.PICP - d.level).abs())
          .groupby(["subset", "seed", "model", "method"]).abs_dev.mean().rename("CE").reset_index())
    ce.groupby(["subset", "model", "method"]).CE.agg(["mean", "std"]).to_csv(TAB / "calibration_error.csv")

    # engine-level bootstrap CIs: per-engine values averaged over seeds (test engines identical across seeds)
    ci_rows = []
    pe = E[E.method == "point"].groupby(["subset", "model", "unit"])[["abs_err", "sq_err"]].mean().reset_index()
    for (fd, m), g in pe.groupby(["subset", "model"]):
        lo, hi = boot_ci(g.abs_err); ci_rows.append({"subset": fd, "model": m, "metric": "MAE", "level": np.nan, "method": "point",
                                                     "value": g.abs_err.mean(), "lo95": lo, "hi95": hi})
        b = g.sq_err.values[np.random.default_rng(0).integers(0, len(g), (2000, len(g)))].mean(1) ** .5
        ci_rows.append({"subset": fd, "model": m, "metric": "RMSE", "level": np.nan, "method": "point",
                        "value": float(np.sqrt(g.sq_err.mean())), "lo95": float(np.quantile(b, .025)), "hi95": float(np.quantile(b, .975))})
    ie = E[E.method != "point"].groupby(["subset", "model", "method", "level", "unit"])[["cov", "width", "is"]].mean().reset_index()
    for (fd, m, meth, L), g in ie.groupby(["subset", "model", "method", "level"]):
        for met, col in (("PICP", "cov"), ("MPIW", "width"), ("IS", "is")):
            lo, hi = boot_ci(g[col]); ci_rows.append({"subset": fd, "model": m, "metric": met, "level": L, "method": meth,
                                                      "value": g[col].mean(), "lo95": lo, "hi95": hi})
    CI = pd.DataFrame(ci_rows); CI.to_csv(TAB / "bootstrap_ci.csv", index=False)

    # paired model differences (engine bootstrap)
    pair_rows = []
    pairs = [("gbm", "ridge"), ("gbm", "lstm"), ("lstm", "ridge")]
    for fd in SUBSETS:
        for a, b in pairs:
            A = pe[(pe.subset == fd) & (pe.model == a)].set_index("unit"); B = pe[(pe.subset == fd) & (pe.model == b)].set_index("unit")
            d = (A.abs_err - B.abs_err).values; lo, hi = boot_ci(d)
            pair_rows.append({"subset": fd, "A": a, "B": b, "metric": "MAE", "method": "point", "level": np.nan, "diff": d.mean(), "lo95": lo, "hi95": hi})
            for meth in ("native", "conformal"):
                for L in HIGH:
                    sel = (ie.subset == fd) & (ie.method == meth) & (ie.level == L)
                    A = ie[sel & (ie.model == a)].set_index("unit"); B = ie[sel & (ie.model == b)].set_index("unit")
                    for met, col in (("PICP", "cov"), ("MPIW", "width"), ("IS", "is")):
                        d = (A[col] - B[col]).values; lo, hi = boot_ci(d)
                        pair_rows.append({"subset": fd, "A": a, "B": b, "metric": met, "method": meth, "level": L,
                                          "diff": d.mean(), "lo95": lo, "hi95": hi})
    P = pd.DataFrame(pair_rows); P.to_csv(TAB / "paired_differences.csv", index=False)
    return pt, iv, ce, CI, P


def hypotheses(D, ce, P):
    out = []
    acc = D[D.method == "point"].groupby(["subset", "model"]).RMSE.mean()
    mae = D[D.method == "point"].groupby(["subset", "model"]).MAE.mean()
    cem = ce.groupby(["subset", "model", "method"]).CE.mean()
    iv = D[D.method != "point"].groupby(["subset", "model", "method", "level"])[["PICP", "MPIW"]].mean()
    for fd in SUBSETS:
        r_rmse = acc[fd].sort_values().index.tolist()
        r_ce = cem.xs((fd, "native"), level=("subset", "method")).sort_values().index.tolist()
        out.append({"hypothesis": "H1", "subset": fd, "detail": f"RMSE rank {r_rmse}; native CE rank {r_ce}",
                    "result": "supported" if r_rmse != r_ce else "not supported"})
        bad = [(m, L, round(iv.loc[(fd, m, 'conformal', L), 'PICP'], 3)) for m in MODELS for L in HIGH
               if abs(iv.loc[(fd, m, "conformal", L), "PICP"] - L) > 0.05]
        out.append({"hypothesis": "H2", "subset": fd, "detail": f"outside ±0.05: {bad}" if bad else "all within ±0.05",
                    "result": "supported" if not bad else "not supported"})
        r_mae = mae[fd].sort_values().index.tolist()
        for L in HIGH:
            r_w = iv.xs((fd, "conformal", L), level=("subset", "method", "level")).MPIW.sort_values().index.tolist()
            # separation: are the MPIW differences between adjacent models resolved by the paired CI?
            sep = []
            for a, b in zip(r_w[:-1], r_w[1:]):
                row = P[(P.subset == fd) & (P.metric == "MPIW") & (P.method == "conformal") & (P.level == L) &
                        (((P.A == a) & (P.B == b)) | ((P.A == b) & (P.B == a)))]
                sep.append(bool(((row.lo95 > 0) | (row.hi95 < 0)).iloc[0]))
            res = "supported" if r_w == r_mae else "not supported"
            if r_w != r_mae and not all(sep):
                res = "inconclusive"
            out.append({"hypothesis": "H3", "subset": fd, "detail": f"L={L}: MPIW rank {r_w}; MAE rank {r_mae}; adjacent pairs separated: {sep}",
                        "result": res})
    H = pd.DataFrame(out); H.to_csv(TAB / "hypotheses_point_estimate_rule_v1.csv", index=False)
    return H


# ---------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COL = {"ridge": "#2a78d6", "gbm": "#eb6834", "lstm": "#1baf7a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, n):
    fig.savefig(FIG / f"{n}.png"); fig.savefig(FIG / f"{n}.pdf"); plt.close(fig)


def fig_calibration_curves(D, CI):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4), sharex=True, sharey=True, constrained_layout=True)
    for i, fd in enumerate(SUBSETS):
        for j, meth in enumerate(("native", "conformal")):
            ax = axes[i, j]; ax.plot([0.45, 1], [0.45, 1], color=MUTED, lw=1, ls="--")
            for m in MODELS:
                g = CI[(CI.subset == fd) & (CI.model == m) & (CI.method == meth) & (CI.metric == "PICP")].sort_values("level")
                ax.fill_between(g.level, g.lo95, g.hi95, color=COL[m], alpha=0.15, lw=0)
                ax.plot(g.level, g.value, color=COL[m], lw=2, marker="o", ms=4.5, label=NAME[m])
            ax.set_title(f"{fd} — {'native intervals' if meth == 'native' else 'split conformal'}")
            ax.set_xlim(0.45, 1.0); ax.set_ylim(0.2, 1.0)
            if i == 1: ax.set_xlabel("nominal coverage")
            if j == 0: ax.set_ylabel("empirical coverage (test engines)")
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=3)
    fig.suptitle("Calibration curves: dashed line = perfect calibration; bands = 95% engine bootstrap", color=INK)
    save(fig, "fig1_calibration_curves")


def fig_accuracy_vs_calibration(D, ce):
    acc = D[D.method == "point"][["subset", "seed", "model", "RMSE"]]
    M = acc.merge(ce, on=["subset", "seed", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, fd in zip(axes, SUBSETS):
        g = M[M.subset == fd]
        for m in MODELS:
            for meth, mk in (("native", "o"), ("conformal", "s")):
                h = g[(g.model == m) & (g.method == meth)]
                if meth == "native":
                    ax.scatter(h.RMSE, h.CE, marker=mk, s=30, facecolor=COL[m], edgecolor=SURF, lw=1, zorder=3)
                    ax.scatter(h.RMSE.mean(), h.CE.mean(), marker=mk, s=120, facecolor=COL[m], edgecolor=INK, lw=1, zorder=4)
                else:
                    ax.scatter(h.RMSE, h.CE, marker=mk, s=26, facecolor="none", edgecolor=COL[m], lw=1, zorder=3)
                    ax.scatter(h.RMSE.mean(), h.CE.mean(), marker=mk, s=120, facecolor=SURF, edgecolor=COL[m], lw=2.2, zorder=4)
        ax.set_title(fd); ax.set_xlabel("RMSE (cycles) — lower is more accurate")
    axes[0].set_ylabel("calibration error  mean|PICP − nominal|")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL[m], marker="o", ls="", ms=7, label=NAME[m]) for m in MODELS] + \
         [Line2D([], [], color=INK2, marker="o", ls="", ms=7, label="native"),
          Line2D([], [], color=INK2, marker="s", ls="", ms=7, mfc="none", label="split conformal")]
    fig.legend(handles=hs, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle("Point accuracy vs calibration error (small marks: seeds; large: mean over seeds)", color=INK)
    save(fig, "fig3_accuracy_vs_calibration")


def fig_width(D):
    iv = D[D.method != "point"].groupby(["subset", "model", "method", "level"])[["MPIW", "PICP"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True)
    for ax, fd in zip(axes, SUBSETS):
        x0 = np.arange(len(HIGH))
        for k, m in enumerate(MODELS):
            for meth, alpha, off in (("native", 0.45, -0.07), ("conformal", 1.0, 0.07)):
                g = iv[(iv.subset == fd) & (iv.model == m) & (iv.method == meth) & iv.level.isin(HIGH)].sort_values("level")
                ax.bar(x0 + (k - 1) * 0.27 + off, g.MPIW, 0.13, color=COL[m], alpha=alpha,
                       label=f"{NAME[m]} ({meth})" if fd == "FD001" else None)
        ax.set_xticks(x0, [f"{int(L*100)}%" for L in HIGH]); ax.set_title(fd); ax.set_xlabel("nominal level")
    axes[0].set_ylabel("mean interval width (cycles)")
    fig.legend(loc="outside lower center", ncol=3, fontsize=7.5)
    fig.suptitle("Interval width: native (light) vs split conformal (solid)", color=INK)
    save(fig, "fig2_width")


def fig_bands(D):
    cols = ["PICP_band_0-40", "PICP_band_40-80", "PICP_band_80-125"]
    iv = D[(D.method != "point") & (D.level == 0.90)].groupby(["subset", "model", "method"])[cols].mean()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6), sharey=True, constrained_layout=True)
    for i, fd in enumerate(SUBSETS):
        for j, meth in enumerate(("native", "conformal")):
            ax = axes[i, j]
            for k, m in enumerate(MODELS):
                ax.bar(np.arange(3) + (k - 1) * 0.26, iv.loc[(fd, m, meth)].values, 0.24, color=COL[m], label=NAME[m])
            ax.axhline(0.9, color=MUTED, lw=1, ls="--"); ax.set_ylim(0, 1.05)
            nb = D[(D.subset == fd) & (D.method != "point")].iloc[0]
            ax.set_xticks(range(3), [f"RUL ≤ 40\n(n={int(nb['n_band_0-40'])})", f"40–80\n(n={int(nb['n_band_40-80'])})",
                                     f"80–125\n(n={int(nb['n_band_80-125'])})"]); ax.set_title(f"{fd} — {meth}")
    axes[0, 0].set_ylabel("coverage at nominal 90%"); axes[1, 0].set_ylabel("coverage at nominal 90%")
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=3)
    fig.suptitle("Coverage by true remaining life (exploratory, H4)", color=INK)
    save(fig, "fig4_bands")


if __name__ == "__main__":
    D, E = collect()
    D.to_csv(TAB / "metrics_per_seed.csv", index=False)
    pt, iv, ce, CI, P = summarise(D, E)
    H = hypotheses(D, ce, P)
    fig_calibration_curves(D, CI); fig_accuracy_vs_calibration(D, ce); fig_width(D); fig_bands(D)
    print(H.to_string(index=False))
    print("done")
