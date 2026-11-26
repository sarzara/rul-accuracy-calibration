"""Exploratory deviation D8: evaluation of deep ensembles, MC dropout and locally adaptive conformal.
Same evaluation as the main study (clipping to [0,125], seed-averaged per-engine values, engine bootstrap).
Writes results/tables/uq_*.csv and results/figures/fig6_uq.*   Usage: python analyze_uq.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_main as AM
from inference import load_arrays, diff_ci, value_ci, resolved, LEVELS, HIGH, SUBSETS, CAP, BANDS

ROOT = Path(__file__).resolve().parent
RAWU, TAB, FIG = ROOT / "results" / "raw_uq", ROOT / "results" / "tables", ROOT / "results" / "figures"
ALL = ["ridge", "gbm", "lstm", "ens", "mcd"]
NAME = {**AM.NAME, "ens": "LSTM deep ensemble (M=5)", "mcd": "LSTM MC dropout (T=50)"}
COL = {**AM.COL, "ens": "#7c5cd6", "mcd": "#b8860b"}
METHS = {"ridge": ["native", "conformal"], "gbm": ["native", "conformal", "adapt"],
         "lstm": ["native", "conformal", "adapt"], "ens": ["native", "conformal", "adapt"], "mcd": ["native", "conformal", "adapt"]}
ADAPT = {"gbm": "CQR", "lstm": "normalised conformal", "ens": "normalised conformal", "mcd": "normalised conformal"}


def load_all(fd):
    A, y = load_arrays(fd)
    main = sorted((ROOT / "results" / "raw").glob(f"{fd}_seed*.npz"))
    uq = sorted(RAWU.glob(f"{fd}_seed*.npz"))
    assert [f.name for f in main] == [f.name for f in uq], "D8 runs missing"
    for fm, fu in zip(main, uq):
        z, r = dict(np.load(fm)), dict(np.load(fu))
        assert np.array_equal(z["unit"], r["unit"]) and np.array_equal(z["y"], y)
        for m in ("ens", "mcd"):
            p = r[f"{m}_point"]; pc = np.clip(p, 0, CAP)
            A.setdefault((m, "abs"), []).append(np.abs(pc - y))
            A.setdefault((m, "sq"), []).append((pc - y) ** 2)
            A.setdefault((m, "abs_raw"), []).append(np.abs(p - y))
        for m in ALL:
            for meth in METHS[m]:
                for L in LEVELS:
                    if m in ("ens", "mcd") and meth == "native":
                        lo, hi = r[f"{m}_native_lo_{L}"], r[f"{m}_native_hi_{L}"]
                    elif m in ("ens", "mcd") and meth == "conformal":
                        q = float(r[f"{m}_cq_{L}"]); lo, hi = r[f"{m}_point"] - q, r[f"{m}_point"] + q
                    elif meth == "adapt":
                        lo, hi = r[f"{m}_adapt_lo_{L}"], r[f"{m}_adapt_hi_{L}"]
                    elif meth == "native":
                        lo, hi = z[f"{m}_native_lo_{L}"], z[f"{m}_native_hi_{L}"]
                    else:
                        q = float(z[f"{m}_cq_{L}"]); lo, hi = z[f"{m}_point"] - q, z[f"{m}_point"] + q
                    lc, hc = np.clip(lo, 0, CAP), np.clip(hi, 0, CAP)
                    if m in ("ens", "mcd") or meth == "adapt":
                        A.setdefault((m, "cov", meth, L), []).append(((y >= lc) & (y <= hc)).astype(float))
                        A.setdefault((m, "w", meth, L), []).append(hc - lc)
                    A.setdefault((m, "is", meth, L), []).append(AM.interval_score(lc, hc, y, 1 - L))
    A = {k: (np.stack(v) if isinstance(v, list) else v) for k, v in A.items()}
    return A, y


def boot_mean(X, seed=0):
    rng = np.random.default_rng(seed)
    b = X[rng.integers(0, len(X), (2000, len(X)))].mean(1)
    return float(X.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def method_diff_ci(A, n_s, n_e, m, ma, mb, what, L=None, mask=None):
    B = dict(A)
    for l in LEVELS:
        for kind in ("cov", "w"):
            B[("__a", kind, "x", l)] = A[(m, kind, ma, l)]
            B[("__b", kind, "x", l)] = A[(m, kind, mb, l)]
    return diff_ci(B, n_s, n_e, "__a", "__b", what, "x", L, mask=mask)


def row(fd, what, d, **kw):
    return {"subset": fd, **kw, "metric": what, "diff": d[0], "lo95": d[1], "hi95": d[2], "resolved": resolved(*d[1:])}


def run():
    vals, pairs, bands = [], [], []
    for fd in SUBSETS:
        A, y = load_all(fd)
        n_s, n_e = A[("ridge", "abs")].shape
        for m in ALL:
            for what in ("RMSE", "MAE"):
                v = value_ci(A, n_s, n_e, m, what)
                vals.append({"subset": fd, "model": m, "method": "point", "metric": what, "level": np.nan,
                             "value": v[0], "lo95": v[1], "hi95": v[2]})
            for meth in METHS[m]:
                v = value_ci(A, n_s, n_e, m, "CE", meth)
                vals.append({"subset": fd, "model": m, "method": meth, "metric": "CE", "level": np.nan,
                             "value": v[0], "lo95": v[1], "hi95": v[2]})
                for L in LEVELS:
                    for what in ("PICP", "MPIW"):
                        v = value_ci(A, n_s, n_e, m, what, meth, L)
                        vals.append({"subset": fd, "model": m, "method": meth, "metric": what, "level": L,
                                     "value": v[0], "lo95": v[1], "hi95": v[2]})
                    v = boot_mean(A[(m, "is", meth, L)].mean(0))
                    vals.append({"subset": fd, "model": m, "method": meth, "metric": "IS", "level": L,
                                 "value": v[0], "lo95": v[1], "hi95": v[2]})
                for lo_b, hi_b in BANDS:
                    mask = (y > lo_b) & (y <= hi_b) if lo_b > 0 else (y <= hi_b)
                    bands.append({"subset": fd, "model": m, "method": meth, "band": f"{lo_b}-{hi_b}", "n": int(mask.sum()),
                                  "PICP90": float(A[(m, "cov", meth, 0.90)][:, mask].mean()),
                                  "MPIW90": float(A[(m, "w", meth, 0.90)][:, mask].mean())})
        # (i) native CE and (ii) RMSE between models
        for a, b in (("ens", "lstm"), ("mcd", "lstm"), ("ens", "gbm")):
            pairs.append(row(fd, "CE", diff_ci(A, n_s, n_e, a, b, "CE", "native"), A=a, B=b, method="native", level=np.nan))
        for a, b in (("ens", "lstm"), ("ens", "gbm"), ("mcd", "lstm")):
            pairs.append(row(fd, "RMSE", diff_ci(A, n_s, n_e, a, b, "RMSE"), A=a, B=b, method="point", level=np.nan))
        # (iii) adaptive vs standard conformal per model
        for m in ("gbm", "lstm", "ens", "mcd"):
            pairs.append(row(fd, "CE", method_diff_ci(A, n_s, n_e, m, "adapt", "conformal", "CE"),
                             A=f"{m}:adapt", B=f"{m}:conformal", method="-", level=np.nan))
            for L in HIGH:
                pairs.append(row(fd, "MPIW", method_diff_ci(A, n_s, n_e, m, "adapt", "conformal", "MPIW", L),
                                 A=f"{m}:adapt", B=f"{m}:conformal", method="-", level=L))
                d = A[(m, "is", "adapt", L)].mean(0) - A[(m, "is", "conformal", L)].mean(0)
                pairs.append(row(fd, "IS", boot_mean(d), A=f"{m}:adapt", B=f"{m}:conformal", method="-", level=L))
            pairs.append(row(fd, "PICP_RUL<=40", method_diff_ci(A, n_s, n_e, m, "adapt", "conformal", "PICP", 0.90, mask=y <= 40),
                             A=f"{m}:adapt", B=f"{m}:conformal", method="-", level=0.90))
    V, P, Bd = pd.DataFrame(vals), pd.DataFrame(pairs), pd.DataFrame(bands)
    V.to_csv(TAB / "uq_values.csv", index=False); P.to_csv(TAB / "uq_paired.csv", index=False)
    Bd.to_csv(TAB / "uq_bands.csv", index=False)
    meta = [json.loads(f.read_text()) for f in sorted(RAWU.glob("*.json"))]
    M = pd.DataFrame([{"subset": d["subset"], "seed": d["seed"], "repro_max": max(d["reproduction_max_abs_diff"].values()),
                       "ens_epistemic_share": d["ens_test_epistemic_share"], "mcd_epistemic_share": d["mcd_test_epistemic_share"],
                       "lstm_mean_sd": d["lstm_test_mean_sd"], "ens_mean_sd": d["ens_test_mean_sd"], "mcd_mean_sd": d["mcd_test_mean_sd"],
                       "member_epochs": "/".join(str(d[f"member{k}_epochs"]) for k in range(d["M"])), "seconds": d["seconds"]}
                      for d in meta])
    M.to_csv(TAB / "uq_runs.csv", index=False)
    return V, P, Bd, M


def figure(V, Bd):
    plt = AM.plt
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    show = ["gbm", "lstm", "ens", "mcd"]
    for i, fd in enumerate(SUBSETS):
        ax = axes[i, 0]; ax.plot([0.45, 1], [0.45, 1], color=AM.MUTED, lw=1)
        for m in show:
            g = V[(V.subset == fd) & (V.model == m) & (V.method == "native") & (V.metric == "PICP")].sort_values("level")
            ax.fill_between(g.level, g.lo95, g.hi95, color=COL[m], alpha=0.12, lw=0)
            ax.plot(g.level, g.value, color=COL[m], lw=2, marker="o", ms=4.5, label=NAME[m])
        ax.set_xlim(0.45, 1.0); ax.set_ylim(0.2, 1.0); ax.set_title(f"{fd}: native intervals")
        ax.set_ylabel("empirical coverage (test engines)")
        if i == 1: ax.set_xlabel("nominal coverage")
        ax = axes[i, 1]
        x = np.arange(len(show))
        for k, (meth, alpha) in enumerate((("conformal", 0.4), ("adapt", 1.0))):
            g = Bd[(Bd.subset == fd) & (Bd.method == meth) & (Bd.band == "0-40")].set_index("model")
            wv, cv = g.loc[show, "MPIW90"].values, g.loc[show, "PICP90"].values
            ax.bar(x + (k - 0.5) * 0.38, wv, 0.36, color=[COL[m] for m in show], alpha=alpha, edgecolor=AM.SURF, lw=2)
            for xi, wi, ci in zip(x, wv, cv):
                ax.text(xi + (k - 0.5) * 0.38, wi + 1.2, f"{ci:.2f}", ha="center", va="bottom", fontsize=7.5, color=AM.INK2)
        ax.set_xticks(x, ["GBM", "LSTM", "Ensemble", "MC dropout"]); ax.set_ylim(0, 105)
        ax.set_ylabel("mean 90% interval width (cycles)")
        ax.set_title(f"{fd}: near failure (true RUL \u2264 40), n = {int(g.n.iloc[0])} engines")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4)
    fig.suptitle("Exploratory (D8). Left: native calibration curves (band = 95% engine bootstrap). Right: width near failure, "
                 "light = split conformal, solid = adaptive (CQR for GBM, normalised for LSTMs); labels = coverage", color=AM.INK, fontsize=9)
    AM.save(fig, "fig6_uq")


if __name__ == "__main__":
    V, P, Bd, M = run()
    figure(V, Bd)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    print(M.round(3).to_string(index=False))
    print(V[V.metric.isin(["RMSE", "MAE", "CE"])].round(3).to_string(index=False))
    print(V[(V.metric.isin(["PICP", "MPIW", "IS"])) & (V.level == 0.9)].round(3).to_string(index=False))
    print(P.round(3).to_string(index=False))
    print(Bd.round(3).to_string(index=False))
