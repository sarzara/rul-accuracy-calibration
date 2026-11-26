"""Post-review supporting analyses (descriptive): MAE and native width by true-RUL band,
paired native near-failure coverage CI, calibration-engine coverage of conformal q,
and chosen hyperparameters per run. Writes results/tables/extra_*.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.argv = ["x"]
from run_main import split_engines, LEVELS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load, regime_stats, normalise, windows
import conformal as C
from inference import load_arrays, diff_ci

R = Path(__file__).resolve().parent / "results"; T = R / "tables"
BANDS = [(0, 40), (40, 80), (80, 125)]
rows, cal_rows, hp = [], [], []
for fd in ("FD001", "FD004"):
    for f in sorted((R / "raw").glob(f"{fd}_seed*.npz")):
        seed = int(f.stem.split("seed")[1]); z = dict(np.load(f)); y = z["y"]
        meta = json.loads(f.with_suffix(".json").read_text())
        hp.append({"subset": fd, "seed": seed, "ridge_alpha": meta["ridge"]["alpha"], "ridge_sigma": meta["ridge"]["sigma"],
                   "gbm_point_trees": meta["gbm"]["trees_point"], "lstm_epochs": meta["lstm"]["epochs"],
                   "ridge_frac_pred_above_125": float(np.mean(z["ridge_point"] > 125))})
        for m in ("ridge", "gbm", "lstm"):
            p = np.clip(z[f"{m}_point"], 0, 125)
            for a, b in BANDS:
                s = (y > a) & (y <= b) if a > 0 else (y <= b)
                w = np.clip(z[f"{m}_native_hi_0.9"], 0, 125) - np.clip(z[f"{m}_native_lo_0.9"], 0, 125)
                rows.append({"subset": fd, "seed": seed, "model": m, "band": f"{a}-{b}", "MAE": float(np.mean(np.abs(p - y)[s])),
                             "native_width_90": float(w[s].mean())})
        # calibration-engine coverage of the conformal half-width (reconstruct calibration labels deterministically)
        S = load(fd, "train"); fit_u, val_u, cal_u = split_engines(S.unit.values, seed)
        st = regime_stats(S[S.unit.isin(np.r_[fit_u, val_u])]); Xs, _ = normalise(S, st)
        msk = S.unit.isin(cal_u).values
        _, yc, _, uc = windows(S[msk].reset_index(drop=True), Xs[msk])
        wts = C.unit_weights(uc); wts = wts / wts.sum()
        for m in ("ridge", "gbm", "lstm"):
            pc = z[f"{m}_cal_point"]; assert len(pc) == len(yc)
            for L in LEVELS:
                q = float(z[f"{m}_cq_{L}"])
                cal_rows.append({"subset": fd, "seed": seed, "model": m, "level": L,
                                 "cal_engine_weighted_coverage": float(wts @ (np.abs(yc - pc) <= q)),
                                 "target": min(1.0, L * (1 + 1 / len(cal_u)))})
B = pd.DataFrame(rows).groupby(["subset", "model", "band"])[["MAE", "native_width_90"]].mean().reset_index()
B.to_csv(T / "extra_by_band.csv", index=False)
CC = pd.DataFrame(cal_rows).groupby(["subset", "model", "level"])[["cal_engine_weighted_coverage", "target"]].mean().reset_index()
CC.to_csv(T / "extra_calibration_engine_coverage.csv", index=False)
H = pd.DataFrame(hp); H.to_csv(T / "extra_hyperparameters.csv", index=False)
# paired native near-failure coverage (90%) GBM vs Ridge and vs LSTM
pr = []
for fd in ("FD001", "FD004"):
    A, y = load_arrays(fd); n_s, n_e = A[("ridge", "abs")].shape
    for lo, hi in BANDS:
        mask = (y > lo) & (y <= hi) if lo > 0 else (y <= hi)
        for a, b in (("gbm", "ridge"), ("gbm", "lstm"), ("lstm", "ridge")):
            d = diff_ci(A, n_s, n_e, a, b, "PICP", "native", 0.90, mask=mask)
            pr.append({"subset": fd, "band": f"{lo}-{hi}", "A": a, "B": b, "diff": d[0], "lo95": d[1], "hi95": d[2]})
pd.DataFrame(pr).to_csv(T / "extra_native_band_paired.csv", index=False)
pd.set_option("display.width", 200)
print(B.round(2).to_string(index=False)); print(CC[CC.level.isin([0.8, 0.9, 0.95])].round(3).to_string(index=False))
print(H.round(2).to_string(index=False)); print(pd.DataFrame(pr).round(3).to_string(index=False))
