"""Exploratory deviation D8 (see DEVIATIONS.md): epistemic uncertainty and locally adaptive conformal.

For one (subset, seed), with the main study's splits:
  * deep ensemble of 5 beta-NLL LSTMs (member 0 = the main-study LSTM);
  * MC dropout (T = 50) on the main-study LSTM;
  * CQR on the gradient-boosting native quantiles;
  * normalised conformal (|y - mu| / sigma) for the single LSTM, the ensemble and MC dropout;
  * standard split conformal for the ensemble and MC-dropout means.
Saves results/raw_uq/{FD}_seed{s}.npz and .json. No metric is computed here.

Usage: python run_uq.py FD001 0
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

import run_main as RM
from run_main import LEVELS, QUANTS, SCALE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load, regime_stats, normalise, windows  # noqa: E402
from models import handcrafted  # noqa: E402
import conformal as C  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "raw_uq"
M_ENS, T_MC = 5, 50
QI = {q: i for i, q in enumerate(QUANTS)}


def cq(scores, w, L):
    """Engine-weighted conformal quantile of arbitrary non-conformity scores (same correction as the main study)."""
    return float(C._wq_aug(scores[None, :], w, np.ones(1), L)[0])


@torch.no_grad()
def predict_mc(m, X, T, seed, bs=4096):
    """MC dropout: dropout layers in train mode, everything else as in eval. Returns (T, n) means and variances."""
    torch.manual_seed(seed)
    m.eval()
    m.lstm.train()  # nn.LSTM applies its inter-layer dropout only in training mode
    mus, vs = [], []
    for _ in range(T):
        a, b = [], []
        for i in range(0, len(X), bs):
            mu, lv = m(torch.from_numpy(X[i:i + bs]))
            a.append(mu.numpy() * SCALE); b.append(lv.exp().numpy() * SCALE ** 2)
        mus.append(np.concatenate(a)); vs.append(np.concatenate(b))
    m.eval()
    return np.stack(mus), np.stack(vs)


def mixture(mus, vs):
    mu = mus.mean(0)
    alea, epi = vs.mean(0), mus.var(0)
    return mu, alea, epi


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
    out = {"y": yte, "unit": ute}
    meta = {"subset": fd, "seed": seed, "M": M_ENS, "T": T_MC}

    # ---------------- gradient boosting: CQR on the native quantiles
    gp, gq, _ = RM.fit_lgbm(Ff, yf, Fv, yv, seed)
    check = {"gbm": float(np.max(np.abs(gp(Fte) - saved["gbm_point"])))}
    Qc, Qt = gq(Fc), gq(Fte)
    for L in LEVELS:
        jl, jh = QI[round((1 - L) / 2, 3)], QI[round((1 + L) / 2, 3)]
        E = np.maximum(Qc[:, jl] - yc, yc - Qc[:, jh])
        q = cq(E, w, L)
        out[f"gbm_adapt_lo_{L}"], out[f"gbm_adapt_hi_{L}"] = Qt[:, jl] - q, Qt[:, jh] + q
        out[f"gbm_adapt_q_{L}"] = np.array(q)

    # ---------------- LSTM ensemble (member 0 = main-study network)
    members = []
    for k in range(M_ENS):
        net, info = RM.train_lstm(Xf, yf, Xv, yv, seed=1000 + seed + 100 * k)
        members.append(net)
        meta[f"member{k}_epochs"] = info["epochs"]
    mu0 = RM.predict_lstm(members[0], Xte)[0]
    check["lstm"] = float(np.max(np.abs(mu0 - saved["lstm_point"])))
    meta["reproduction_max_abs_diff"] = check
    if max(check.values()) > 1e-3:
        raise RuntimeError(f"re-fit does not reproduce the main study: {check}")

    def preds(X):
        P = [RM.predict_lstm(n, X) for n in members]
        return np.stack([p[0] for p in P]), np.stack([p[1] for p in P])

    ens_c, ens_t = mixture(*preds(Xc)), mixture(*preds(Xte))
    mc_c = mixture(*predict_mc(members[0], Xc, T_MC, seed=7000 + seed))
    mc_t = mixture(*predict_mc(members[0], Xte, T_MC, seed=8000 + seed))
    single_c, single_t = RM.predict_lstm(members[0], Xc), RM.predict_lstm(members[0], Xte)
    single_c = (single_c[0], single_c[1], np.zeros_like(single_c[1]))
    single_t = (single_t[0], single_t[1], np.zeros_like(single_t[1]))

    for name, (mc_, ac, ec), (mt, at, et) in (("ens", ens_c, ens_t), ("mcd", mc_c, mc_t), ("lstm", single_c, single_t)):
        sc, st = np.sqrt(ac + ec), np.sqrt(at + et)
        if name != "lstm":
            out[f"{name}_point"] = mt
            out[f"{name}_alea_var"], out[f"{name}_epi_var"] = at, et
            out[f"{name}_cal_point"] = mc_
        for L in LEVELS:
            z = norm.ppf(.5 + L / 2)
            if name != "lstm":
                out[f"{name}_native_lo_{L}"], out[f"{name}_native_hi_{L}"] = mt - z * st, mt + z * st
                out[f"{name}_cq_{L}"] = np.array(cq(np.abs(yc - mc_), w, L))
            qn = cq(np.abs(yc - mc_) / sc, w, L)
            out[f"{name}_adapt_lo_{L}"], out[f"{name}_adapt_hi_{L}"] = mt - qn * st, mt + qn * st
            out[f"{name}_adapt_q_{L}"] = np.array(qn)
        meta[f"{name}_test_mean_sd"] = float(np.mean(st))
        meta[f"{name}_test_epistemic_share"] = float(np.mean(et / (at + et)))

    np.savez_compressed(OUT / f"{fd}_seed{seed}.npz", **out)
    meta["seconds"] = time.time() - t0
    (OUT / f"{fd}_seed{seed}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"[{fd} s{seed}] uq done {meta['seconds']:.0f}s repro={check} "
          f"epi_share ens={meta['ens_test_epistemic_share']:.3f} mcd={meta['mcd_test_epistemic_share']:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
