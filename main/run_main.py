"""Main study (see PREREGISTRATION.md). One call = one (subset, seed).

Saves, for the official test set (last window per engine):
  point predictions, native interval bounds at every nominal level, and the
  split-conformal half-width q_L for every level, for Ridge, LightGBM and LSTM.
No metric is computed here.

Usage: python run_main.py FD001 0
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load, regime_stats, normalise, windows, RUL_CAP  # noqa: E402
from models import handcrafted, beta_nll  # noqa: E402
import conformal as C  # noqa: E402

torch.set_num_threads(2)
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
QUANTS = [0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975]
OUT = Path(__file__).resolve().parent / "results" / "raw"
SCALE = float(RUL_CAP)


# ---------------------------------------------------------------- splits
def split_engines(units, seed):
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(units))
    n_tr = int(round(0.70 * len(u)))
    train, cal = u[:n_tr], u[n_tr:]
    n_val = max(3, int(round(0.15 * len(train))))
    val, fit = train[:n_val], train[n_val:]
    assert not (set(fit) & set(val)) and not (set(train) & set(cal))
    return fit, val, cal


# ---------------------------------------------------------------- LSTM
class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=32, p=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU())
        self.mu, self.logvar = nn.Linear(32, 1), nn.Linear(32, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        z = self.head(h[:, -1])
        return self.mu(z).squeeze(-1), self.logvar(z).squeeze(-1).clamp(-10, 6)


def train_lstm(Xf, yf, Xv, yv, seed, max_epochs=60, patience=8, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LSTMNet(Xf.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xf_t, yf_t, Xv_t = torch.from_numpy(Xf), torch.from_numpy(yf / SCALE), torch.from_numpy(Xv)
    g = torch.Generator().manual_seed(seed)
    best, state, bad, ep = np.inf, None, 0, 0
    for ep in range(max_epochs):
        m.train()
        perm = torch.randperm(len(Xf_t), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            mu, lv = m(Xf_t[b])
            loss = beta_nll(mu, lv, yf_t[b], 0.5)
            opt.zero_grad(); loss.backward(); opt.step()
        mu, var = predict_lstm(m, Xv)
        nll = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (yv - mu) ** 2 / var))
        if nll < best - 1e-4:
            best, bad = nll, 0
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return m, {"val_nll": best, "epochs": ep + 1}


@torch.no_grad()
def predict_lstm(m, X, bs=4096):
    m.eval()
    mus, vs = [], []
    for i in range(0, len(X), bs):
        mu, lv = m(torch.from_numpy(X[i:i + bs]))
        mus.append(mu.numpy() * SCALE); vs.append(lv.exp().numpy() * SCALE ** 2)
    return np.concatenate(mus), np.concatenate(vs)


# ---------------------------------------------------------------- Ridge / LightGBM
def fit_ridge(Ff, yf, Fv, yv):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ff)
    best = None
    for a in (0.01, 0.1, 1, 10, 100, 1000):
        r = Ridge(alpha=a).fit(sc.transform(Ff), yf)
        mse = float(np.mean((r.predict(sc.transform(Fv)) - yv) ** 2))
        if best is None or mse < best[0]:
            best = (mse, a, r)
    mse, a, r = best
    return (lambda F: r.predict(sc.transform(F))), {"alpha": a, "sigma": float(np.sqrt(mse))}


def lgb_params(seed, **kw):
    return dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2, **kw)


def fit_lgbm(Ff, yf, Fv, yv, seed):
    import lightgbm as lgb
    cb = [lgb.early_stopping(50, verbose=False)]
    point = lgb.LGBMRegressor(**lgb_params(seed, objective="l2")).fit(Ff, yf, eval_set=[(Fv, yv)], callbacks=cb)
    # DEVIATION D1 (see DEVIATIONS.md): with LightGBM's quantile objective the upper-quantile models
    # (q >= 0.75) stay a constant 125 for every input, because boosting starts from the q-quantile of the
    # labels, which is the cap (about 30% of labels tie at 125), and never moves. Native quantiles are
    # therefore fitted with scikit-learn's histogram gradient boosting (same algorithm family), with the
    # pre-registered settings where they translate: learning rate 0.03, 31 leaves, min 50 samples per leaf,
    # fixed 500 iterations (no early stopping). Chosen on validation engines only.
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    qm = {q: HGB(loss="quantile", quantile=q, learning_rate=0.03, max_iter=500, max_leaf_nodes=31,
                 min_samples_leaf=50, early_stopping=False, random_state=seed).fit(Ff, yf) for q in QUANTS}
    info = {"trees_point": int(point.best_iteration_), "quantile_impl": "sklearn HistGradientBoosting, 500 iter"}

    def quantiles(F):
        Q = np.stack([qm[q].predict(F) for q in QUANTS], 1)
        info.setdefault("crossing_rate", []).append(float(np.mean(np.any(np.diff(Q, axis=1) < 0, axis=1))))
        return np.sort(Q, 1)  # repair crossing (rate logged)
    return point.predict, quantiles, info


# ---------------------------------------------------------------- conformal
def conformal_q(pred_cal, y_cal, units_cal, level):
    """Engine-weighted split-conformal half-width with the +inf atom (finite-sample
    correction (1-a)(1+1/n_engines); returns inf when too few engines). Coverage is
    approximate/conservative for engine-weighted pooling, not exact."""
    R = np.abs(y_cal - pred_cal)
    return float(C._wq_aug(R[None, :], C.unit_weights(units_cal), np.ones(1), level)[0])


# ---------------------------------------------------------------- main
def main(fd, seed):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    S, T = load(fd, "train"), load(fd, "test")
    fit_u, val_u, cal_u = split_engines(S.unit.values, seed)
    stats = regime_stats(S[S.unit.isin(np.r_[fit_u, val_u])])   # training engines only
    Xs, _ = normalise(S, stats)
    Xt, _ = normalise(T, stats)

    def sub(units):
        m = S.unit.isin(units).values
        return windows(S[m].reset_index(drop=True), Xs[m])

    (Xf, yf, _, _), (Xv, yv, _, _), (Xc, yc, _, uc) = sub(fit_u), sub(val_u), sub(cal_u)
    Xte, yte, yte_raw, ute = windows(T, Xt, last_only=True)
    Ff, Fv, Fc, Fte = map(handcrafted, (Xf, Xv, Xc, Xte))

    out = {"y": yte, "y_raw": yte_raw, "unit": ute}
    meta = {"subset": fd, "seed": seed, "n_fit": len(fit_u), "n_val": len(val_u), "n_cal": len(cal_u),
            "cal_units": cal_u.tolist(), "levels": LEVELS}

    def store(name, pt_te, pt_cal, native):  # native: dict level -> (lo, hi) on test
        out[f"{name}_point"] = pt_te
        for L in LEVELS:
            lo, hi = native[L]
            out[f"{name}_native_lo_{L}"], out[f"{name}_native_hi_{L}"] = lo, hi
            out[f"{name}_cq_{L}"] = np.array(conformal_q(pt_cal, yc, uc, L))
        out[f"{name}_cal_point"] = pt_cal

    # Ridge
    rp, rinfo = fit_ridge(Ff, yf, Fv, yv)
    p_te = rp(Fte)
    store("ridge", p_te, rp(Fc), {L: (p_te - norm.ppf(.5 + L / 2) * rinfo["sigma"],
                                      p_te + norm.ppf(.5 + L / 2) * rinfo["sigma"]) for L in LEVELS})
    meta["ridge"] = rinfo

    # LightGBM
    gp, gq, ginfo = fit_lgbm(Ff, yf, Fv, yv, seed)
    Q = gq(Fte); qi = {q: i for i, q in enumerate(QUANTS)}
    p_te = gp(Fte)
    store("gbm", p_te, gp(Fc), {L: (Q[:, qi[round((1 - L) / 2, 3)]], Q[:, qi[round((1 + L) / 2, 3)]]) for L in LEVELS})
    meta["gbm"] = ginfo

    # LSTM
    net, linfo = train_lstm(Xf, yf, Xv, yv, seed=1000 + seed)
    mu_te, var_te = predict_lstm(net, Xte)
    mu_c, _ = predict_lstm(net, Xc)
    s = np.sqrt(var_te)
    store("lstm", mu_te, mu_c, {L: (mu_te - norm.ppf(.5 + L / 2) * s, mu_te + norm.ppf(.5 + L / 2) * s) for L in LEVELS})
    out["lstm_sd"] = s
    meta["lstm"] = linfo

    # validation-set point error (for reporting model fit; never used for selection beyond pre-registered rules)
    meta["val_rmse"] = {"ridge": float(np.sqrt(np.mean((rp(Fv) - yv) ** 2))),
                        "gbm": float(np.sqrt(np.mean((gp(Fv) - yv) ** 2))),
                        "lstm": float(np.sqrt(np.mean((predict_lstm(net, Xv)[0] - yv) ** 2)))}
    vcov = {}
    pv_r = rp(Fv); Qv = gq(Fv); mv, vv = predict_lstm(net, Xv)
    for L in LEVELS:
        z_ = norm.ppf(.5 + L / 2)
        vcov[str(L)] = {"ridge": float(np.mean(np.abs(yv - pv_r) <= z_ * rinfo["sigma"])),
                        "gbm": float(np.mean((yv >= Qv[:, qi[round((1 - L) / 2, 3)]]) & (yv <= Qv[:, qi[round((1 + L) / 2, 3)]]))),
                        "lstm": float(np.mean(np.abs(yv - mv) <= z_ * np.sqrt(vv)))}
    meta["val_native_coverage_windows"] = vcov
    np.savez_compressed(OUT / f"{fd}_seed{seed}.npz", **out)
    meta["seconds"] = time.time() - t0
    (OUT / f"{fd}_seed{seed}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"[{fd} s{seed}] done {meta['seconds']:.0f}s val_rmse={meta['val_rmse']} lstm_ep={linfo['epochs']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
