"""Models: 1D-CNN with heteroscedastic Gaussian head, and LightGBM quantile baseline.

v2 changes (after review): dropout in every conv block (needed for MC dropout to
perturb the whole feature extractor); early stopping on validation Gaussian NLL
(a proper scoring rule) instead of RMSE, so the variance head is selected for;
architecture width/dropout/beta chosen by a documented grid on source-validation
units only (tune.py); LightGBM uses early stopping on validation pinball loss.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data import RUL_CAP

SCALE = float(RUL_CAP)  # targets are trained on y / 125
CFG = {"width": 32, "p": 0.1, "beta": 0.5, "lr": 1e-3}  # selected by tune.py (results/tuning.json)


class CNN(nn.Module):
    """Small temporal CNN in the spirit of Li et al. (2018), with a mean and a
    log-variance head."""

    def __init__(self, n_feat: int, width: int = 64, p: float = 0.1):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            nn.Conv1d(n_feat, w, 5, padding=2), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(w, w, 5, padding=2), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(w, w, 3, padding=1), nn.ReLU(), nn.Dropout(p),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(4 * w, 64), nn.ReLU(), nn.Dropout(p),
        )
        self.mu = nn.Linear(64, 1)
        self.logvar = nn.Linear(64, 1)

    def forward(self, x):  # x: (B, T, F)
        h = self.body(x.transpose(1, 2))
        return self.mu(h).squeeze(-1), self.logvar(h).squeeze(-1).clamp(-10, 6)


def beta_nll(mu, logvar, y, beta=0.5):
    """beta-NLL (Seitzer et al., ICLR 2022): plain Gaussian NLL lets the variance
    absorb hard points and starves the mean of gradient; beta in (0,1] rebalances.
    beta=0 recovers the standard NLL."""
    var = logvar.exp()
    nll = 0.5 * (logvar + (y - mu) ** 2 / var)
    return (nll * var.detach() ** beta).mean() if beta > 0 else nll.mean()


def _val_nll(m, Xva_t, yva):
    with torch.no_grad():
        mu, lv = m(Xva_t)
    mu, var = mu.numpy() * SCALE, lv.exp().numpy() * SCALE ** 2
    nll = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (yva - mu) ** 2 / var))
    rmse = float(np.sqrt(np.mean((mu - yva) ** 2)))
    return nll, rmse


def train_cnn(Xtr, ytr, Xva, yva, seed, cfg=None, max_epochs=60, patience=8, bs=256):
    cfg = {**CFG, **(cfg or {})}
    torch.manual_seed(seed); np.random.seed(seed)
    m = CNN(Xtr.shape[-1], cfg["width"], cfg["p"])
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr / SCALE)
    Xva_t = torch.from_numpy(Xva)
    best, best_state, bad, ep = np.inf, None, 0, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        m.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            mu, lv = m(Xtr_t[b])
            loss = beta_nll(mu, lv, ytr_t[b], cfg["beta"])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        nll, rmse = _val_nll(m, Xva_t, yva)
        if nll < best - 1e-4:
            best, bad, best_rmse = nll, 0, rmse
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, {"val_nll": best, "val_rmse": best_rmse, "epochs": ep + 1}


@torch.no_grad()
def predict(m, X, bs=4096):
    m.eval()
    out_mu, out_var = [], []
    for i in range(0, len(X), bs):
        mu, lv = m(torch.from_numpy(X[i:i + bs]))
        out_mu.append(mu.numpy() * SCALE); out_var.append(lv.exp().numpy() * SCALE ** 2)
    return np.concatenate(out_mu), np.concatenate(out_var)


@torch.no_grad()
def predict_mc(m, X, T=100, seed=0, bs=4096):
    """MC dropout (Gal & Ghahramani, 2016) on a heteroscedastic net: dropout active,
    T passes. Returns per-pass means (T, n) and aleatoric variances (T, n) so that
    T-sensitivity can be computed afterwards without re-running."""
    torch.manual_seed(seed)
    m.train()  # dropout on (no batch-norm in the model)
    mus, vs = [], []
    for _ in range(T):
        mu_b, v_b = [], []
        for i in range(0, len(X), bs):
            mu, lv = m(torch.from_numpy(X[i:i + bs]))
            mu_b.append(mu.numpy() * SCALE); v_b.append(lv.exp().numpy() * SCALE ** 2)
        mus.append(np.concatenate(mu_b)); vs.append(np.concatenate(v_b))
    m.eval()
    return np.stack(mus).astype(np.float32), np.stack(vs).astype(np.float32)


# ---------------- LightGBM ----------------
def handcrafted(Xw):
    """Window features: last value, mean, std, OLS slope per sensor."""
    t = np.arange(Xw.shape[1], dtype=np.float32); t = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    slope = np.einsum("t,ntf->nf", t, Xw)
    return np.concatenate([Xw[:, -1], Xw.mean(1), Xw.std(1), slope], 1)


def train_lgbm_quantiles(F, y, Fva, yva, seed, qs=(0.05, 0.5, 0.95)):
    """Quantile GBMs with early stopping on validation pinball loss (same
    validation units as the CNN)."""
    import lightgbm as lgb
    models = {}
    for q in qs:
        models[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=2000,
                                      learning_rate=0.03, num_leaves=31, min_child_samples=50,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                      random_state=seed, verbose=-1, n_jobs=2).fit(
            F, y, eval_set=[(Fva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return models


def domain_logit(F_src, F_tgt, F_query_list, seed):
    """Probabilistic domain classifier source(0) vs target(1) on unlabelled inputs,
    for weighted conformal prediction (Tibshirani et al., 2019). Returns the
    log-odds for each query array, rebalanced for class sizes."""
    import lightgbm as lgb
    X = np.concatenate([F_src, F_tgt]); y = np.r_[np.zeros(len(F_src)), np.ones(len(F_tgt))]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=2).fit(X, y)
    prior = np.log(len(F_src) / len(F_tgt))
    return [clf.predict(Fq, raw_score=True) + prior for Fq in F_query_list]
