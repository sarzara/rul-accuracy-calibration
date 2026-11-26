"""Sanity checks that must pass before any result is trusted. Run: python test_sanity.py"""
from pathlib import Path

import numpy as np
import pandas as pd

import conformal as C
from data import load, windows, regime_stats, normalise, unit_folds, WINDOW

RAW = Path(__file__).resolve().parents[1] / "data_raw"


def _grouped(rng, n_units, sd_u=3.0):
    eff = rng.normal(0, sd_u, n_units)
    units = np.repeat(np.arange(n_units), rng.integers(5, 40, n_units))
    return units, eff[units] + rng.normal(0, 1, len(units))


def test_cvplus_grouped_with_fold_models():
    """Units iid, windows correlated, fold models = noisy constants that differ per
    fold (exercises the fold indexing). One fresh engine, one random window."""
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(400):
        n_units, K = 60, 5
        units, y = _grouped(rng, n_units)
        fmap = np.empty(n_units, int)
        for k, f in enumerate(np.array_split(rng.permutation(n_units), K)):
            fmap[f] = k
        fold = fmap[units]
        bias = rng.normal(0, 0.5, K)                    # fold-specific predictor
        oof_mu = bias[fold]
        mu_k = bias[:, None]                            # (K, 1) predictions at the test point
        lo, hi = C.cvplus_abs(mu_k, oof_mu, y, fold, C.unit_weights(units), 0.1)
        yt = rng.normal(0, 3) + rng.normal(0, 1)
        covs.append(lo[0] <= yt <= hi[0])
    cov = float(np.mean(covs))
    assert cov >= 0.85, cov
    return cov


def test_small_n_gives_infinite_interval():
    """k=5 engines at a=0.1: ceil(0.9*6)=6>5 -> valid interval must be infinite (review C2)."""
    units = np.repeat(np.arange(5), 10)
    lo, hi = C.split_abs(np.zeros(3), np.zeros(50), np.random.default_rng(1).normal(size=50), units, 0.1)
    assert np.all(np.isinf(hi)) and np.all(np.isinf(lo))
    units = np.repeat(np.arange(9), 10)   # ceil(0.9*10)=9 <= 9 -> finite
    lo, hi = C.split_abs(np.zeros(3), np.zeros(90), np.random.default_rng(1).normal(size=90), units, 0.1)
    assert np.all(np.isfinite(hi))


def test_weighted_reduces_to_unweighted():
    rng = np.random.default_rng(2)
    units, y = _grouped(rng, 40)
    fold = units % 5
    mu_k = np.zeros((5, 4))
    a = C.cvplus_abs(mu_k, np.zeros_like(y), y, fold, C.unit_weights(units), 0.1)
    b = C.weighted_cvplus_abs(mu_k, np.zeros_like(y), y, fold, C.unit_weights(units),
                              np.zeros_like(y), np.zeros(4), 0.1)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_data_integrity():
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        df = load(fd, "train")
        assert (df.groupby("unit")["rul"].min() == 0).all()
        t = load(fd, "test")
        X, _ = normalise(t, regime_stats(df))
        Xw, yc, yr, u = windows(t, X, last_only=True)
        assert len(Xw) == t.unit.nunique() and Xw.shape[1] == WINDOW
        rul = pd.read_csv(RAW / f"RUL_{fd}.txt", header=None)[0].values
        assert np.allclose(yr, rul), fd
    folds = unit_folds(np.arange(1, 101), 5, 0)
    assert len(set(np.concatenate(folds))) == 100 and sum(map(len, folds)) == 100


if __name__ == "__main__":
    test_small_n_gives_infinite_interval(); test_weighted_reduces_to_unweighted(); test_data_integrity()
    print("grouped CV+ coverage (nominal 0.90):", test_cvplus_grouped_with_fold_models())
    print("ALL SANITY TESTS PASSED")
