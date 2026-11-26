"""C-MAPSS data loading, regime-wise normalisation and windowing.

Design rules (enforced here, audited in review):
  * All splits are by engine unit, never by window.
  * Normalisation statistics are fit on training units only (per fold).
  * Regimes unseen in the source are normalised with statistics from the
    *unlabelled* target inputs (label-free, transductive). This is stated
    explicitly in the write-up as a form of unsupervised alignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data_raw"
COLS = ["unit", "cycle", "s1", "s2", "s3"] + [f"x{i}" for i in range(1, 22)]
# 14 informative sensors used throughout the C-MAPSS literature
SENSORS = [f"x{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
ALTITUDES = np.array([0.0, 10.0, 20.0, 25.0, 35.0, 42.0])  # kft, one per regime
RUL_CAP = 125
WINDOW = 30


def load(fd: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{split}_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    df["regime"] = np.abs(df["s1"].values[:, None] - ALTITUDES[None, :]).argmin(1)
    if split == "train":
        eol = df.groupby("unit")["cycle"].transform("max")
        df["rul"] = eol - df["cycle"]
    else:
        rul = pd.read_csv(RAW / f"RUL_{fd}.txt", header=None)[0].values
        eol = df.groupby("unit")["cycle"].transform("max")
        df["rul"] = (eol - df["cycle"]) + rul[df["unit"].values - 1]
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    return df


def regime_stats(df: pd.DataFrame) -> dict:
    g = df.groupby("regime")[SENSORS]
    mu, sd = g.mean(), g.std().replace(0, 1.0).fillna(1.0)
    return {r: (mu.loc[r].values, sd.loc[r].values) for r in mu.index}


def normalise(df: pd.DataFrame, src_stats: dict, tgt_stats: dict | None = None):
    """Per-regime z-score. Source stats preferred; unseen regimes fall back to
    label-free target stats. Returns array and fraction of rows using fallback."""
    X = df[SENSORS].values.astype(np.float32).copy()
    fallback = 0
    for r in np.unique(df["regime"].values):
        m = df["regime"].values == r
        if r in src_stats:
            mu, sd = src_stats[r]
        elif tgt_stats is not None and r in tgt_stats:
            mu, sd = tgt_stats[r]
            fallback += m.sum()
        else:
            raise ValueError(f"No statistics for regime {r}")
        X[m] = (X[m] - mu) / sd
    return X, fallback / len(df)


def windows(df: pd.DataFrame, X: np.ndarray, last_only: bool = False):
    """Sliding windows per unit; units shorter than WINDOW are edge-padded with
    their first observation. Returns (Xw, y_capped, y_raw, unit_ids, rul_true)."""
    xs, yc, yr, us = [], [], [], []
    for u, idx in df.groupby("unit").indices.items():
        xu = X[idx]
        if len(xu) < WINDOW:
            xu = np.vstack([np.repeat(xu[:1], WINDOW - len(xu), 0), xu])
            pad = WINDOW - len(idx)
        else:
            pad = 0
        rc = df["rul_capped"].values[idx]
        rr = df["rul"].values[idx]
        ends = [len(xu)] if last_only else range(WINDOW, len(xu) + 1)
        for e in ends:
            xs.append(xu[e - WINDOW:e])
            j = e - 1 - pad  # index into the original (unpadded) rows
            yc.append(rc[j]); yr.append(rr[j]); us.append(u)
    return (np.stack(xs).astype(np.float32), np.array(yc, np.float32),
            np.array(yr, np.float32), np.array(us))


def unit_folds(units: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(units))
    return np.array_split(u, k)
