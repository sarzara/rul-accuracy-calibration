"""Hypothesis tests with explicit decision rules (deviation D2, see DEVIATIONS.md).

Pre-registered wording: a hypothesis is 'inconclusive when the 95% intervals do not separate
the models'. The first analysis script decided H1-H3 from point estimates. Here every decision
uses paired bootstrap intervals of the model differences on BOTH rankings involved.

Primary bootstrap (pre-registered): resample test engines (2,000x) using per-engine values
averaged over the 5 seeds. Sensitivity: two-level bootstrap resampling seeds and engines.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW, TAB = ROOT / "results" / "raw", ROOT / "results" / "tables"
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
HIGH = [0.80, 0.90, 0.95]
MODELS = ["ridge", "gbm", "lstm"]
SUBSETS = ["FD001", "FD004"]
CAP, B = 125.0, 2000
BANDS = [(0, 40), (40, 80), (80, 125)]


def load_arrays(fd):
    """A[(model, kind)] -> array (seeds, engines). kinds: abs, sq, abs_raw, ('cov'|'w', method, L)."""
    A, y = {}, None
    files = sorted(RAW.glob(f"{fd}_seed*.npz"))
    for f in files:
        z = dict(np.load(f)); y = z["y"]
        for m in MODELS:
            p = z[f"{m}_point"]; pc = np.clip(p, 0, CAP)
            A.setdefault((m, "abs"), []).append(np.abs(pc - y))
            A.setdefault((m, "sq"), []).append((pc - y) ** 2)
            A.setdefault((m, "abs_raw"), []).append(np.abs(p - y))
            for meth in ("native", "conformal"):
                for L in LEVELS:
                    if meth == "native":
                        lo, hi = z[f"{m}_native_lo_{L}"], z[f"{m}_native_hi_{L}"]
                    else:
                        q = float(z[f"{m}_cq_{L}"]); lo, hi = p - q, p + q
                    lc, hc = np.clip(lo, 0, CAP), np.clip(hi, 0, CAP)
                    A.setdefault((m, "cov", meth, L), []).append(((y >= lc) & (y <= hc)).astype(float))
                    A.setdefault((m, "w", meth, L), []).append(hc - lc)
    return {k: np.stack(v) for k, v in A.items()}, y


def stat(A, m, what, idx_s, idx_e, meth=None, L=None, mask=None):
    """Statistic on a (seed, engine) resample. idx_s: seed indices; idx_e: engine indices."""
    def take(k):
        X = A[k][idx_s][:, idx_e]
        if mask is not None:
            X = X[:, mask[idx_e]]
        return X
    if what == "MAE":
        return take((m, "abs")).mean()
    if what == "MAE_raw":
        return take((m, "abs_raw")).mean()
    if what == "RMSE":
        return np.sqrt(take((m, "sq")).mean())
    if what == "PICP":
        return take((m, "cov", meth, L)).mean()
    if what == "MPIW":
        return take((m, "w", meth, L)).mean()
    if what == "CE":
        return np.mean([abs(take((m, "cov", meth, l)).mean() - l) for l in LEVELS])
    raise ValueError(what)


def boot(A, n_s, n_e, fn, two_level, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    all_s = np.arange(n_s)
    for b in range(B):
        idx_s = rng.integers(0, n_s, n_s) if two_level else all_s
        idx_e = rng.integers(0, n_e, n_e)
        out[b] = fn(idx_s, idx_e)
    return out


def diff_ci(A, n_s, n_e, a, b, what, meth=None, L=None, mask=None, two_level=False, seed=0):
    f = lambda s, e: stat(A, a, what, s, e, meth, L, mask) - stat(A, b, what, s, e, meth, L, mask)
    full = f(np.arange(n_s), np.arange(n_e))
    bs = boot(A, n_s, n_e, f, two_level, seed)
    lo, hi = np.quantile(bs, [.025, .975])
    return float(full), float(lo), float(hi)


def value_ci(A, n_s, n_e, m, what, meth=None, L=None, two_level=False, seed=0):
    f = lambda s, e: stat(A, m, what, s, e, meth, L)
    bs = boot(A, n_s, n_e, f, two_level, seed)
    return float(f(np.arange(n_s), np.arange(n_e))), *map(float, np.quantile(bs, [.025, .975]))


def resolved(lo, hi):
    return lo > 0 or hi < 0


def decide_ranking(pairs_x, pairs_y):
    """pairs_*: {(a,b): (diff, lo, hi)} for 'lower is better' metrics x and y.
    supported-as-reversal if some pair has resolved opposite signs; 'agree' if every pair is resolved
    on both and signs agree; otherwise inconclusive."""
    reversal = any(resolved(*pairs_x[k][1:]) and resolved(*pairs_y[k][1:]) and
                   np.sign(pairs_x[k][0]) != np.sign(pairs_y[k][0]) for k in pairs_x)
    agree = all(resolved(*pairs_x[k][1:]) and resolved(*pairs_y[k][1:]) and
                np.sign(pairs_x[k][0]) == np.sign(pairs_y[k][0]) for k in pairs_x)
    return "reversal" if reversal else ("agree" if agree else "unresolved")


def run(two_level):
    rows, tests = [], []
    for fd in SUBSETS:
        A, y = load_arrays(fd)
        n_s, n_e = A[(MODELS[0], "abs")].shape
        sd = 0 if not two_level else 1
        pairs = list(combinations(MODELS, 2))
        # value CIs
        for m in MODELS:
            for what in ("MAE", "RMSE"):
                rows.append({"subset": fd, "model": m, "metric": what, "method": "point", "level": np.nan,
                             **dict(zip(("value", "lo95", "hi95"), value_ci(A, n_s, n_e, m, what, two_level=two_level, seed=sd)))})
            for meth in ("native", "conformal"):
                rows.append({"subset": fd, "model": m, "metric": "CE", "method": meth, "level": np.nan,
                             **dict(zip(("value", "lo95", "hi95"), value_ci(A, n_s, n_e, m, "CE", meth, two_level=two_level, seed=sd)))})
        # H1: RMSE vs native CE
        px = {p: diff_ci(A, n_s, n_e, *p, "RMSE", two_level=two_level, seed=sd) for p in pairs}
        py = {p: diff_ci(A, n_s, n_e, *p, "CE", "native", two_level=two_level, seed=sd) for p in pairs}
        d = decide_ranking(px, py)
        tests.append({"hypothesis": "H1", "subset": fd, "level": np.nan,
                      "result": {"reversal": "supported", "agree": "not supported", "unresolved": "inconclusive"}[d],
                      "detail": "; ".join(f"{a}-{b}: dRMSE {px[(a,b)][0]:+.2f} [{px[(a,b)][1]:+.2f},{px[(a,b)][2]:+.2f}], "
                                          f"dCE {py[(a,b)][0]:+.3f} [{py[(a,b)][1]:+.3f},{py[(a,b)][2]:+.3f}]" for a, b in pairs)})
        # H2: conformal PICP within +-0.05 (point estimate, as pre-registered) + CIs reported
        for L in HIGH:
            parts, ok = [], True
            for m in MODELS:
                v, lo, hi = value_ci(A, n_s, n_e, m, "PICP", "conformal", L, two_level=two_level, seed=sd)
                ok &= abs(v - L) <= 0.05
                parts.append(f"{m} {v:.3f} [{lo:.3f},{hi:.3f}]")
            tests.append({"hypothesis": "H2", "subset": fd, "level": L, "result": "supported" if ok else "not supported",
                          "detail": "; ".join(parts)})
        # H3: conformal MPIW vs MAE (clipped, primary) and vs MAE_raw (sensitivity)
        for mae_kind in ("MAE", "MAE_raw"):
            px = {p: diff_ci(A, n_s, n_e, *p, mae_kind, two_level=two_level, seed=sd) for p in pairs}
            for L in HIGH:
                py = {p: diff_ci(A, n_s, n_e, *p, "MPIW", "conformal", L, two_level=two_level, seed=sd) for p in pairs}
                d = decide_ranking(px, py)
                tests.append({"hypothesis": "H3" if mae_kind == "MAE" else "H3 (sens.: unclipped MAE)", "subset": fd, "level": L,
                              "result": {"reversal": "not supported", "agree": "supported", "unresolved": "inconclusive"}[d],
                              "detail": "; ".join(f"{a}-{b}: d{mae_kind} {px[(a,b)][0]:+.2f} [{px[(a,b)][1]:+.2f},{px[(a,b)][2]:+.2f}], "
                                                  f"dMPIW {py[(a,b)][0]:+.2f} [{py[(a,b)][1]:+.2f},{py[(a,b)][2]:+.2f}]" for a, b in pairs)})
        # H4: conformal 90% coverage within true-RUL bands, paired model differences
        res, parts = False, []
        for lo_b, hi_b in BANDS:
            mask = (y > lo_b) & (y <= hi_b) if lo_b > 0 else (y <= hi_b)
            for a, b in pairs:
                dd = diff_ci(A, n_s, n_e, a, b, "PICP", "conformal", 0.90, mask=mask, two_level=two_level, seed=sd)
                if resolved(*dd[1:]):
                    res = True
                parts.append(f"band {lo_b}-{hi_b} (n={int(mask.sum())}) {a}-{b}: {dd[0]:+.3f} [{dd[1]:+.3f},{dd[2]:+.3f}]")
        tests.append({"hypothesis": "H4", "subset": fd, "level": 0.90, "result": "supported" if res else "inconclusive",
                      "detail": "; ".join(parts)})
    return pd.DataFrame(rows), pd.DataFrame(tests)


if __name__ == "__main__":
    V, T = run(two_level=False)
    V.to_csv(TAB / "value_ci_primary.csv", index=False); T.to_csv(TAB / "hypotheses_primary.csv", index=False)
    V2, T2 = run(two_level=True)
    V2.to_csv(TAB / "value_ci_two_level.csv", index=False); T2.to_csv(TAB / "hypotheses_two_level.csv", index=False)
    M = T[["hypothesis", "subset", "level", "result"]].merge(
        T2[["hypothesis", "subset", "level", "result"]], on=["hypothesis", "subset", "level"], suffixes=("_primary", "_two_level"))
    print(M.to_string(index=False))
