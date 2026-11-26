# Pre-registration — main study

**Written:** 2026-11-16, before any Ridge or LSTM model was trained and before any main-study result existed.
**Frozen by:** SHA-256 of this file, recorded in `PREREGISTRATION.sha256` at the time of writing.
Any change after that point is listed under *Deviations* at the bottom, with date and reason.

## Research question

Do RUL models with lower point-prediction error also give better prediction intervals?
Two parts:

- **Native intervals:** the uncertainty each model produces by itself.
- **Conformal intervals:** the same model wrapped in one common method, split conformal prediction.

## Status of the data (honest disclosure)

A pilot study (10 Nov 2026) already evaluated a 1D-CNN and LightGBM (quantile regression and CQR) on the official test sets of all four C-MAPSS subsets, including FD001 and FD004. Therefore:

- **FD001** results are reported as **exploratory** (replication of the pilot setting).
- **FD004** results are **pre-specified but not blind**: the settings below are fixed before running, but LightGBM-type results on this test set were seen in the pilot.
- **Ridge and LSTM** have never been run on any subset. For these two models both subsets are unseen.

## Data

- Subsets: FD001 (1 operating condition, 1 fault mode) and FD004 (6 conditions, 2 fault modes).
- 14 sensors: x2, x3, x4, x7, x8, x9, x11, x12, x13, x14, x15, x17, x20, x21.
- Regime from altitude setting; z-score per regime using statistics of the training engines of each seed.
- Target: RUL capped at 125 cycles (piecewise linear). Window: 30 cycles. Test engines shorter than 30 cycles are edge-padded.

## Splits (per seed s ∈ {0, 1, 2, 3, 4})

- Engines of the official training file are shuffled with seed s.
- **70% training engines**; of these, 15% are held out as **validation engines** for early stopping and Ridge α.
- **30% calibration engines**, used only for split conformal.
- **Test set:** the official NASA test file, one prediction per engine at its last cycle (FD001: 100 engines, FD004: 248).
- No engine appears in more than one of training, validation, calibration.

## Models

| Model | Input | Point prediction | Native interval |
|---|---|---|---|
| Ridge | 56 window features (last value, mean, s.d., OLS slope per sensor), standardised | Ridge regression, α ∈ {0.01, 0.1, 1, 10, 100, 1000} chosen by validation MSE | Gaussian, homoscedastic: σ = RMSE on validation engines |
| Gradient boosting (LightGBM) | Same 56 features | L2 objective, learning rate 0.03, 31 leaves, min 50 samples per leaf, subsample 0.8, colsample 0.8, ≤ 2000 trees, early stopping (50 rounds) on validation loss | Quantile regression with identical settings at q ∈ {0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975}; crossing quantiles sorted |
| Small LSTM | Raw 30 × 14 window | 2-layer LSTM, 32 hidden units, dropout 0.1 between layers, dense 32 → mean and log-variance; β-NLL (β = 0.5); Adam lr 1e-3, weight decay 1e-4, batch 256, ≤ 60 epochs, early stopping on validation Gaussian NLL (patience 8) | Gaussian, heteroscedastic (predicted variance) |

No hyperparameter is tuned on calibration or test engines. No setting will be changed after test results are seen.

## Interval methods

- **Native** as in the table above.
- **Split conformal** (identical for all models): score = |y − ŷ| on all windows of the calibration engines, each engine given total weight 1; interval ŷ ± q, where q is the engine-weighted quantile at level (1 − α)(1 + 1/n_engines), returning an infinite interval when that level exceeds 1.
- Points and intervals are intersected with the target support [0, 125] for evaluation; unclipped values are also stored.

## Metrics

- Accuracy: **MAE**, **RMSE** (primary); mean NASA score on raw predictions vs official RUL (secondary).
- Calibration at nominal levels L ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}, highlighted 0.80 / 0.90 / 0.95:
  - **PICP** (empirical coverage), **MPIW** (mean width), **interval score** (Winkler).
  - **Calibration curve**: PICP against L.
  - **Calibration error (CE)**: mean |PICP − L| over the six levels.
- Exploratory: coverage by true RUL band (≤ 40, 40–80, 80–125); coverage on test engines with true RUL < 125.

## Statistics

- Mean and seed s.d. over 5 seeds.
- 95% intervals by engine-level bootstrap (2,000 resamples) of per-engine values averaged over seeds.
- Model comparisons by paired engine-level bootstrap; results described per subset; no p-value thresholds beyond the 95% intervals.

## Hypotheses

- **H1 (accuracy and native calibration disagree).** The ranking of the three models by RMSE differs from their ranking by native CE on at least one subset. Expected from the pilot: gradient boosting is the most accurate model, and its native quantile intervals under-cover.
- **H2 (conformal equalises coverage).** After split conformal, every model's PICP at L = 0.80, 0.90 and 0.95 lies within ±0.05 of L on both subsets.
- **H3 (under conformal, accuracy decides width).** After split conformal, the ranking of models by MPIW at each highlighted level equals their ranking by MAE.
- **H4 (exploratory).** After split conformal, coverage differs between models within true-RUL bands even when marginal coverage is similar.

A hypothesis is reported as supported, not supported, or inconclusive (when the 95% intervals do not separate the models).

## Deviations

(None at the time of writing.)
