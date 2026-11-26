# Deviations from the pre-registration

`PREREGISTRATION.md` is kept unchanged so that its recorded SHA-256 stays valid; deviations are listed here.
All were found by an independent code review carried out after the first analysis of the main-study results (16 Nov 2026).

| # | What the pre-registration said | What was done | Why | Effect |
|---|---|---|---|---|
| D1 | Native gradient-boosting intervals from LightGBM quantile models with early stopping | Native quantiles from scikit-learn `HistGradientBoostingRegressor` (quantile loss; learning rate 0.03, 31 leaves, min 50 samples per leaf, fixed 500 iterations) | With LightGBM, the q = 0.75 model stopped after one tree and predicted a constant 125 for every input in all 10 runs; after the pre-registered sorting this constant became the upper bound of every interval at 80% and above. Boosting starts from the q-quantile of the labels, which is the cap (30% of FD001 and 43% of FD004 training windows have the label 125). A fixed number of trees did not help. The replacement was chosen on validation engines only. | Changes all native gradient-boosting results. Pre-registered (degenerate) outputs kept in `results/raw_prereg_v1/` and `results/tables_prereg_v1/` |
| D2 | "Inconclusive when the 95% intervals do not separate the models" | Decision rules implemented explicitly (`inference.py`): a ranking claim is decided by paired engine-bootstrap intervals of the differences on **both** rankings involved (supported / not supported only if the relevant pairs are resolved; otherwise inconclusive). H4 tested by paired band-coverage differences at 90% | The first analysis script decided H1–H3 from point estimates and checked separation only for interval width | Changed two pre-registered outcomes after the results were known: H1 on FD004 (supported → inconclusive) and H3 on FD001 (mixed → inconclusive). The point-estimate version is kept in `tables/hypotheses_point_estimate_rule_v1.csv` |
| D3 | Engine bootstrap of seed-averaged values | Kept as primary; added (a) bootstrap CIs for calibration error, (b) paired RMSE differences, (c) a two-level bootstrap over seeds and engines as sensitivity, (d) H3 with unclipped MAE as sensitivity | Calibration error had no interval; the engine bootstrap ignores training and calibration-split variability | No hypothesis outcome changes under (c) or (d) |
| D4 | — | LSTM log-variance clamped to [−10, 6] | Numerical safety; an implementation detail not written down | Bounds the predicted s.d. at 0.84–2,500 cycles; never binding in practice |
| D5 | — | Figure styling and titles in `analyze_main.py` edited after the first results; one figure title that stated a conclusion was replaced by a neutral one | Readability; reviewer request | No metric changed |
| D6 | Crossing quantiles sorted | Kept; crossing rate now logged | Separate quantile models with 12 closely spaced levels cross often | At least one crossing pair among the 12 quantiles in 85–94% of test engines (per run); sorting is the pre-registered repair |

## D7 — Exploratory addition: post-hoc recalibration (added 19 Nov 2026, after the main results were known)

**Status: exploratory, not pre-registered.** H1–H4 and every number in the main analysis are unchanged.

**Why.** Model-based post-hoc recalibration is the main alternative to conformal prediction for repairing miscalibrated RUL intervals in the prognostics literature (isotonic recalibration of Kuleshov et al., 2018; its use for Bayesian RUL models by Lin & Li, 2022; variance scaling of Levi et al., 2019). Without it a reader cannot tell whether the miscalibration of the native gradient-boosting intervals could have been fixed more cheaply.

**Plan, fixed before running (`run_recal.py`, `analyze_recal.py`).**
1. Re-fit the three models with the same seeds and splits as the main study; abort if the re-fitted test-set point predictions differ from the saved ones (max abs. difference > 1e-3 cycles).
2. On the same calibration engines used for conformal, compute the probability integral transform PIT_i = F_i(y_i) of every calibration window under each model's native predictive distribution: Gaussian N(mu, sigma^2) for Ridge (sigma = validation RMSE) and the LSTM; for gradient boosting the piecewise-linear CDF through its 12 sorted quantiles, extended linearly to tau = 0 and tau = 1 with the slopes of the two outermost quantile pairs.
3. **Isotonic recalibration (primary exploratory method).** The recalibration map is the engine-weighted empirical CDF of the PIT values (the isotonic fit of Kuleshov et al. on its own PIT values). The recalibrated central interval at level L is [F^-1(tau_lo), F^-1(tau_hi)] with tau_lo, tau_hi the engine-weighted (1-L)/2 and (1+L)/2 quantiles of PIT. No finite-sample correction.
4. **Variance scaling (secondary; Ridge and LSTM only, because gradient boosting has no variance parameter).** sigma is multiplied by s, where s^2 is the engine-weighted mean of the squared standardised calibration residuals (the maximum-likelihood scale).
5. Evaluation exactly as in the main study: test engines, clipping to [0, 125], PICP/MPIW/IS at 50–95%, calibration error CE over the six levels, seed-averaged per-engine values, 2,000-sample engine bootstrap; paired engine-bootstrap differences in CE between recalibrated and native, and between recalibrated and conformal intervals, per model; coverage at 90% by true-RUL band.
6. Reporting: all results labelled exploratory; no new hypothesis is claimed as confirmed.

**Effect.** Adds `results/raw_recal/`, `results/tables/recal_*.csv` and `results/figures/fig5_recalibration.*`; Section 7.8 of the thesis.

**Post-plan diagnostic (added after the D7 results were seen).** `analyze_recal.py` also writes `recal_lstm_tail_diagnostic.csv`, which compares the LSTM's maximum-likelihood variance-scaling factor with the factor needed for 90% coverage on the calibration windows. It explains a result; it changes none.

## D8 — Exploratory addition: epistemic uncertainty and locally adaptive conformal (added 21 Nov 2026, after the main and D7 results were known)

**Status: exploratory, not pre-registered.** H1–H4 and every number in the main analysis and in D7 are unchanged. The plan below was written before any D8 code was run.

**Why.** All native uncertainty in the main study is aleatoric (a noise model fitted to residuals). Two questions a reviewer will ask are therefore open: (a) is the accuracy–calibration gap a property of accurate models, or of the uncertainty mechanism attached to them? (b) Marginal split conformal repairs coverage on average but not where gradient boosting fails (near failure, one-sided); does a locally adaptive conformal method do better?

**Plan, fixed before running (`run_uq.py`, `analyze_uq.py`).**
1. Same subsets, seeds, splits, windows, features and test engines as the main study. Abort if the re-fitted single LSTM (member 0) or gradient-boosting point predictions differ from the saved ones by more than 1e-3 cycles.
2. **Deep ensemble** (Lakshminarayanan et al., 2017): M = 5 LSTMs with the main-study architecture, loss (beta-NLL, beta = 0.5), optimiser and early stopping; member k uses torch seed 1000 + seed + 100k (k = 0 is the main-study LSTM). Predictive mean = mean of member means; predictive variance = mean of member variances (aleatoric) + variance of member means (epistemic). Native interval: Gaussian with that mean and variance.
3. **MC dropout** (Gal & Ghahramani, 2016): the main-study LSTM (trained with dropout p = 0.1 between its two LSTM layers) evaluated with dropout active, T = 50 stochastic passes (torch seed fixed per run). Predictive mean and variance as for the ensemble. No retraining, so the comparison with the single LSTM isolates the inference method.
4. **Locally adaptive conformal** on the calibration engines, engine-weighted, with the same finite-sample correction as the main study:
   - gradient boosting: conformalized quantile regression (CQR; Romano et al., 2019) on the native quantiles, score E = max(q_lo(x) − y, y − q_hi(x)), interval [q_lo − Q, q_hi + Q];
   - LSTM, ensemble and MC dropout: normalised conformal, score |y − mu| / sigma, interval mu ± Q·sigma.
   Standard (absolute-residual) split conformal is also computed for the ensemble and MC-dropout means.
5. Evaluation exactly as in the main study (clipping to [0, 125], PICP/MPIW/IS at 50–95%, CE over six levels, seed-averaged per-engine values, 2,000-sample engine bootstrap; coverage at 90% by true-RUL band).
6. Pre-specified comparisons (paired engine bootstrap): (i) native CE: ensemble vs single LSTM, MC dropout vs single LSTM, ensemble vs gradient boosting; (ii) RMSE: ensemble vs single LSTM, ensemble vs gradient boosting; (iii) adaptive vs standard conformal per model: CE, MPIW and IS at 80/90/95%, and coverage at 90% for true RUL <= 40.
7. Reporting: all D8 results labelled exploratory; no hypothesis is claimed as confirmed; a resolved difference means a 95% interval excluding zero.

**Effect.** Adds `results/raw_uq/`, `results/tables/uq_*.csv`, `results/figures/fig6_uq.*`; Section 7.9 of the thesis.
