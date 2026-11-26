# Main study — accuracy vs calibration of RUL models (C-MAPSS FD001, FD004)

Order of work (timestamps in file metadata):
1. `PREREGISTRATION.md` written and hashed (`PREREGISTRATION.sha256`) before any main-study model was run.
2. `run_main.py FD00x <seed>` for seeds 0-4  -> `results/raw/`
3. `analyze_main.py` -> metrics tables and figures in `results/tables/`, `results/figures/`
4. `inference.py` -> hypothesis tests with bootstrap intervals (`hypotheses_primary.csv`, `hypotheses_two_level.csv`); `extra_checks.py` -> supporting analyses (`extra_*.csv`)
5. Exploratory (deviation D7): `run_recal.py <FD001|FD004> <seed>` for seeds 0-4 -> `results/raw_recal/`; then `analyze_recal.py` -> `recal_*.csv` and `fig5_recalibration`
6. Exploratory (deviation D8): `run_uq.py <FD001|FD004> <seed>` for seeds 0-4 -> `results/raw_uq/` (deep ensemble, MC dropout, CQR, normalised conformal; ~30 min on 2 cores); then `analyze_uq.py` -> `uq_*.csv` and `fig6_uq`
7. Deviations (D1-D6 from independent review; D7, D8 exploratory): `DEVIATIONS.md`. Pre-registered outputs before the fix: `results/raw_prereg_v1/`, `results/tables_prereg_v1/`.

Requires the C-MAPSS text files in `../data_raw/` (see `../SHA256SUMS.txt`) and the shared modules in `../src/`.
