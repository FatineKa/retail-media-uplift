# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Causal-inference / uplift-modeling project on the Criteo randomized ad experiment dataset (13.9M users, treatment/control). The core question: does advertising cause conversions, for whom, and how should a media budget be allocated. See `README.md` for the full problem framing and `PLAN.md` / `EXECUTION_PLAN.md` for the phased build plan and current status.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Download + sample the dataset (streams from Hugging Face; see Conventions below)
python -m src.download --rows 1000000

# Full pipeline: randomization checks -> ATE -> train all uplift models -> Qini/AUUC table
python -m src.train

# Tests
python -m pytest tests/ -q
python -m pytest tests/test_uplift.py::test_policy_profit_has_ci -q   # single test

# Data profile / EDA helper (prints shape, rates, SMD balance, ATE)
python -m src.data_prep

# Dashboard (WIP, see Architecture)
streamlit run app/streamlit_app.py
```

## Architecture

- `src/data_prep.py` — loads the sampled parquet, randomization balance check (standardized mean difference per feature), ATE with Welch CI.
- `src/uplift.py` — uplift meta-learners: `TLearner`, `SLearner`, `ClassTransformation` (IPW-rebalanced). All share the `fit(X, treatment, y)` / `predict_uplift(X)` interface.
- `src/evaluate.py` — counterfactual-appropriate metrics only (never AUC/accuracy — both potential outcomes are never observed for the same user): `qini_curve`, `auuc`, `qini_coefficient`, `uplift_at_k`.
- `src/policy.py` — turns uplift scores into a business decision: `policy_profit(...)` estimates $ profit and a 95% CI for targeting the top-k% of users by score; `profit_curve(...)` sweeps that across budget levels for one scoring policy at a time (call once for propensity scores, once for uplift scores — "target everyone" is just any policy's profit at `budget_frac=1.0`, since ranking stops mattering once everyone is targeted).
- `src/train.py` — end-to-end entrypoint wiring `data_prep` → `uplift` → `evaluate` together; prints the AUUC/uplift@10% comparison table used in the README results section.
- `app/streamlit_app.py` — budget-allocation dashboard; currently a stub, not implemented.
- `notebooks/01_experiment_eda` → `02_uplift_models` → `03_budget_allocation` — exploratory work; reusable logic is meant to be pushed into `src/`, with notebooks importing from it rather than duplicating logic.
- `tests/test_uplift.py` — validates methods against synthetic data with a *known* ground-truth uplift, not against real-world accuracy metrics (which don't exist for this problem — see `src/evaluate.py`'s docstring).

## Conventions specific to this codebase

- Every statistical function returns a plain dict of results **and** prints a human-readable diagnostic line (e.g. `[ATE:conversion] effect = ... 95% CI [...] p = ...`). Keep both when adding new metrics/policies — `python -m src.train`'s console output relies on the printed diagnostics, not just the return values.
- CIs are built with Welch's method (unequal-variance approximation, `±1.96*SE`) throughout — `data_prep.ate()` is the canonical pattern, reused in `policy.policy_profit()`.
- Treatment is randomized (real RCT), so propensity is always known exactly (`e = treatment.mean()`) rather than estimated — used directly in `uplift.ClassTransformation`'s IPW weights.
- Conversion is rare (~0.3%): LightGBM models default to `is_unbalance=True`, and splits are stratified on `treatment x conversion` (see `train_test_split` in `src/train.py`), never plain random.
- `data/raw/` is gitignored — the sampled parquet must be regenerated locally via `python -m src.download` before training or running the notebooks.
- `src/download.py` streams the dataset row-by-row with per-row Bernoulli sampling instead of materializing the full 13.9M-row dataset in pandas, to bound peak memory on low-RAM machines. No partial results are written until the full stream completes, so an interrupted download must restart from scratch.
