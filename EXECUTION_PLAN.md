# Execution Plan — Retail Media Incrementality Project

End-to-end, from empty Linux terminal to published results. Assumes Ubuntu/Debian and the scaffold in this repo. Budget: ~2–4 weeks at ~1–2 hrs/day.

---

## Phase 0 — Environment & repo setup (Day 1)

```bash
# 1. System prerequisites
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

# 2. Project setup
cd ~/projects
unzip retail-media-uplift-scaffold.zip && cd retail-media-uplift

# 3. Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # add this to your daily routine
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify the scaffold works BEFORE writing any code
python -m pytest tests/ -q         # expect: 3 passed

# 5. Git + GitHub
git init && git add -A && git commit -m "Project scaffold"
gh repo create retail-media-uplift --public --source=. --push
# (or create the repo on github.com and: git remote add origin ... && git push -u origin main)
```

**Checkpoint:** tests pass, repo is on GitHub. Commit at least daily from here on.

---

## Phase 1 — Data acquisition & sanity checks (Days 2–4)

```bash
python -m src.download --rows 1000000   # ~10 min; needs ~2GB RAM for the full pull
```

If RAM is tight (<8GB): stream instead — in `src/download.py`, use
`load_dataset(..., streaming=True)` and take the first N shuffled rows, or download
the CSV from https://ailab.criteo.com/criteo-uplift-prediction-dataset/ and
`pd.read_csv(..., chunksize=500_000)`.

Then work through `notebooks/01_experiment_eda.ipynb`:

1. **Shape & rates** — rows, treatment share (~85% treated), visit rate, conversion rate (~0.3%). Write these numbers down; they drive every later decision.
2. **Randomization check** — feature means by treatment group should be near-identical (it was a real RCT). Standardized mean differences < 0.05 = healthy.
3. **Raw ATE** — `ate(df, 'conversion')` and `ate(df, 'visit')` with 95% CIs. This is your first *finding*: "advertising lifts conversion by X pp (Y% relative)."
4. **EDA plots** — feature distributions, outcome correlations. Features are anonymized (f0–f11), so keep this section short; the interesting analysis is causal, not descriptive.

**Checkpoint / commit:** notebook 01 runs top-to-bottom; ATE table with CIs in the README.

---

## Phase 2 — Uplift models (Days 5–10)

Work in `notebooks/02_uplift_models.ipynb`, then push stable code into `src/`.

1. **Split once, save indices** — 70/30, stratified on treatment × conversion (`src/train.py` does this). Never touch the test set until evaluation.
2. **Baselines first:**
   - T-learner + logistic regression (already in `src/uplift.py`)
   - S-learner + LightGBM
3. **Stronger models:**
   - T-learner + LightGBM
   - Class transformation via `sklift.models.ClassTransformation`
   - Optional: X-learner (`causalml`) if time allows
4. **Evaluate with uplift metrics only** — Qini curves on one plot, AUUC table, uplift@10%/20%. `src/evaluate.py` has qini_curve/auuc.
5. **Tune the best model** — RandomizedSearchCV on the base learner (depth, n_estimators, learning_rate). Don't over-invest: +0.001 AUUC matters less than Phase 3.
6. **Run `python -m src.train`** and paste its output into the README results table.

Known gotchas:
- Conversion is ~0.3% → use `scale_pos_weight` or `is_unbalance=True` in LightGBM.
- S-learner often collapses to near-zero uplift (treatment feature gets ignored) — that's a *finding*, mention it.
- If Qini curves look like noise: sample more rows (2–4M) — uplift signals are weak.

**Checkpoint / commit:** AUUC table with ≥3 models; Qini plot saved to `reports/figures/`.

---

## Phase 3 — Budget allocation, the business layer (Days 11–16)

Work in `notebooks/03_budget_allocation.ipynb` using `src/policy.py`.

1. **Set assumptions** (document them!): cost per exposed user (e.g., $0.10 CPM-ish), value per incremental conversion (e.g., $40 margin). Do a sensitivity check with ±50%.
2. **Build the naive competitor:** a plain conversion-propensity model (LightGBM on control-group data, predicting conversion). This is what most marketing teams actually target with.
3. **The money chart:** `profit_curve()` for (a) target everyone, (b) propensity scores, (c) uplift scores — three lines, incremental profit vs. % of audience targeted. Mark the optimum of each.
4. **Quantify the gap:** "At 20% budget, uplift targeting yields N more incremental conversions than propensity targeting, worth $X per million users."
5. **Segment quadrants:** cross conversion-propensity × predicted-uplift → persuadables (target), sure things (don't pay for them), lost causes (skip), do-not-disturbs (negative uplift — actively avoid).
6. **Sensitivity analysis:** rerun the money chart at 3 cost/value assumptions; show the conclusion is robust.

**Checkpoint / commit:** money chart in `reports/figures/`; one-paragraph recommendation written for a non-technical media buyer.

---

## Phase 4 — App & polish (Days 17–21)

1. **Streamlit dashboard** (`app/streamlit_app.py`): budget slider → expected incremental conversions + profit under each policy; Qini curve tab; example "customer card" with uplift score.
2. **Refactor:** everything reusable lives in `src/`; notebooks import from it. `python -m src.train` reproduces the results table end-to-end.
3. **Tests:** keep the 3 scaffold tests green; add 2–3 more (e.g., T-learner recovers positive uplift on synthetic persuadables).
4. **README final pass** — order matters:
   - One-sentence problem + money chart image at the top
   - Headline results table
   - 3 findings in plain English
   - Method summary, quickstart, repo structure
5. **Optional deploy:** Streamlit Community Cloud (free) — link it in the README header.

---

## Phase 5 — Results & findings write-up (Days 21+)

Fill this template in the README (and optionally a short `reports/findings.md`):

**Finding 1 — Ads work, on average:** raw ATE = X pp lift in conversion (95% CI [a, b]), a Y% relative lift over control.

**Finding 2 — But the average hides the story:** the top 10% of users by predicted uplift account for Z% of all incremental conversions; the bottom ~30% show near-zero or negative uplift.

**Finding 3 — Uplift targeting beats propensity targeting:** at a 20% audience budget, uplift targeting delivers N vs. M incremental conversions — propensity targeting spends heavily on "sure things" who convert regardless.

**Recommendation:** target segment X first, exclude do-not-disturbs, expected ROI improvement of P% under the stated cost assumptions (sensitivity: holds from $c1 to $c2 per exposure).

**Limitations (include these — they signal maturity):** anonymized features limit interpretation; single experiment/time period; cost assumptions are illustrative; conversion is rare so uplift estimates are noisy at small budgets.

---

## Timeline at a glance

| Days | Phase | Deliverable |
|---|---|---|
| 1 | Env + repo | venv, tests pass, GitHub repo live |
| 2–4 | Data + sanity | ATE with CIs, notebook 01 |
| 5–10 | Uplift models | AUUC table, Qini plot, notebook 02 |
| 11–16 | Business layer | Money chart, segments, notebook 03 |
| 17–21 | App + polish | Streamlit app, refactor, README |
| 21+ | Findings | Results write-up, optional deploy |

**If you fall behind:** cut the app, then extra models — never cut Phase 3 or the write-up.
