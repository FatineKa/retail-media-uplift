# Project Plan — Retail Media Incrementality (4 weeks)

**Goal:** Answer the question every retail media network gets asked: *"Did the ads actually cause the sales, and which shoppers are worth the ad spend?"*

**Why this project?** Retail media budgets are exploding, and incrementality is the #1 credibility problem in the industry. Last-click attribution over-credits ads shown to people who would have bought anyway. Uplift modeling finds the **persuadables** — customers whose behavior ads actually change. Very few portfolio projects do this; it signals causal-inference literacy, not just model fitting.

**Dataset:** Criteo Uplift (https://huggingface.co/datasets/criteo/criteo-uplift) — ~13.9M users from a real randomized ad experiment: 12 anonymized features, `treatment` flag, `visit`/`conversion`/`exposure` outcomes. **Sample 1–2M rows** for fast iteration; scale up at the end.

---

## Week 1 — Framing + experiment sanity checks
- README problem statement: why last-click attribution misleads retail media buyers; what incrementality means in $.
- Download data (Hugging Face), sample it, EDA.
- **Randomization checks:** treatment/control balance on features, treatment share, raw ATE (average treatment effect) on visits and conversions with confidence intervals. This *is* the analysis a real measurement team does first.
- Notebook `01_experiment_eda.ipynb`.

## Week 2 — Uplift models
- Implement 3 approaches and compare:
  1. **T-learner** (two models: treated vs. control, subtract predictions)
  2. **S-learner** (one model with treatment as a feature)
  3. **Class transformation** (Athey-Imbens style, or use `scikit-uplift`)
- Base learners: logistic regression → LightGBM/XGBoost.
- **Uplift metrics, not AUC:** Qini curve, AUUC, uplift@k. (Standard metrics don't work — you never observe both outcomes for one person.)
- Notebook `02_uplift_models.ipynb`.

## Week 3 — The budget-allocation layer (what makes it "smart")
- Simulate a media budget: cost per exposed user vs. value per incremental conversion.
- Compare 3 targeting policies on held-out data: (a) target everyone, (b) target by *conversion propensity* (what naive teams do), (c) target by *predicted uplift*.
- Show the money chart: **incremental profit vs. % of budget spent** for each policy. Propensity targeting wastes money on "sure things" — prove it.
- Segment users into persuadables / sure things / lost causes / do-not-disturbs.
- Notebook `03_budget_allocation.ipynb`.

## Week 4 — Ship it
- Streamlit dashboard: budget slider → who to target, expected incremental conversions and profit under each policy.
- Refactor into `src/`, `python -m src.train` entrypoint, unit tests on the Qini/policy code.
- README polish: money chart at the top, plain-English findings for a media buyer, quickstart.
- Stretch: write a short "measurement methodology" doc — great interview artifact.

---

## Definition of done
- [ ] Randomization checks + raw ATE with confidence intervals
- [ ] ≥3 uplift approaches compared on Qini/AUUC
- [ ] Profit-vs-budget chart comparing uplift vs. propensity targeting
- [ ] Streamlit dashboard
- [ ] Reproducible: requirements.txt + `python -m src.train`
- [ ] README leads with the business finding, not the tech stack

## Rules of thumb
- Conversion rate is tiny (~0.3%) — always use stratified splits and report CIs.
- If behind schedule, cut the dashboard before the budget-allocation analysis.
- The story is "ads shown ≠ sales caused" — repeat it in every chart title.
