# Persuadables — Retail Media Incrementality with Uplift Modeling

> Which shoppers does advertising *actually* change — and how should a retail media budget be spent?

Last-click attribution over-credits ads shown to people who would have bought anyway. Using Criteo's 13.9M-user randomized ad experiment, this project measures true ad incrementality and builds uplift models that target **persuadables** — then shows in dollars why uplift targeting beats propensity targeting.

## Headline results (fill in)
| Targeting policy | Incremental conversions @ 20% budget | Est. profit |
|---|---|---|
| Everyone | – | – |
| Conversion propensity (naive) | – | – |
| **Uplift model (ours)** | – | – |

## Structure
```
retail-media-uplift/
├── data/raw/          # criteo-uplift-v2.1.csv (see data/raw/README.md)
├── notebooks/         # 01_experiment_eda → 02_uplift_models → 03_budget_allocation
├── src/               # data prep, uplift learners, qini/policy evaluation
├── app/               # Streamlit budget-allocation dashboard
└── PLAN.md            # 4-week roadmap
```

## Quickstart
```bash
pip install -r requirements.txt
python -m src.download          # fetches + samples the Criteo uplift dataset
python -m src.train             # trains T/S-learners, prints Qini/AUUC
streamlit run app/streamlit_app.py
```

## Method
1. **Experiment sanity checks** — randomization balance, raw ATE with CIs
2. **Uplift models** — T-learner, S-learner, class transformation; Qini/AUUC evaluation
3. **Budget allocation** — profit curves per targeting policy; persuadable segmentation
