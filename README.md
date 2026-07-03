# Ads Shown ≠ Sales Caused

**Retail media budgets reward the wrong ads.** When a shopper sees an ad and later buys, attribution tools credit the ad, even if that shopper was going to buy anyway. The result: campaigns look great on paper while a large share of spend changes nothing. Measuring what advertising actually causes is the central credibility problem in retail media, and it can't be solved with attribution modeling. It requires a randomized experiment.

This project uses one: Criteo's 13.9M-user ad experiment, where users were randomly held out of advertising. That randomization makes true causal measurement possible — the difference between treated and control users *is* the effect of ads.

## The three questions this project answers

**1. Does advertising cause conversions, on average?**
Raw treatment-vs-control comparison with confidence intervals — the number an attribution report can't give you.

**2. For whom?**
Uplift models (T-learner, S-learner, class transformation) rank users by how much ads change their behavior.

**3. How should a media budget be allocated?**
A profit simulation compares three targeting policies at every budget level: target everyone, target likely buyers (the industry default), and target by predicted uplift. The output is a single chart: incremental profit vs. % of audience targeted, per policy.

## Status

**Work in progress**

| Milestone | Status |
|---|---|
| Randomization checks + raw treatment effect | in progress |
| Uplift models compared on Qini / AUUC | — |
| Budget-allocation profit curves | — |
| Interactive dashboard | — |

| Model | AUUC | Uplift@10% |
|---|---|---|
| T-learner / LightGBM | ⟨—⟩ | ⟨—⟩ |
| Class transformation | ⟨—⟩ | ⟨—⟩ |
| S-learner / LightGBM | ⟨—⟩ | ⟨—⟩ |

## Method

| Step | Approach |
|---|---|
| Ground truth | Criteo randomized experiment: 13.9M users, 12 features, treatment/control |
| Sanity checks | Randomization balance, raw ATE with 95% CIs |
| Uplift models | Meta-learners over LightGBM and logistic regression |
| Evaluation | Qini curves, AUUC, uplift@k — counterfactual-appropriate metrics |
| Business layer | Profit simulation: media cost per exposure vs. margin per incremental conversion |

## Reproduce it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download          # fetch + sample the Criteo uplift dataset
python -m src.train             # train models, print AUUC table
```

```
notebooks/   01_experiment_eda → 02_uplift_models → 03_budget_allocation
src/         data prep · uplift learners · Qini/AUUC · profit simulation
app/         budget-allocation dashboard (week 4)
```
