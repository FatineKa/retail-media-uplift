# Ads Shown ≠ Sales Caused

[![tests](https://github.com/FatineKa/retail-media-uplift/actions/workflows/tests.yml/badge.svg)](https://github.com/FatineKa/retail-media-uplift/actions/workflows/tests.yml)

**[Live dashboard →](https://retail-media-uplift-plus.streamlit.app/)**

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

**Results in, on a 3M-row sample (8,770 converters).**

| Milestone | Status |
|---|---|
| Randomization checks + raw treatment effect | done |
| Uplift models compared on Qini / AUUC | done |
| Budget-allocation profit curves | done |
| Interactive dashboard | done |

Raw ATE: advertising lifts conversion +0.00113 (95% CI [+0.00098, +0.00127], p < 1e-300), a
+57.5% relative lift over the 0.20% control-group baseline.

| Model | Qini coefficient | AUUC | Uplift@10% |
|---|---|---|---|
| S-learner / LightGBM | **+0.000352** | 748.64 | 0.00802 |
| T-learner / logistic | +0.000352 | 748.31 | 0.00802 |
| T-learner / LightGBM | +0.000220 | 629.46 | 0.00674 |
| Class transformation | −0.000133 | 311.90 | 0.00286 |

The best model is chosen on a validation slice (20% of the training data); the table
reports the held-out test set (900,481 users).

**Key finding:** three of the four uplift learners rank users better than random targeting,
but ranking well is not the same as earning more. On the same held-out users, uplift
targeting does not beat propensity targeting (the industry default) at any budget level:
at a 2% budget it nets $13,669 vs. $10,635, a difference of +$3,034 with a 95% CI of
[−$4,617, +$8,665] (paired bootstrap, 200 resamples). From 35% to 84% of the audience,
propensity targeting earns significantly more. With conversion this rare (0.3%), the
people most moved by ads are hard to separate from the people likely to buy anyway, so
before replacing a simple targeting rule with an uplift model, the profit gain has to be
tested, not assumed.

## Method

| Step | Approach |
|---|---|
| Ground truth | Criteo randomized experiment: 13.9M users, 12 features, treatment/control |
| Sanity checks | Randomization balance, raw ATE with 95% CIs |
| Uplift models | Meta-learners over LightGBM and logistic regression |
| Evaluation | Qini curves, AUUC, uplift@k — counterfactual-appropriate metrics |
| Business layer | Profit simulation: media cost per exposure vs. margin per incremental conversion |
| Policy comparison | Paired bootstrap CI on the profit difference between policies, same users |

## Cloud data pipeline

Separate from the modeling above, a small automated pipeline re-runs the randomization check as a deployable quality gate, built to practice running this kind of check in the cloud rather than only in a notebook:

- **Snowflake** (`snowflake/`) stores the 3M-row sample, loaded from S3 through a storage integration, with the per-feature standardized-mean-difference check as SQL.
- **AWS Lambda** (`lambda/quality_gate.py`) re-runs that check on demand, authenticating to Snowflake with an RSA key pair (the account enforces MFA, which blocks password-based service accounts).
- **Step Functions** (`step-functions/`) calls the Lambda, then publishes a pass/fail notification over SNS: `RunQualityGate → CheckResult → NotifyPass / NotifyFail`.

This pipeline doesn't feed the local model training above, it's a standalone exercise in shipping one specific check (data still looks randomized) as an automated, cloud-deployed gate rather than a cell someone has to remember to re-run by hand.

## Reproduce it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download          # fetch + sample the Criteo uplift dataset
python -m src.train             # train models, print AUUC table
streamlit run app/streamlit_app.py   # or just use the live dashboard above
```

```
notebooks/   01_experiment_eda → 02_uplift_models → 03_budget_allocation
src/         data prep · uplift learners · Qini/AUUC · profit simulation
app/         budget-allocation dashboard, live at retail-media-uplift-plus.streamlit.app
```
