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
| T-learner / logistic | **+0.000356** | 751.90 | 0.00770 |
| T-learner / LightGBM | −0.000141 | 304.96 | 0.00176 |
| Class transformation | −0.000166 | 282.47 | 0.00267 |
| S-learner / LightGBM | −0.000296 | 165.07 | 0.00129 |

**Key finding:** every LightGBM-based uplift learner scores *worse than random targeting*
(negative Qini coefficient); only a plain logistic-regression T-learner beats random. This
holds at both 1M and 3M rows, so it isn't a sample-size artifact — the LightGBM T-learner's
individual uplift estimates swing to the ±1.0 extremes for 30% of users, a textbook sign of
overfitting a 300-tree/31-leaf model to a ~0.3%-rate label. The lesson: with conversion this
rare, model capacity needs to be matched to the signal, and a heavily regularized/linear
learner can beat a more "powerful" one. The budget-allocation layer uses the winning model
(T-learner/logistic) and beats propensity-based targeting (today's industry default) at every
budget level, most clearly at small budgets — e.g. at a 2% budget, uplift targeting nets
$14,113 (95% CI [$6,281, $21,944]) vs. propensity targeting's $6,746 (95% CI [$2,217,
$11,275]) on the same held-out users.

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
