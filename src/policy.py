"""Profit simulation for budget allocation.

Math notes
----------
A targeting policy shows ads to the top k of the users. A score sets the order
of the users. The expected profit is:

  profit(k) = N_targeted(k) * (uplift(k) * value_per_conversion - cost_per_exposure)

- uplift(k) = P(Y=1|top-k, T=1) - P(Y=1|top-k, T=0). This is the observed
  treatment effect in the top k (the same value as evaluate.uplift_at_k).
- N_targeted(k) = k * n. This is the number of users that get the ad with this policy.
- The CI of the profit uses the Welch SE of the uplift in the top k
  (the same method as data_prep.ate).

profit_curve() calculates policy_profit() at many budget levels for one score
(for example, propensity or uplift). To compare two policies, call it one time
for each score. The policy "target everyone" does not need a curve. It is the
profit of any policy at budget_frac=1.0, because the order has no effect when
all users get the ad.
"""
import numpy as np
import pandas as pd


def policy_profit(y, treatment, scores, budget_frac: float,
                   cost_per_exposure: float = 0.10,
                   value_per_conversion: float = 40.0) -> dict:
    y, t = np.asarray(y), np.asarray(treatment)
    n = len(y)
    idx = np.argsort(-np.asarray(scores))[: int(n * budget_frac)]
    yt, tt = y[idx], t[idx]

    t_conv, c_conv = yt[tt == 1], yt[tt == 0]
    uplift = t_conv.mean() - c_conv.mean()
    se = np.sqrt(t_conv.var(ddof=1) / len(t_conv) + c_conv.var(ddof=1) / len(c_conv))

    n_targeted = int(n * budget_frac)
    per_user = lambda u: u * value_per_conversion - cost_per_exposure
    profit = n_targeted * per_user(uplift)
    ci_low = n_targeted * per_user(uplift - 1.96 * se)
    ci_high = n_targeted * per_user(uplift + 1.96 * se)

    print(f"[policy] budget {budget_frac:.0%} -> {n_targeted:,} targeted, "
          f"uplift {uplift:+.5f}, profit ${profit:,.0f} "
          f"95% CI [${ci_low:,.0f}, ${ci_high:,.0f}]")
    return {"budget_frac": budget_frac, "n_targeted": n_targeted,
            "uplift": uplift, "profit": profit,
            "ci_low": ci_low, "ci_high": ci_high}


def profit_curve(y, treatment, scores, budgets=None,
                  cost_per_exposure: float = 0.10,
                  value_per_conversion: float = 40.0) -> pd.DataFrame:
    if budgets is None:
        budgets = np.linspace(0.02, 1.0, 25)
    rows = [policy_profit(y, treatment, scores, b,
                           cost_per_exposure, value_per_conversion)
            for b in budgets]
    return pd.DataFrame(rows)


def _profits(y, t, order, w, budgets, cost_per_exposure, value_per_conversion):
    """Calculate the profit at each budget for one order of users. Each user has the weight w."""
    y, t, w = y[order], t[order], w[order]
    n = len(y)
    k = (np.asarray(budgets) * n).astype(int)
    conv_t, n_t = np.cumsum(y * t * w)[k - 1], np.cumsum(t * w)[k - 1]
    conv_c, n_c = np.cumsum(y * (1 - t) * w)[k - 1], np.cumsum((1 - t) * w)[k - 1]
    uplift = conv_t / n_t - conv_c / n_c
    return k * (uplift * value_per_conversion - cost_per_exposure)


def profit_difference(y, treatment, scores_a, scores_b, budgets=None, n_boot: int = 200,
                      seed: int = 0, cost_per_exposure: float = 0.10,
                      value_per_conversion: float = 40.0) -> pd.DataFrame:
    """Calculate profit(policy a) - profit(policy b) on the same users, with a 95% CI.

    - The two profits use the same users. Thus you cannot compare their separate CIs.
    - Bootstrap: take a new sample of the users (Poisson(1) weights).
      Then calculate the two profits again.
    - If the CI is above 0, policy a gives more profit at that budget."""
    if budgets is None:
        budgets = np.linspace(0.02, 1.0, 25)
    y, t = np.asarray(y, dtype=float), np.asarray(treatment, dtype=float)
    order_a, order_b = np.argsort(-np.asarray(scores_a)), np.argsort(-np.asarray(scores_b))
    one = np.ones(len(y))
    diff = (_profits(y, t, order_a, one, budgets, cost_per_exposure, value_per_conversion)
            - _profits(y, t, order_b, one, budgets, cost_per_exposure, value_per_conversion))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        w = rng.poisson(1.0, len(y)).astype(float)
        boots.append(_profits(y, t, order_a, w, budgets, cost_per_exposure, value_per_conversion)
                     - _profits(y, t, order_b, w, budgets, cost_per_exposure, value_per_conversion))
    low, high = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    out = pd.DataFrame({"budget_frac": budgets, "difference": diff, "ci_low": low, "ci_high": high})
    for r in out.itertuples():
        print(f"[difference] budget {r.budget_frac:.0%}: ${r.difference:,.0f} "
              f"95% CI [${r.ci_low:,.0f}, ${r.ci_high:,.0f}]"
              f"{'  <- significant' if r.ci_low > 0 or r.ci_high < 0 else ''}")
    return out


def segment_users(propensity, uplift) -> pd.Series:
    """Put each user in one of four business segments. Use the propensity and the uplift.

    - do-not-disturb : uplift < 0. The ads can decrease conversion. Do not target these users.
    - sure thing     : uplift >= 0 and propensity >= median. These users convert without ads.
                       Do not pay to show ads to them.
    - persuadable    : uplift >= 0, propensity < median, and uplift >= the median of that group.
                       The ads change what these users do. Target these users first.
    - lost cause     : all other users. These users probably do not convert, with or without ads.

    Use the percentile rank of the propensity, not its value.
    For a rare outcome, many users get a propensity of exactly 0. Then the median
    of the values is 0, and all users go into "high propensity".
    """
    propensity, uplift = np.asarray(propensity), np.asarray(uplift)
    high_propensity = pd.Series(propensity).rank(pct=True).to_numpy() >= 0.5
    positive = uplift >= 0
    uplift_med_pos = np.median(uplift[positive]) if positive.any() else 0.0

    segment = np.full(len(uplift), "lost cause", dtype=object)
    segment[positive & high_propensity] = "sure thing"
    segment[positive & ~high_propensity & (uplift >= uplift_med_pos)] = "persuadable"
    segment[~positive] = "do-not-disturb"

    counts = pd.Series(segment).value_counts()
    print("[segments] " + ", ".join(
        f"{k}: {v:,} ({v / len(segment):.1%})" for k, v in counts.items()))
    return pd.Series(segment, name="segment")
