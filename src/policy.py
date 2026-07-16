"""Budget-allocation profit simulation.

Math notes
----------
Given a targeting policy that exposes the top-k fraction of users (ranked by
some score) to ads, the expected profit is:

  profit(k) = N_targeted(k) * (uplift(k) * value_per_conversion - cost_per_exposure)

where uplift(k) = P(Y=1|top-k, T=1) - P(Y=1|top-k, T=0) is the observed
treatment effect inside the top-k slice (same quantity as evaluate.uplift_at_k)
and N_targeted(k) = k * n is how many users would be exposed if this policy
were deployed. CI on profit comes from propagating the Welch SE of the
in-slice uplift estimate (see data_prep.ate for the same construction).

profit_curve() sweeps policy_profit() across budget levels for one scoring
policy (e.g. propensity or uplift), so callers can compare policies by
calling it once per set of scores. "Target everyone" needs no curve of its
own -- it's just any policy's profit at budget_frac=1.0, since ranking stops
mattering once 100% of users are targeted.
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


def segment_users(propensity, uplift) -> pd.Series:
    """Cross conversion-propensity x predicted-uplift into four business segments.

    - do-not-disturb : uplift < 0 -- ads may reduce conversion; actively avoid.
    - sure thing      : uplift >= 0, propensity >= median -- would convert anyway,
                          don't pay to reach them.
    - persuadable      : uplift >= 0, propensity < median, uplift >= median of that
                          group -- ads change their behavior; target these first.
    - lost cause        : everything else -- unlikely to convert either way.

    Propensity is ranked by percentile rather than split on the raw value:
    rare-outcome propensity models routinely emit a large tied plateau at 0
    (many leaves predict "never converts"), which would otherwise pin the raw
    median at 0 and push every row into "high propensity."
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
