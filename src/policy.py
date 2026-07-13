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
