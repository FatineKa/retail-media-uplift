"""Budget allocation: compare targeting policies in dollars.

Assumptions to tune:
- COST_PER_USER: media cost to expose one targeted user
- VALUE_PER_CONVERSION: margin from one incremental conversion
"""
import numpy as np

COST_PER_USER = 0.10
VALUE_PER_CONVERSION = 40.0


def policy_profit(y, treatment, scores, target_frac: float) -> float:
    """Estimated profit from targeting the top `target_frac` by `scores`,
    using the randomized holdout to estimate incremental conversions."""
    n = len(y)
    k = int(n * target_frac)
    idx = np.argsort(-scores)[:k]
    yt, tt = np.asarray(y)[idx], np.asarray(treatment)[idx]
    nt, nc = tt.sum(), (1 - tt).sum()
    if nt == 0 or nc == 0:
        return 0.0
    incr_rate = yt[tt == 1].mean() - yt[tt == 0].mean()
    incremental_conversions = incr_rate * k
    return incremental_conversions * VALUE_PER_CONVERSION - k * COST_PER_USER


def profit_curve(y, treatment, scores, fracs=None):
    fracs = fracs if fracs is not None else np.linspace(0.02, 1.0, 50)
    return fracs, np.array([policy_profit(y, treatment, scores, f) for f in fracs])
