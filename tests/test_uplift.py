import numpy as np

from src.evaluate import auuc, qini_curve
from src.policy import policy_profit


def _fake_experiment(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    persuadable = rng.random(n) < 0.3
    # persuadables convert only if treated; others convert at base rate
    y = np.where(persuadable & (t == 1), 1, (rng.random(n) < 0.05).astype(int))
    return y, t, persuadable.astype(float)


def test_qini_rewards_true_uplift_scores():
    y, t, true_uplift = _fake_experiment()
    good = auuc(y, t, true_uplift)
    random = auuc(y, t, np.random.default_rng(1).random(len(y)))
    assert good > random


def test_qini_curve_starts_at_zero():
    y, t, s = _fake_experiment()
    fracs, qini = qini_curve(y, t, s)
    assert fracs[0] == 0.0 and qini[0] == 0.0


def test_policy_profit_finite():
    y, t, s = _fake_experiment()
    assert np.isfinite(policy_profit(y, t, s, 0.2))
