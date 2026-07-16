"""Tests on synthetic experiments where the true uplift is KNOWN."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluate import auuc, qini_curve, uplift_at_k
from src.policy import policy_profit, profit_curve, segment_users
from src.uplift import TLearner


def _fake_experiment(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    x = rng.normal(size=(n, 3))
    persuadable = x[:, 0] > 0.5          # true persuadables are identifiable
    base = rng.random(n) < 0.05
    y = np.where(persuadable & (t == 1), 1, base.astype(int))
    return x, y, t, persuadable.astype(float)


def test_qini_rewards_true_uplift_scores():
    _, y, t, true_uplift = _fake_experiment()
    assert auuc(y, t, true_uplift) > auuc(
        y, t, np.random.default_rng(1).random(len(y)))


def test_tlearner_recovers_persuadables():
    x, y, t, persuadable = _fake_experiment()
    import pandas as pd
    X = pd.DataFrame(x, columns=list("abc"))
    tau = TLearner(LogisticRegression()).fit(X, t, y).predict_uplift(X)
    assert tau[persuadable == 1].mean() > tau[persuadable == 0].mean() + 0.05


def test_uplift_at_k_beats_average_for_good_scores():
    _, y, t, true_uplift = _fake_experiment()
    top = uplift_at_k(y, t, true_uplift, 0.1)
    overall = y[t == 1].mean() - y[t == 0].mean()
    assert top > overall


def test_policy_profit_has_ci():
    _, y, t, s = _fake_experiment()
    out = policy_profit(y, t, s, 0.2)
    assert out["ci_low"] <= out["profit"] <= out["ci_high"]


def test_profit_curve_targets_more_users_at_higher_budget():
    _, y, t, s = _fake_experiment()
    curve = profit_curve(y, t, s, budgets=[0.1, 0.5, 1.0])
    assert list(curve["n_targeted"]) == sorted(curve["n_targeted"])
    assert curve["n_targeted"].iloc[-1] == len(y)


def test_profit_curve_true_uplift_beats_random_at_same_budget():
    _, y, t, true_uplift = _fake_experiment()
    random_scores = np.random.default_rng(1).random(len(y))
    good = profit_curve(y, t, true_uplift, budgets=[0.3])
    bad = profit_curve(y, t, random_scores, budgets=[0.3])
    assert good["profit"].iloc[0] > bad["profit"].iloc[0]


def test_segment_users_covers_all_users_with_no_overlap():
    rng = np.random.default_rng(2)
    propensity = rng.random(2000)
    uplift = rng.normal(size=2000)
    segment = segment_users(propensity, uplift)
    assert set(segment) <= {"do-not-disturb", "sure thing", "persuadable", "lost cause"}
    assert len(segment) == 2000
    assert segment.notna().all()


def test_segment_users_flags_negative_uplift_as_do_not_disturb():
    propensity = np.array([0.8, 0.2, 0.8, 0.2])
    uplift = np.array([-0.1, -0.05, 0.1, 0.1])
    segment = segment_users(propensity, uplift)
    assert (segment.iloc[:2] == "do-not-disturb").all()
    assert (segment.iloc[2:] != "do-not-disturb").all()