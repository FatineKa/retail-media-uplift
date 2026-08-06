"""Tests on synthetic experiments where the true treatment effect and
feature balance are KNOWN, mirroring the approach in test_uplift.py."""
import numpy as np
import pandas as pd

from src.data_prep import ate, randomization_check


def _fake_experiment(n=20000, true_effect=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    base_rate = 0.10
    y = (rng.random(n) < base_rate + true_effect * t).astype(int)
    return pd.DataFrame({"treatment": t, "conversion": y})


def test_ate_recovers_known_effect():
    df = _fake_experiment(true_effect=0.05, seed=0)
    out = ate(df, "conversion")
    assert out["ci_low"] < 0.05 < out["ci_high"]
    assert out["p_value"] < 0.01


def test_ate_no_effect_is_not_significant():
    df = _fake_experiment(true_effect=0.0, seed=1)
    out = ate(df, "conversion")
    assert out["ci_low"] < 0 < out["ci_high"]
    assert out["p_value"] > 0.05


def test_randomization_check_passes_on_balanced_features():
    # n large enough that per-feature SMD standard error (~sqrt(2/n)) sits well
    # under the 0.05 threshold, so this doesn't flake on random noise across
    # 12 features.
    rng = np.random.default_rng(2)
    n = 50000
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    df["treatment"] = rng.integers(0, 2, n)
    out = randomization_check(df)
    assert out["ok"].all()


def test_randomization_check_flags_imbalanced_feature():
    rng = np.random.default_rng(3)
    n = 50000
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    df["treatment"] = rng.integers(0, 2, n)
    # Inject a large, deliberate imbalance on one feature.
    df.loc[df.treatment == 1, "f0"] += 1.0
    out = randomization_check(df)
    row = out.loc[out.feature == "f0"].iloc[0]
    assert not row["ok"]
    assert abs(row["smd"]) > 0.05
