"""End-to-end training run: sanity checks -> models -> uplift metrics -> saved artifacts.

Usage: python -m src.train
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_prep import FEATURES, ate, load_sample, randomization_check
from src.evaluate import qini_coefficient, qini_curve, uplift_at_k
from src.policy import profit_curve, segment_users
from src.uplift import ClassTransformation, SLearner, TLearner

REPORTS_DIR = Path("reports")


def lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          is_unbalance=True, verbose=-1)


def propensity_model(X_tr, t_tr, y_tr):
    """Naive competitor: conversion probability fit on control-group data only
    (what most marketing teams target with, in place of an uplift model)."""
    ctrl = (t_tr == 0).to_numpy()
    model = lgbm()
    print(f"[propensity] fitting conversion model on {ctrl.sum():,} control rows")
    model.fit(X_tr[ctrl], y_tr[ctrl])
    return model


def main():
    df = load_sample()
    randomization_check(df)
    print()
    ate(df, "conversion")
    ate(df, "visit")
    print()

    X, t, y = df[FEATURES], df["treatment"], df["conversion"]
    X_tr, X_te, t_tr, t_te, y_tr, y_te = train_test_split(
        X, t, y, test_size=0.3, random_state=42,
        stratify=t.astype(str) + y.astype(str))
    print(f"[split] train {len(X_tr):,} / test {len(X_te):,} "
          "(stratified on treatment x conversion)\n")

    models = {
        "T-learner / logistic": TLearner(LogisticRegression(max_iter=1000)),
        "T-learner / LightGBM": TLearner(lgbm()),
        "S-learner / LightGBM": SLearner(lgbm()),
        "Class transformation": ClassTransformation(lgbm()),
    }
    results = {}
    test_scores = pd.DataFrame({"conversion": y_te.to_numpy(), "treatment": t_te.to_numpy()})
    qini_rows = []
    for name, model in models.items():
        print(f"--- {name} ---")
        t0 = time.time()
        model.fit(X_tr, t_tr, y_tr)
        tau = model.predict_uplift(X_te)
        q = qini_coefficient(y_te, t_te, tau)
        u10 = uplift_at_k(y_te, t_te, tau, 0.1)
        results[name] = (q, u10)
        test_scores[name] = tau
        fracs, qini_vals = qini_curve(y_te, t_te, tau)
        qini_rows.append(pd.DataFrame({"model": name, "frac": fracs, "qini": qini_vals}))
        print(f"    done in {time.time() - t0:.1f}s\n")

    print("=" * 58)
    print(f"{'model':28s} {'Qini coef':>12s} {'uplift@10%':>12s}")
    for name, (q, u) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:28s} {q:12.6f} {u:12.5f}")
    print("=" * 58)

    best_model = max(results, key=lambda k: results[k][0])
    print(f"\n[best] {best_model} wins on Qini coefficient\n")

    # Naive competitor for the budget-allocation layer: propensity-to-convert,
    # trained on control-group data only (what most teams target with today).
    prop_model = propensity_model(X_tr, t_tr, y_tr)
    propensity_scores = prop_model.predict_proba(X_te)[:, 1]
    test_scores["propensity"] = propensity_scores
    print()

    # Money chart: propensity targeting vs. uplift targeting (best model) across
    # budget levels. "Target everyone" is just either curve's value at frac=1.0.
    budgets = np.linspace(0.02, 1.0, 25)
    print("[profit] propensity targeting:")
    prop_curve = profit_curve(y_te, t_te, propensity_scores, budgets)
    prop_curve["policy"] = "propensity"
    print("[profit] uplift targeting (best model):")
    uplift_curve = profit_curve(y_te, t_te, test_scores[best_model].to_numpy(), budgets)
    uplift_curve["policy"] = "uplift"
    profit_table = pd.concat([prop_curve, uplift_curve], ignore_index=True)
    print()

    segments = segment_users(propensity_scores, test_scores[best_model].to_numpy())
    test_scores["segment"] = segments.to_numpy()

    REPORTS_DIR.mkdir(exist_ok=True)
    test_scores.to_parquet(REPORTS_DIR / "test_scores.parquet")
    pd.concat(qini_rows, ignore_index=True).to_parquet(REPORTS_DIR / "qini_curves.parquet")
    profit_table.to_parquet(REPORTS_DIR / "profit_curves.parquet")
    summary = {
        "best_model": best_model,
        "n_test": int(len(y_te)),
        "results": {k: {"qini_coefficient": v[0], "uplift_at_10pct": v[1]}
                    for k, v in results.items()},
        "segment_counts": segments.value_counts().to_dict(),
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[artifacts] saved test_scores / qini_curves / profit_curves / "
          f"summary.json to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()