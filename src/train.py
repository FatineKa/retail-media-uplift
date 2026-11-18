"""Do the full training: checks -> models -> uplift metrics -> saved files.

To run: python -m src.train
On SageMaker, sagemaker/train_entry.py calls main(). SageMaker gives the data and
output folders in environment variables, and the LightGBM settings as arguments.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_prep import FEATURES, SAMPLE_PATH, ate, load_sample, randomization_check
from src.evaluate import qini_coefficient, qini_curve, uplift_at_k
from src.policy import profit_curve, profit_difference, segment_users
from src.uplift import ClassTransformation, SLearner, TLearner

REPORTS_DIR = Path("reports")
DASHBOARD_ROWS = 1_000_000  # The dashboard recalculates profits live. More rows is too slow there.
# Found by SageMaker automatic model tuning, on validation Qini (pipeline run of 2026-11-18,
# step-functions/quality_gate_workflow.json). The reports in reports/ use these settings.
# Before tuning: n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20.
LGBM_PARAMS = {"n_estimators": 777, "learning_rate": 0.01015573888115397, "num_leaves": 15,
               "min_child_samples": 22}


def lgbm():
    # Do not use is_unbalance. It changes the weight of the rare converters.
    # Then the outputs are not correct probabilities, and uplift = p1 - p0 is wrong.
    return LGBMClassifier(**LGBM_PARAMS, verbose=-1)


def propensity_model(X_tr, t_tr, y_tr):
    """Fit the comparison model: the conversion probability, on the control group only.
    Most marketing teams use this model, not an uplift model."""
    ctrl = (t_tr == 0).to_numpy()
    model = lgbm()
    print(f"[propensity] fitting conversion model on {ctrl.sum():,} control rows")
    model.fit(X_tr[ctrl], y_tr[ctrl])
    return model


def split(df):
    X, t, y = df[FEATURES], df["treatment"], df["conversion"]
    X_tr, X_te, t_tr, t_te, y_tr, y_te = train_test_split(
        X, t, y, test_size=0.3, random_state=42,
        stratify=t.astype(str) + y.astype(str))
    print(f"[split] train {len(X_tr):,} / test {len(X_te):,} "
          "(stratified on treatment x conversion)\n")
    return X_tr, X_te, t_tr, t_te, y_tr, y_te


def fit_uplift_models(X_tr, t_tr, y_tr):
    models = {
        "T-learner / logistic": TLearner(LogisticRegression(max_iter=1000)),
        "T-learner / LightGBM": TLearner(lgbm()),
        "S-learner / LightGBM": SLearner(lgbm()),
        "Class transformation": ClassTransformation(lgbm()),
    }
    for name, model in models.items():
        print(f"--- {name} ---")
        t0 = time.time()
        model.fit(X_tr, t_tr, y_tr)
        print(f"    fit in {time.time() - t0:.1f}s\n")
    return models


def evaluate_uplift_models(models, X_te, t_te, y_te):
    results = {}
    test_scores = pd.DataFrame({"conversion": y_te.to_numpy(), "treatment": t_te.to_numpy()})
    qini_rows = []
    for name, model in models.items():
        print(f"--- {name} ---")
        tau = model.predict_uplift(X_te)
        q = qini_coefficient(y_te, t_te, tau)
        u10 = uplift_at_k(y_te, t_te, tau, 0.1)
        results[name] = (q, u10)
        test_scores[name] = tau
        fracs, qini_vals = qini_curve(y_te, t_te, tau)
        qini_rows.append(pd.DataFrame({"model": name, "frac": fracs, "qini": qini_vals}))
        print()

    print("=" * 58)
    print(f"{'model':28s} {'Qini coef':>12s} {'uplift@10%':>12s}")
    for name, (q, u) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:28s} {q:12.6f} {u:12.5f}")
    print("=" * 58 + "\n")
    return results, test_scores, qini_rows


def pick_best_on_validation(models, X_val, t_val, y_val):
    """Select the best model on a validation sample. Use the test set only for the report."""
    print("[validation] choosing the best model")
    q = {name: qini_coefficient(y_val, t_val, m.predict_uplift(X_val)) for name, m in models.items()}
    best = max(q, key=q.get)
    print(f"[best] {best} wins on validation Qini coefficient\n")
    return best


def build_profit_analysis(y_te, t_te, test_scores, propensity_scores, best_model):
    budgets = np.linspace(0.02, 1.0, 25)
    print("[profit] propensity targeting:")
    prop_curve = profit_curve(y_te, t_te, propensity_scores, budgets)
    prop_curve["policy"] = "propensity"
    print("[profit] uplift targeting (best model):")
    uplift_curve = profit_curve(y_te, t_te, test_scores[best_model].to_numpy(), budgets)
    uplift_curve["policy"] = "uplift"
    profit_table = pd.concat([prop_curve, uplift_curve], ignore_index=True)
    print("[profit] uplift minus propensity, same users, bootstrap 95% CI:")
    diff_table = profit_difference(y_te, t_te, test_scores[best_model].to_numpy(),
                                   propensity_scores, budgets)
    print()
    return profit_table, diff_table


def save_reports(test_scores, qini_rows, profit_table, diff_table, summary):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # All results above use the full test set. Only the dashboard file is a random sample.
    if len(test_scores) > DASHBOARD_ROWS:
        test_scores = test_scores.sample(DASHBOARD_ROWS, random_state=42)
        print(f"[artifacts] test_scores.parquet: random sample of {DASHBOARD_ROWS:,} test users")
    test_scores.to_parquet(REPORTS_DIR / "test_scores.parquet")
    pd.concat(qini_rows, ignore_index=True).to_parquet(REPORTS_DIR / "qini_curves.parquet")
    profit_table.to_parquet(REPORTS_DIR / "profit_curves.parquet")
    diff_table.to_parquet(REPORTS_DIR / "profit_difference.parquet")
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[artifacts] saved test_scores / qini_curves / profit_curves / "
          f"summary.json to {REPORTS_DIR}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("SM_CHANNEL_TRAIN", str(SAMPLE_PATH)))
    p.add_argument("--reports-dir", default=os.environ.get("SM_MODEL_DIR", str(REPORTS_DIR)))
    # --tune-only: fit only the S-learner and print its validation Qini (for SageMaker tuning).
    p.add_argument("--tune-only", type=int, default=0)
    p.add_argument("--n-estimators", type=int, default=LGBM_PARAMS["n_estimators"])
    p.add_argument("--learning-rate", type=float, default=LGBM_PARAMS["learning_rate"])
    p.add_argument("--num-leaves", type=int, default=LGBM_PARAMS["num_leaves"])
    p.add_argument("--min-child-samples", type=int, default=LGBM_PARAMS["min_child_samples"])
    return p.parse_known_args()[0]


def main():
    global REPORTS_DIR
    args = parse_args()
    REPORTS_DIR = Path(args.reports_dir)
    LGBM_PARAMS.update(n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                       num_leaves=args.num_leaves, min_child_samples=args.min_child_samples)
    print(f"[lightgbm] {LGBM_PARAMS}\n")

    df = load_sample(Path(args.data))
    randomization_check(df)
    print()
    ate(df, "conversion")
    ate(df, "visit")
    print()

    X_tr, X_te, t_tr, t_te, y_tr, y_te = split(df)

    # Validation sample (20% of the training data). Use it only to select the best model.
    X_fit, X_val, t_fit, t_val, y_fit, y_val = train_test_split(
        X_tr, t_tr, y_tr, test_size=0.2, random_state=42,
        stratify=t_tr.astype(str) + y_tr.astype(str))
    print(f"[split] fit {len(X_fit):,} / validation {len(X_val):,}\n")

    if args.tune_only:
        # The test set is not used here. The tuner compares settings on the validation sample.
        model = SLearner(lgbm()).fit(X_fit, t_fit, y_fit)
        print(f"valid_qini={qini_coefficient(y_val, t_val, model.predict_uplift(X_val)):.8f}")
        return

    models = fit_uplift_models(X_fit, t_fit, y_fit)
    best_model = pick_best_on_validation(models, X_val, t_val, y_val)
    results, test_scores, qini_rows = evaluate_uplift_models(models, X_te, t_te, y_te)

    # Comparison policy for the budget allocation: the propensity to convert.
    # Fit it on the control group only. Most teams use this policy today.
    prop_model = propensity_model(X_fit, t_fit, y_fit)
    propensity_scores = prop_model.predict_proba(X_te)[:, 1]
    test_scores["propensity"] = propensity_scores
    print()

    # Profit chart: propensity targeting and uplift targeting (best model) at each budget.
    # "Target everyone" is the value of each curve at frac=1.0.
    profit_table, diff_table = build_profit_analysis(
        y_te, t_te, test_scores, propensity_scores, best_model)

    segments = segment_users(propensity_scores, test_scores[best_model].to_numpy())
    test_scores["segment"] = segments.to_numpy()

    summary = {
        "best_model": best_model,
        "n_test": int(len(y_te)),
        "results": {k: {"qini_coefficient": v[0], "uplift_at_10pct": v[1]}
                    for k, v in results.items()},
        "segment_counts": segments.value_counts().to_dict(),
        "profit_difference": diff_table.to_dict(orient="records"),
    }
    save_reports(test_scores, qini_rows, profit_table, diff_table, summary)


if __name__ == "__main__":
    main()
