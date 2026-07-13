"""End-to-end training run: sanity checks -> models -> uplift metrics.

Usage: python -m src.train
"""
import time

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_prep import FEATURES, ate, load_sample, randomization_check
from src.evaluate import qini_coefficient, uplift_at_k
from src.uplift import ClassTransformation, SLearner, TLearner


def lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          is_unbalance=True, verbose=-1)


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
    for name, model in models.items():
        print(f"--- {name} ---")
        t0 = time.time()
        model.fit(X_tr, t_tr, y_tr)
        tau = model.predict_uplift(X_te)
        q = qini_coefficient(y_te, t_te, tau)
        u10 = uplift_at_k(y_te, t_te, tau, 0.1)
        results[name] = (q, u10)
        print(f"    done in {time.time() - t0:.1f}s\n")

    print("=" * 58)
    print(f"{'model':28s} {'Qini coef':>12s} {'uplift@10%':>12s}")
    for name, (q, u) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:28s} {q:12.6f} {u:12.5f}")
    print("=" * 58)


if __name__ == "__main__":
    main()