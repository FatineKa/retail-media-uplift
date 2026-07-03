"""Train uplift models and report Qini/AUUC. Usage: python -m src.train"""
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_prep import FEATURES, ate, load_sample
from src.evaluate import auuc
from src.uplift import SLearner, TLearner


def main():
    df = load_sample()
    print("Raw ATE (conversion):", ate(df))

    X, t, y = df[FEATURES], df["treatment"], df["conversion"]
    X_tr, X_te, t_tr, t_te, y_tr, y_te = train_test_split(
        X, t, y, test_size=0.3, stratify=t.astype(str) + y.astype(str), random_state=42
    )

    models = {
        "T-learner / logreg": TLearner(LogisticRegression(max_iter=1000)),
        "T-learner / lgbm": TLearner(LGBMClassifier(n_estimators=200, verbose=-1)),
        "S-learner / lgbm": SLearner(LGBMClassifier(n_estimators=200, verbose=-1)),
    }
    for name, m in models.items():
        m.fit(X_tr, t_tr, y_tr)
        score = auuc(y_te, t_te, m.predict_uplift(X_te))
        print(f"{name:22s} AUUC = {score:.4f}")


if __name__ == "__main__":
    main()
