"""Uplift learners.

----------
Individual uplift tau(x) = P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0).
- T-learner : fit mu1 on treated, mu0 on control; tau_hat = mu1(x) - mu0(x).
- S-learner : one model f(x, t); tau_hat = f(x,1) - f(x,0). Can shrink tau to 0
  when the treatment feature carries little split gain -- report this if seen.
- Class transformation (Jaskowski & Jaroszewicz): with balanced treatment,
  Z = Y*T + (1-Y)*(1-T) satisfies tau(x) = 2*P(Z=1|x) - 1. Criteo is ~85%
  treated, so we rebalance with inverse-propensity weights w = T/e + (1-T)/(1-e),
  where e = P(T=1) is known exactly (randomized experiment).
"""
import numpy as np
from sklearn.base import clone


class TLearner:
    def __init__(self, base_model):
        self.m1, self.m0 = clone(base_model), clone(base_model)

    def fit(self, X, treatment, y):
        t = np.asarray(treatment)
        print(f"[T-learner] fitting: {int(t.sum()):,} treated "
              f"(rate {y[t == 1].mean():.5f}) | {int((1 - t).sum()):,} control "
              f"(rate {y[t == 0].mean():.5f})")
        self.m1.fit(X[t == 1], y[t == 1])
        self.m0.fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        tau = self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]
        print(f"[T-learner] uplift: mean {tau.mean():+.5f}, "
              f"range [{tau.min():+.4f}, {tau.max():+.4f}], "
              f"{(tau < 0).mean():.1%} of users negative")
        return tau


class SLearner:
    def __init__(self, base_model):
        self.model = clone(base_model)

    def fit(self, X, treatment, y):
        Xt = X.copy()
        Xt["treatment"] = np.asarray(treatment)
        print(f"[S-learner] fitting one model on {len(Xt):,} rows "
              f"(treatment as feature)")
        self.model.fit(Xt, y)
        return self

    def predict_uplift(self, X):
        X1, X0 = X.copy(), X.copy()
        X1["treatment"], X0["treatment"] = 1, 0
        tau = self.model.predict_proba(X1)[:, 1] - self.model.predict_proba(X0)[:, 1]
        print(f"[S-learner] uplift: mean {tau.mean():+.5f} "
              f"(if ~0 everywhere, the model ignored the treatment feature)")
        return tau


class ClassTransformation:
    """Z = Y*T + (1-Y)*(1-T), IPW-rebalanced; tau_hat = 2*P(Z=1|x) - 1."""

    def __init__(self, base_model):
        self.model = clone(base_model)

    def fit(self, X, treatment, y):
        t, y = np.asarray(treatment), np.asarray(y)
        e = t.mean()  # known propensity in an RCT
        z = (y * t + (1 - y) * (1 - t)).astype(int)
        w = t / e + (1 - t) / (1 - e)
        print(f"[class-transform] propensity e = {e:.3f}, "
              f"P(Z=1) = {z.mean():.3f}, fitting with IPW weights")
        self.model.fit(X, z, sample_weight=w)
        return self

    def predict_uplift(self, X):
        tau = 2 * self.model.predict_proba(X)[:, 1] - 1
        print(f"[class-transform] uplift: mean {tau.mean():+.5f}")
        return tau