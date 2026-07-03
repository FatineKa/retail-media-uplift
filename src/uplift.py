"""Uplift learners: T-learner and S-learner over any sklearn-style base model."""
import numpy as np
import pandas as pd
from sklearn.base import clone


class TLearner:
    """Fit separate models on treated and control; uplift = p1 - p0."""

    def __init__(self, base_model):
        self.m1, self.m0 = clone(base_model), clone(base_model)

    def fit(self, X, treatment, y):
        self.m1.fit(X[treatment == 1], y[treatment == 1])
        self.m0.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class SLearner:
    """One model with treatment as a feature; uplift = p(t=1) - p(t=0)."""

    def __init__(self, base_model):
        self.model = clone(base_model)

    def fit(self, X, treatment, y):
        Xt = X.copy()
        Xt["treatment"] = treatment.values
        self.model.fit(Xt, y)
        return self

    def predict_uplift(self, X):
        X1, X0 = X.copy(), X.copy()
        X1["treatment"], X0["treatment"] = 1, 0
        return self.model.predict_proba(X1)[:, 1] - self.model.predict_proba(X0)[:, 1]
