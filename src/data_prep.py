"""Load data, verify randomization, estimate the average treatment effect.

Math notes
----------
- ATE  = E[Y|T=1] - E[Y|T=0]. Valid causally only because treatment was randomized.
- SE   = sqrt(s1^2/n1 + s0^2/n0)  (Welch, unequal variances); 95% CI = ATE +/- 1.96*SE.
- SMD  = (mean_t - mean_c) / pooled_std : standardized mean difference per feature.
  |SMD| < 0.05 on every feature => randomization looks healthy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SAMPLE_PATH = Path("data/raw/criteo_uplift_sample.parquet")
FEATURES = [f"f{i}" for i in range(12)]


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[data] loaded {len(df):,} rows x {df.shape[1]} cols from {path}")
    print(f"[data] treatment share : {df.treatment.mean():.3f}")
    print(f"[data] visit rate      : {df.visit.mean():.4f}")
    print(f"[data] conversion rate : {df.conversion.mean():.5f}  "
          f"({df.conversion.sum():,} converters)")
    return df


def randomization_check(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Standardized mean differences of every feature across arms."""
    t, c = df[df.treatment == 1], df[df.treatment == 0]
    rows = []
    for f in FEATURES:
        pooled = np.sqrt((t[f].var() + c[f].var()) / 2)
        smd = (t[f].mean() - c[f].mean()) / pooled if pooled > 0 else 0.0
        rows.append({"feature": f, "smd": smd, "ok": abs(smd) < threshold})
    out = pd.DataFrame(rows)
    worst = out.loc[out.smd.abs().idxmax()]
    print(f"[randomization] max |SMD| = {abs(worst.smd):.4f} on {worst.feature} "
          f"(threshold {threshold}) -> {'PASS' if out.ok.all() else 'FAIL'}")
    return out


def ate(df: pd.DataFrame, outcome: str = "conversion") -> dict:
    """Difference in means with Welch SE, 95% CI, z-test p-value, relative lift."""
    t = df.loc[df.treatment == 1, outcome].to_numpy()
    c = df.loc[df.treatment == 0, outcome].to_numpy()
    effect = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    z = effect / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rel = effect / c.mean() if c.mean() > 0 else np.nan
    print(f"[ATE:{outcome}] treated {t.mean():.5f} vs control {c.mean():.5f}")
    print(f"[ATE:{outcome}] effect = {effect:+.5f}  "
          f"95% CI [{effect - 1.96*se:+.5f}, {effect + 1.96*se:+.5f}]  "
          f"p = {p:.2e}  relative lift = {rel:+.1%}")
    return {"ate": effect, "se": se, "ci_low": effect - 1.96 * se,
            "ci_high": effect + 1.96 * se, "p_value": p, "relative_lift": rel}


def profile(df: pd.DataFrame):
    """Quick data profile — run: python -m src.data_prep"""
    print("\n[profile] first rows:")
    print(df.head(5).to_string())
    print("\n[profile] dtypes and non-null counts:")
    df.info()
    print("\n[profile] feature summary:")
    print(df[FEATURES].describe().T.round(3).to_string())
    print("\n[profile] outcome counts:")
    for col in ("treatment", "visit", "conversion", "exposure"):
        print(f"  {col:12s} {df[col].value_counts().to_dict()}")


if __name__ == "__main__":
    df = load_sample()
    profile(df)
    randomization_check(df)
    ate(df, "conversion")
    ate(df, "visit")