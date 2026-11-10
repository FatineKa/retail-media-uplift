"""Load the data, check the randomization and calculate the average treatment effect.

Math notes
----------
- ATE  = E[Y|T=1] - E[Y|T=0]. This value is causal only because the treatment is random.
- SE   = sqrt(s1^2/n1 + s0^2/n0) (Welch, unequal variances). 95% CI = ATE +/- 1.96*SE.
- SMD  = (mean_t - mean_c) / pooled_std. This is the standardized mean difference of one feature.
  If |SMD| < 0.05 for all features, the randomization is correct.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SAMPLE_PATH = Path("data/raw/criteo_uplift_sample.parquet")
FEATURES = [f"f{i}" for i in range(12)]


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Change the columns to smaller types (float32, int8). This uses half of the memory.
    # This is important on a computer with little memory.
    df[FEATURES] = df[FEATURES].astype("float32")
    for col in ("treatment", "visit", "conversion", "exposure"):
        if col in df.columns:
            df[col] = df[col].astype("int8")
    print(f"[data] loaded {len(df):,} rows x {df.shape[1]} cols from {path}")
    print(f"[data] treatment share : {df.treatment.mean():.3f}")
    print(f"[data] visit rate      : {df.visit.mean():.4f}")
    print(f"[data] conversion rate : {df.conversion.mean():.5f}  "
          f"({df.conversion.sum():,} converters)")
    return df


def randomization_check(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Calculate the standardized mean difference of each feature between the two groups."""
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
    """Calculate the difference in means, the Welch SE, the 95% CI, the p-value and the relative lift."""
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
    """Show a short profile of the data. To run: python -m src.data_prep"""
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