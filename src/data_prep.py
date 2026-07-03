from pathlib import Path

import pandas as pd

SAMPLE_PATH = Path("data/raw/criteo_uplift_sample.parquet")
FEATURES = [f"f{i}" for i in range(12)]


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def ate(df: pd.DataFrame, outcome: str = "conversion") -> dict:
    """Raw average treatment effect with a normal-approx 95% CI."""
    import numpy as np

    t, c = df[df.treatment == 1][outcome], df[df.treatment == 0][outcome]
    effect = t.mean() - c.mean()
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))
    return {"ate": effect, "ci_low": effect - 1.96 * se, "ci_high": effect + 1.96 * se,
            "treated_rate": t.mean(), "control_rate": c.mean()}
