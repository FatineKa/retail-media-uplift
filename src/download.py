"""Download the Criteo uplift dataset, stratified-sample it, save parquet.

Stratifying on treatment x conversion preserves the rare converters
(conversion ~ 0.3%) that a plain random sample would decimate.

Usage: python -m src.download --rows 1000000
"""
import argparse
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/raw/criteo_uplift_sample.parquet")


def main(rows: int = 1_000_000):
    print("[download] fetching criteo/criteo-uplift from Hugging Face "
          "(~14M rows, first run takes a while)...")
    df = load_dataset("criteo/criteo-uplift", split="train").to_pandas()
    print(f"[download] full dataset: {len(df):,} rows, "
          f"conversion rate {df.conversion.mean():.5f}")
    frac = min(1.0, rows / len(df))
    sample = df.groupby(["treatment", "conversion"], group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=42))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT)
    print(f"[download] saved {len(sample):,} rows to {OUT}")
    print(f"[download] sample conversion rate {sample.conversion.mean():.5f} "
          "(should match the full dataset above)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1_000_000)
    main(p.parse_args().rows)