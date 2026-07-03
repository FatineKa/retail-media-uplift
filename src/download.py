"""Download the Criteo uplift dataset and save a stratified sample.

Usage: python -m src.download [--rows 1000000]
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path("data/raw/criteo_uplift_sample.parquet")


def main(rows: int = 1_000_000):
    ds = load_dataset("criteo/criteo-uplift", split="train")
    df = ds.to_pandas()
    # stratify on treatment x conversion so rare converters survive sampling
    frac = min(1.0, rows / len(df))
    sample = df.groupby(["treatment", "conversion"], group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=42)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT)
    print(f"Saved {len(sample):,} rows to {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1_000_000)
    main(p.parse_args().rows)
