"""Download the Criteo uplift dataset, sample it, save parquet.

Streams the dataset row-by-row instead of materializing all ~13.9M rows in
pandas first -- keeps peak memory at O(sample size) instead of O(full
dataset), for low-RAM machines. Each row is kept independently with
probability `rows / TOTAL_ROWS` (a fixed, documented estimate of the
dataset's size, since HF doesn't expose row counts without a full read for
this dataset). Applying the same keep-probability regardless of treatment/
conversion is stratification-neutral: every stratum is subsampled at the same
rate in expectation, so the rare converters (conversion ~ 0.3%) survive at
their true proportion rather than being at the mercy of one global draw.

Usage: python -m src.download --rows 1000000
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path("data/raw/criteo_uplift_sample.parquet")
TOTAL_ROWS = 13_979_592  # published size of criteo/criteo-uplift train split
PROGRESS_EVERY = 1_000_000


def main(rows: int = 1_000_000, seed: int = 42):
    keep_p = min(1.0, rows / TOTAL_ROWS)
    print(f"[download] streaming criteo/criteo-uplift from Hugging Face "
          f"(~{TOTAL_ROWS:,} rows, keeping ~{keep_p:.1%} -> ~{rows:,} rows)...")

    rng = random.Random(seed)
    stream = load_dataset("criteo/criteo-uplift", split="train", streaming=True)

    sampled_rows = []
    n_seen = n_treated = n_conversions = 0
    for row in stream:
        n_seen += 1
        n_treated += row["treatment"]
        n_conversions += row["conversion"]
        if rng.random() < keep_p:
            sampled_rows.append(row)
        if n_seen % PROGRESS_EVERY == 0:
            print(f"[download] ...{n_seen:,} rows streamed, "
                  f"{len(sampled_rows):,} kept so far")

    print(f"[download] full stream: {n_seen:,} rows, "
          f"treatment share {n_treated / n_seen:.3f}, "
          f"conversion rate {n_conversions / n_seen:.5f}")

    sample = pd.DataFrame(sampled_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT)
    print(f"[download] saved {len(sample):,} rows to {OUT}")
    print(f"[download] sample conversion rate {sample.conversion.mean():.5f} "
          "(should match the full stream above)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1_000_000)
    main(p.parse_args().rows)