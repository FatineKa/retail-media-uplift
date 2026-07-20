"""Download the Criteo uplift dataset, sample it, save parquet.

Streams the dataset row-by-row instead of materializing all ~13.9M rows in
pandas first, and flushes kept rows to disk every BATCH_SIZE rows as a
Parquet row group instead of buffering the whole sample -- peak memory is
bounded to O(batch size), not O(sample size), so requesting a bigger sample
doesn't cost more RAM on low-RAM machines. Each row is kept independently
with probability `rows / TOTAL_ROWS` (a fixed, documented estimate of the
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
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

OUT = Path("data/raw/criteo_uplift_sample.parquet")
TOTAL_ROWS = 13_979_592  # published size of criteo/criteo-uplift train split
PROGRESS_EVERY = 1_000_000
BATCH_SIZE = 50_000  # flush every this many kept rows -- caps peak memory


def main(rows: int = 1_000_000, seed: int = 42):
    keep_p = min(1.0, rows / TOTAL_ROWS)
    print(f"[download] streaming criteo/criteo-uplift from Hugging Face "
          f"(~{TOTAL_ROWS:,} rows, keeping ~{keep_p:.1%} -> ~{rows:,} rows)...")

    rng = random.Random(seed)
    stream = load_dataset("criteo/criteo-uplift", split="train", streaming=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    buffer = []
    n_seen = n_kept = n_treated = n_conversions = 0

    def flush():
        nonlocal writer, buffer
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT, table.schema)
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        buffer = []

    try:
        for row in stream:
            n_seen += 1
            n_treated += row["treatment"]
            n_conversions += row["conversion"]
            if rng.random() < keep_p:
                buffer.append(row)
                n_kept += 1
                if len(buffer) >= BATCH_SIZE:
                    flush()
            if n_seen % PROGRESS_EVERY == 0:
                print(f"[download] ...{n_seen:,} rows streamed, "
                      f"{n_kept:,} kept so far")
        flush()
    finally:
        if writer is not None:
            writer.close()

    print(f"[download] full stream: {n_seen:,} rows, "
          f"treatment share {n_treated / n_seen:.3f}, "
          f"conversion rate {n_conversions / n_seen:.5f}")
    print(f"[download] saved {n_kept:,} rows to {OUT} "
          f"(flushed in batches of {BATCH_SIZE:,})")

    return {
        "n_seen": n_seen,
        "n_kept": n_kept,
        "treatment_share": n_treated / n_seen,
        "conversion_rate": n_conversions / n_seen,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1_000_000)
    main(p.parse_args().rows)