"""Download the Criteo uplift dataset, take a sample and save it as parquet.

- The script reads the dataset one row at a time.
  It does not load all ~13.9M rows into pandas.
- After each BATCH_SIZE kept rows, it writes them to disk as one Parquet row group.
  Thus the memory use stays small, also for a large sample.
- The script keeps each row with the probability rows / TOTAL_ROWS.
  TOTAL_ROWS is a fixed value. HF does not give the row count of this dataset
  without a full read.
- All rows have the same probability: treated, control and converters.
  Thus the rare converters (approximately 0.3%) keep their true proportion.

To run: python -m src.download --rows 1000000
"""
import argparse
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

OUT = Path("data/raw/criteo_uplift_sample.parquet")
TOTAL_ROWS = 13_979_592  # Number of rows in the criteo/criteo-uplift train split
PROGRESS_EVERY = 1_000_000
BATCH_SIZE = 50_000  # Write to disk after this number of kept rows. This limits the memory use.


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