"""Runs inside the SageMaker processing job: download the full Criteo uplift dataset
(~13.9M users) and save it as one Parquet file. SageMaker then copies it to S3.

The randomization check runs after this step, in Snowflake (see ../../snowflake/).
"""
import argparse
from pathlib import Path

from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--out", default="/opt/ml/processing/raw")
out = Path(p.parse_args().out)
out.mkdir(parents=True, exist_ok=True)

# Without streaming, Hugging Face downloads the dataset files once (Arrow format, on disk).
# This is much faster than reading 13.9M rows one by one.
ds = load_dataset("criteo/criteo-uplift", split="train")
print(f"[prepare] {ds.num_rows:,} rows, columns: {ds.column_names}")
ds.to_parquet(str(out / "criteo_uplift_full.parquet"))
print(f"[prepare] saved to {out / 'criteo_uplift_full.parquet'}")
