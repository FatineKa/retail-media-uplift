"""Start the final SageMaker training job: 4 uplift models + profit analysis, on all 13.9M users.
The LightGBM settings are the defaults in src/train.py (found by scripts/launch_tuning.py).
The reports are saved in s3://sagemaker-retail-media-uplift-3769/models/<job>/output/model.tar.gz
To run, from the project folder, after aws login:  python scripts/launch_training.py
"""
import argparse

from sagemaker.sklearn.estimator import SKLearn

BUCKET = "sagemaker-retail-media-uplift-3769"
ROLE = "arn:aws:iam::376943093487:role/service-role/AmazonSageMaker-ExecutionRole-20261116T221626"

p = argparse.ArgumentParser()
# --baseline: only measure the validation Qini of the settings used before tuning.
p.add_argument("--baseline", action="store_true")
baseline = p.parse_args().baseline

hyperparameters = {}
if baseline:
    hyperparameters = {"tune-only": 1, "n-estimators": 300, "learning-rate": 0.05,
                       "num-leaves": 31, "min-child-samples": 20}

estimator = SKLearn(
    entry_point="train_entry.py",
    source_dir="sagemaker/training",
    dependencies=["src"],
    role=ROLE,
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    framework_version="1.4-2",
    py_version="py3",
    output_path=f"s3://{BUCKET}/models/",
    hyperparameters=hyperparameters,
    base_job_name="uplift-baseline" if baseline else "uplift-train",
)

estimator.fit({"train": f"s3://{BUCKET}/raw/"}, wait=False)
print("training started:", estimator.latest_training_job.name)
