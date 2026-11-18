"""Start SageMaker automatic model tuning for the S-learner (LightGBM), on all 13.9M users.
The goal is the Qini coefficient on the validation sample; the test set is never used here.
To run, from the project folder, after aws login:  python scripts/launch_tuning.py
"""
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter

BUCKET = "sagemaker-retail-media-uplift-3769"
ROLE = "arn:aws:iam::376943093487:role/service-role/AmazonSageMaker-ExecutionRole-20261116T221626"

estimator = SKLearn(
    entry_point="train_entry.py",
    source_dir="sagemaker/training",  # also sends requirements.txt (lightgbm)
    dependencies=["src"],             # copies src/ next to train_entry.py
    role=ROLE,
    instance_type="ml.m5.2xlarge",    # 32 GB of memory for the full dataset
    instance_count=1,
    framework_version="1.4-2",
    py_version="py3",
    output_path=f"s3://{BUCKET}/tuning/",
    hyperparameters={"tune-only": 1},
    base_job_name="uplift-tune",
)

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name="valid_qini",
    objective_type="Maximize",
    metric_definitions=[{"Name": "valid_qini", "Regex": "valid_qini=(-?[0-9.]+)"}],
    hyperparameter_ranges={
        "n-estimators": IntegerParameter(100, 800),
        "learning-rate": ContinuousParameter(0.01, 0.2, scaling_type="Logarithmic"),
        "num-leaves": IntegerParameter(15, 127),
        # Converters are rare (0.3%): large leaves average over more of them.
        "min-child-samples": IntegerParameter(20, 2000, scaling_type="Logarithmic"),
    },
    strategy="Bayesian",
    max_jobs=12,
    max_parallel_jobs=3,
)

tuner.fit({"train": f"s3://{BUCKET}/raw/"}, wait=False)
print("tuning started:", tuner.latest_tuning_job.name)
