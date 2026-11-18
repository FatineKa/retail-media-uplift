"""Start the SageMaker processing job: full dataset -> s3://sagemaker-retail-media-uplift-3769/raw/
To run, from the project folder, after aws login:  python scripts/launch_processing.py
"""
from sagemaker.processing import FrameworkProcessor, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn

BUCKET = "sagemaker-retail-media-uplift-3769"
ROLE = "arn:aws:iam::376943093487:role/service-role/AmazonSageMaker-ExecutionRole-20261116T221626"

processor = FrameworkProcessor(
    estimator_cls=SKLearn,
    framework_version="1.4-2",
    py_version="py3",
    role=ROLE,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size_in_gb=30,  # disk space for the Hugging Face download
    base_job_name="uplift-prepare",
)

processor.run(
    code="prepare_data.py",
    source_dir="sagemaker/processing",  # also sends requirements.txt (datasets)
    outputs=[ProcessingOutput(source="/opt/ml/processing/raw",
                              destination=f"s3://{BUCKET}/raw/")],
)
