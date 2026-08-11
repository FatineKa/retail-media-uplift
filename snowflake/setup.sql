-- Warehouse (compute), database, and schema for this project.
CREATE WAREHOUSE IF NOT EXISTS RETAIL_MEDIA_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS RETAIL_MEDIA_UPLIFT;

CREATE SCHEMA IF NOT EXISTS RETAIL_MEDIA_UPLIFT.PUBLIC;

-- Trust connection to the retail-media-uplift-fatin S3 bucket, via the
-- snowflake-retail-media-role IAM role (see ../aws/iam-policy-s3-only.json
-- for the AWS-side permissions).
CREATE STORAGE INTEGRATION retail_media_s3_integration
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = 'S3'
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::425178888541:role/snowflake-retail-media-role'
  STORAGE_ALLOWED_LOCATIONS = ('s3://retail-media-uplift-fatin/');

-- Run `DESC INTEGRATION retail_media_s3_integration;` to get
-- STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID, then use those two
-- values to build the AWS IAM role's trust policy before this integration
-- can actually read anything.

-- Pointers to the bucket's two folders.
CREATE STAGE retail_media_uplift.public.raw_stage
  URL = 's3://retail-media-uplift-fatin/raw/'
  STORAGE_INTEGRATION = retail_media_s3_integration;

CREATE STAGE retail_media_uplift.public.processed_stage
  URL = 's3://retail-media-uplift-fatin/processed/'
  STORAGE_INTEGRATION = retail_media_s3_integration;

-- Load the raw experiment data (3,001,602 rows) into a real table.
CREATE FILE FORMAT retail_media_uplift.public.parquet_format
  TYPE = PARQUET;

CREATE TABLE retail_media_uplift.public.experiment_data
  USING TEMPLATE (
    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
    FROM TABLE(
      INFER_SCHEMA(
        LOCATION => '@retail_media_uplift.public.raw_stage/criteo_uplift_sample.parquet',
        FILE_FORMAT => 'retail_media_uplift.public.parquet_format'
      )
    )
  );

COPY INTO retail_media_uplift.public.experiment_data
  FROM @retail_media_uplift.public.raw_stage/criteo_uplift_sample.parquet
  FILE_FORMAT = retail_media_uplift.public.parquet_format
  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- Note: INFER_SCHEMA preserved the original lowercase column names as
-- case-sensitive, so every column reference elsewhere needs double quotes,
-- e.g. "treatment", "conversion", "f0" -- not TREATMENT/CONVERSION/F0.
