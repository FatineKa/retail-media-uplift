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

-- Dedicated, narrow-permission login for the Week 3 Lambda function
-- (../lambda/quality_gate.py) -- separate from your own personal Snowflake
-- login, and can only read experiment_data via this one warehouse.
CREATE ROLE IF NOT EXISTS lambda_quality_check_role;

GRANT USAGE ON WAREHOUSE RETAIL_MEDIA_WH TO ROLE lambda_quality_check_role;
GRANT USAGE ON DATABASE RETAIL_MEDIA_UPLIFT TO ROLE lambda_quality_check_role;
GRANT USAGE ON SCHEMA RETAIL_MEDIA_UPLIFT.PUBLIC TO ROLE lambda_quality_check_role;
GRANT SELECT ON TABLE RETAIL_MEDIA_UPLIFT.PUBLIC.EXPERIMENT_DATA TO ROLE lambda_quality_check_role;

CREATE USER IF NOT EXISTS lambda_service_user
  DEFAULT_ROLE = lambda_quality_check_role
  DEFAULT_WAREHOUSE = RETAIL_MEDIA_WH
  MUST_CHANGE_PASSWORD = FALSE;

GRANT ROLE lambda_quality_check_role TO USER lambda_service_user;

-- This account enforces MFA on password logins, which blocks unattended
-- service accounts entirely -- so this user authenticates via RSA key-pair
-- instead of a password (generated locally, private key never committed):
--   openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt
--   openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub
-- Then register the public half here (public keys aren't secret):
ALTER USER lambda_service_user SET RSA_PUBLIC_KEY='MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEApn686ihFWUT+kpQrGtjrUMF1anhCKszKABW75a3TjTDhtbjUZVuEKJdKFhgHJLzpzYxrnE4JVZenh/yblX0YohQw5Y1Kg/VD+mv3YWQg/t17NdAFZnPUahdXB7OL2O/GJ53vAz28XkmCtYH7DPuRNQJlr9b+KGHgqzWlyUXxQPqhi+W8ypG0WEegCTQfJUPcs/w65B+bZpCA9Uj20bd4v7s45v6NWI9lTHGNbs4LY+P583jPDA1UnLXTTfh3pFw3onb2+kZ+2VzX9dxr5guCqpjs4m/Prjw0NoT0cg6kI0QRHzbdjQeGnsIOkO3g4VWFAkQStpPmz4uJVuWVYtgeIQIDAQAB';
-- The Lambda function loads the matching private key from a base64-encoded
-- environment variable (SNOWFLAKE_PRIVATE_KEY_B64) -- never committed here.
