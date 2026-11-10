-- Make the warehouse (compute), the database and the schema for this project.
CREATE WAREHOUSE IF NOT EXISTS RETAIL_MEDIA_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS RETAIL_MEDIA_UPLIFT;

CREATE SCHEMA IF NOT EXISTS RETAIL_MEDIA_UPLIFT.PUBLIC;

-- Connect to the retail-media-uplift-fatin S3 bucket through the
-- snowflake-retail-media-role IAM role.
-- For the AWS permissions, see ../aws/iam-policy-s3-only.json.
CREATE STORAGE INTEGRATION retail_media_s3_integration
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = 'S3'
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::425178888541:role/snowflake-retail-media-role'
  STORAGE_ALLOWED_LOCATIONS = ('s3://retail-media-uplift-fatin/');

-- Run `DESC INTEGRATION retail_media_s3_integration;`. It gives
-- STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID.
-- Use these two values to make the trust policy of the AWS IAM role.
-- The integration cannot read data before you do this step.

-- Stages for the two folders of the bucket.
CREATE STAGE retail_media_uplift.public.raw_stage
  URL = 's3://retail-media-uplift-fatin/raw/'
  STORAGE_INTEGRATION = retail_media_s3_integration;

CREATE STAGE retail_media_uplift.public.processed_stage
  URL = 's3://retail-media-uplift-fatin/processed/'
  STORAGE_INTEGRATION = retail_media_s3_integration;

-- Load the raw experiment data (3,001,602 rows) into a table.
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

-- Note: INFER_SCHEMA keeps the lowercase column names, and they are case-sensitive.
-- Thus, put double quotes around each column name in other queries,
-- for example "treatment", "conversion", "f0". Do not write TREATMENT/CONVERSION/F0.

-- Make a separate login for the Lambda function (../lambda/quality_gate.py).
-- This login is not your personal Snowflake login.
-- It can only read experiment_data, through this one warehouse.
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

-- This account requires MFA for password logins.
-- Thus, a service account cannot use a password. This user uses an RSA key pair.
-- Make the keys on your computer. Do not commit the private key:
--   openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt
--   openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub
-- Then add the public key here (a public key is not secret):
ALTER USER lambda_service_user SET RSA_PUBLIC_KEY='MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEApn686ihFWUT+kpQrGtjrUMF1anhCKszKABW75a3TjTDhtbjUZVuEKJdKFhgHJLzpzYxrnE4JVZenh/yblX0YohQw5Y1Kg/VD+mv3YWQg/t17NdAFZnPUahdXB7OL2O/GJ53vAz28XkmCtYH7DPuRNQJlr9b+KGHgqzWlyUXxQPqhi+W8ypG0WEegCTQfJUPcs/w65B+bZpCA9Uj20bd4v7s45v6NWI9lTHGNbs4LY+P583jPDA1UnLXTTfh3pFw3onb2+kZ+2VzX9dxr5guCqpjs4m/Prjw0NoT0cg6kI0QRHzbdjQeGnsIOkO3g4VWFAkQStpPmz4uJVuWVYtgeIQIDAQAB';
-- The Lambda function reads the private key from a base64 environment
-- variable (SNOWFLAKE_PRIVATE_KEY_B64). Do not commit this key.
