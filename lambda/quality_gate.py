"""AWS Lambda function: re-runs the Week 2 Snowflake randomization-check
quality gate (see ../snowflake/quality_checks.sql) and returns pass/fail.

Reads Snowflake connection details from environment variables so no
credentials are hardcoded here -- see ../snowflake/setup.sql for the
dedicated lambda_service_user this is meant to authenticate as.
"""
import base64
import os

import snowflake.connector
from cryptography.hazmat.primitives import serialization

QUALITY_GATE_QUERY = """
SELECT BOOLOR_AGG(ok) AS randomization_check_passed
FROM (
  SELECT feature, smd, ABS(smd) < 0.05 AS ok
  FROM (
    SELECT 'f0' AS feature, (AVG(CASE WHEN "treatment"=1 THEN "f0" END) - AVG(CASE WHEN "treatment"=0 THEN "f0" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f0" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f0" END))/2) AS smd FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f1', (AVG(CASE WHEN "treatment"=1 THEN "f1" END) - AVG(CASE WHEN "treatment"=0 THEN "f1" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f1" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f1" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f2', (AVG(CASE WHEN "treatment"=1 THEN "f2" END) - AVG(CASE WHEN "treatment"=0 THEN "f2" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f2" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f2" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f3', (AVG(CASE WHEN "treatment"=1 THEN "f3" END) - AVG(CASE WHEN "treatment"=0 THEN "f3" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f3" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f3" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f4', (AVG(CASE WHEN "treatment"=1 THEN "f4" END) - AVG(CASE WHEN "treatment"=0 THEN "f4" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f4" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f4" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f5', (AVG(CASE WHEN "treatment"=1 THEN "f5" END) - AVG(CASE WHEN "treatment"=0 THEN "f5" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f5" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f5" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f6', (AVG(CASE WHEN "treatment"=1 THEN "f6" END) - AVG(CASE WHEN "treatment"=0 THEN "f6" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f6" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f6" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f7', (AVG(CASE WHEN "treatment"=1 THEN "f7" END) - AVG(CASE WHEN "treatment"=0 THEN "f7" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f7" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f7" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f8', (AVG(CASE WHEN "treatment"=1 THEN "f8" END) - AVG(CASE WHEN "treatment"=0 THEN "f8" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f8" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f8" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f9', (AVG(CASE WHEN "treatment"=1 THEN "f9" END) - AVG(CASE WHEN "treatment"=0 THEN "f9" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f9" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f9" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f10', (AVG(CASE WHEN "treatment"=1 THEN "f10" END) - AVG(CASE WHEN "treatment"=0 THEN "f10" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f10" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f10" END))/2) FROM retail_media_uplift.public.experiment_data
    UNION ALL
    SELECT 'f11', (AVG(CASE WHEN "treatment"=1 THEN "f11" END) - AVG(CASE WHEN "treatment"=0 THEN "f11" END)) / SQRT((VARIANCE(CASE WHEN "treatment"=1 THEN "f11" END) + VARIANCE(CASE WHEN "treatment"=0 THEN "f11" END))/2) FROM retail_media_uplift.public.experiment_data
  )
);
"""


def _load_private_key():
    # SNOWFLAKE_PRIVATE_KEY_B64 holds the PEM key base64-encoded, to survive
    # being pasted through Lambda's console env-var text box unmangled.
    pem = base64.b64decode(os.environ["SNOWFLAKE_PRIVATE_KEY_B64"])
    key = serialization.load_pem_private_key(pem, password=None)
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def handler(event, context):
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        private_key=_load_private_key(),
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute(QUALITY_GATE_QUERY)
        passed = cursor.fetchone()[0]
    finally:
        conn.close()
    return {"passed": bool(passed)}
