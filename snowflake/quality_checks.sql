-- Same checks as src/data_prep.py's ate() and randomization_check(),
-- recreated as SQL running directly on the Snowflake-warehoused data.

-- Average treatment effect: does advertising cause more purchases?
-- Expect ~+0.00113, matching README.md.
SELECT
  AVG(CASE WHEN "treatment" = 1 THEN "conversion" END) AS conversion_rate_treated,
  AVG(CASE WHEN "treatment" = 0 THEN "conversion" END) AS conversion_rate_control,
  AVG(CASE WHEN "treatment" = 1 THEN "conversion" END)
    - AVG(CASE WHEN "treatment" = 0 THEN "conversion" END) AS ate
FROM retail_media_uplift.public.experiment_data;

-- Randomization check: standardized mean difference (SMD) per feature.
-- |SMD| < 0.05 on every feature means the treated/control groups were
-- balanced before ads ran, confirming the randomization worked.
SELECT
  feature,
  smd,
  ABS(smd) < 0.05 AS ok
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
ORDER BY ABS(smd) DESC;

-- Quality gate: a single TRUE/FALSE that a future automated step (Week 3
-- orchestration) can check to halt the pipeline if randomization broke.
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
