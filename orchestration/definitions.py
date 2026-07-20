"""Dagster entrypoint. Run: dagster dev -f orchestration/definitions.py"""
import dagster as dg

from orchestration.assets import (
    evaluation_results,
    fitted_models,
    profit_analysis,
    propensity_scores,
    raw_sample,
    raw_sample_check,
    reports,
    segments,
    split_data,
    treatment_effect,
)

defs = dg.Definitions(
    assets=[
        raw_sample,
        treatment_effect,
        split_data,
        fitted_models,
        evaluation_results,
        propensity_scores,
        profit_analysis,
        segments,
        reports,
    ],
    asset_checks=[raw_sample_check],
)
