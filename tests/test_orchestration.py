"""Cheap check that the Dagster asset graph is wired correctly -- catches broken
dependencies/typos without paying for a real download or multi-minute training run."""
from orchestration.definitions import defs


def test_definitions_build():
    assert defs is not None


def test_all_assets_present():
    job = defs.resolve_implicit_global_asset_job_def()
    asset_names = {key.to_user_string() for key in job.asset_layer.asset_graph.get_all_asset_keys()}
    assert asset_names == {
        "raw_sample",
        "treatment_effect",
        "split_data",
        "fitted_models",
        "evaluation_results",
        "propensity_scores",
        "profit_analysis",
        "segments",
        "reports",
    }


def test_randomization_check_is_registered():
    check_names = {c.check_key.name for c in defs.asset_checks}
    assert "raw_sample_check" in check_names
