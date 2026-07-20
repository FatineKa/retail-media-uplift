"""Dagster asset graph over the retail-media-uplift pipeline.

No business logic lives here -- every asset body is a thin wrapper calling into the
same, independently-tested functions in src/ that `python -m src.download` and
`python -m src.train` already use. This module only adds structure (dependencies,
config, observability metadata, one data-quality gate) on top of what already exists.

Run it: `dagster dev -f orchestration/definitions.py`
"""
import dagster as dg

import src.download as download
from src.data_prep import ate, load_sample, randomization_check
from src.policy import segment_users
from src.train import (
    build_profit_analysis,
    evaluate_uplift_models,
    fit_uplift_models,
    propensity_model,
    save_reports,
    split,
)


class DownloadConfig(dg.Config):
    rows: int = 3_000_000  # matches the sample size reports/ was actually built from


@dg.asset(
    description="Streams + samples the Criteo uplift dataset to data/raw/ (src.download.main).",
)
def raw_sample(config: DownloadConfig) -> dg.MaterializeResult:
    stats = download.main(rows=config.rows)
    return dg.MaterializeResult(
        metadata={
            "rows_seen": stats["n_seen"],
            "rows_kept": stats["n_kept"],
            "treatment_share": dg.MetadataValue.float(float(stats["treatment_share"])),
            "conversion_rate": dg.MetadataValue.float(float(stats["conversion_rate"])),
        }
    )


@dg.asset_check(
    asset=raw_sample,
    blocking=True,
    description="Randomization held: every feature's |SMD| must be < 0.05 (src.data_prep.randomization_check).",
)
def raw_sample_check() -> dg.AssetCheckResult:
    df = load_sample()
    out = randomization_check(df)
    worst = out.loc[out.smd.abs().idxmax()]
    return dg.AssetCheckResult(
        passed=bool(out.ok.all()),
        metadata={
            "max_abs_smd": dg.MetadataValue.float(float(abs(worst.smd))),
            "worst_feature": worst.feature,
        },
    )


@dg.asset(
    deps=[raw_sample],
    description="Raw ATE (conversion, visit) with Welch 95% CIs (src.data_prep.ate).",
)
def treatment_effect() -> dg.MaterializeResult:
    df = load_sample()
    conversion = ate(df, "conversion")
    visit = ate(df, "visit")
    return dg.MaterializeResult(
        metadata={
            "conversion_ate": dg.MetadataValue.float(float(conversion["ate"])),
            "conversion_relative_lift": dg.MetadataValue.float(float(conversion["relative_lift"])),
            "conversion_p_value": dg.MetadataValue.float(float(conversion["p_value"])),
            "visit_ate": dg.MetadataValue.float(float(visit["ate"])),
        }
    )


@dg.asset(
    deps=[raw_sample],
    description="Stratified 70/30 train/test split, stratified on treatment x conversion (src.train.split).",
)
def split_data() -> dict:
    df = load_sample()
    X_tr, X_te, t_tr, t_te, y_tr, y_te = split(df)
    return {"X_tr": X_tr, "X_te": X_te, "t_tr": t_tr, "t_te": t_te, "y_tr": y_tr, "y_te": y_te}


@dg.asset(description="Fits all 4 uplift meta-learners (src.train.fit_uplift_models).")
def fitted_models(split_data: dict) -> dict:
    return fit_uplift_models(split_data["X_tr"], split_data["t_tr"], split_data["y_tr"])


@dg.asset(
    description="Qini coefficient / AUUC / uplift@10% for every model, plus per-model "
    "Qini curves (src.train.evaluate_uplift_models)."
)
def evaluation_results(fitted_models: dict, split_data: dict) -> dg.Output:
    results, test_scores, qini_rows, best_model = evaluate_uplift_models(
        fitted_models, split_data["X_te"], split_data["t_te"], split_data["y_te"]
    )
    payload = {
        "results": results,
        "test_scores": test_scores,
        "qini_rows": qini_rows,
        "best_model": best_model,
    }
    return dg.Output(
        payload,
        metadata={
            "best_model": best_model,
            "best_qini_coefficient": dg.MetadataValue.float(float(results[best_model][0])),
        },
    )


@dg.asset(description="Naive competitor: propensity-to-convert scores on the test set (src.train.propensity_model).")
def propensity_scores(split_data: dict) -> dg.Output:
    model = propensity_model(split_data["X_tr"], split_data["t_tr"], split_data["y_tr"])
    scores = model.predict_proba(split_data["X_te"])[:, 1]
    return dg.Output(scores, metadata={"mean_score": dg.MetadataValue.float(float(scores.mean()))})


@dg.asset(
    description="Money chart: uplift targeting vs. propensity targeting profit across "
    "budget levels (src.train.build_profit_analysis)."
)
def profit_analysis(evaluation_results: dict, propensity_scores, split_data: dict) -> dg.Output:
    table = build_profit_analysis(
        split_data["y_te"],
        split_data["t_te"],
        evaluation_results["test_scores"],
        propensity_scores,
        evaluation_results["best_model"],
    )
    at_2pct = table[(table.policy == "uplift") & (table.budget_frac.round(2) == 0.02)]
    metadata = {}
    if not at_2pct.empty:
        metadata["uplift_profit_at_2pct"] = dg.MetadataValue.float(float(at_2pct.profit.iloc[0]))
    return dg.Output(table, metadata=metadata)


@dg.asset(
    description="Persuadable / sure-thing / lost-cause / do-not-disturb segments (src.policy.segment_users)."
)
def segments(evaluation_results: dict, propensity_scores) -> dg.Output:
    best_model = evaluation_results["best_model"]
    uplift_scores = evaluation_results["test_scores"][best_model].to_numpy()
    seg = segment_users(propensity_scores, uplift_scores)
    return dg.Output(seg, metadata={"counts": dg.MetadataValue.json(seg.value_counts().to_dict())})


@dg.asset(
    description="Persists test_scores / qini_curves / profit_curves / summary.json to reports/ "
    "(src.train.save_reports) -- same 4 artifacts `python -m src.train` produces."
)
def reports(
    evaluation_results: dict,
    profit_analysis,
    segments,
    propensity_scores,
    split_data: dict,
) -> dg.MaterializeResult:
    test_scores = evaluation_results["test_scores"].copy()
    test_scores["propensity"] = propensity_scores
    test_scores["segment"] = segments.to_numpy()

    results = evaluation_results["results"]
    summary = {
        "best_model": evaluation_results["best_model"],
        "n_test": int(len(split_data["y_te"])),
        "results": {
            k: {"qini_coefficient": v[0], "uplift_at_10pct": v[1]} for k, v in results.items()
        },
        "segment_counts": segments.value_counts().to_dict(),
    }
    save_reports(test_scores, evaluation_results["qini_rows"], profit_analysis, summary)
    return dg.MaterializeResult(
        metadata={
            "best_model": summary["best_model"],
            "n_test": summary["n_test"],
        }
    )
