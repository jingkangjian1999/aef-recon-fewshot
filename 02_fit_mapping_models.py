from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (
    CACHE_DIR,
    MAPPING_DIR,
    MODEL_LABELS,
    MODEL_NAMES,
    PARALLEL_JOBS,
    create_run_metadata,
    ensure_dir,
    fit_regression_with_group_cv,
    get_aef_columns,
    get_feature_columns,
    load_pickle,
    predict_linear_matrix,
    save_dataframe,
    save_json,
    save_pickle,
)


DATASET_KEYS = {
    "training_pool_2023": CACHE_DIR / "training_pool_2023.pkl",
    "same_year_eval_2023": CACHE_DIR / "same_year_eval_2023.pkl",
    "transfer_eval_2024": CACHE_DIR / "transfer_eval_2024.pkl",
}


def fit_one_target(model_name: str, target_name: str, feature_matrix: np.ndarray, target_values: np.ndarray, groups: np.ndarray) -> dict:
    result = fit_regression_with_group_cv(feature_matrix, target_values, groups, model_name)
    result["target"] = target_name
    result["model_name"] = model_name
    return result


def reconstruct_dataset(dataframe: pd.DataFrame, feature_columns: list[str], target_columns: list[str], model_store: dict[str, dict]) -> pd.DataFrame:
    feature_matrix = dataframe[feature_columns].to_numpy(dtype=float)
    coefficient_matrix = np.vstack([model_store[target]["coef"] for target in target_columns])
    intercepts = np.array([model_store[target]["intercept"] for target in target_columns], dtype=float)
    scaler_mean = np.array(model_store[target_columns[0]]["scaler_mean"], dtype=float)
    scaler_scale = np.array(model_store[target_columns[0]]["scaler_scale"], dtype=float)
    predictions = predict_linear_matrix(feature_matrix, scaler_mean, scaler_scale, coefficient_matrix, intercepts)
    reconstructed_df = pd.DataFrame(predictions, columns=[f"recon_{target}" for target in target_columns])
    for column in ["system:index", "row_id", "crop_type", "fold_id", "block_id"]:
        if column in dataframe.columns:
            reconstructed_df[column] = dataframe[column].to_numpy()
    return reconstructed_df


def main() -> None:
    for directory in [MAPPING_DIR, MAPPING_DIR / "metrics", MAPPING_DIR / "coefficients", MAPPING_DIR / "predictions", MAPPING_DIR / "models", MAPPING_DIR / "params", MAPPING_DIR / "features"]:
        ensure_dir(directory)

    lasso_fit_df = load_pickle(CACHE_DIR / "lasso_fit.pkl")
    feature_columns = get_feature_columns()
    aef_columns = get_aef_columns()
    feature_matrix = lasso_fit_df[feature_columns].to_numpy(dtype=float)
    group_ids = lasso_fit_df["fold_id"].to_numpy()

    model_summary_rows = []

    for model_name in MODEL_NAMES:
        target_results = Parallel(n_jobs=PARALLEL_JOBS, prefer="threads", verbose=10)(
            delayed(fit_one_target)(model_name, target_name, feature_matrix, lasso_fit_df[target_name].to_numpy(dtype=float), group_ids)
            for target_name in aef_columns
        )

        prediction_df = pd.DataFrame({"system:index": lasso_fit_df["system:index"], "fold_id": lasso_fit_df["fold_id"]})
        metric_rows = []
        coefficient_rows = []
        candidate_rows = []
        model_store = {}

        for result in target_results:
            target_name = result["target"]
            prediction_df[f"{target_name}_true"] = lasso_fit_df[target_name].to_numpy(dtype=float)
            prediction_df[f"{target_name}_pred"] = result["cv_prediction"]
            metric_rows.append(
                {
                    "model_name": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "target": target_name,
                    "r2": result["cv_metrics"]["r2"],
                    "rmse": result["cv_metrics"]["rmse"],
                    "best_params_json": json.dumps(result["best_params"], ensure_ascii=False),
                }
            )
            coefficient_row = {
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "target": target_name,
                "intercept": result["intercept"],
                "best_params_json": json.dumps(result["best_params"], ensure_ascii=False),
            }
            coefficient_row.update({feature_name: coefficient for feature_name, coefficient in zip(feature_columns, result["coef"])})
            coefficient_rows.append(coefficient_row)
            for candidate in result["candidate_scores"]:
                candidate_rows.append(
                    {
                        "model_name": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "target": target_name,
                        "params_json": json.dumps(candidate["params"], ensure_ascii=False),
                        "mean_r2": candidate["mean_r2"],
                        "sem_r2": candidate["sem_r2"],
                    }
                )
            model_store[target_name] = {
                "coef": result["coef"],
                "intercept": result["intercept"],
                "best_params": result["best_params"],
                "scaler_mean": result["scaler_mean"],
                "scaler_scale": result["scaler_scale"],
                "fold_records": result["fold_records"],
            }

        metric_df = pd.DataFrame(metric_rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
        ranking_df = metric_df[["target", "r2", "rmse"]].sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
        coefficient_df = pd.DataFrame(coefficient_rows)
        candidate_df = pd.DataFrame(candidate_rows)

        save_dataframe(prediction_df, MAPPING_DIR / "predictions" / f"{model_name}_cv_predictions.csv")
        save_dataframe(metric_df, MAPPING_DIR / "metrics" / f"{model_name}_per_target_metrics.csv")
        save_dataframe(ranking_df, MAPPING_DIR / "metrics" / f"{model_name}_dimension_ranking.csv")
        save_dataframe(coefficient_df, MAPPING_DIR / "coefficients" / f"{model_name}_coefficients.csv")
        save_dataframe(candidate_df, MAPPING_DIR / "params" / f"{model_name}_candidate_scores.csv")
        save_pickle(model_store, MAPPING_DIR / "models" / f"{model_name}_model_store.pkl")

        for dataset_key, dataset_path in DATASET_KEYS.items():
            dataset_df = load_pickle(dataset_path)
            reconstructed_df = reconstruct_dataset(dataset_df, feature_columns, aef_columns, model_store)
            save_dataframe(reconstructed_df, MAPPING_DIR / "features" / f"{dataset_key}_{model_name}_reconstructed.csv")

        model_summary_rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "mean_r2": metric_df["r2"].mean(),
                "median_r2": metric_df["r2"].median(),
                "mean_rmse": metric_df["rmse"].mean(),
                "positive_r2_dims": int((metric_df["r2"] > 0).sum()),
                "avg_nonzero_coef": float((coefficient_df[feature_columns].abs() > 1e-12).sum(axis=1).mean()),
            }
        )

    model_summary_df = pd.DataFrame(model_summary_rows).sort_values(["mean_r2", "median_r2"], ascending=[False, False]).reset_index(drop=True)
    save_dataframe(model_summary_df, MAPPING_DIR / "metrics" / "mapping_model_summary.csv")
    save_json(create_run_metadata("02_fit_mapping_models.py"), MAPPING_DIR / "run_metadata.json")


if __name__ == "__main__":
    main()
