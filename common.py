from __future__ import annotations

import json
import math
import os
import pickle
import platform
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


def resolve_evaluation_csv(year: int) -> Path:
    sample_root = PROJECT_ROOT / "样点"
    pattern = f"Dataset_AEF_S1S2_CDL{year}_CornSoy_withBlocks*.csv"
    candidates = sorted(path for path in sample_root.rglob(pattern) if "5000_each" not in path.name)
    nested_candidates = [path for path in candidates if path.parent != sample_root]
    if len(nested_candidates) == 1:
        return nested_candidates[0]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one evaluation CSV for {year}, found {len(candidates)}: {candidates}")
    return candidates[0]


LASSO_FIT_CSV = PROJECT_ROOT / "lasso" / "Dataset_AEF_S1S2_CDL2023_CornSoy_lasso.csv"
TRAINING_POOL_2023_CSV = PROJECT_ROOT / "样点" / "Dataset_AEF_S1S2_CDL2023_CornSoy_withBlocks_5000_each.csv"
SAME_YEAR_EVAL_2023_CSV = resolve_evaluation_csv(2023)
TRANSFER_EVAL_2024_CSV = resolve_evaluation_csv(2024)

MANIFEST_DIR = OUTPUT_ROOT / "01_manifest"
MAPPING_DIR = OUTPUT_ROOT / "02_mapping_models"
MODEL_SELECTION_DIR = OUTPUT_ROOT / "03_model_k_selection"
BASELINE_K_DIR = OUTPUT_ROOT / "04_baseline_k_selection"
SAME_YEAR_DIR = OUTPUT_ROOT / "05_same_year_baselines"
TRANSFER_DIR = OUTPUT_ROOT / "06_transfer_baselines"
BUDGET_DIR = OUTPUT_ROOT / "07_budget_ablation"
TRANSFER_BUDGET_DIR = OUTPUT_ROOT / "08_transfer_budget_ablation"
MODEL_K_TRANSFER_DIR = OUTPUT_ROOT / "09_model_k_transfer"
STATISTICS_DIR = OUTPUT_ROOT / "10_statistics"
LOG_DIR = OUTPUT_ROOT / "logs"

CACHE_DIR = MANIFEST_DIR / "cache"
TRAIN_INDEX_DIR = MANIFEST_DIR / "train_indices"

METADATA_COLUMNS = [
    "system:index",
    "block_id",
    "class_id",
    "col_id",
    "crop_type",
    "fold_id",
    "grid_size_m",
    "label",
    "row_id",
    ".geo",
    "lat",
    "lon",
]

BUDGETS = [10, 20, 50, 100, 200]
DEVELOPMENT_BUDGETS = [50, 100]
K_VALUES = [5, 10, 20, 30, 64]
REPEATS = list(range(1, 11))
RF_TREES = 100
PARALLEL_JOBS = max(1, int(os.environ.get("EXPERIMENT_JOBS", "1")))
MODEL_NAMES = ["lasso", "ridge", "elastic_net", "pls", "mlr"]
MODEL_LABELS = {
    "lasso": "Lasso",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "pls": "PLSRegression",
    "mlr": "MLR",
}
MODEL_PREFERENCE = ["lasso", "elastic_net", "ridge", "pls", "mlr"]
BASELINE_LABELS = {
    "raw_rf": "Multi-modal Satellite Features",
    "reconstructed_rf": "Reconstructed Features",
    "aef64_rf": "AEF Embeddings",
    "pca_rf": "PCA",
    "pls_label_rf": "Supervised PLS",
    "anova_rf": "ANOVA",
    "mi_rf": "MI",
    "rf_importance_rf": "RF Importance",
    "coral_rf": "CORAL",
    "tca_rf": "TCA",
}
BASELINE_K_METHODS = ["pca_rf", "pls_label_rf", "anova_rf", "mi_rf", "rf_importance_rf", "tca_rf"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_crop_labels(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    values = values.replace({"soybean": "soy", "soybeans": "soy", "maize": "corn"})
    return values


def load_dataset(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=usecols)
    if "crop_type" in df.columns:
        df["crop_type"] = normalize_crop_labels(df["crop_type"])
    if "fold_id" in df.columns:
        df["fold_id"] = df["fold_id"].astype(int)
    return df


def infer_columns(path: str | Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, nrows=1)
    columns = df.columns.tolist()
    aef_columns = sorted([col for col in columns if len(col) == 3 and col.startswith("A") and col[1:].isdigit()], key=lambda x: int(x[1:]))
    metadata_columns = [col for col in columns if col in METADATA_COLUMNS]
    feature_columns = [col for col in columns if col not in set(aef_columns) | set(metadata_columns)]
    return {
        "all_columns": columns,
        "aef_columns": aef_columns,
        "feature_columns": feature_columns,
        "metadata_columns": metadata_columns,
    }


def load_manifest() -> dict[str, Any]:
    return load_json(MANIFEST_DIR / "dataset_manifest.json")


def get_aef_columns() -> list[str]:
    return load_manifest()["aef_columns"]


def get_feature_columns() -> list[str]:
    return load_manifest()["feature_columns"]


def get_metadata_columns() -> list[str]:
    return load_manifest()["metadata_columns"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "oa": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "aa": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def summarize_grouped_metrics(dataframe: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        dataframe.groupby(group_columns)
        .agg(
            oa_mean=("oa", "mean"),
            oa_std=("oa", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            aa_mean=("aa", "mean"),
            aa_std=("aa", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            n_runs=("oa", "size"),
        )
        .reset_index()
    )
    for metric in ["oa", "macro_f1", "aa", "kappa"]:
        summary[f"{metric}_sem"] = summary[f"{metric}_std"] / np.sqrt(summary["n_runs"])
        interval = 1.96 * summary[f"{metric}_sem"]
        summary[f"{metric}_ci_low"] = summary[f"{metric}_mean"] - interval
        summary[f"{metric}_ci_high"] = summary[f"{metric}_mean"] + interval
    return summary


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=RF_TREES, random_state=seed, n_jobs=1)


def fit_standardizer(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    return train_scaled, test_scaled, scaler


def get_regression_param_grid(model_name: str) -> list[dict[str, float | int]]:
    if model_name == "lasso":
        return [{"alpha": float(value)} for value in np.logspace(-4, -1, 10)]
    if model_name == "ridge":
        return [{"alpha": float(value)} for value in np.logspace(-3, 3, 13)]
    if model_name == "elastic_net":
        return [{"alpha": float(alpha), "l1_ratio": float(ratio)} for alpha in np.logspace(-4, -1, 7) for ratio in [0.2, 0.5, 0.8]]
    if model_name == "pls":
        return [{"n_components": int(value)} for value in [2, 5, 10, 20, 30, 40]]
    if model_name == "mlr":
        return [{}]
    raise ValueError(f"Unknown model: {model_name}")


def _prepare_regressor_params(model_name: str, params: dict[str, Any], n_rows: int, n_cols: int) -> dict[str, Any]:
    adjusted = dict(params)
    if model_name == "pls":
        adjusted["n_components"] = max(1, min(int(adjusted["n_components"]), n_rows - 1, n_cols))
    return adjusted


def build_regressor(model_name: str, params: dict[str, Any], n_rows: int, n_cols: int) -> Any:
    adjusted = _prepare_regressor_params(model_name, params, n_rows, n_cols)
    if model_name == "lasso":
        return Lasso(max_iter=20000, tol=1e-4, **adjusted)
    if model_name == "ridge":
        return Ridge(**adjusted)
    if model_name == "elastic_net":
        return ElasticNet(max_iter=20000, tol=1e-4, **adjusted)
    if model_name == "pls":
        return PLSRegression(scale=False, **adjusted)
    if model_name == "mlr":
        return LinearRegression()
    raise ValueError(f"Unknown model: {model_name}")


def parameter_complexity_key(model_name: str, params: dict[str, Any]) -> tuple[Any, ...]:
    if model_name in {"lasso", "ridge", "elastic_net"}:
        return (-params.get("alpha", 0.0), -params.get("l1_ratio", 0.0))
    if model_name == "pls":
        return (params.get("n_components", 9999),)
    return (0,)


def fit_regression_with_group_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str) -> dict[str, Any]:
    group_kfold = GroupKFold(n_splits=len(np.unique(groups)))
    candidate_rows: list[dict[str, Any]] = []
    for params in get_regression_param_grid(model_name):
        fold_scores = []
        for train_index, val_index in group_kfold.split(x, y, groups):
            train_x = x[train_index]
            val_x = x[val_index]
            train_y = y[train_index]
            val_y = y[val_index]
            train_scaled, val_scaled, _ = fit_standardizer(train_x, val_x)
            estimator = build_regressor(model_name, params, train_scaled.shape[0], train_scaled.shape[1])
            estimator.fit(train_scaled, train_y)
            predictions = np.asarray(estimator.predict(val_scaled)).reshape(-1)
            fold_scores.append(r2_score(val_y, predictions))
        candidate_rows.append(
            {
                "params": params,
                "fold_scores": fold_scores,
                "mean_r2": float(np.mean(fold_scores)),
                "sem_r2": float(stats.sem(fold_scores)) if len(fold_scores) > 1 else 0.0,
            }
        )

    best_mean = max(row["mean_r2"] for row in candidate_rows)
    best_sem = next(row["sem_r2"] for row in candidate_rows if row["mean_r2"] == best_mean)
    admissible_rows = [row for row in candidate_rows if row["mean_r2"] >= best_mean - best_sem]
    selected_candidate = sorted(admissible_rows, key=lambda row: parameter_complexity_key(model_name, row["params"]))[0]
    best_params = selected_candidate["params"]

    cv_prediction = np.zeros_like(y, dtype=float)
    fold_records = []
    for fold_number, (train_index, val_index) in enumerate(group_kfold.split(x, y, groups), start=1):
        train_x = x[train_index]
        val_x = x[val_index]
        train_y = y[train_index]
        val_y = y[val_index]
        train_scaled, val_scaled, scaler = fit_standardizer(train_x, val_x)
        estimator = build_regressor(model_name, best_params, train_scaled.shape[0], train_scaled.shape[1])
        estimator.fit(train_scaled, train_y)
        predictions = np.asarray(estimator.predict(val_scaled)).reshape(-1)
        cv_prediction[val_index] = predictions
        fold_records.append(
            {
                "fold": fold_number,
                "train_size": int(len(train_index)),
                "val_size": int(len(val_index)),
                "r2": float(r2_score(val_y, predictions)),
                "rmse": float(math.sqrt(mean_squared_error(val_y, predictions))),
            }
        )

    full_scaler = StandardScaler().fit(x)
    full_x = full_scaler.transform(x)
    final_estimator = build_regressor(model_name, best_params, full_x.shape[0], full_x.shape[1])
    final_estimator.fit(full_x, y)
    coefficients = np.asarray(getattr(final_estimator, "coef_")).reshape(-1)
    intercept = float(np.asarray(getattr(final_estimator, "intercept_")).reshape(-1)[0])
    return {
        "best_params": best_params,
        "candidate_scores": candidate_rows,
        "cv_prediction": cv_prediction,
        "cv_metrics": regression_metrics(y, cv_prediction),
        "fold_records": fold_records,
        "coef": coefficients,
        "intercept": intercept,
        "scaler_mean": full_scaler.mean_.tolist(),
        "scaler_scale": full_scaler.scale_.tolist(),
    }


def predict_linear_matrix(x: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray, coefficient_matrix: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    safe_scale = np.where(np.asarray(scaler_scale, dtype=float) == 0, 1.0, np.asarray(scaler_scale, dtype=float))
    scaled_x = (x - np.asarray(scaler_mean, dtype=float)) / safe_scale
    return scaled_x @ coefficient_matrix.T + intercepts


def coral_align(source_x: np.ndarray, target_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    source_cov = np.cov(source_x, rowvar=False) + np.eye(source_x.shape[1])
    target_cov = np.cov(target_x, rowvar=False) + np.eye(target_x.shape[1])
    source_mean = source_x.mean(axis=0, keepdims=True)
    target_mean = target_x.mean(axis=0, keepdims=True)
    transform = linalg.fractional_matrix_power(source_cov, -0.5) @ linalg.fractional_matrix_power(target_cov, 0.5)
    transform = np.real_if_close(transform, tol=1000)
    aligned_source = (source_x - source_mean) @ transform + target_mean
    metadata = {
        "transform": np.asarray(np.real(transform), dtype=float),
        "source_mean": source_mean.ravel(),
        "target_mean": target_mean.ravel(),
    }
    return aligned_source, target_x.copy(), metadata


def tca_transform(source_x: np.ndarray, target_x: np.ndarray, dim: int, reg: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n_source = source_x.shape[0]
    n_target = target_x.shape[0]
    stacked = np.vstack([source_x, target_x]).astype(float)
    n_total = n_source + n_target
    e = np.vstack([np.full((n_source, 1), 1.0 / n_source), np.full((n_target, 1), -1.0 / n_target)])
    m_matrix = e @ e.T
    h_matrix = np.eye(n_total) - np.ones((n_total, n_total)) / n_total
    xhx = stacked.T @ h_matrix @ stacked
    xmx = stacked.T @ m_matrix @ stacked
    d = stacked.shape[1]
    eigvals, eigvecs = linalg.eigh(xhx, xmx + reg * np.eye(d))
    order = np.argsort(eigvals)[::-1]
    dim = max(1, min(int(dim), stacked.shape[1]))
    projection = eigvecs[:, order[:dim]]
    transformed = stacked @ projection
    metadata = {
        "projection": projection,
        "eigvals": eigvals[order[:dim]],
    }
    return transformed[:n_source], transformed[n_source:], metadata


def load_sampling_indices(fold_id: int, budget: int, repeat_id: int) -> pd.DataFrame:
    path = TRAIN_INDEX_DIR / f"train2023_fold{fold_id}_budget{budget}_repeat{repeat_id:02d}.csv"
    return pd.read_csv(path)


def find_training_rows(df: pd.DataFrame, index_df: pd.DataFrame) -> np.ndarray:
    key = "system:index" if "system:index" in df.columns and "system:index" in index_df.columns else "row_id"
    lookup = pd.Series(np.arange(len(df)), index=df[key]).to_dict()
    return np.array([lookup[value] for value in index_df[key].tolist()], dtype=int)


def create_run_metadata(script_name: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(PROJECT_ROOT)
    metadata = {
        "script": script_name,
        "package_root": str(PACKAGE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "python_executable": os.environ.get("PYTHON_EXECUTABLE", sys.executable),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "disk": {
            "total_gb": round(total_bytes / 1024**3, 2),
            "used_gb": round(used_bytes / 1024**3, 2),
            "free_gb": round(free_bytes / 1024**3, 2),
        },
    }
    if extra:
        metadata.update(extra)
    return metadata
