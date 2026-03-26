from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from common import (
    BASELINE_K_DIR,
    BASELINE_K_METHODS,
    BASELINE_LABELS,
    MAPPING_DIR,
    MODEL_SELECTION_DIR,
    MODEL_LABELS,
    classification_metrics,
    coral_align,
    find_training_rows,
    fit_standardizer,
    get_aef_columns,
    get_feature_columns,
    load_json,
    load_pickle,
    load_sampling_indices,
    make_rf,
    predict_linear_matrix,
    tca_transform,
)


@lru_cache(maxsize=1)
def load_selected_model_config() -> dict:
    config = load_json(MODEL_SELECTION_DIR / "selected_model_k.json")
    ranking_df = pd.read_csv(MAPPING_DIR / "metrics" / f"{config['selected_model']}_dimension_ranking.csv")
    config["selected_dimensions"] = ranking_df["target"].tolist()[: int(config["selected_k"])]
    return config


@lru_cache(maxsize=1)
def load_selected_baseline_k_config() -> dict[str, int]:
    return load_json(BASELINE_K_DIR / "selected_baseline_k.json")


def load_reconstructed_dataset(dataset_key: str, model_name: str) -> pd.DataFrame:
    path = MAPPING_DIR / "features" / f"{dataset_key}_{model_name}_reconstructed.csv"
    return pd.read_csv(path)


def encode_labels(train_y: np.ndarray, val_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    train_numeric = encoder.fit_transform(train_y)
    val_numeric = encoder.transform(val_y)
    return train_numeric, val_numeric


def _feature_names(prefix: str, dimensions: list[str]) -> list[str]:
    return [f"{prefix}{dimension}" for dimension in dimensions]


def build_model_k_features(
    train_reconstructed_df: pd.DataFrame,
    val_reconstructed_df: pd.DataFrame,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    ranked_dimensions: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected_dimensions = ranked_dimensions[:k]
    selected_columns = _feature_names("recon_", selected_dimensions)
    train_x = train_reconstructed_df.iloc[train_rows][selected_columns].to_numpy(dtype=float)
    val_x = val_reconstructed_df.iloc[val_rows][selected_columns].to_numpy(dtype=float)
    return train_x, val_x, selected_dimensions


def build_baseline_feature_views(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_reconstructed_df: pd.DataFrame,
    val_reconstructed_df: pd.DataFrame,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    baseline_k_config: dict[str, int],
) -> dict[str, dict]:
    feature_columns = get_feature_columns()
    aef_columns = get_aef_columns()
    selected_model_config = load_selected_model_config()
    reconstructed_dimensions = selected_model_config["selected_dimensions"]
    reconstructed_columns = _feature_names("recon_", reconstructed_dimensions)

    train_y = train_df.iloc[train_rows]["crop_type"].to_numpy()
    val_y = val_df.iloc[val_rows]["crop_type"].to_numpy()
    train_y_numeric, _ = encode_labels(train_y, val_y)

    raw_train = train_df.iloc[train_rows][feature_columns].to_numpy(dtype=float)
    raw_val = val_df.iloc[val_rows][feature_columns].to_numpy(dtype=float)
    aef_train = train_df.iloc[train_rows][aef_columns].to_numpy(dtype=float)
    aef_val = val_df.iloc[val_rows][aef_columns].to_numpy(dtype=float)
    reconstructed_train = train_reconstructed_df.iloc[train_rows][reconstructed_columns].to_numpy(dtype=float)
    reconstructed_val = val_reconstructed_df.iloc[val_rows][reconstructed_columns].to_numpy(dtype=float)

    methods: dict[str, dict] = {
        "raw_rf": {
            "train_x": raw_train,
            "val_x": raw_val,
            "metadata": {"n_features": raw_train.shape[1]},
        },
        "reconstructed_rf": {
            "train_x": reconstructed_train,
            "val_x": reconstructed_val,
            "metadata": {"n_features": reconstructed_train.shape[1], "selected_dimensions": "|".join(reconstructed_dimensions)},
        },
        "aef64_rf": {
            "train_x": aef_train,
            "val_x": aef_val,
            "metadata": {"n_features": aef_train.shape[1]},
        },
    }

    raw_train_scaled, raw_val_scaled, _ = fit_standardizer(raw_train, raw_val)

    pca_k = max(1, min(int(baseline_k_config["pca_rf"]), raw_train.shape[0] - 1, raw_train.shape[1]))
    pca = PCA(n_components=pca_k, random_state=0)
    methods["pca_rf"] = {
        "train_x": pca.fit_transform(raw_train_scaled),
        "val_x": pca.transform(raw_val_scaled),
        "metadata": {"n_features": pca_k, "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())},
    }

    pls_k = max(1, min(int(baseline_k_config["pls_label_rf"]), raw_train.shape[0] - 1, raw_train.shape[1]))
    pls = PLSRegression(n_components=pls_k, scale=False)
    pls.fit(raw_train_scaled, train_y_numeric.astype(float))
    methods["pls_label_rf"] = {
        "train_x": pls.transform(raw_train_scaled),
        "val_x": pls.transform(raw_val_scaled),
        "metadata": {"n_features": pls_k},
    }

    anova_k = max(1, min(int(baseline_k_config["anova_rf"]), raw_train.shape[1]))
    anova_scores, _ = f_classif(raw_train, train_y_numeric)
    anova_index = np.argsort(np.nan_to_num(anova_scores, nan=-np.inf))[::-1][:anova_k]
    methods["anova_rf"] = {
        "train_x": raw_train[:, anova_index],
        "val_x": raw_val[:, anova_index],
        "metadata": {"n_features": len(anova_index), "selected_raw_features": "|".join([feature_columns[index] for index in anova_index])},
    }

    mi_k = max(1, min(int(baseline_k_config["mi_rf"]), raw_train.shape[1]))
    mi_scores = mutual_info_classif(raw_train, train_y_numeric, random_state=0)
    mi_index = np.argsort(np.nan_to_num(mi_scores, nan=-np.inf))[::-1][:mi_k]
    methods["mi_rf"] = {
        "train_x": raw_train[:, mi_index],
        "val_x": raw_val[:, mi_index],
        "metadata": {"n_features": len(mi_index), "selected_raw_features": "|".join([feature_columns[index] for index in mi_index])},
    }

    rf_selector_k = max(1, min(int(baseline_k_config["rf_importance_rf"]), raw_train.shape[1]))
    rf_selector = make_rf(seed=0)
    rf_selector.fit(raw_train, train_y)
    rf_importance = rf_selector.feature_importances_
    rf_index = np.argsort(rf_importance)[::-1][:rf_selector_k]
    methods["rf_importance_rf"] = {
        "train_x": raw_train[:, rf_index],
        "val_x": raw_val[:, rf_index],
        "metadata": {"n_features": len(rf_index), "selected_raw_features": "|".join([feature_columns[index] for index in rf_index])},
    }

    coral_train, coral_val, coral_meta = coral_align(raw_train_scaled, raw_val_scaled)
    methods["coral_rf"] = {
        "train_x": coral_train,
        "val_x": coral_val,
        "metadata": {"n_features": coral_train.shape[1], "coral_transform_trace": float(np.trace(coral_meta["transform"]))},
    }

    tca_k = max(1, min(int(baseline_k_config["tca_rf"]), raw_train.shape[1]))
    tca_train, tca_val, tca_meta = tca_transform(raw_train_scaled, raw_val_scaled, dim=tca_k, reg=1.0)
    methods["tca_rf"] = {
        "train_x": tca_train,
        "val_x": tca_val,
        "metadata": {"n_features": tca_train.shape[1], "tca_eigval_sum": float(np.sum(tca_meta["eigvals"]))},
    }

    return methods


def build_single_baseline_method(
    method_name: str,
    k: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
) -> dict:
    feature_columns = get_feature_columns()
    train_y = train_df.iloc[train_rows]["crop_type"].to_numpy()
    val_y = val_df.iloc[val_rows]["crop_type"].to_numpy()
    train_y_numeric, _ = encode_labels(train_y, val_y)

    raw_train = train_df.iloc[train_rows][feature_columns].to_numpy(dtype=float)
    raw_val = val_df.iloc[val_rows][feature_columns].to_numpy(dtype=float)
    raw_train_scaled, raw_val_scaled, _ = fit_standardizer(raw_train, raw_val)

    if method_name == "pca_rf":
        effective_k = max(1, min(int(k), raw_train.shape[0] - 1, raw_train.shape[1]))
        model = PCA(n_components=effective_k, random_state=0)
        return {
            "train_x": model.fit_transform(raw_train_scaled),
            "val_x": model.transform(raw_val_scaled),
            "metadata": {"n_features": effective_k, "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum())},
        }
    if method_name == "pls_label_rf":
        effective_k = max(1, min(int(k), raw_train.shape[0] - 1, raw_train.shape[1]))
        model = PLSRegression(n_components=effective_k, scale=False)
        model.fit(raw_train_scaled, train_y_numeric.astype(float))
        return {
            "train_x": model.transform(raw_train_scaled),
            "val_x": model.transform(raw_val_scaled),
            "metadata": {"n_features": effective_k},
        }
    if method_name == "anova_rf":
        effective_k = max(1, min(int(k), raw_train.shape[1]))
        scores, _ = f_classif(raw_train, train_y_numeric)
        selected_index = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1][:effective_k]
        return {
            "train_x": raw_train[:, selected_index],
            "val_x": raw_val[:, selected_index],
            "metadata": {"n_features": len(selected_index), "selected_raw_features": "|".join([feature_columns[index] for index in selected_index])},
        }
    if method_name == "mi_rf":
        effective_k = max(1, min(int(k), raw_train.shape[1]))
        scores = mutual_info_classif(raw_train, train_y_numeric, random_state=0)
        selected_index = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1][:effective_k]
        return {
            "train_x": raw_train[:, selected_index],
            "val_x": raw_val[:, selected_index],
            "metadata": {"n_features": len(selected_index), "selected_raw_features": "|".join([feature_columns[index] for index in selected_index])},
        }
    if method_name == "rf_importance_rf":
        effective_k = max(1, min(int(k), raw_train.shape[1]))
        selector = make_rf(seed=0)
        selector.fit(raw_train, train_y)
        scores = selector.feature_importances_
        selected_index = np.argsort(scores)[::-1][:effective_k]
        return {
            "train_x": raw_train[:, selected_index],
            "val_x": raw_val[:, selected_index],
            "metadata": {"n_features": len(selected_index), "selected_raw_features": "|".join([feature_columns[index] for index in selected_index])},
        }
    if method_name == "tca_rf":
        effective_k = max(1, min(int(k), raw_train.shape[1]))
        transformed_train, transformed_val, metadata = tca_transform(raw_train_scaled, raw_val_scaled, dim=effective_k, reg=1.0)
        return {
            "train_x": transformed_train,
            "val_x": transformed_val,
            "metadata": {"n_features": transformed_train.shape[1], "tca_eigval_sum": float(np.sum(metadata["eigvals"]))},
        }
    raise ValueError(f"Unsupported method for k selection: {method_name}")


def evaluate_model_k_same_year_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_reconstructed_df: pd.DataFrame,
    val_reconstructed_df: pd.DataFrame,
    model_name: str,
    ranked_dimensions: list[str],
    budget: int,
    fold_id: int,
    repeat_id: int,
) -> list[dict]:
    index_df = load_sampling_indices(fold_id, budget, repeat_id)
    train_rows = find_training_rows(train_df, index_df)
    val_rows = np.where(val_df["fold_id"].to_numpy() != fold_id)[0]
    train_y = train_df.iloc[train_rows]["crop_type"].to_numpy()
    val_y = val_df.iloc[val_rows]["crop_type"].to_numpy()

    rows = []
    for k in [5, 10, 20, 30, 64]:
        train_x, val_x, selected_dimensions = build_model_k_features(train_reconstructed_df, val_reconstructed_df, train_rows, val_rows, ranked_dimensions, k)
        seed = budget * 1000 + fold_id * 100 + repeat_id
        classifier = make_rf(seed)
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(val_x)
        metrics = classification_metrics(val_y, predictions)
        rows.append(
            {
                "evaluation": "same_year",
                "year": 2023,
                "fold_id": fold_id,
                "budget": budget,
                "repeat_id": repeat_id,
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "k": k,
                "selected_dimensions": "|".join(selected_dimensions),
                **metrics,
            }
        )
    return rows


def evaluate_model_k_transfer_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_reconstructed_df: pd.DataFrame,
    val_reconstructed_df: pd.DataFrame,
    model_name: str,
    ranked_dimensions: list[str],
    budget: int,
    fold_id: int,
    repeat_id: int,
) -> list[dict]:
    index_df = load_sampling_indices(fold_id, budget, repeat_id)
    train_rows = find_training_rows(train_df, index_df)
    val_rows = np.where(val_df["fold_id"].to_numpy() != fold_id)[0]
    train_y = train_df.iloc[train_rows]["crop_type"].to_numpy()
    val_y = val_df.iloc[val_rows]["crop_type"].to_numpy()

    rows = []
    for k in [5, 10, 20, 30, 64]:
        train_x, val_x, selected_dimensions = build_model_k_features(train_reconstructed_df, val_reconstructed_df, train_rows, val_rows, ranked_dimensions, k)
        seed = budget * 1000 + fold_id * 100 + repeat_id
        classifier = make_rf(seed)
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(val_x)
        metrics = classification_metrics(val_y, predictions)
        rows.append(
            {
                "evaluation": "transfer",
                "source_year": 2023,
                "target_year": 2024,
                "fold_id": fold_id,
                "budget": budget,
                "repeat_id": repeat_id,
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "k": k,
                "selected_dimensions": "|".join(selected_dimensions),
                **metrics,
            }
        )
    return rows


def evaluate_baseline_same_year_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_reconstructed_df: pd.DataFrame,
    val_reconstructed_df: pd.DataFrame,
    budget: int,
    fold_id: int,
    repeat_id: int,
) -> list[dict]:
    baseline_k_config = load_selected_baseline_k_config()
    index_df = load_sampling_indices(fold_id, budget, repeat_id)
    train_rows = find_training_rows(train_df, index_df)
    val_rows = np.where(val_df["fold_id"].to_numpy() != fold_id)[0]
    train_y = train_df.iloc[train_rows]["crop_type"].to_numpy()
    val_y = val_df.iloc[val_rows]["crop_type"].to_numpy()
    methods = build_baseline_feature_views(train_df, val_df, train_reconstructed_df, val_reconstructed_df, train_rows, val_rows, baseline_k_config)

    rows = []
    for method_name, method_bundle in methods.items():
        seed = budget * 1000 + fold_id * 100 + repeat_id
        classifier = make_rf(seed)
        classifier.fit(method_bundle["train_x"], train_y)
        predictions = classifier.predict(method_bundle["val_x"])
        metrics = classification_metrics(val_y, predictions)
        rows.append(
            {
                "evaluation": "same_year",
                "year": 2023,
                "fold_id": fold_id,
                "budget": budget,
                "repeat_id": repeat_id,
                "method_name": method_name,
                "method_label": BASELINE_LABELS[method_name],
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                **metrics,
                "metadata_json": json.dumps(method_bundle["metadata"], ensure_ascii=False),
            }
        )
    return rows


def evaluate_baseline_transfer_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_reconstructed_df: pd.DataFrame,
    val_reconstructed_df: pd.DataFrame,
    budget: int,
    fold_id: int,
    repeat_id: int,
) -> list[dict]:
    baseline_k_config = load_selected_baseline_k_config()
    index_df = load_sampling_indices(fold_id, budget, repeat_id)
    train_rows = find_training_rows(train_df, index_df)
    val_rows = np.where(val_df["fold_id"].to_numpy() != fold_id)[0]
    train_y = train_df.iloc[train_rows]["crop_type"].to_numpy()
    val_y = val_df.iloc[val_rows]["crop_type"].to_numpy()
    methods = build_baseline_feature_views(train_df, val_df, train_reconstructed_df, val_reconstructed_df, train_rows, val_rows, baseline_k_config)

    rows = []
    for method_name, method_bundle in methods.items():
        seed = budget * 1000 + fold_id * 100 + repeat_id
        classifier = make_rf(seed)
        classifier.fit(method_bundle["train_x"], train_y)
        predictions = classifier.predict(method_bundle["val_x"])
        metrics = classification_metrics(val_y, predictions)
        rows.append(
            {
                "evaluation": "transfer",
                "source_year": 2023,
                "target_year": 2024,
                "fold_id": fold_id,
                "budget": budget,
                "repeat_id": repeat_id,
                "method_name": method_name,
                "method_label": BASELINE_LABELS[method_name],
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                **metrics,
                "metadata_json": json.dumps(method_bundle["metadata"], ensure_ascii=False),
            }
        )
    return rows
