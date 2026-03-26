from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (
    CACHE_DIR,
    DEVELOPMENT_BUDGETS,
    K_VALUES,
    MODEL_LABELS,
    MODEL_NAMES,
    MODEL_PREFERENCE,
    MODEL_SELECTION_DIR,
    MAPPING_DIR,
    PARALLEL_JOBS,
    REPEATS,
    create_run_metadata,
    ensure_dir,
    save_dataframe,
    save_json,
)
from evaluation_utils import evaluate_model_k_same_year_split, load_reconstructed_dataset


def select_model(summary_df: pd.DataFrame) -> str:
    model_best_df = (
        summary_df.sort_values(["model_name", "macro_f1_mean", "oa_mean"], ascending=[True, False, False])
        .groupby("model_name", as_index=False)
        .first()
    )
    overall_best = model_best_df["macro_f1_mean"].max()
    overall_best_sem = model_best_df.loc[model_best_df["macro_f1_mean"].idxmax(), "macro_f1_sem"]
    margin = max(float(overall_best_sem), 0.004)
    candidate_df = model_best_df[model_best_df["macro_f1_mean"] >= overall_best - margin].copy()
    candidate_df["model_rank"] = candidate_df["model_name"].map({name: index for index, name in enumerate(MODEL_PREFERENCE)})
    candidate_df = candidate_df.sort_values(["model_rank", "macro_f1_mean", "oa_mean"], ascending=[True, False, False])
    return str(candidate_df.iloc[0]["model_name"])


def select_k(summary_df: pd.DataFrame, model_name: str) -> dict:
    model_df = summary_df[summary_df["model_name"] == model_name].copy().sort_values(["macro_f1_mean", "oa_mean"], ascending=[False, False]).reset_index(drop=True)
    best_row = model_df.iloc[0]
    margin = max(float(best_row["macro_f1_sem"]), 0.001)
    candidate_df = model_df[model_df["macro_f1_mean"] >= best_row["macro_f1_mean"] - margin].copy()
    candidate_df = candidate_df.sort_values(["k", "macro_f1_mean", "oa_mean"], ascending=[True, False, False])
    return candidate_df.iloc[0].to_dict()


def main() -> None:
    for directory in [MODEL_SELECTION_DIR, MODEL_SELECTION_DIR / "metrics"]:
        ensure_dir(directory)

    train_df = pd.read_pickle(CACHE_DIR / "training_pool_2023.pkl")
    val_df = pd.read_pickle(CACHE_DIR / "same_year_eval_2023.pkl")

    tasks = []
    rankings = {}
    for model_name in MODEL_NAMES:
        ranking_df = pd.read_csv(MAPPING_DIR / "metrics" / f"{model_name}_dimension_ranking.csv")
        rankings[model_name] = ranking_df["target"].tolist()
        train_reconstructed_df = load_reconstructed_dataset("training_pool_2023", model_name)
        val_reconstructed_df = load_reconstructed_dataset("same_year_eval_2023", model_name)
        for budget in DEVELOPMENT_BUDGETS:
            for fold_id in sorted(train_df["fold_id"].unique()):
                for repeat_id in REPEATS:
                    tasks.append((train_df, val_df, train_reconstructed_df, val_reconstructed_df, model_name, rankings[model_name], budget, int(fold_id), repeat_id))

    result_rows = Parallel(n_jobs=PARALLEL_JOBS, prefer="threads", verbose=10)(
        delayed(evaluate_model_k_same_year_split)(*task) for task in tasks
    )
    splitwise_df = pd.DataFrame([row for rows in result_rows for row in rows])
    save_dataframe(splitwise_df, MODEL_SELECTION_DIR / "metrics" / "model_k_same_year_splitwise.csv")

    summary_df = (
        splitwise_df.groupby(["model_name", "model_label", "k"])
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
    summary_df["oa_sem"] = summary_df["oa_std"] / np.sqrt(summary_df["n_runs"])
    summary_df["macro_f1_sem"] = summary_df["macro_f1_std"] / np.sqrt(summary_df["n_runs"])
    summary_df["aa_sem"] = summary_df["aa_std"] / np.sqrt(summary_df["n_runs"])
    summary_df["kappa_sem"] = summary_df["kappa_std"] / np.sqrt(summary_df["n_runs"])
    save_dataframe(summary_df, MODEL_SELECTION_DIR / "metrics" / "model_k_same_year_summary.csv")

    selected_model = select_model(summary_df)
    selected_k_row = select_k(summary_df, selected_model)
    ranking_df = pd.read_csv(MAPPING_DIR / "metrics" / f"{selected_model}_dimension_ranking.csv")
    selected_dimensions = ranking_df["target"].tolist()[: int(selected_k_row["k"])]
    selected_config = {
        "selected_model": selected_model,
        "selected_model_label": MODEL_LABELS[selected_model],
        "selected_k": int(selected_k_row["k"]),
        "selected_macro_f1_mean": float(selected_k_row["macro_f1_mean"]),
        "selected_oa_mean": float(selected_k_row["oa_mean"]),
        "development_budgets": DEVELOPMENT_BUDGETS,
        "repeats": REPEATS,
        "selected_dimensions": selected_dimensions,
        "selection_rule": "Model selected within max(one standard error, 0.4 percentage points) of the best Macro-F1 with model simplicity preference; k selected as the smallest value within max(one standard error, 0.1 percentage points) of the best k inside the selected model.",
    }
    save_json(selected_config, MODEL_SELECTION_DIR / "selected_model_k.json")
    save_json(create_run_metadata("03_select_model_k.py"), MODEL_SELECTION_DIR / "run_metadata.json")


if __name__ == "__main__":
    main()
