from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import CACHE_DIR, DEVELOPMENT_BUDGETS, MODEL_K_TRANSFER_DIR, MODEL_NAMES, PARALLEL_JOBS, REPEATS, create_run_metadata, ensure_dir, save_dataframe, save_json
from evaluation_utils import evaluate_model_k_transfer_split, load_reconstructed_dataset


def main() -> None:
    for directory in [MODEL_K_TRANSFER_DIR, MODEL_K_TRANSFER_DIR / "metrics"]:
        ensure_dir(directory)

    train_df = pd.read_pickle(CACHE_DIR / "training_pool_2023.pkl")
    val_df = pd.read_pickle(CACHE_DIR / "transfer_eval_2024.pkl")

    tasks = []
    for model_name in MODEL_NAMES:
        ranking_df = pd.read_csv(MODEL_K_TRANSFER_DIR.parent / "02_mapping_models" / "metrics" / f"{model_name}_dimension_ranking.csv")
        ranked_dimensions = ranking_df["target"].tolist()
        train_reconstructed_df = load_reconstructed_dataset("training_pool_2023", model_name)
        val_reconstructed_df = load_reconstructed_dataset("transfer_eval_2024", model_name)
        for budget in DEVELOPMENT_BUDGETS:
            for fold_id in sorted(train_df["fold_id"].unique()):
                for repeat_id in REPEATS:
                    tasks.append((train_df, val_df, train_reconstructed_df, val_reconstructed_df, model_name, ranked_dimensions, budget, int(fold_id), repeat_id))

    result_rows = Parallel(n_jobs=PARALLEL_JOBS, prefer="threads", verbose=10)(
        delayed(evaluate_model_k_transfer_split)(*task) for task in tasks
    )
    splitwise_df = pd.DataFrame([row for rows in result_rows for row in rows])
    save_dataframe(splitwise_df, MODEL_K_TRANSFER_DIR / "metrics" / "model_k_transfer_splitwise.csv")

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
    for metric in ["oa", "macro_f1", "aa", "kappa"]:
        summary_df[f"{metric}_sem"] = summary_df[f"{metric}_std"] / np.sqrt(summary_df["n_runs"])
    save_dataframe(summary_df, MODEL_K_TRANSFER_DIR / "metrics" / "model_k_transfer_summary.csv")
    save_json(create_run_metadata("09_run_model_k_transfer.py"), MODEL_K_TRANSFER_DIR / "run_metadata.json")


if __name__ == "__main__":
    main()
