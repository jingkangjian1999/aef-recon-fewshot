from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from common import CACHE_DIR, PARALLEL_JOBS, REPEATS, TRANSFER_DIR, create_run_metadata, ensure_dir, load_pickle, save_dataframe, save_json, summarize_grouped_metrics
from evaluation_utils import evaluate_baseline_transfer_split, load_reconstructed_dataset, load_selected_model_config


TRANSFER_BUDGETS = [100]


def main() -> None:
    for directory in [TRANSFER_DIR, TRANSFER_DIR / "metrics"]:
        ensure_dir(directory)

    train_df = load_pickle(CACHE_DIR / "training_pool_2023.pkl")
    val_df = load_pickle(CACHE_DIR / "transfer_eval_2024.pkl")
    selected_model = load_selected_model_config()["selected_model"]
    train_reconstructed_df = load_reconstructed_dataset("training_pool_2023", selected_model)
    val_reconstructed_df = load_reconstructed_dataset("transfer_eval_2024", selected_model)

    tasks = []
    for budget in TRANSFER_BUDGETS:
        for fold_id in sorted(train_df["fold_id"].unique()):
            for repeat_id in REPEATS:
                tasks.append((train_df, val_df, train_reconstructed_df, val_reconstructed_df, budget, int(fold_id), repeat_id))

    result_rows = Parallel(n_jobs=PARALLEL_JOBS, prefer="threads", verbose=10)(
        delayed(evaluate_baseline_transfer_split)(*task) for task in tasks
    )
    splitwise_df = pd.DataFrame([row for rows in result_rows for row in rows])
    save_dataframe(splitwise_df, TRANSFER_DIR / "metrics" / "transfer_baselines_splitwise.csv")

    summary_df = summarize_grouped_metrics(splitwise_df, ["method_name", "method_label", "budget"])
    save_dataframe(summary_df, TRANSFER_DIR / "metrics" / "transfer_baselines_summary.csv")
    save_json(create_run_metadata("06_run_transfer_baselines.py"), TRANSFER_DIR / "run_metadata.json")


if __name__ == "__main__":
    main()
