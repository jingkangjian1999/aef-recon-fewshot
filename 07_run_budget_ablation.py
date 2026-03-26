from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from common import BUDGETS, CACHE_DIR, BUDGET_DIR, PARALLEL_JOBS, REPEATS, create_run_metadata, ensure_dir, load_pickle, save_dataframe, save_json, summarize_grouped_metrics
from evaluation_utils import evaluate_baseline_same_year_split, load_reconstructed_dataset, load_selected_model_config


def main() -> None:
    for directory in [BUDGET_DIR, BUDGET_DIR / "metrics"]:
        ensure_dir(directory)

    train_df = load_pickle(CACHE_DIR / "training_pool_2023.pkl")
    val_df = load_pickle(CACHE_DIR / "same_year_eval_2023.pkl")
    selected_model = load_selected_model_config()["selected_model"]
    train_reconstructed_df = load_reconstructed_dataset("training_pool_2023", selected_model)
    val_reconstructed_df = load_reconstructed_dataset("same_year_eval_2023", selected_model)

    tasks = []
    for budget in BUDGETS:
        for fold_id in sorted(train_df["fold_id"].unique()):
            for repeat_id in REPEATS:
                tasks.append((train_df, val_df, train_reconstructed_df, val_reconstructed_df, budget, int(fold_id), repeat_id))

    result_rows = Parallel(n_jobs=PARALLEL_JOBS, prefer="threads", verbose=10)(
        delayed(evaluate_baseline_same_year_split)(*task) for task in tasks
    )
    splitwise_df = pd.DataFrame([row for rows in result_rows for row in rows])
    save_dataframe(splitwise_df, BUDGET_DIR / "metrics" / "same_year_budget_ablation_splitwise.csv")

    summary_df = summarize_grouped_metrics(splitwise_df, ["method_name", "method_label", "budget"])
    save_dataframe(summary_df, BUDGET_DIR / "metrics" / "same_year_budget_ablation_summary.csv")
    save_json(create_run_metadata("07_run_budget_ablation.py"), BUDGET_DIR / "run_metadata.json")


if __name__ == "__main__":
    main()
