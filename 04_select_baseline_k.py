from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (
    BASELINE_K_DIR,
    BASELINE_K_METHODS,
    DEVELOPMENT_BUDGETS,
    K_VALUES,
    PARALLEL_JOBS,
    REPEATS,
    create_run_metadata,
    ensure_dir,
    find_training_rows,
    load_pickle,
    load_sampling_indices,
    make_rf,
    save_dataframe,
    save_json,
)
from evaluation_utils import build_single_baseline_method


def evaluate_one_split(method_name: str, k: int, train_df: pd.DataFrame, val_df: pd.DataFrame, budget: int, fold_id: int, repeat_id: int) -> dict:
    index_df = load_sampling_indices(fold_id, budget, repeat_id)
    train_rows = find_training_rows(train_df, index_df)
    val_rows = np.where(val_df["fold_id"].to_numpy() != fold_id)[0]
    train_y = train_df.iloc[train_rows]["crop_type"].to_numpy()
    val_y = val_df.iloc[val_rows]["crop_type"].to_numpy()
    method_bundle = build_single_baseline_method(method_name, k, train_df, val_df, train_rows, val_rows)
    seed = budget * 1000 + fold_id * 100 + repeat_id
    classifier = make_rf(seed)
    classifier.fit(method_bundle["train_x"], train_y)
    predictions = classifier.predict(method_bundle["val_x"])
    from common import classification_metrics

    return {
        "method_name": method_name,
        "k": int(k),
        "budget": int(budget),
        "fold_id": int(fold_id),
        "repeat_id": int(repeat_id),
        **classification_metrics(val_y, predictions),
    }


def select_k(summary_df: pd.DataFrame, method_name: str) -> int:
    method_df = summary_df[summary_df["method_name"] == method_name].copy().sort_values(["macro_f1_mean", "oa_mean"], ascending=[False, False]).reset_index(drop=True)
    best_row = method_df.iloc[0]
    margin = max(float(best_row["macro_f1_sem"]), 0.001)
    candidate_df = method_df[method_df["macro_f1_mean"] >= best_row["macro_f1_mean"] - margin].copy()
    candidate_df = candidate_df.sort_values(["k", "macro_f1_mean", "oa_mean"], ascending=[True, False, False])
    return int(candidate_df.iloc[0]["k"])


def main() -> None:
    for directory in [BASELINE_K_DIR, BASELINE_K_DIR / "metrics"]:
        ensure_dir(directory)

    train_df = load_pickle(BASELINE_K_DIR.parent / "01_manifest" / "cache" / "training_pool_2023.pkl")
    val_df = load_pickle(BASELINE_K_DIR.parent / "01_manifest" / "cache" / "same_year_eval_2023.pkl")

    tasks = []
    for method_name in BASELINE_K_METHODS:
        for k in K_VALUES:
            for budget in DEVELOPMENT_BUDGETS:
                for fold_id in sorted(train_df["fold_id"].unique()):
                    for repeat_id in REPEATS:
                        tasks.append((method_name, k, train_df, val_df, budget, int(fold_id), repeat_id))

    result_rows = Parallel(n_jobs=PARALLEL_JOBS, prefer="threads", verbose=10)(
        delayed(evaluate_one_split)(*task) for task in tasks
    )
    splitwise_df = pd.DataFrame(result_rows)
    save_dataframe(splitwise_df, BASELINE_K_DIR / "metrics" / "baseline_k_selection_splitwise.csv")

    summary_df = (
        splitwise_df.groupby(["method_name", "k"])
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
    save_dataframe(summary_df, BASELINE_K_DIR / "metrics" / "baseline_k_selection_summary.csv")

    selected_config = {method_name: select_k(summary_df, method_name) for method_name in BASELINE_K_METHODS}
    save_json(selected_config, BASELINE_K_DIR / "selected_baseline_k.json")
    save_json(create_run_metadata("04_select_baseline_k.py"), BASELINE_K_DIR / "run_metadata.json")


if __name__ == "__main__":
    main()
