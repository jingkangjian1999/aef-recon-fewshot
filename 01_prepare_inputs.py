from __future__ import annotations

import shutil

import pandas as pd

from common import (
    BUDGETS,
    CACHE_DIR,
    DEVELOPMENT_BUDGETS,
    LASSO_FIT_CSV,
    LOG_DIR,
    MANIFEST_DIR,
    REPEATS,
    SAME_YEAR_EVAL_2023_CSV,
    TRAINING_POOL_2023_CSV,
    TRAIN_INDEX_DIR,
    TRANSFER_EVAL_2024_CSV,
    create_run_metadata,
    ensure_dir,
    infer_columns,
    load_dataset,
    save_dataframe,
    save_json,
    save_pickle,
)


def summarize_dataset(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    return dataframe.groupby(["fold_id", "crop_type"]).size().rename("count").reset_index().assign(dataset=dataset_name)


def generate_training_indices(train_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for fold_id in sorted(train_df["fold_id"].unique()):
        fold_df = train_df[train_df["fold_id"] == fold_id].copy()
        for budget in BUDGETS:
            per_class = budget // 2
            for repeat_id in REPEATS:
                sampled_parts = []
                sampling_seed = budget * 1000 + int(fold_id) * 100 + repeat_id
                for crop_type, class_df in fold_df.groupby("crop_type"):
                    sampled_parts.append(class_df.sample(n=per_class, random_state=sampling_seed, replace=False).copy())
                sampled_df = pd.concat(sampled_parts, ignore_index=True).sort_values(["crop_type", "row_id"]).reset_index(drop=True)
                sampled_df["train_fold"] = int(fold_id)
                sampled_df["budget"] = int(budget)
                sampled_df["repeat_id"] = int(repeat_id)
                sampled_df["sampling_seed"] = int(sampling_seed)
                out_path = TRAIN_INDEX_DIR / f"train2023_fold{int(fold_id)}_budget{int(budget)}_repeat{int(repeat_id):02d}.csv"
                save_dataframe(
                    sampled_df[["system:index", "row_id", "crop_type", "fold_id", "train_fold", "budget", "repeat_id", "sampling_seed"]],
                    out_path,
                )
                records.append(
                    {
                        "train_fold": int(fold_id),
                        "budget": int(budget),
                        "repeat_id": int(repeat_id),
                        "train_index_path": str(out_path),
                        "n_samples": int(len(sampled_df)),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    for directory in [MANIFEST_DIR, CACHE_DIR, TRAIN_INDEX_DIR, LOG_DIR]:
        ensure_dir(directory)

    manifest = infer_columns(LASSO_FIT_CSV)
    manifest["data_files"] = {
        "lasso_fit_csv": str(LASSO_FIT_CSV),
        "training_pool_2023_csv": str(TRAINING_POOL_2023_CSV),
        "same_year_eval_2023_csv": str(SAME_YEAR_EVAL_2023_CSV),
        "transfer_eval_2024_csv": str(TRANSFER_EVAL_2024_CSV),
    }
    manifest["budgets"] = BUDGETS
    manifest["development_budgets"] = DEVELOPMENT_BUDGETS
    manifest["repeats"] = REPEATS
    save_json(manifest, MANIFEST_DIR / "dataset_manifest.json")

    lasso_fit_df = load_dataset(LASSO_FIT_CSV)
    training_pool_2023_df = load_dataset(TRAINING_POOL_2023_CSV)
    same_year_eval_2023_df = load_dataset(SAME_YEAR_EVAL_2023_CSV)
    transfer_eval_2024_df = load_dataset(TRANSFER_EVAL_2024_CSV)

    save_pickle(lasso_fit_df, CACHE_DIR / "lasso_fit.pkl")
    save_pickle(training_pool_2023_df, CACHE_DIR / "training_pool_2023.pkl")
    save_pickle(same_year_eval_2023_df, CACHE_DIR / "same_year_eval_2023.pkl")
    save_pickle(transfer_eval_2024_df, CACHE_DIR / "transfer_eval_2024.pkl")

    summary_df = pd.concat(
        [
            summarize_dataset(lasso_fit_df, "lasso_fit_2023"),
            summarize_dataset(training_pool_2023_df, "training_pool_2023"),
            summarize_dataset(same_year_eval_2023_df, "same_year_eval_2023"),
            summarize_dataset(transfer_eval_2024_df, "transfer_eval_2024"),
        ],
        ignore_index=True,
    )
    save_dataframe(summary_df, MANIFEST_DIR / "dataset_fold_summary.csv")

    sampling_df = generate_training_indices(training_pool_2023_df)
    save_dataframe(sampling_df, MANIFEST_DIR / "training_index_manifest.csv")

    disk_total, disk_used, disk_free = shutil.disk_usage(MANIFEST_DIR.parent)
    save_json(
        create_run_metadata(
            "01_prepare_inputs.py",
            {
                "disk_total_gb": round(disk_total / 1024**3, 2),
                "disk_free_gb": round(disk_free / 1024**3, 2),
            },
        ),
        MANIFEST_DIR / "run_metadata.json",
    )


if __name__ == "__main__":
    main()
