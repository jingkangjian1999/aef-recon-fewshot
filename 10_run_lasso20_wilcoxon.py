from __future__ import annotations

import pandas as pd
from scipy.stats import wilcoxon

from common import MODEL_SELECTION_DIR, STATISTICS_DIR, create_run_metadata, ensure_dir, save_dataframe, save_json


COMPARISONS = [
    ("lasso", 20, "mlr", 20),
    ("lasso", 20, "ridge", 20),
    ("lasso", 20, "pls", 20),
    ("lasso", 20, "elastic_net", 20),
    ("lasso", 20, "lasso", 30),
]


def run_comparison(dataframe: pd.DataFrame, left_model: str, left_k: int, right_model: str, right_k: int, budget: int | None) -> list[dict]:
    work_df = dataframe.copy()
    if budget is not None:
        work_df = work_df[work_df["budget"] == budget].copy()

    left_df = work_df[(work_df["model_name"] == left_model) & (work_df["k"] == left_k)].copy()
    right_df = work_df[(work_df["model_name"] == right_model) & (work_df["k"] == right_k)].copy()
    merged_df = left_df.merge(
        right_df,
        on=["evaluation", "year", "fold_id", "budget", "repeat_id"],
        suffixes=("_left", "_right"),
    ).sort_values(["year", "fold_id", "budget", "repeat_id"]).reset_index(drop=True)

    records = []
    for metric in ["macro_f1", "oa"]:
        stat, p_value = wilcoxon(merged_df[f"{metric}_left"], merged_df[f"{metric}_right"])
        records.append(
            {
                "budget_scope": "all" if budget is None else int(budget),
                "left": f"{left_model}@{left_k}",
                "right": f"{right_model}@{right_k}",
                "metric": metric,
                "n_pairs": int(len(merged_df)),
                "mean_left": float(merged_df[f"{metric}_left"].mean()),
                "mean_right": float(merged_df[f"{metric}_right"].mean()),
                "mean_difference": float((merged_df[f"{metric}_left"] - merged_df[f"{metric}_right"]).mean()),
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
            }
        )
    return records


def main() -> None:
    ensure_dir(STATISTICS_DIR)
    input_df = pd.read_csv(MODEL_SELECTION_DIR / "metrics" / "model_k_same_year_splitwise.csv")
    input_df["evaluation"] = "same_year"

    rows = []
    for left_model, left_k, right_model, right_k in COMPARISONS:
        rows.extend(run_comparison(input_df, left_model, left_k, right_model, right_k, budget=None))
        for budget in sorted(input_df["budget"].unique()):
            rows.extend(run_comparison(input_df, left_model, left_k, right_model, right_k, budget=int(budget)))

    result_df = pd.DataFrame(rows)
    save_dataframe(result_df, STATISTICS_DIR / "lasso20_wilcoxon.csv")
    save_json(create_run_metadata("10_run_lasso20_wilcoxon.py"), STATISTICS_DIR / "lasso20_wilcoxon_metadata.json")


if __name__ == "__main__":
    main()
