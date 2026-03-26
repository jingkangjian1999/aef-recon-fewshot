from __future__ import annotations

import json

import pandas as pd
from scipy.stats import wilcoxon

from common import (
    BASELINE_K_DIR,
    BUDGET_DIR,
    MAPPING_DIR,
    MODEL_K_TRANSFER_DIR,
    MODEL_SELECTION_DIR,
    SAME_YEAR_DIR,
    STATISTICS_DIR,
    TRANSFER_BUDGET_DIR,
    TRANSFER_DIR,
    create_run_metadata,
    ensure_dir,
    load_json,
    save_dataframe,
    save_json,
    save_text,
)


def compute_reconstructed_pairwise_tests(splitwise_df: pd.DataFrame, evaluation_name: str) -> pd.DataFrame:
    comparison_methods = ["raw_rf", "pca_rf", "pls_label_rf", "anova_rf", "mi_rf", "rf_importance_rf", "coral_rf", "tca_rf"]
    rows = []
    for budget in sorted(splitwise_df["budget"].unique()):
        budget_df = splitwise_df[splitwise_df["budget"] == budget].copy()
        left_df = budget_df[budget_df["method_name"] == "reconstructed_rf"].sort_values(["fold_id", "repeat_id"]).reset_index(drop=True)
        for method_name in comparison_methods:
            right_df = budget_df[budget_df["method_name"] == method_name].sort_values(["fold_id", "repeat_id"]).reset_index(drop=True)
            merged_df = left_df.merge(right_df, on=["evaluation", "fold_id", "budget", "repeat_id"], suffixes=("_left", "_right"))
            for metric in ["macro_f1", "oa"]:
                stat, p_value = wilcoxon(merged_df[f"{metric}_left"], merged_df[f"{metric}_right"])
                rows.append(
                    {
                        "evaluation": evaluation_name,
                        "budget": int(budget),
                        "left": "reconstructed_rf",
                        "right": method_name,
                        "metric": metric,
                        "n_pairs": int(len(merged_df)),
                        "mean_difference": float((merged_df[f"{metric}_left"] - merged_df[f"{metric}_right"]).mean()),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p_value),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dir(STATISTICS_DIR)

    mapping_summary_df = pd.read_csv(MAPPING_DIR / "metrics" / "mapping_model_summary.csv")
    model_k_same_year_summary_df = pd.read_csv(MODEL_SELECTION_DIR / "metrics" / "model_k_same_year_summary.csv")
    model_k_transfer_summary_df = pd.read_csv(MODEL_K_TRANSFER_DIR / "metrics" / "model_k_transfer_summary.csv")
    baseline_k_summary_df = pd.read_csv(BASELINE_K_DIR / "metrics" / "baseline_k_selection_summary.csv")
    same_year_summary_df = pd.read_csv(SAME_YEAR_DIR / "metrics" / "same_year_baselines_summary.csv")
    transfer_summary_df = pd.read_csv(TRANSFER_DIR / "metrics" / "transfer_baselines_summary.csv")
    same_year_budget_summary_df = pd.read_csv(BUDGET_DIR / "metrics" / "same_year_budget_ablation_summary.csv")
    same_year_budget_split_df = pd.read_csv(BUDGET_DIR / "metrics" / "same_year_budget_ablation_splitwise.csv")
    transfer_budget_summary_df = pd.read_csv(TRANSFER_BUDGET_DIR / "metrics" / "transfer_budget_ablation_summary.csv")
    transfer_budget_split_df = pd.read_csv(TRANSFER_BUDGET_DIR / "metrics" / "transfer_budget_ablation_splitwise.csv")

    selected_model_config = load_json(MODEL_SELECTION_DIR / "selected_model_k.json")
    selected_baseline_k = load_json(BASELINE_K_DIR / "selected_baseline_k.json")

    save_dataframe(mapping_summary_df, STATISTICS_DIR / "table_mapping_model_summary.csv")
    save_dataframe(model_k_same_year_summary_df, STATISTICS_DIR / "table_model_k_same_year.csv")
    save_dataframe(model_k_transfer_summary_df, STATISTICS_DIR / "table_model_k_transfer.csv")
    save_dataframe(baseline_k_summary_df, STATISTICS_DIR / "table_baseline_k_selection.csv")
    save_dataframe(pd.DataFrame([selected_baseline_k]), STATISTICS_DIR / "table_selected_baseline_k.csv")
    save_dataframe(same_year_summary_df, STATISTICS_DIR / "table_same_year_budget100.csv")
    save_dataframe(transfer_summary_df, STATISTICS_DIR / "table_transfer_budget100.csv")
    save_dataframe(same_year_budget_summary_df, STATISTICS_DIR / "table_same_year_budget_ablation.csv")
    save_dataframe(transfer_budget_summary_df, STATISTICS_DIR / "table_transfer_budget_ablation.csv")

    same_year_pairwise_df = compute_reconstructed_pairwise_tests(same_year_budget_split_df, "same_year")
    transfer_pairwise_df = compute_reconstructed_pairwise_tests(transfer_budget_split_df, "transfer")
    save_dataframe(same_year_pairwise_df, STATISTICS_DIR / "table_same_year_pairwise_wilcoxon.csv")
    save_dataframe(transfer_pairwise_df, STATISTICS_DIR / "table_transfer_pairwise_wilcoxon.csv")

    summary_lines = [
        "# Non-pure Experiment Package",
        "",
        "## Selected Reconstruction Configuration",
        f"- Selected model: {selected_model_config['selected_model_label']}",
        f"- Selected k: {selected_model_config['selected_k']}",
        f"- Selected Macro-F1: {selected_model_config['selected_macro_f1_mean']:.6f}",
        f"- Selected OA: {selected_model_config['selected_oa_mean']:.6f}",
        "",
        "## Selected Baseline k",
        *[f"- {method_name}: {selected_baseline_k[method_name]}" for method_name in sorted(selected_baseline_k)],
        "",
        "## Generated Tables",
        "- table_mapping_model_summary.csv",
        "- table_model_k_same_year.csv",
        "- table_model_k_transfer.csv",
        "- table_baseline_k_selection.csv",
        "- table_selected_baseline_k.csv",
        "- table_same_year_budget100.csv",
        "- table_transfer_budget100.csv",
        "- table_same_year_budget_ablation.csv",
        "- table_transfer_budget_ablation.csv",
        "- table_same_year_pairwise_wilcoxon.csv",
        "- table_transfer_pairwise_wilcoxon.csv",
        "- lasso20_wilcoxon.csv",
    ]
    save_text("\n".join(summary_lines), STATISTICS_DIR / "experiment_summary.md")
    save_json(
        create_run_metadata(
            "11_make_tables.py",
            {
                "selected_model_config": selected_model_config,
                "selected_baseline_k": selected_baseline_k,
            },
        ),
        STATISTICS_DIR / "run_metadata.json",
    )


if __name__ == "__main__":
    main()
