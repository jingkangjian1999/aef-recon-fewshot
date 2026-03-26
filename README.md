# aef-recon-fewshot

A reproducible remote sensing experiment package for learning Sentinel-to-AEF mappings, selecting compact reconstructed features, and evaluating same-year and cross-year few-shot crop classification under spatial block validation.

## Scope

This package reproduces the core experiment pipeline:

- linear AEF mapping comparison on the 2023 fitting dataset
- reconstruction-model and retained-dimension selection for same-year evaluation
- baseline-specific `k` selection
- same-year baseline comparison at budget `50` `100`
- `2023 -> 2024` transfer baseline comparison at budget `50` `100`
- same-year sample-budget ablation (`10, 20, 50, 100, 200`)
- transfer sample-budget ablation (`10, 20, 50, 100, 200`)
- transfer-side model-and-dimension supplementary comparison
- paired Wilcoxon tests for `Lasso@20`
- final tabular summaries

## Input File Configuration

This package requires four logical inputs:

- one AEF fitting CSV
- one `2023` block-partitioned training CSV
- one `2023` evaluation CSV
- one `2024` evaluation CSV

The input paths are configured in:

- [common.py](/D:/AlphaEarth/返修实验/github/common.py)

If your data are stored in different locations, update the following entries in [common.py](/D:/AlphaEarth/返修实验/github/common.py):

- `LASSO_FIT_CSV`
- `TRAINING_POOL_2023_CSV`
- `resolve_evaluation_csv(year)`

By default, the package resolves these paths relative to the parent project directory. The same-year and transfer evaluation files are located through `resolve_evaluation_csv(year)`, so if your folder layout differs, that function is the place to modify.

## Environment

Recommended Python version:

- Python `3.11`

Required packages:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `joblib`

Install with:

```bash
pip install -r requirements.txt
```

## How to Run

Run the full pipeline with:

```bash
python run_all.py
```

You can also execute each stage separately:

1. `01_prepare_inputs.py`
2. `02_fit_mapping_models.py`
3. `03_select_model_k.py`
4. `04_select_baseline_k.py`
5. `05_run_same_year_baselines.py`
6. `06_run_transfer_baselines.py`
7. `07_run_budget_ablation.py`
8. `08_run_transfer_budget_ablation.py`
9. `09_run_model_k_transfer.py`
10. `10_run_lasso20_wilcoxon.py`
11. `11_make_tables.py`

## Output Structure

All generated outputs are written under `outputs/`:

- `outputs/01_manifest`
- `outputs/02_mapping_models`
- `outputs/03_model_k_selection`
- `outputs/04_baseline_k_selection`
- `outputs/05_same_year_baselines`
- `outputs/06_transfer_baselines`
- `outputs/07_budget_ablation`
- `outputs/08_transfer_budget_ablation`
- `outputs/09_model_k_transfer`
- `outputs/10_statistics`

## Experimental Protocol

### AEF Mapping

- Fitting data: `2023` fitting file in `lasso/`
- Candidate linear models: `Lasso`, `Ridge`, `Elastic Net`, `PLSRegression`, `MLR`
- Validation for mapping quality: grouped spatial validation using `fold_id`

### Same-Year Classification

- Training pool: `2023` sample file with block partitions
- Evaluation pool: `2023` evaluation sample file
- Spatial protocol: one `fold_id` is used for training, all other folds are used for evaluation
- Random repetition count per fold: `10`
- Total runs per budget: `5 folds x 10 repeats = 50`

### Transfer Classification

- Training pool: `2023` sample file with block partitions
- Evaluation pool: `2024` evaluation sample file
- Spatial protocol: one `fold_id` is used for training, all other folds in the target year are used for evaluation
- Random repetition count per fold: `10`
- Total runs per budget: `5 folds x 10 repeats = 50`

### Sample Budgets

- `10`
- `20`
- `50`
- `100`
- `200`

The budget is defined as the total number of labeled samples, with class-balanced sampling inside the selected training fold.

### Downstream Classifier

- Random Forest with `100` trees

## Key Result Files

After a successful run, the main result files are:

- `outputs/02_mapping_models/metrics/mapping_model_summary.csv`
- `outputs/03_model_k_selection/selected_model_k.json`
- `outputs/04_baseline_k_selection/selected_baseline_k.json`
- `outputs/05_same_year_baselines/metrics/same_year_baselines_summary.csv`
- `outputs/06_transfer_baselines/metrics/transfer_baselines_summary.csv`
- `outputs/07_budget_ablation/metrics/same_year_budget_ablation_summary.csv`
- `outputs/08_transfer_budget_ablation/metrics/transfer_budget_ablation_summary.csv`
- `outputs/09_model_k_transfer/metrics/model_k_transfer_summary.csv`
- `outputs/10_statistics/lasso20_wilcoxon.csv`
- `outputs/10_statistics/experiment_summary.md`

## Reproducibility Notes

- Sampling indices are generated once in `01_prepare_inputs.py` and reused by all later stages.
- All methods use the same training indices within the same `fold x budget x repeat` combination.
- The reconstruction model is selected only from the development budgets `50` and `100`.
- Baseline-specific `k` values are also selected only from the development budgets `50` and `100`.
- The chosen reconstruction model and selected baseline `k` values are frozen before the final same-year, transfer, and sample-budget experiments are executed.
