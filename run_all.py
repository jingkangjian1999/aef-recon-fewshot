from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "01_prepare_inputs.py",
    "02_fit_mapping_models.py",
    "03_select_model_k.py",
    "04_select_baseline_k.py",
    "05_run_same_year_baselines.py",
    "06_run_transfer_baselines.py",
    "07_run_budget_ablation.py",
    "08_run_transfer_budget_ablation.py",
    "09_run_model_k_transfer.py",
    "10_run_lasso20_wilcoxon.py",
    "11_make_tables.py",
]


def main() -> None:
    package_root = Path(__file__).resolve().parent
    for script_name in SCRIPT_ORDER:
        script_path = package_root / script_name
        print(f"[run_all] Running {script_name}")
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(package_root.parent))


if __name__ == "__main__":
    main()
