"""Reproduce the cohort-selection experiments from public inputs (Python 3.11)."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "arc_lesion_image_benchmark/src"


def run(script, *args):
    subprocess.run([sys.executable, str(script), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare", help="download public inputs and construct features/cohorts")
    prepare.add_argument("--extra-atlases", action="store_true", help="also prepare the atlas-sweep inputs")
    grid = commands.add_parser("grid", help="train the 20-repeat crossed cohort/model grid")
    grid.add_argument("--tabpfn", action="store_true", help="include TabPFN in separate processes (full paper grid)")
    commands.add_parser("fixed", help="train RF/XGBoost on the full pool, then rescore and resample")
    commands.add_parser("statistics", help="variance decomposition and model-set sensitivity")
    commands.add_parser("figures", help="generate four figures and the fixed-prediction tables")
    args = parser.parse_args()
    if args.stage == "prepare":
        run(ROOT / "scripts/prepare_data.py", *(["--extra-atlases"] if args.extra_atlases else []))
    elif args.stage == "grid":
        for task in ["regression", "classification"]:
            run(SRC / "s18_crossed_grid.py", "--task", task)
            if args.tabpfn:
                run(SRC / "s18_crossed_grid.py", "--task", task, "--models", "TabPFN")
    elif args.stage == "fixed":
        for task in ["regression", "classification"]:
            if not (SRC.parent / f"results/runs/grid_{task}_fast.csv").exists():
                parser.error("Run 'python reproduce.py grid' first; its full-pool scores verify the fixed predictions.")
        run(SRC / "s31_frozen_prediction_control.py", "train")
        run(SRC / "s31_frozen_prediction_control.py", "analyze")
        run(SRC / "s31_resampling_tie_fix.py")
    elif args.stage == "statistics":
        for task in ["regression", "classification"]:
            run(SRC / "s19_variance_decomp.py", "--task", task)
            if (SRC.parent / f"results/runs/grid_{task}_TabPFN.csv").exists():
                run(SRC / "s19b_variance_decomp_with_tabpfn.py", "--task", task)
        run(SRC / "s19c_mae_competitive.py")
        run(ROOT / "scripts/model_set_sensitivity.py")
    else:
        run(SRC / "fig_manuscript_rewrite_20260922.py")
        run(ROOT / "scripts/build_manuscript_tables.py")


if __name__ == "__main__":
    main()
