"""Entry point: run the ARC WAB-AQ regression benchmark.

Usage (from anywhere, with the data-analysis env):
    PYENV_VERSION=data-analysis python src/run_benchmark.py
    PYENV_VERSION=data-analysis python src/run_benchmark.py --models SVR_rbf TabPFN
    PYENV_VERSION=data-analysis python src/run_benchmark.py --protocol repeated_kfold
    PYENV_VERSION=data-analysis python src/run_benchmark.py --dry-run

Outputs:
    results/runs/<model>_loo_predictions.csv   per-patient y_true/y_pred
    results/summary/benchmark_<protocol>.csv   one metric row per model
    reports/benchmark_<protocol>.md            human-readable report
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd

# Allow running as `python src/run_benchmark.py` (no package install).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as config_mod
import data as data_mod
import evaluate as ev
from pipelines import build_pipeline


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    ap = argparse.ArgumentParser(description="ARC WAB-AQ regression benchmark")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--protocol", default=None, choices=["loo", "repeated_kfold"])
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--device", default=None, help="override TabPFN device")
    ap.add_argument("--dry-run", action="store_true", help="print plan and exit")
    return ap.parse_args()


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def run_loo(cfg, X, y, models, device):
    rows = []
    for name in models:
        t0 = time.time()
        y_true, y_pred = ev.leave_one_out(
            lambda: build_pipeline(name, cfg["seed"], device), X, y
        )
        m = ev.regression_metrics(y_true, y_pred)
        ci = ev.bootstrap_ci(y_true, y_pred, cfg["bootstrap"]["n"], cfg["seed"])
        elapsed = time.time() - t0

        pred_df = pd.DataFrame(
            {"participant_id": X.index, "y_true": y_true, "y_pred": y_pred}
        )
        pred_path = os.path.join(
            cfg["output"]["runs_dir"], f"{name}_loo_predictions.csv"
        )
        pred_df.to_csv(pred_path, index=False)

        rows.append({
            "model": name,
            "pearson_r": m["pearson_r"],
            "pearson_r_CI": fmt_ci(ci["pearson_r"]),
            "pearson_p": m["pearson_p"],
            "mae": m["mae"],
            "mae_CI": fmt_ci(ci["mae"]),
            "rmse": m["rmse"],
            "rmse_CI": fmt_ci(ci["rmse"]),
            "r2": m["r2"],
            "r2_CI": fmt_ci(ci["r2"]),
            "seconds": round(elapsed, 1),
        })
        print(f"  {name:14s} r={m['pearson_r']:.3f} {fmt_ci(ci['pearson_r'])}  "
              f"MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}  "
              f"({elapsed:.1f}s)")
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def run_kfold(cfg, X, y, models, device):
    ns, nr = cfg["cv"]["kfold_splits"], cfg["cv"]["kfold_repeats"]
    rows = []
    for name in models:
        t0 = time.time()
        per_repeat = ev.repeated_kfold(
            lambda: build_pipeline(name, cfg["seed"], device),
            X, y, ns, nr, cfg["seed"],
        )
        elapsed = time.time() - t0
        agg = {}
        for key in ("pearson_r", "mae", "rmse", "r2"):
            vals = np.array([d[key] for d in per_repeat], dtype=float)
            agg[key + "_mean"] = float(vals.mean())
            agg[key + "_std"] = float(vals.std(ddof=0))
        rows.append({
            "model": name,
            "pearson_r": f"{agg['pearson_r_mean']:.3f} ± {agg['pearson_r_std']:.3f}",
            "mae": f"{agg['mae_mean']:.2f} ± {agg['mae_std']:.2f}",
            "rmse": f"{agg['rmse_mean']:.2f} ± {agg['rmse_std']:.2f}",
            "r2": f"{agg['r2_mean']:.3f} ± {agg['r2_std']:.3f}",
            "pearson_r_sort": agg["pearson_r_mean"],
            "seconds": round(elapsed, 1),
        })
        print(f"  {name:14s} r={agg['pearson_r_mean']:.3f}±{agg['pearson_r_std']:.3f}  "
              f"MAE={agg['mae_mean']:.2f}  RMSE={agg['rmse_mean']:.2f}  "
              f"R2={agg['r2_mean']:.3f}  ({elapsed:.1f}s)")
    df = pd.DataFrame(rows).sort_values("pearson_r_sort", ascending=False)
    return df.drop(columns=["pearson_r_sort"])


def write_report(cfg, summary, protocol, ds_info, device):
    base = cfg["data"]["path"].split("derived_features_arc_demo/")[-1]
    lines = []
    lines.append(f"# ARC WAB-AQ Regression Benchmark ({protocol})\n")
    lines.append(f"- Data: `{base}`  ({ds_info['n_patients']} patients, "
                 f"{ds_info['n_features']} features)")
    lines.append(f"- Target: WAB-AQ continuous, range "
                 f"{ds_info['target_min']:.1f}–{ds_info['target_max']:.1f}, "
                 f"mean {ds_info['target_mean']:.1f} ± {ds_info['target_std']:.1f}")
    lines.append(f"- Protocol: {protocol}; StandardScaler fit inside each fold "
                 f"(no leakage); seed={cfg['seed']}; TabPFN device={device}")
    bl = cfg.get("baselines", {})
    if bl:
        bl_str = ", ".join(f"{k} r={v}" for k, v in bl.items())
        lines.append(f"- Published baseline on this table (leave-one-out): {bl_str}")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    report_path = os.path.join(
        cfg["output"]["reports_dir"], f"benchmark_{protocol}.md"
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return report_path


def main():
    args = parse_args()
    cfg = config_mod.load_config(args.config)
    if args.protocol:
        cfg["cv"]["protocol"] = args.protocol
    if args.device:
        cfg["device"] = args.device
    device = cfg["device"]
    protocol = cfg["cv"]["protocol"]
    models = args.models if args.models else cfg["models"]

    set_global_seed(cfg["seed"])
    X, y, feat = data_mod.load_data(cfg)
    ds_info = data_mod.describe(X, y)

    print(f"Dataset: {ds_info['n_patients']} patients x {ds_info['n_features']} "
          f"features | target WAB-AQ {ds_info['target_min']:.0f}-"
          f"{ds_info['target_max']:.0f}")
    print(f"Protocol: {protocol} | models: {models} | seed: {cfg['seed']}")

    if args.dry_run:
        print("[dry-run] features:", feat)
        print("[dry-run] exiting before fitting.")
        return

    print(f"\nRunning {protocol} ...")
    if protocol == "loo":
        summary = run_loo(cfg, X, y, models, device)
    else:
        summary = run_kfold(cfg, X, y, models, device)

    summary_path = os.path.join(
        cfg["output"]["summary_dir"], f"benchmark_{protocol}.csv"
    )
    summary.to_csv(summary_path, index=False)
    report_path = write_report(cfg, summary, protocol, ds_info, device)

    print(f"\nSummary -> {summary_path}")
    print(f"Report  -> {report_path}")


if __name__ == "__main__":
    main()
