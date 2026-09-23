"""Aggregate per-model LOO predictions into the final summary + report.

Reads every results/runs/<model>_loo_predictions.csv, recomputes pooled metrics
and bootstrap CIs, writes results/summary/benchmark_loo.csv and
reports/benchmark_loo.md. Decoupled from run_benchmark.py so a slow model
(TabPFN) can be run on its own and folded in afterwards.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as config_mod
import data as data_mod
import evaluate as ev


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    cfg = config_mod.load_config()
    X, y, _ = data_mod.load_data(cfg)
    ds = data_mod.describe(X, y)

    rows = []
    for path in sorted(glob.glob(os.path.join(cfg["output"]["runs_dir"], "*_loo_predictions.csv"))):
        name = os.path.basename(path).replace("_loo_predictions.csv", "")
        d = pd.read_csv(path)
        m = ev.regression_metrics(d["y_true"], d["y_pred"])
        ci = ev.bootstrap_ci(d["y_true"].to_numpy(), d["y_pred"].to_numpy(),
                             cfg["bootstrap"]["n"], cfg["seed"])
        rows.append({
            "model": name,
            "pearson_r": round(m["pearson_r"], 3),
            "pearson_r_CI": fmt_ci(ci["pearson_r"]),
            "mae": round(m["mae"], 2),
            "mae_CI": fmt_ci(ci["mae"]),
            "rmse": round(m["rmse"], 2),
            "r2": round(m["r2"], 3),
            "r2_CI": fmt_ci(ci["r2"]),
        })
    summary = pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)

    summary_path = os.path.join(cfg["output"]["summary_dir"], "benchmark_loo.csv")
    summary.to_csv(summary_path, index=False)

    bl = cfg.get("baselines", {})
    lines = [
        "# ARC WAB-AQ Regression Benchmark (leave-one-out)\n",
        f"- Data: `merged_artery_participants.tsv` ({ds['n_patients']} patients, {ds['n_features']} features)",
        f"- Target: WAB-AQ {ds['target_min']:.0f}-{ds['target_max']:.0f}, mean {ds['target_mean']:.1f} ± {ds['target_std']:.1f}",
        "- Protocol: leave-one-out, StandardScaler fit inside each fold (no leakage), "
        f"seed={cfg['seed']}, TabPFN device={cfg['device']}",
        f"- Published baseline on this table (LOO): " + ", ".join(f"{k} r={v}" for k, v in bl.items()),
        "",
        summary.to_markdown(index=False),
        "",
    ]
    report_path = os.path.join(cfg["output"]["reports_dir"], "benchmark_loo.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(summary.to_string(index=False))
    print(f"\nSummary -> {summary_path}\nReport  -> {report_path}")


if __name__ == "__main__":
    main()
