#!/usr/bin/env python
"""
s9_chronicity_sensitivity.py -- chronicity sensitivity analysis (RF regression + classification)

Background
----------
ARC describes itself as a "chronic stroke repository", and we follow that definition of
chronic. However, wab_days in participants.tsv (days from stroke to WAB assessment) is
fairly short for some participants in the modelling cohort (minimum 42 days). This script
checks whether the headline scores hold when only a more chronic subset is kept.

What it does
------------
Joins each feature table (JHU fine / arterial coarse) with wab_days from participants.tsv
and builds two subsets:
  * wab_days >= 180
  * wab_days >= 365
On each subset RandomForest is re-run with exactly the same protocol as the main analysis:
  * Regression: reuses run_default / run_tuned from s5_nested_tuning_regression (5x10
          repeated stratified CV, in-fold StandardScaler->RF, tuned uses an inner
          GridSearchCV). Metrics: Pearson r / MAE / RMSE / R2.
  * Classification: reuses run_outer from s3b_classification_nested_dump (5x10 repeated
          stratified CV, in-fold StandardScaler->SMOTE->RF). Metrics: balanced-acc / AUC /
          F1 / MCC.
The full cohort (no wab_days filter) is run the same way as an anchor for direct comparison.

Leakage guards
--------------
1) The wab_days filter is a pure row selection applied before CV: participants are kept
   only by "wab_days >= threshold". It does not touch the target, uses no feature/label
   statistic and is not a data-dependent transform -> no leakage.
2) The filtered subset goes through the original fold logic unchanged: scaler/SMOTE are
   still fit_transform on the inner training folds only, and the outer test fold is only
   transformed/predicted. The protocol is identical to the main analysis (functions are
   imported and reused, not copied).
3) wab_days is used only for row selection and never enters the feature matrix X (the
   column is dropped right after filtering).

Usage
-----
  # SMOKE (RF only, tiny grid, few repeats, checks that it runs end to end):
  python src/s9_chronicity_sensitivity.py --smoke

  # Full run (default + tuned, JHU + arterial, >=180 + >=365 + full-cohort anchor):
  python src/s9_chronicity_sensitivity.py
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent  # repository root
SRC = PKG / "src"
sys.path.insert(0, str(SRC))

# Reuse the core functions of the main scripts directly (same fold/leakage logic, not copied)
import s5_nested_tuning_regression as reg   # noqa: E402
import s3b_classification_nested_dump as clf  # noqa: E402

PARTICIPANTS = ROOT / "datasets" / "arc_ds004884" / "participants.tsv"
TABLES = {
    "JHU_fine": PKG / "data" / "jhu156_features_merged.tsv",
    "arterial_coarse": PKG / "data" / "ArterialAtlas156_features_merged.tsv",
}
THRESHOLDS = [None, 180, 365]   # None = full-cohort anchor
SEED = reg.SEED


def load_wab_days():
    """participant_id -> wab_days (numeric)."""
    pt = pd.read_csv(PARTICIPANTS, sep="\t")
    wd = pd.to_numeric(pt["wab_days"], errors="coerce")
    return pd.Series(wd.values, index=pt["participant_id"].values, name="wab_days")


def filtered_ids(table_path, wab_days, min_days):
    """Return the participant_ids in this table with a non-NA target and wab_days >= min_days
    (table order preserved). min_days=None means no filtering (full cohort). The filter uses
    only the wab_days column and never touches features or labels."""
    df = pd.read_csv(table_path, sep="\t")
    ids = df["participant_id"]
    keep_target = df["wab_aq"].notna().values   # as in the main analysis: drop unlabelled participants first
    if min_days is None:
        mask = keep_target
    else:
        d = ids.map(wab_days).values
        mask = keep_target & (d >= min_days)
    return ids[mask].tolist()


# ---------------- Regression (s5 protocol) ----------------
def run_regression(name, table_path, keep_ids, do_tuned, n_repeats, smoke):
    """Reuse load_xy / run_default / run_tuned / metrics_with_ci from s5 and run
    RandomForest on the filtered subset. X/y are loaded by reg.load_xy, then subset by keep_ids."""
    X, y, feat = reg.load_xy(str(table_path))   # same cleaning as main analysis (drop NA target -> fillna 0 -> drop sparse columns)
    keep = [i for i in keep_ids if i in X.index]
    Xs = X.loc[keep]
    ys = y.loc[keep]
    outer = reg.RepeatedStratifiedKFold(
        n_splits=reg.N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    out = {}

    t0 = time.time()
    yt_d, yp_d, _ = reg.run_default("RandomForest", Xs, ys, outer, n_repeats)
    m_d, ci_d = reg.metrics_with_ci(yt_d, yp_d)
    out["default"] = (m_d, ci_d, time.time() - t0)
    print(f"    [reg/{name}] DEFAULT r={m_d['pearson_r']:.3f} "
          f"{reg.fmt_ci(ci_d['pearson_r'])} MAE={m_d['mae']:.2f} "
          f"R2={m_d['r2']:.3f}", flush=True)

    if do_tuned:
        t1 = time.time()
        yt_t, yp_t, _, _ = reg.run_tuned("RandomForest", Xs, ys, outer, n_repeats, smoke)
        m_t, ci_t = reg.metrics_with_ci(yt_t, yp_t)
        out["tuned"] = (m_t, ci_t, time.time() - t1)
        print(f"    [reg/{name}] TUNED   r={m_t['pearson_r']:.3f} "
              f"{reg.fmt_ci(ci_t['pearson_r'])} MAE={m_t['mae']:.2f} "
              f"R2={m_t['r2']:.3f}", flush=True)
    return len(Xs), Xs.shape[1], out


# ---------------- Classification (s3b protocol) ----------------
def run_classification(name, table_path, keep_ids, do_tuned, smoke):
    """Reuse load_xy / run_outer / bootstrap_ci from s3b and run RandomForest
    (in-fold SMOTE) on the filtered subset."""
    X, y, wab, pids, feat = clf.load_xy(str(table_path))  # numpy arrays + list of pids
    pid_idx = {p: k for k, p in enumerate(pids)}
    sel = [pid_idx[i] for i in keep_ids if i in pid_idx]
    sel = np.asarray(sorted(sel))
    Xs, ys, wabs = X[sel], y[sel], wab[sel]
    pids_s = [pids[k] for k in sel]
    out = {}

    for tuned in ([False, True] if do_tuned else [False]):
        tag = "tuned" if tuned else "default"
        t0 = time.time()
        summary, _, agg = clf.run_outer(
            "RandomForest", Xs, ys, wabs, pids_s, tuned, smoke=smoke, verbose=False)
        ci = clf.bootstrap_ci(agg["y_true"], agg["y_pred"], agg["y_prob"])
        out[tag] = (summary, ci, time.time() - t0)
        ba, _ = summary["balanced_acc"]
        auc, _ = summary["auc"]
        print(f"    [clf/{name}] {tag:7s} bAcc={clf.fmt_ci(ba, ci['balanced_acc'])} "
              f"AUC={auc:.3f} F1={summary['f1_severe'][0]:.3f} "
              f"MCC={summary['mcc'][0]:.3f}", flush=True)
    return len(ys), int(ys.sum()), Xs.shape[1], out


def main():
    ap = argparse.ArgumentParser(description="Chronicity sensitivity: RF regression + classification on wab_days subsets")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: tiny grid + few repeats (classification still 5x10, small grid)")
    ap.add_argument("--no-tuned", action="store_true", help="Run the default configuration only")
    args = ap.parse_args()

    do_tuned = not args.no_tuned
    n_repeats_reg = 2 if args.smoke else reg.N_REPEATS
    wab_days = load_wab_days()

    rows = []
    for gran, table in TABLES.items():
        for thr in THRESHOLDS:
            subset = "full" if thr is None else f"wab_days_ge_{thr}"
            keep = filtered_ids(table, wab_days, thr)
            print(f"\n=== {gran} | {subset} | n={len(keep)} ===", flush=True)

            # Regression
            n_reg, nfeat_reg, reg_out = run_regression(
                gran, table, keep, do_tuned, n_repeats_reg, args.smoke)
            # Classification
            n_clf, n_sev, nfeat_clf, clf_out = run_classification(
                gran, table, keep, do_tuned, args.smoke)

            for cfg in (["default", "tuned"] if do_tuned else ["default"]):
                row = {
                    "granularity": gran,
                    "subset": subset,
                    "min_wab_days": (0 if thr is None else thr),
                    "config": cfg,
                    "n": n_clf,
                    "n_severe": n_sev,
                    "severe_prevalence": round(n_sev / n_clf, 4),
                    "n_features": nfeat_clf,
                }
                # Regression
                if cfg in reg_out:
                    m, ci, sec = reg_out[cfg]
                    row.update({
                        "reg_pearson_r": round(m["pearson_r"], 4),
                        "reg_r_CI": reg.fmt_ci(ci["pearson_r"]),
                        "reg_mae": round(m["mae"], 3),
                        "reg_rmse": round(m["rmse"], 3),
                        "reg_r2": round(m["r2"], 4),
                        "reg_seconds": round(sec, 1),
                    })
                # Classification
                if cfg in clf_out:
                    summ, ci, sec = clf_out[cfg]
                    row.update({
                        "clf_balanced_acc": round(summ["balanced_acc"][0], 4),
                        "clf_bAcc_CI": clf.fmt_ci(summ["balanced_acc"][0], ci["balanced_acc"]),
                        "clf_auc": round(summ["auc"][0], 4),
                        "clf_auc_CI": clf.fmt_ci(summ["auc"][0], ci["auc"]),
                        "clf_f1_severe": round(summ["f1_severe"][0], 4),
                        "clf_mcc": round(summ["mcc"][0], 4),
                        "clf_seconds": round(sec, 1),
                    })
                rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = PKG / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"chronicity_sensitivity{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsummary -> {out_path}", flush=True)
    # Short comparison printout
    cols = ["granularity", "subset", "n", "n_severe", "config",
            "reg_pearson_r", "clf_balanced_acc"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
