#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s20_matchedN_permutation.py
===========================
Matched-N composition control (formal permutation test). Tests whether the score
differences between subsets can be explained by smaller sample size alone.

Design (fixed before running):
  For each standard subset (size n_k), draw B random subsamples from Full, stratified
  by the WAB-AQ histogram -- bin edges include 50 (severe boundary) and 90 (ceiling),
  so N, the severe prior and the target SD are matched at once.
  Each random subsample retrains the whole pipeline from scratch (fresh folds, scaler,
  SMOTE and model; no fit or fold indices are shared).
  Build an empirical null distribution -> report the subset metric's two-sided empirical
  p, the composition effect (subset score - null mean), and whether the score falls
  outside the null band.
  For aq_le90 the matched draws are automatically restricted to the y <= 90 support,
  which also controls for range restriction.

Decision rule (fixed before running): alpha = 0.05; outside the band (p < 0.05) = the
difference is due to composition, not sample size; results inside the band are reported
as they are.
RNG: the sampling RNG and the CV RNG are independent and fixed; each draw is retrained
independently.

Usage: python src/s20_matchedN_permutation.py --task regression --B 1000
"""
from __future__ import annotations
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Reuse s18's pipeline / models / cell evaluation so that the same fixed pipeline is used
import s18_crossed_grid as g

PKG = g.PKG
SUBSET_DIR = g.SUBSET_DIR
SUMM = os.path.join(PKG, "results", "summary")
BIN_EDGES = [-0.1, 25, 50, 70, 90, 100.1]   # 50 = severe boundary, 90 = ceiling boundary
CV_SEED = 2024
DRAW_BASE = 7000


def bin_counts(y_aq, edges):
    b = pd.cut(y_aq, bins=edges, labels=False, include_lowest=True)
    return b, pd.Series(b).value_counts().sort_index()


def eval_subset(X, y_aq, model, task, seed=CV_SEED):
    runner = g.run_cell_regression if task == "regression" else g.run_cell_classification
    return runner(X, y_aq, model, seed)["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    models = args.models or (["RandomForest", "Ridge"] if args.task == "regression"
                             else ["RandomForest", "LogReg"])

    X_full, y_full, keep = g.load_full_pool()
    full_bins, _ = bin_counts(y_full, BIN_EDGES)
    full_bins = pd.Series(full_bins.values, index=X_full.index)
    # Members of each bin in Full (keys forced to int so pd.cut float codes match the int keys of need)
    bin_members = {int(b): full_bins.index[full_bins == b].tolist()
                   for b in sorted(full_bins.dropna().unique())}

    man = pd.read_csv(os.path.join(SUBSET_DIR, "subset_manifest.csv"))
    # Run the control for standard subsets only (excluding the full baseline); the exact
    # Teghipco cohort is also run (as an anchor)
    targets = [s for s in man["subset"].tolist() if s != "full"]

    rng = np.random.default_rng(DRAW_BASE)
    rows = []
    t0 = time.time()
    for sub in targets:
        ids = [i for i in g.load_subset_ids(sub) if i in X_full.index]
        Xs, ys = X_full.loc[ids], y_full.loc[ids]
        # Bin histogram of the subset
        sub_bins, sub_hist = bin_counts(ys, BIN_EDGES)
        need = {int(b): int(c) for b, c in sub_hist.items()}
        for model in models:
            obs = eval_subset(Xs, ys, model, args.task)
            # Null distribution: B random draws matching the subset histogram
            null = []
            for _ in range(args.B):
                draw_ids = []
                for b, c in need.items():
                    pool_b = bin_members.get(b, [])
                    if c > len(pool_b):
                        c = len(pool_b)
                    draw_ids += list(rng.choice(pool_b, size=c, replace=False))
                Xd, yd = X_full.loc[draw_ids], y_full.loc[draw_ids]
                null.append(eval_subset(Xd, yd, model, args.task))
            null = np.array(null)
            null_mean = float(null.mean())
            lo, hi = float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))
            # Two-sided empirical p
            p_two = 2 * min(np.mean(null >= obs), np.mean(null <= obs))
            p_two = float(min(1.0, p_two))
            comp = obs - null_mean
            outside = not (lo <= obs <= hi)
            rows.append({"task": args.task, "subset": sub, "model": model,
                         "n": len(ids), "obs_score": round(obs, 4),
                         "null_mean": round(null_mean, 4),
                         "null_ci_lo": round(lo, 4), "null_ci_hi": round(hi, 4),
                         "composition_effect": round(comp, 4),
                         "p_two_sided": round(p_two, 4),
                         "outside_band": outside, "B": args.B})
            print(f"  [{sub:18s}|{model:13s}] obs={obs:.3f} null={null_mean:.3f}"
                  f"[{lo:.3f},{hi:.3f}] comp={comp:+.3f} p={p_two:.3f} "
                  f"{'OUT' if outside else 'in'} ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(SUMM, exist_ok=True)
    out.to_csv(os.path.join(SUMM, f"matchedN_perm_{args.task}.csv"), index=False)
    n_out = int(out["outside_band"].sum())
    print(f"\n[write] matchedN_perm_{args.task}.csv  "
          f"{n_out}/{len(out)} (subset x model) outside the null band (significant composition effect)", flush=True)


if __name__ == "__main__":
    main()
