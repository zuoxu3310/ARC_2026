#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s30_mechanism_partition.py
==========================
Falsification check: split "why does cohort selection raise scores" into (A) range restriction
[known] vs (B) feature-space separability [possibly new].
Single key question: after controlling for how widely severity is spread (aq_sd etc. = range
restriction), does feature-space complexity (n3_1nn / nearest-neighbour separability / volume
structure) still contribute independently to the score?

Method (on the 300 random subsamples from s28):
  - Hierarchical R^2: add predictors block by block and record each block's incremental R^2
    (= its independent explanatory power after controlling for the earlier blocks).
      block1 range             = aq_sd, aq_iqr            (range restriction)
      block2 prior/size        = + severe_prev, n          (severe prior + sample size)
      block3 feature structure = + n3_1nn, mean_logvol, sd_logvol, mean_bdist, frac_near15
  - Partial correlation: after controlling for block1 (range), the partial correlation of each
    "feature structure" variable with the score.
Decision: if the block3 incremental R^2 and the n3_1nn partial correlation remain significant ->
      feature-space separability is a lever independent of range (new);
      if they are close to zero -> "selecting easy patients" is only range restriction in another
      form (known effect, not claimed as new).

Usage: python src/s30_mechanism_partition.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMM = os.path.join(PKG, "results", "summary")

BLOCK1 = ["aq_sd", "aq_iqr"]                                   # range restriction
BLOCK2 = ["severe_prev", "n"]                                  # prior + sample size
BLOCK3 = ["mean_logvol", "sd_logvol", "mean_bdist", "frac_near15"]  # feature / boundary structure
N3 = "n3_1nn"                                                  # classification only: nearest-neighbour separability in feature space


def r2(Z, y):
    return float(LinearRegression().fit(Z, y).score(Z, y))


def partial_corr(x, y, Z):
    """Partial correlation of x and y controlling for Z = correlation of their residuals on Z."""
    Z = np.asarray(Z, float)
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = y - LinearRegression().fit(Z, y).predict(Z)
    return float(np.corrcoef(rx, ry)[0, 1])


def hier(df, score_col, has_n3):
    blocks = [("range (aq_sd,aq_iqr)", BLOCK1),
              ("+prior+size (severe_prev,n)", BLOCK2),
              ("+feature/boundary structure", BLOCK3 + ([N3] if has_n3 else []))]
    preds_all = BLOCK1 + BLOCK2 + BLOCK3 + ([N3] if has_n3 else [])
    sub = df.dropna(subset=preds_all + [score_col])
    y = sub[score_col].values
    used, prev = [], 0.0
    print(f"\n===== {score_col}: hierarchical R^2 (n={len(sub)} draws) =====", flush=True)
    for name, blk in blocks:
        used += blk
        Z = StandardScaler().fit_transform(sub[used].values)
        cur = r2(Z, y)
        print(f"  {name:28s} cumulative R^2={cur:.3f}  incremental dR^2={cur-prev:+.3f}", flush=True)
        prev = cur
    # partial correlation: control for block1 (range)
    Zc = StandardScaler().fit_transform(sub[BLOCK1].values)
    print(f"  -- partial correlations after controlling for range (block1) --", flush=True)
    for p in (BLOCK3 + ([N3] if has_n3 else [])):
        pc = partial_corr(sub[p].values.astype(float), y, Zc)
        print(f"     {p:14s} partial r={pc:+.3f}", flush=True)


def main():
    df = pd.read_csv(os.path.join(SUMM, "cohort_explainers_draws.csv"))
    hier(df, "clf_score", has_n3=True)
    hier(df, "reg_score", has_n3=False)

    # 8 real subsets (teghipco anchor excluded): subset score vs nearest-neighbour separability,
    # controlling for boundary distance (range proxy)
    comp = pd.read_csv(os.path.join(SUMM, "subset_difficulty_composition.csv"))
    print("\n===== real inclusion-criterion subsets (excl. teghipco): is NN separability independent of boundary/range? =====", flush=True)
    for task in ["classification", "regression"]:
        t = comp[(comp["task"] == task) & (comp["subset"] != "teghipco_idlist")]
        y = t["score"].values
        kd = t["mean_knn_disagree"].values
        bd = t["mean_boundary_dist"].values
        raw = float(np.corrcoef(y, kd)[0, 1])
        pc = partial_corr(kd, y, bd.reshape(-1, 1))
        print(f"  [{task}] score ~ kNN label disagreement: raw r={raw:+.3f}  partial r given mean boundary distance={pc:+.3f}  (n={len(t)})", flush=True)


if __name__ == "__main__":
    main()
