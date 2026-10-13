#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s29_difficulty_composition.py
=============================
Mechanism analysis (part 2): patient-level difficulty composition.
Question: do some inclusion criteria happen to select patients who are inherently easy to
classify (i.e. rarely misclassified)?

Avoiding circularity (hard constraint): difficulty is defined only from the patients' own
attributes, never from whether a baseline model got them right.
  - boundary_dist = |WAB-AQ - 50|      distance from the severe cut-off (label-derived only)
  - knn_disagree  = on standardized full-pool features, the fraction of each patient's 10
                    nearest neighbours with the opposite label (severe / non-severe)
                    (model-free local class overlap, no baseline model involved)
Per-patient model correctness (full-pool OOF) is used only in step 2 as a validation target,
to show that these difficulty scores do predict who gets misclassified; it never enters the
definition of difficulty.

Three steps:
  1. Compute two model-free difficulty scores for every patient in the full 226 pool.
  2. Validation: difficulty score -> per-patient misclassification / absolute error of the
     full-pool RF (binned error rate + AUC + correlation), showing the scores are informative.
  3. Composition: mean difficulty of the patients selected by each inclusion-criterion subset
     vs the full pool; then correlate subset scores with subset mean difficulty to see whether
     higher-scoring subsets systematically select easier patients.

Usage: python src/s29_difficulty_composition.py
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

import s18_crossed_grid as g

PKG = g.PKG
SUMM = os.path.join(PKG, "results", "summary")
RUNS = os.path.join(PKG, "results", "runs")
SUBSET_DIR = g.SUBSET_DIR
SEVERE_THR = 50.0
KNN = 10


def knn_disagreement(Xstd, ylab, k=KNN):
    """Fraction of opposite-label neighbours among each patient's k nearest neighbours
    (leave-self-out). Model-free local class overlap."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xstd)
    _, idx = nn.kneighbors(Xstd)
    nb = idx[:, 1:]                        # drop the patient itself
    return (ylab[nb] != ylab[:, None]).mean(axis=1)


def binned_rate(x, hit, edges, labels):
    out = []
    b = pd.cut(x, bins=edges, labels=False, include_lowest=True)
    for i, lab in enumerate(labels):
        m = (b == i)
        out.append((lab, int(m.sum()), float(hit[m].mean()) if m.sum() else np.nan))
    return out


def main():
    X, y_aq, keep = g.load_full_pool()
    aq = y_aq.values.astype(float)
    ids = np.array(X.index)
    ylab = (aq <= SEVERE_THR).astype(int)
    Xstd = StandardScaler().fit_transform(X.values)

    # ---- 1. model-free difficulty ----
    boundary_dist = np.abs(aq - SEVERE_THR)
    knn_dis = knn_disagreement(Xstd, ylab)
    pat = pd.DataFrame({"participant_id": ids, "wab_aq": aq, "severe": ylab,
                        "boundary_dist": boundary_dist, "knn_disagree": knn_dis})

    # ---- 2. Validation: difficulty -> per-patient correctness of the full-pool RF ----
    clf = pd.read_csv(os.path.join(RUNS, "clf_JHU_fine_RandomForest_default_patient_probs.csv"))
    clf["misclass"] = (clf["y_pred"] != clf["y_true"]).astype(int)
    clf["margin"] = (clf["y_prob_mean"] - 0.5).abs()
    reg = pd.read_csv(os.path.join(RUNS, "reg_JHU_fine_RandomForest_default_patient_pred.csv"))
    reg["abs_err"] = (reg["y_true"] - reg["y_pred_mean"]).abs()
    pat = pat.merge(clf[["participant_id", "misclass", "margin"]], on="participant_id", how="left")
    pat = pat.merge(reg[["participant_id", "abs_err"]], on="participant_id", how="left")
    pat.to_csv(os.path.join(SUMM, "patient_difficulty.csv"), index=False)

    mis = pat["misclass"].values.astype(float)
    print("===== Validation: does model-free difficulty predict classification errors? =====", flush=True)
    # Boundary distance: closer = harder; use -boundary_dist as the difficulty score
    auc_bd = roc_auc_score(mis, -pat["boundary_dist"].values)
    auc_kd = roc_auc_score(mis, pat["knn_disagree"].values)
    auc_both = roc_auc_score(mis, (-pat["boundary_dist"].rank() + pat["knn_disagree"].rank()).values)
    print(f"  AUC(boundary dist -> misclass)   = {auc_bd:.3f}", flush=True)
    print(f"  AUC(kNN disagreement -> misclass) = {auc_kd:.3f}", flush=True)
    print(f"  AUC(rank sum of both -> misclass) = {auc_both:.3f}", flush=True)
    print("  Misclassification rate by boundary-distance bin:", flush=True)
    for lab, n, rate in binned_rate(pat["boundary_dist"].values, mis,
                                    [-0.1, 15, 30, 60], ["<=15", "15-30", ">30"]):
        print(f"    |AQ-50| {lab:6s}  n={n:3d}  misclass rate={rate:.3f}", flush=True)
    print("  Misclassification rate by kNN-disagreement bin (tertiles):", flush=True)
    q = pat["knn_disagree"].quantile([1/3, 2/3]).values
    for lab, n, rate in binned_rate(pat["knn_disagree"].values, mis,
                                    [-0.001, q[0], q[1], 1.001], ["low", "mid", "high"]):
        print(f"    kNN disagreement {lab:3s}  n={n:3d}  misclass rate={rate:.3f}", flush=True)

    # Regression error vs difficulty (the regression score is r, which depends more on the
    # AQ range; correlations are reported here only as supporting evidence)
    ae = pat["abs_err"].values
    print("\n  Regression absolute-error correlations (supporting): "
          f"corr(abs err, AQ; higher = nearer top end)={np.corrcoef(ae, aq)[0,1]:+.3f}  "
          f"corr(abs err, kNN disagreement)={np.corrcoef(ae, knn_dis)[0,1]:+.3f}", flush=True)

    # ---- 3. Subset difficulty composition + correlation with score ----
    pat_idx = pat.set_index("participant_id")
    comp_rows = []
    for task, mn_csv, score_col in [
        ("classification", "matchedN_perm_classification.csv", "obs_score"),
        ("regression", "matchedN_perm_regression.csv", "obs_score")]:
        mn = pd.read_csv(os.path.join(SUMM, mn_csv))
        mn = mn[mn["model"] == "RandomForest"]
        for _, r in mn.iterrows():
            sids = [i for i in g.load_subset_ids(r["subset"]) if i in pat_idx.index]
            sub = pat_idx.loc[sids]
            comp_rows.append({
                "task": task, "subset": r["subset"], "n": len(sids),
                "score": round(float(r[score_col]), 4),
                "mean_boundary_dist": round(float(sub["boundary_dist"].mean()), 3),
                "frac_near15": round(float((sub["boundary_dist"] <= 15).mean()), 3),
                "mean_knn_disagree": round(float(sub["knn_disagree"].mean()), 4),
                "severe_prev": round(float(sub["severe"].mean()), 3),
            })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(SUMM, "subset_difficulty_composition.csv"), index=False)

    print("\n===== Subset difficulty composition vs score (do higher scores come from easier patients?) =====", flush=True)
    full_bd = float(pat["boundary_dist"].mean())
    full_kd = float(pat["knn_disagree"].mean())
    print(f"  Full-pool baseline: mean boundary dist={full_bd:.2f}  mean kNN disagreement={full_kd:.3f}", flush=True)
    for task in ["classification", "regression"]:
        t = comp[comp["task"] == task]
        # Exclude the exact same-patient anchor (teghipco) to look at the inclusion criteria themselves
        tc = t[t["subset"] != "teghipco_idlist"]
        r_bd = float(np.corrcoef(tc["score"], tc["mean_boundary_dist"])[0, 1])
        r_kd = float(np.corrcoef(tc["score"], tc["mean_knn_disagree"])[0, 1])
        print(f"  [{task}] subset score vs mean boundary dist r={r_bd:+.3f}; "
              f"vs mean kNN disagreement r={r_kd:+.3f}  (n={len(tc)} subsets, teghipco anchor excluded)", flush=True)
    print(f"\n[write] patient_difficulty.csv / subset_difficulty_composition.csv", flush=True)


if __name__ == "__main__":
    main()
