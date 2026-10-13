#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s28_cohort_explainers.py
========================
Mechanism analysis (part 1): cohort-level "distribution -> score" explanatory model.
Question: why does the choice of cohort move the score so much? Scores are regressed on
cohort-structure indicators that do not depend on whether the baseline model is right,
to measure which lever raises the score and how much variance each one explains.

Avoiding circularity (hard constraint): every explanatory variable is a patient attribute
(label- or feature-derived); baseline-model correctness is never used to define a variable.
  - n              sample size
  - severe_prev    severe prior (fraction with AQ<=50)
  - aq_sd          severity spread (WAB-AQ SD)  -> the range-restriction pathway
  - aq_iqr         severity interquartile range
  - mean_bdist     mean distance from the cut-off, mean(|AQ-50|)
  - frac_near15    fraction within 15 points of the severe cut-off (boundary density)
  - mean_logvol    mean lesion volume (log1p)
  - sd_logvol      spread of lesion volume (log1p)
  - n3_1nn (classification only) 1-NN opposite-label rate (model-free data complexity,
                   not a baseline model)

Method: draw B random subsamples of varying size and composition from the full 226 pool
(random weights on |AQ-50| spread the draws over the indicator space); run the same fixed
pipeline (s18) on each to get the RandomForest regression r and classification balanced-acc;
then regress the scores on the indicators above (standardized beta + univariate R2 + RF
permutation importance + full-model R2).
Finally the 10 real inclusion-criterion subsets are overlaid on the same explanatory
surface, to see whether the random-draw surface predicts the real subsets' scores.

Usage: python src/s28_cohort_explainers.py --B 200
Smoke: python src/s28_cohort_explainers.py --B 6
"""
from __future__ import annotations
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Reuse the fixed pipeline / model / cell evaluation from s18
import s18_crossed_grid as g

PKG = g.PKG
SUMM = os.path.join(PKG, "results", "summary")
CV_SEED = 2024        # fixed fold split, so score changes come only from which patients are drawn
DRAW_SEED = 8128
SEVERE_THR = 50.0
MODEL = "RandomForest"
MIN_CLASS = 10        # classification needs >=10 per class, else that draw's clf score is NaN (regression only)

PREDICTORS = ["n", "severe_prev", "aq_sd", "aq_iqr", "mean_bdist",
              "frac_near15", "mean_logvol", "sd_logvol"]


def n3_complexity(Xstd, ylab):
    """1-NN opposite-label rate (leave-self-out nearest neighbour). Model-free data
    complexity; no baseline-model predictions are used."""
    nn = NearestNeighbors(n_neighbors=2).fit(Xstd)
    _, idx = nn.kneighbors(Xstd)
    nb = idx[:, 1]   # nearest neighbour after excluding the patient itself
    return float(np.mean(ylab[nb] != ylab))


def cohort_stats(aq, vol):
    bdist = np.abs(aq - SEVERE_THR)
    q25, q75 = np.percentile(aq, [25, 75])
    lv = np.log1p(vol)
    return {
        "n": int(len(aq)),
        "severe_prev": float(np.mean(aq <= SEVERE_THR)),
        "aq_sd": float(np.std(aq, ddof=1)),
        "aq_iqr": float(q75 - q25),
        "mean_bdist": float(np.mean(bdist)),
        "frac_near15": float(np.mean(bdist <= 15)),
        "mean_logvol": float(np.mean(lv)),
        "sd_logvol": float(np.std(lv, ddof=1)),
    }


def safe_cell(fn, Xs, ys):
    try:
        return float(fn(Xs, ys, MODEL, CV_SEED)["score"])
    except Exception as e:
        print(f"    [warn] cell failed: {e}", flush=True)
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200)
    args = ap.parse_args()

    X, y_aq, keep = g.load_full_pool()
    aq_all = y_aq.values.astype(float)
    bdist_all = np.abs(aq_all - SEVERE_THR)
    ids_all = np.array(X.index)
    N = len(ids_all)
    print(f"[pool] N={N}, features={X.shape[1]} columns, severe={np.mean(aq_all<=SEVERE_THR)*100:.1f}%", flush=True)

    rng = np.random.default_rng(DRAW_SEED)
    rows = []
    t0 = time.time()
    for b in range(args.B):
        n = int(rng.integers(120, N + 1))
        tilt = 0.0 if (b % 7 == 0) else float(rng.uniform(-2.5, 2.5))  # every 7th draw is a purely random anchor
        w = np.exp(tilt * (bdist_all / 50.0))
        w = w / w.sum()
        sel = rng.choice(N, size=n, replace=False, p=w)
        idx = ids_all[sel]
        Xs = X.loc[idx]
        ys = y_aq.loc[idx]
        st = cohort_stats(ys.values, Xs["lesion_volume"].values)

        ylab = (ys.values <= SEVERE_THR).astype(int)
        both = (ylab.sum() >= MIN_CLASS) and ((len(ylab) - ylab.sum()) >= MIN_CLASS)
        if both:
            Xstd = StandardScaler().fit_transform(Xs.values)
            st["n3_1nn"] = n3_complexity(Xstd, ylab)
        else:
            st["n3_1nn"] = np.nan

        st["reg_score"] = safe_cell(g.run_cell_regression, Xs, ys)
        st["clf_score"] = safe_cell(g.run_cell_classification, Xs, ys) if both else np.nan
        rows.append(st)
        if b < 3 or b % 25 == 0 or b == args.B - 1:
            print(f"  draw {b+1}/{args.B} n={n} tilt={tilt:+.1f} sev={st['severe_prev']:.2f} "
                  f"aqsd={st['aq_sd']:.1f} reg={st['reg_score']:.3f} "
                  f"clf={st['clf_score'] if not np.isnan(st['clf_score']) else float('nan'):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(SUMM, exist_ok=True)
    draws_path = os.path.join(SUMM, "cohort_explainers_draws.csv")
    df.to_csv(draws_path, index=False)
    print(f"[write] {draws_path} ({len(df)} draws)", flush=True)

    # ---- Explanatory model: score ~ cohort structure ----
    def explain(score_col, preds):
        sub = df.dropna(subset=preds + [score_col])
        Xp = sub[preds].values
        ysc = sub[score_col].values
        Z = StandardScaler().fit_transform(Xp)
        lin = LinearRegression().fit(Z, ysc)
        r2_full = float(lin.score(Z, ysc))
        betas = dict(zip(preds, lin.coef_))
        uni = {}
        for j, p in enumerate(preds):
            uni[p] = float(LinearRegression().fit(Z[:, [j]], ysc).score(Z[:, [j]], ysc))
        rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1).fit(Xp, ysc)
        pi = permutation_importance(rf, Xp, ysc, n_repeats=20, random_state=0, n_jobs=-1)
        imp = dict(zip(preds, pi.importances_mean))
        return r2_full, betas, uni, imp, len(sub)

    out_rows = []
    for task, score_col in [("regression", "reg_score"), ("classification", "clf_score")]:
        preds = PREDICTORS + (["n3_1nn"] if task == "classification" else [])
        r2_full, betas, uni, imp, nrow = explain(score_col, preds)
        print(f"\n===== {task}: score ~ cohort structure (n={nrow} draws) =====", flush=True)
        print(f"  full-model R2 = {r2_full:.3f}", flush=True)
        for p in sorted(preds, key=lambda q: -uni[q]):
            print(f"  {p:14s} univariate R2={uni[p]:.3f}  std beta={betas[p]:+.4f}  "
                  f"RF perm importance={imp[p]:+.4f}", flush=True)
            out_rows.append({"task": task, "predictor": p, "univariate_r2": round(uni[p], 4),
                             "std_beta": round(betas[p], 4), "rf_perm_importance": round(imp[p], 4),
                             "full_model_r2": round(r2_full, 4), "n_draws": nrow})
    pd.DataFrame(out_rows).to_csv(os.path.join(SUMM, "cohort_explainers_importance.csv"), index=False)
    print(f"\n[write] cohort_explainers_importance.csv", flush=True)

    # ---- Overlay of the 10 real subsets: does the random-draw surface predict their scores? ----
    overlay = []
    for task, score_col, mn_csv in [
        ("regression", "reg_score", "matchedN_perm_regression.csv"),
        ("classification", "clf_score", "matchedN_perm_classification.csv")]:
        preds = PREDICTORS + (["n3_1nn"] if task == "classification" else [])
        sub = df.dropna(subset=preds + [score_col])
        sc = StandardScaler().fit(sub[preds].values)
        lin = LinearRegression().fit(sc.transform(sub[preds].values), sub[score_col].values)
        mn = pd.read_csv(os.path.join(SUMM, mn_csv))
        mn = mn[mn["model"] == "RandomForest"]
        # Drop the prior study's exact same-patient anchor (teghipco) and keep only the 8
        # inclusion-criterion subsets (as in s29/s30), so that the prior study's exact
        # patient list / scores never enter our overlay statistics.
        mn = mn[mn["subset"] != "teghipco_idlist"]
        for _, r in mn.iterrows():
            ids = [i for i in g.load_subset_ids(r["subset"]) if i in X.index]
            Xs = X.loc[ids]
            ys = y_aq.loc[ids]
            st = cohort_stats(ys.values, Xs["lesion_volume"].values)
            if task == "classification":
                ylab = (ys.values <= SEVERE_THR).astype(int)
                Xstd = StandardScaler().fit_transform(Xs.values)
                st["n3_1nn"] = n3_complexity(Xstd, ylab)
            xrow = np.array([[st[p] for p in preds]])
            pred_score = float(lin.predict(sc.transform(xrow))[0])
            overlay.append({"task": task, "subset": r["subset"], "n": len(ids),
                            "actual_score": round(float(r["obs_score"]), 4),
                            "predicted_score": round(pred_score, 4),
                            "abs_err": round(abs(pred_score - float(r["obs_score"])), 4)})
    ov = pd.DataFrame(overlay)
    ov.to_csv(os.path.join(SUMM, "cohort_explainers_subset_overlay.csv"), index=False)
    for task in ["regression", "classification"]:
        t = ov[ov["task"] == task]
        rr = float(np.corrcoef(t["actual_score"], t["predicted_score"])[0, 1])
        print(f"[overlay] {task}: real subsets, actual vs surface-predicted r={rr:.3f}, "
              f"mean |err|={t['abs_err'].mean():.4f}", flush=True)
    print(f"[write] cohort_explainers_subset_overlay.csv", flush=True)


if __name__ == "__main__":
    main()
