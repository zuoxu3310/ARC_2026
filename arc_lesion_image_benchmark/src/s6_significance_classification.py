#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s6_significance_classification.py
=================================
Paired significance tests (classification task). Reuses the existing per-patient dumps; no retraining.

Main contrasts (key null hypotheses for the "lesion-only performance ceiling" argument):
  1) fine vs coarse   : RandomForest, JHU_fine vs arterial_coarse (same patients)
  2) tuned vs default : RandomForest, JHU_fine
Secondary:
  3) RandomForest vs XGBoost : JHU_fine, default

Metrics and tests:
  - AUC difference: paired DeLong test (same patients -> ROC covariance matrix from each model's
    probabilities).
  - balanced-accuracy difference: patient-level paired bootstrap (resample patients, B=2000)
    -> distribution of differences -> 95% CI + two-sided p.
    Threshold fixed at 0.5 (consistent with the patient-level aggregation in the dump script:
    mean_pred = (mean_prob >= 0.5)).

Rules:
  - Use patient-level predictions (one value per patient: y_prob_mean in _patient_probs.csv), so
    correlated repeated-CV predictions do not inflate the effective sample size.
  - For each contrast, align the two prediction columns on participant_id before the paired test.
  - Report only the raw p + difference + 95% CI per contrast; no multiple-comparison correction
    here (Holm/FDR is applied jointly downstream).

DeLong implementation (own code): nonparametric covariance estimator of DeLong, DeLong &
Clarke-Pearson (1988) Biometrics 44:837-845, with the O(N log N) midrank /
structural-component algorithm of Sun & Xu (2014) IEEE SPL 21:1389-1393. Both ROC curves share
the same samples (same patients, same y), so the paired (correlated) DeLong test is used:
var(AUC1-AUC2) = S11 + S22 - 2*S12, z = (AUC1-AUC2) / sqrt(var).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

# ---------------------------------------------------------------------------
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "runs")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "summary")
OUT_CSV = os.path.join(OUT_DIR, "significance_classification.csv")

B_BOOT = 2000
SEED = 20260604
THRESHOLD = 0.5  # matches the patient-level aggregation in the dump


# ============================================================================
# Fast DeLong algorithm (Sun & Xu 2014 structural components)
# ============================================================================
def _compute_midrank(x):
    """midrank with ties averaged (Sun & Xu 2014, Algorithm)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based midrank
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong (Sun & Xu 2014): returns (aucs, covariance_matrix).
    predictions_sorted_transposed: shape (k, n), columns ordered [positive samples first,
      negative samples after]; rows = scores of the k classifiers.
      label_1_count = number of positive (severe=1) samples m.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n               # V10: structural comp over positives
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m         # V01: structural comp over negatives
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:  # np.cov returns scalar for 1 row
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_paired_test(y_true, prob1, prob2):
    """
    Paired (correlated) DeLong test: both ROC curves share the same samples (same patients, same y).
    Returns dict: auc1, auc2, diff (= auc1 - auc2), se_diff, z, p (two-sided).
    Positive class = 1 (severe).
    """
    y_true = np.asarray(y_true, dtype=int)
    order = (-y_true).argsort(kind="mergesort")  # positives (1) first, negatives (0) after; stable sort
    label_1_count = int(y_true.sum())
    preds = np.vstack((np.asarray(prob1, dtype=float)[order],
                       np.asarray(prob2, dtype=float)[order]))
    aucs, cov = _fast_delong(preds, label_1_count)
    auc1, auc2 = float(aucs[0]), float(aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    diff = auc1 - auc2
    if var <= 0:
        # the two ROC curves are numerically identical -> degenerate, p=1
        return dict(auc1=auc1, auc2=auc2, diff=diff, se_diff=0.0, z=0.0, p=1.0)
    se = np.sqrt(var)
    z = diff / se
    p = 2.0 * stats.norm.sf(abs(z))
    return dict(auc1=auc1, auc2=auc2, diff=diff, se_diff=float(se),
                z=float(z), p=float(p))


# ============================================================================
# Patient-level paired bootstrap (balanced-accuracy difference)
# ============================================================================
def paired_bootstrap_bacc(y_true, prob1, prob2, threshold=THRESHOLD,
                          B=B_BOOT, seed=SEED):
    """
    Resample patients B times -> recompute both balanced accuracies each time -> distribution of
    differences -> 95% CI + two-sided p.
    Fixed threshold (default 0.5). The point estimate uses the full sample (not the bootstrap mean).
    Two-sided p: share of the difference distribution on each side of 0 (Davison & Hinkley
    empirical p), p = 2*min(P(d<=0), P(d>=0)), with +1 smoothing to avoid p=0.
    """
    y_true = np.asarray(y_true, dtype=int)
    p1 = np.asarray(prob1, dtype=float)
    p2 = np.asarray(prob2, dtype=float)
    pred1 = (p1 >= threshold).astype(int)
    pred2 = (p2 >= threshold).astype(int)

    bacc1 = balanced_accuracy_score(y_true, pred1)
    bacc2 = balanced_accuracy_score(y_true, pred2)
    diff_point = bacc1 - bacc2

    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            diffs[b] = np.nan
            continue
        d = balanced_accuracy_score(yt, pred1[idx]) - balanced_accuracy_score(yt, pred2[idx])
        diffs[b] = d
    diffs = diffs[~np.isnan(diffs)]
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    n_eff = len(diffs)
    p_lo = (np.sum(diffs <= 0) + 1) / (n_eff + 1)
    p_hi = (np.sum(diffs >= 0) + 1) / (n_eff + 1)
    p = 2.0 * min(p_lo, p_hi)
    p = min(p, 1.0)
    return dict(bacc1=float(bacc1), bacc2=float(bacc2), diff=float(diff_point),
                ci_lo=float(ci_lo), ci_hi=float(ci_hi), p=float(p),
                n_boot_eff=int(n_eff))


# ============================================================================
# Loading + alignment
# ============================================================================
def load_patient(granularity, model, config):
    f = os.path.join(RUNS_DIR, f"clf_{granularity}_{model}_{config}_patient_probs.csv")
    df = pd.read_csv(f)[["participant_id", "y_true", "y_prob_mean"]]
    return df


def align_two(spec_a, spec_b):
    """spec = (granularity, model, config, label). Align on participant_id."""
    ga, ma, ca, la = spec_a
    gb, mb, cb, lb = spec_b
    a = load_patient(ga, ma, ca).rename(columns={"y_prob_mean": "prob_a", "y_true": "yt_a"})
    b = load_patient(gb, mb, cb).rename(columns={"y_prob_mean": "prob_b", "y_true": "yt_b"})
    m = a.merge(b, on="participant_id", how="inner")
    assert (m.yt_a == m.yt_b).all(), f"y_true mismatch between {la} and {lb}"
    return m["yt_a"].values, m["prob_a"].values, m["prob_b"].values, len(m)


def run_contrast(name, spec_a, spec_b):
    la, lb = spec_a[3], spec_b[3]
    y, pa, pb, n = align_two(spec_a, spec_b)
    rows = []

    d = delong_paired_test(y, pa, pb)
    rows.append(dict(
        contrast=name, comparison=f"{la} vs {lb}", metric="AUC",
        test="paired DeLong (1988; Sun&Xu 2014)", threshold="n/a",
        n_patients=n,
        value_a=round(d["auc1"], 4), value_b=round(d["auc2"], 4),
        diff=round(d["diff"], 4), ci_lo="", ci_hi="",
        se_or_z=round(d["z"], 4), p_value=d["p"],
    ))

    bb = paired_bootstrap_bacc(y, pa, pb)
    rows.append(dict(
        contrast=name, comparison=f"{la} vs {lb}", metric="balanced_acc",
        test=f"paired patient-level bootstrap (B={B_BOOT})", threshold=THRESHOLD,
        n_patients=n,
        value_a=round(bb["bacc1"], 4), value_b=round(bb["bacc2"], 4),
        diff=round(bb["diff"], 4),
        ci_lo=round(bb["ci_lo"], 4), ci_hi=round(bb["ci_hi"], 4),
        se_or_z="", p_value=bb["p"],
    ))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    contrasts = [
        # 1) fine vs coarse (RandomForest)
        ("fine_vs_coarse",
         ("JHU_fine", "RandomForest", "default", "JHU_fine/RF/default"),
         ("arterial_coarse", "RandomForest", "default", "arterial_coarse/RF/default")),
        # 2) tuned vs default (RandomForest, JHU)
        ("tuned_vs_default",
         ("JHU_fine", "RandomForest", "tuned", "JHU_fine/RF/tuned"),
         ("JHU_fine", "RandomForest", "default", "JHU_fine/RF/default")),
        # 3) RF vs XGB (JHU, default) - secondary
        ("RF_vs_XGB",
         ("JHU_fine", "RandomForest", "default", "JHU_fine/RF/default"),
         ("JHU_fine", "XGBoost", "default", "JHU_fine/XGB/default")),
    ]
    all_rows = []
    for name, sa, sb in contrasts:
        all_rows.extend(run_contrast(name, sa, sb))

    out = pd.DataFrame(all_rows, columns=[
        "contrast", "comparison", "metric", "test", "threshold", "n_patients",
        "value_a", "value_b", "diff", "ci_lo", "ci_hi", "se_or_z", "p_value"])
    out.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"[written] {OUT_CSV}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
