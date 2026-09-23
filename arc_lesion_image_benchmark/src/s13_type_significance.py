#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s13_type_significance.py — 类型【二分类】配对显著性检验(复用 s12 的逐人 dump, 不重训)。

镜像 s6_significance_classification.py。对每个二分任务(fluent / broca)做三组对比:
  1) fine_vs_coarse  : RandomForest, JHU_fine vs arterial_coarse(同一批病人)
  2) tuned_vs_default: RandomForest, JHU_fine
  3) RF_vs_LogReg    : JHU_fine, default(树 vs 线性, 次要)

指标与检验(与 s6 同):
  - AUC 差: 成对(correlated)DeLong(DeLong 1988 + Sun&Xu 2014 快速算法)。
  - balanced-acc 差: 患者级配对 bootstrap(对人重抽样 B=2000), 阈值 0.5。
只输出原始 p + 差 + 95%CI; 多重比较校正由主控统一做。

读: results/runs/clf_type_{task}_{gran}_{model}_{tag}_patient_probs.csv (s12 产)
写: results/summary/type_significance.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runs")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "summary")
OUT_CSV = os.path.join(OUT_DIR, "type_significance.csv")
B_BOOT = 2000
SEED = 20260605
THRESHOLD = 0.5


# ---------------- DeLong 快速算法(Sun & Xu 2014) ----------------
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def _fast_delong(preds_sorted_t, m):
    n = preds_sorted_t.shape[1] - m
    pos = preds_sorted_t[:, :m]; neg = preds_sorted_t[:, m:]; k = preds_sorted_t.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_t[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    return aucs, sx / m + sy / n


def delong_paired_test(y_true, p1, p2):
    y_true = np.asarray(y_true, dtype=int)
    order = (-y_true).argsort(kind="mergesort")
    m = int(y_true.sum())
    preds = np.vstack((np.asarray(p1, float)[order], np.asarray(p2, float)[order]))
    aucs, cov = _fast_delong(preds, m)
    a1, a2 = float(aucs[0]), float(aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]; diff = a1 - a2
    if var <= 0:
        return dict(auc1=a1, auc2=a2, diff=diff, z=0.0, p=1.0)
    z = diff / np.sqrt(var); p = 2.0 * stats.norm.sf(abs(z))
    return dict(auc1=a1, auc2=a2, diff=diff, z=float(z), p=float(p))


def paired_bootstrap_bacc(y_true, p1, p2, threshold=THRESHOLD, B=B_BOOT, seed=SEED):
    y_true = np.asarray(y_true, int)
    pred1 = (np.asarray(p1, float) >= threshold).astype(int)
    pred2 = (np.asarray(p2, float) >= threshold).astype(int)
    b1 = balanced_accuracy_score(y_true, pred1); b2 = balanced_accuracy_score(y_true, pred2)
    n = len(y_true); rng = np.random.default_rng(seed); diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n); yt = y_true[idx]
        if yt.sum() in (0, n):
            continue
        diffs.append(balanced_accuracy_score(yt, pred1[idx]) - balanced_accuracy_score(yt, pred2[idx]))
    diffs = np.array(diffs); lo, hi = np.percentile(diffs, [2.5, 97.5]); ne = len(diffs)
    p_lo = (np.sum(diffs <= 0) + 1) / (ne + 1); p_hi = (np.sum(diffs >= 0) + 1) / (ne + 1)
    return dict(bacc1=float(b1), bacc2=float(b2), diff=float(b1 - b2),
                ci_lo=float(lo), ci_hi=float(hi), p=float(min(2.0 * min(p_lo, p_hi), 1.0)))


def load_patient(task, gran, model, config):
    f = os.path.join(RUNS_DIR, f"clf_type_{task}_{gran}_{model}_{config}_patient_probs.csv")
    return pd.read_csv(f)[["participant_id", "y_true", "y_prob_mean"]]


def align_two(task, sa, sb):
    ga, ma, ca, la = sa; gb, mb, cb, lb = sb
    a = load_patient(task, ga, ma, ca).rename(columns={"y_prob_mean": "pa", "y_true": "ya"})
    b = load_patient(task, gb, mb, cb).rename(columns={"y_prob_mean": "pb", "y_true": "yb"})
    m = a.merge(b, on="participant_id", how="inner")
    assert (m.ya == m.yb).all(), f"y_true mismatch {la} vs {lb}"
    return m["ya"].values, m["pa"].values, m["pb"].values, len(m)


def run_contrast(task, name, sa, sb):
    la, lb = sa[3], sb[3]
    y, pa, pb, n = align_two(task, sa, sb)
    rows = []
    d = delong_paired_test(y, pa, pb)
    rows.append(dict(task=task, contrast=name, comparison=f"{la} vs {lb}", metric="AUC",
                     test="paired DeLong", n=n, value_a=round(d["auc1"], 4), value_b=round(d["auc2"], 4),
                     diff=round(d["diff"], 4), ci_lo="", ci_hi="", p_value=d["p"]))
    bb = paired_bootstrap_bacc(y, pa, pb)
    rows.append(dict(task=task, contrast=name, comparison=f"{la} vs {lb}", metric="balanced_acc",
                     test=f"paired bootstrap B={B_BOOT}", n=n,
                     value_a=round(bb["bacc1"], 4), value_b=round(bb["bacc2"], 4),
                     diff=round(bb["diff"], 4), ci_lo=round(bb["ci_lo"], 4), ci_hi=round(bb["ci_hi"], 4),
                     p_value=bb["p"]))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    for task in ["fluent", "broca"]:
        contrasts = [
            ("fine_vs_coarse",
             ("JHU_fine", "RandomForest", "default", "JHU/RF/def"),
             ("arterial_coarse", "RandomForest", "default", "coarse/RF/def")),
            ("tuned_vs_default",
             ("JHU_fine", "RandomForest", "tuned", "JHU/RF/tuned"),
             ("JHU_fine", "RandomForest", "default", "JHU/RF/def")),
            ("RF_vs_LogReg",
             ("JHU_fine", "RandomForest", "default", "JHU/RF/def"),
             ("JHU_fine", "LogReg", "default", "JHU/LogReg/def")),
        ]
        for nm, sa, sb in contrasts:
            try:
                all_rows.extend(run_contrast(task, nm, sa, sb))
            except FileNotFoundError as e:
                print(f"  [skip] {task}/{nm}: 缺 dump ({e})")
    out = pd.DataFrame(all_rows, columns=["task", "contrast", "comparison", "metric", "test",
                                          "n", "value_a", "value_b", "diff", "ci_lo", "ci_hi", "p_value"])
    out.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(f"[written] {OUT_CSV}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
