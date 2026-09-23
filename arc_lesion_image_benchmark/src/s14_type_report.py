#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s14_type_report.py — 类型基准的混淆矩阵 / 逐类报告 / 二分校准(复用 dump, 不重训)。

两部分:
A) 多分类(6 类)诚实报告: 读 s11 dump type_multiclass_{gran}_{model}_oof.csv,
   出混淆矩阵 + 逐类 precision/recall/F1/support。重点: 少数类(Wernicke/TCM)support 极小,
   逐类指标本就不可靠 —— 用 support 列把这点摆明, 不让读者误读 macro 分。
B) 二分校准(fluent / broca, RandomForest): 读 s12 dump clf_type_{task}_{gran}_RF_default_patient_probs.csv,
   在原始患病率上算 Brier/ECE/MCE + 可靠性曲线; 并报 CV 无泄露重校准(Platt / isotonic)可修复性。
   (SMOTE 把训练折平衡到 50/50 -> 折外概率系统偏移, 与 s7 同源问题。)

写:
  results/summary/type_multiclass_confusion_{gran}_{model}.csv
  results/summary/type_multiclass_perclass_{gran}_{model}.csv
  results/summary/type_calibration.csv
  results/summary/type_calibration_reliability.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(PKG, "results", "runs")
SUMMARY = os.path.join(PKG, "results", "summary")
os.makedirs(SUMMARY, exist_ok=True)
N_BINS, CALIB_FOLDS, SEED = 10, 5, 42

GRANS = ["arterial_coarse", "JHU_fine"]
MC_MODELS = ["RandomForest", "LogReg"]
BIN_TASKS = ["fluent", "broca"]


# ---------------- A) 多分类 ----------------
def multiclass_report(gran, model):
    f = os.path.join(RUNS, f"type_multiclass_{gran}_{model}_oof.csv")
    if not os.path.exists(f):
        print(f"  [skip] 缺 {os.path.basename(f)}"); return
    d = pd.read_csv(f)
    classes = [c[len("prob_"):] for c in d.columns if c.startswith("prob_")]
    y_true = d["y_true"].to_numpy(); y_pred = d["y_pred"].to_numpy()
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                         columns=[f"pred_{c}" for c in classes])
    cm_df.to_csv(os.path.join(SUMMARY, f"type_multiclass_confusion_{gran}_{model}.csv"))

    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pc = pd.DataFrame({"class": classes, "precision": pr.round(3), "recall": rc.round(3),
                       "f1": f1.round(3), "support": sup})
    pc = pc.sort_values("support", ascending=False).reset_index(drop=True)
    pc.to_csv(os.path.join(SUMMARY, f"type_multiclass_perclass_{gran}_{model}.csv"), index=False)
    print(f"  [{gran}/{model}] 6类 混淆矩阵 + 逐类 -> 写出。逐类(按 support):")
    print(pc.to_string(index=False))
    small = pc[pc["support"] < 15]["class"].tolist()
    if small:
        print(f"    !! support<15 的类(指标不可靠): {small}")


# ---------------- B) 二分校准(同 s7) ----------------
def brier(y, p): return float(np.mean((p - y) ** 2))


def reliability(y, p, n_bins=N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1); rows = []
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b; n = int(m.sum())
        rows.append((edges[b], edges[b + 1], n,
                     np.nan if n == 0 else float(p[m].mean()),
                     np.nan if n == 0 else float(y[m].mean())))
    return rows


def ece(y, p, n_bins=N_BINS):
    rows = reliability(y, p, n_bins); N = len(y); e = 0.0
    for _, _, n, mp, fp in rows:
        if n > 0:
            e += (n / N) * abs(mp - fp)
    return float(e)


def mce(y, p, n_bins=N_BINS):
    gaps = [abs(mp - fp) for _, _, n, mp, fp in reliability(y, p, n_bins) if n > 0]
    return float(max(gaps)) if gaps else np.nan


def cv_recalibrate(y, p, method, folds=CALIB_FOLDS, seed=SEED):
    y = np.asarray(y, float); p = np.asarray(p, float); out = np.full_like(p, np.nan)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(p.reshape(-1, 1), y):
        if method == "sigmoid":
            ptr = np.clip(p[tr], 1e-6, 1 - 1e-6)
            z = np.log(ptr / (1 - ptr)).reshape(-1, 1)
            lr = LogisticRegression(C=1e10, solver="lbfgs").fit(z, y[tr])
            pte = np.clip(p[te], 1e-6, 1 - 1e-6)
            out[te] = lr.predict_proba(np.log(pte / (1 - pte)).reshape(-1, 1))[:, 1]
        else:
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[tr], y[tr])
            out[te] = ir.predict(p[te])
    assert not np.isnan(out).any()
    return out


def calibration_for(task, gran, summary_rows, rel_rows):
    f = os.path.join(RUNS, f"clf_type_{task}_{gran}_RandomForest_default_patient_probs.csv")
    if not os.path.exists(f):
        print(f"  [skip] 缺 {os.path.basename(f)}"); return
    d = pd.read_csv(f)
    y = d["y_true"].to_numpy(float); p_raw = d["y_prob_mean"].to_numpy(float)
    prev = float(y.mean()); brier_const = brier(y, np.full_like(y, prev))
    p_platt = cv_recalibrate(y, p_raw, "sigmoid"); p_iso = cv_recalibrate(y, p_raw, "isotonic")
    for tag, p in [("raw", p_raw), ("platt", p_platt), ("isotonic", p_iso)]:
        summary_rows.append(dict(task=task, granularity=gran, model="RandomForest", calibration=tag,
                                 n=len(y), prevalence=round(prev, 4), mean_pred=round(float(p.mean()), 4),
                                 brier=round(brier(y, p), 5), ece=round(ece(y, p), 5),
                                 mce=round(mce(y, p), 5), brier_const_baseline=round(brier_const, 5)))
        for lo, hi, n, mp, fp in reliability(y, p):
            rel_rows.append(dict(task=task, granularity=gran, calibration=tag,
                                 bin_mid=round((lo + hi) / 2, 2), n=n,
                                 mean_pred=None if np.isnan(mp) else round(mp, 4),
                                 frac_pos=None if np.isnan(fp) else round(fp, 4)))


def main():
    print("=== A) 多分类(6 类)混淆矩阵 + 逐类报告 ===")
    for gran in GRANS:
        for model in MC_MODELS:
            multiclass_report(gran, model)

    print("\n=== B) 二分校准(RandomForest, 原始患病率上) ===")
    srows, rrows = [], []
    for task in BIN_TASKS:
        for gran in GRANS:
            calibration_for(task, gran, srows, rrows)
    if srows:
        sdf = pd.DataFrame(srows); rdf = pd.DataFrame(rrows)
        sdf.to_csv(os.path.join(SUMMARY, "type_calibration.csv"), index=False)
        rdf.to_csv(os.path.join(SUMMARY, "type_calibration_reliability.csv"), index=False)
        pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
        print(sdf.to_string(index=False))
        print("\nwrote type_calibration.csv + type_calibration_reliability.csv")


if __name__ == "__main__":
    main()
