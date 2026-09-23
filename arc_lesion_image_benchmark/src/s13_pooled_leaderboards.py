#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s13_pooled_leaderboards.py —— 把回归/分类排行榜统一重算成【池化 OOF】口径。

为什么: 全文+图统一到池化(n=226 病人, 患者级 bootstrap CI, 与配对/等价检验同一向量)。
逐折平均口径(原主表)不再混用。本脚本从已存的逐人预测重算, 不重训模型。

输入(逐人, held-out):
  - 回归: results/runs/reg_<atlas>_<model>_<cfg>_patient_pred.csv  (y_true, y_pred_mean)
  - 分类: results/runs/clf_<atlas>_<model>_<cfg>_patient_probs.csv (y_true, y_prob_mean, y_pred)
输出:
  - results/summary/pooled_regression_leaderboard.csv
  - results/summary/pooled_classification_leaderboard.csv
"""
from __future__ import annotations
import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef)

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(PKG, "results", "runs")
OUT = os.path.join(PKG, "results", "summary")
SEED = 42
B = 2000


def boot_ci(fn, *arrays, n=B, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            v = fn(*[a[idx] for a in arrays])
            if v == v:
                vals.append(v)
        except Exception:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def reg_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "reg_*_patient_pred.csv"))):
        base = os.path.basename(f)
        m = re.match(r"reg_(JHU_fine|arterial_coarse)_([A-Za-z_]+?)_(default|tuned)_patient_pred\.csv", base)
        if not m:
            continue
        atlas, model, cfg = m.groups()
        d = pd.read_csv(f)
        yt = d["y_true"].to_numpy(float); yp = d["y_pred_mean"].to_numpy(float)
        r = pearsonr(yt, yp)[0]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
        lo, hi = boot_ci(lambda a, b: pearsonr(a, b)[0], yt, yp)
        rows.append({"atlas": atlas, "model": model, "config": cfg,
                     "pooled_r": round(r, 3), "r_ci95": f"[{lo:.3f}, {hi:.3f}]",
                     "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                     "R2": round(float(ss), 3), "n": len(yt)})
    return pd.DataFrame(rows).sort_values(["atlas", "config", "pooled_r"],
                                          ascending=[True, True, False])


def clf_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "clf_*_patient_probs.csv"))):
        base = os.path.basename(f)
        m = re.match(r"clf_(JHU_fine|arterial_coarse)_([A-Za-z_]+?)_(default|tuned)_patient_probs\.csv", base)
        if not m:
            continue
        atlas, model, cfg = m.groups()
        d = pd.read_csv(f)
        yt = d["y_true"].to_numpy(int)
        prob = d["y_prob_mean"].to_numpy(float)
        pred = d["y_pred"].to_numpy(int) if "y_pred" in d else (prob >= 0.5).astype(int)
        ba = balanced_accuracy_score(yt, pred)
        auc = roc_auc_score(yt, prob)
        f1 = f1_score(yt, pred, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(yt, pred)
        lo, hi = boot_ci(lambda a, b: balanced_accuracy_score(a, b), yt, pred)
        alo, ahi = boot_ci(lambda a, b: roc_auc_score(a, b), yt, prob)
        rows.append({"atlas": atlas, "model": model, "config": cfg,
                     "pooled_bAcc": round(ba, 3), "bAcc_ci95": f"[{lo:.3f}, {hi:.3f}]",
                     "pooled_AUC": round(auc, 3), "AUC_ci95": f"[{alo:.3f}, {ahi:.3f}]",
                     "F1_severe": round(f1, 3), "MCC": round(mcc, 3), "n": len(yt)})
    return pd.DataFrame(rows).sort_values(["atlas", "config", "pooled_bAcc"],
                                          ascending=[True, True, False])


def main():
    reg = reg_rows(); clf = clf_rows()
    reg.to_csv(os.path.join(OUT, "pooled_regression_leaderboard.csv"), index=False)
    clf.to_csv(os.path.join(OUT, "pooled_classification_leaderboard.csv"), index=False)
    print("=== 回归 池化排行榜(default)===")
    print(reg[reg.config == "default"][["atlas", "model", "pooled_r", "r_ci95", "MAE"]].to_string(index=False))
    print("\n=== 回归 池化(tuned, 仅 RF 等)===")
    print(reg[reg.config == "tuned"][["atlas", "model", "pooled_r", "MAE"]].to_string(index=False))
    print("\n=== 分类 池化排行榜(default)===")
    print(clf[clf.config == "default"][["atlas", "model", "pooled_bAcc", "pooled_AUC", "F1_severe"]].to_string(index=False))
    print("\n=== 分类 池化(tuned)===")
    print(clf[clf.config == "tuned"][["atlas", "model", "pooled_bAcc", "pooled_AUC"]].to_string(index=False))


if __name__ == "__main__":
    main()
