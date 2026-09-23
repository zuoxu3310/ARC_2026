#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s29_difficulty_composition.py
=============================
机制拆解(其二): 病人级"难易构成"分析。
回答 Master ZX 的问: 某些纳入标准是不是正好挑到了"本来就容易区分/很少犯错"的病人?

反循环论证(硬约束): "难易"只用病人自带属性定义, 绝不用基准模型答对没来定义。
  - boundary_dist = |WAB-AQ - 50|      离 severe 二分界的距离 (纯标签派生)
  - knn_disagree  = 全池标准化特征上, 每人 10 个最近邻里异标签(severe/非severe)的比例
                    (model-free 局部类重叠, 不用基准模型)
模型的逐人对错(全池 OOF)只在第 2 步当【验证目标】: 证明上面这套难易确实能预测谁被错判;
绝不进入难易的定义。

三步:
  1. 在全 226 池上给每人算两个 model-free 难易分。
  2. 验证: 难易分 -> 全池 RF 的逐人误判/绝对误差 (分箱命中率 + AUC + 相关), 证难易管用。
  3. 构成: 每个纳入标准子集挑到的病人, 平均难易 vs 全池; 再把子集分数对子集平均难易相关,
     看"分高的子集是不是系统性挑到更容易的人"。

用法: PYENV_VERSION=data-analysis python src/s29_difficulty_composition.py
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
    """每人 k 近邻里异标签比例(留一)。model-free 局部类重叠。"""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xstd)
    _, idx = nn.kneighbors(Xstd)
    nb = idx[:, 1:]                        # 去掉自身
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

    # ---- 1. model-free 难易 ----
    boundary_dist = np.abs(aq - SEVERE_THR)
    knn_dis = knn_disagreement(Xstd, ylab)
    pat = pd.DataFrame({"participant_id": ids, "wab_aq": aq, "severe": ylab,
                        "boundary_dist": boundary_dist, "knn_disagree": knn_dis})

    # ---- 2. 验证: 难易 -> 全池 RF 逐人对错 ----
    clf = pd.read_csv(os.path.join(RUNS, "clf_JHU_fine_RandomForest_default_patient_probs.csv"))
    clf["misclass"] = (clf["y_pred"] != clf["y_true"]).astype(int)
    clf["margin"] = (clf["y_prob_mean"] - 0.5).abs()
    reg = pd.read_csv(os.path.join(RUNS, "reg_JHU_fine_RandomForest_default_patient_pred.csv"))
    reg["abs_err"] = (reg["y_true"] - reg["y_pred_mean"]).abs()
    pat = pat.merge(clf[["participant_id", "misclass", "margin"]], on="participant_id", how="left")
    pat = pat.merge(reg[["participant_id", "abs_err"]], on="participant_id", how="left")
    pat.to_csv(os.path.join(SUMM, "patient_difficulty.csv"), index=False)

    mis = pat["misclass"].values.astype(float)
    print("===== 验证: model-free 难易能否预测【分类误判】 =====", flush=True)
    # 离界距离: 近 = 难; 用 -boundary_dist 当"难度分"
    auc_bd = roc_auc_score(mis, -pat["boundary_dist"].values)
    auc_kd = roc_auc_score(mis, pat["knn_disagree"].values)
    auc_both = roc_auc_score(mis, (-pat["boundary_dist"].rank() + pat["knn_disagree"].rank()).values)
    print(f"  AUC(离界距离->误判)   = {auc_bd:.3f}", flush=True)
    print(f"  AUC(近邻异标签->误判) = {auc_kd:.3f}", flush=True)
    print(f"  AUC(两者秩和->误判)   = {auc_both:.3f}", flush=True)
    print("  按离界距离分箱的误判率:", flush=True)
    for lab, n, rate in binned_rate(pat["boundary_dist"].values, mis,
                                    [-0.1, 15, 30, 60], ["<=15", "15-30", ">30"]):
        print(f"    |AQ-50| {lab:6s}  n={n:3d}  误判率={rate:.3f}", flush=True)
    print("  按近邻异标签率分箱(三分位)的误判率:", flush=True)
    q = pat["knn_disagree"].quantile([1/3, 2/3]).values
    for lab, n, rate in binned_rate(pat["knn_disagree"].values, mis,
                                    [-0.001, q[0], q[1], 1.001], ["低", "中", "高"]):
        print(f"    近邻异标签 {lab:3s}  n={n:3d}  误判率={rate:.3f}", flush=True)

    # 回归误差 vs 难易(回归分数是 r, 更受量程影响; 这里只报相关当佐证)
    ae = pat["abs_err"].values
    print("\n  回归绝对误差相关(佐证): "
          f"corr(误差, 越接近高分端)={np.corrcoef(ae, aq)[0,1]:+.3f}  "
          f"corr(误差, 近邻异标签)={np.corrcoef(ae, knn_dis)[0,1]:+.3f}", flush=True)

    # ---- 3. 子集难易构成 + 与分数相关 ----
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

    print("\n===== 子集难易构成 vs 分数(分高=是否挑到更容易的人) =====", flush=True)
    full_bd = float(pat["boundary_dist"].mean())
    full_kd = float(pat["knn_disagree"].mean())
    print(f"  全池基准: 平均离界距离={full_bd:.2f}  平均近邻异标签={full_kd:.3f}", flush=True)
    for task in ["classification", "regression"]:
        t = comp[comp["task"] == task]
        # 排除精确同人锚点(teghipco)看纳入标准本身
        tc = t[t["subset"] != "teghipco_idlist"]
        r_bd = float(np.corrcoef(tc["score"], tc["mean_boundary_dist"])[0, 1])
        r_kd = float(np.corrcoef(tc["score"], tc["mean_knn_disagree"])[0, 1])
        print(f"  [{task}] 子集分数 vs 平均离界距离 r={r_bd:+.3f}; "
              f"vs 平均近邻异标签 r={r_kd:+.3f}  (n={len(tc)} 子集, 去 teghipco 锚)", flush=True)
    print(f"\n[write] patient_difficulty.csv / subset_difficulty_composition.csv", flush=True)


if __name__ == "__main__":
    main()
