#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s30_mechanism_partition.py
==========================
证伪关卡: 把"挑队列为什么抬分数"拆成 (A)量程截断[已知] vs (B)特征空间可分性[可能是新].
回答唯一关键问: 控住严重度摊开幅度(aq_sd 等=range restriction)之后, 特征空间复杂度
(n3_1nn / 近邻可分性 / 体积结构)是否还对分数有独立贡献?

做法(在 s28 的 300 个随机子样上):
  - 分层 R²: 逐块加入预测变量, 看每块带来的增量 R² (= 该块在控住前面之后的独立解释力)。
      block1 量程  = aq_sd, aq_iqr            (range restriction 那条腿)
      block2 先验/量 = + severe_prev, n        (重症先验 + 样本量)
      block3 特征结构 = + n3_1nn, mean_logvol, sd_logvol, mean_bdist, frac_near15
  - 偏相关: 控住 block1(量程)后, 各"特征结构"变量与分数的偏相关。
判决: block3 增量 R² 与 n3_1nn 偏相关若仍显著 -> 特征空间可分性是【独立于量程】的杠杆(新);
      若几乎归零 -> "挑到容易的人"只是量程截断换皮(旧, 不当新卖点)。

用法: PYENV_VERSION=data-analysis python src/s30_mechanism_partition.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMM = os.path.join(PKG, "results", "summary")

BLOCK1 = ["aq_sd", "aq_iqr"]                                   # 量程截断
BLOCK2 = ["severe_prev", "n"]                                  # 先验 + 样本量
BLOCK3 = ["mean_logvol", "sd_logvol", "mean_bdist", "frac_near15"]  # 特征/边界结构
N3 = "n3_1nn"                                                  # 仅分类: 特征空间近邻可分性


def r2(Z, y):
    return float(LinearRegression().fit(Z, y).score(Z, y))


def partial_corr(x, y, Z):
    """控住 Z 后 x 与 y 的偏相关 = 两者对 Z 残差的相关。"""
    Z = np.asarray(Z, float)
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = y - LinearRegression().fit(Z, y).predict(Z)
    return float(np.corrcoef(rx, ry)[0, 1])


def hier(df, score_col, has_n3):
    blocks = [("量程截断(aq_sd,aq_iqr)", BLOCK1),
              ("+先验+样本量(severe_prev,n)", BLOCK2),
              ("+特征/边界结构", BLOCK3 + ([N3] if has_n3 else []))]
    preds_all = BLOCK1 + BLOCK2 + BLOCK3 + ([N3] if has_n3 else [])
    sub = df.dropna(subset=preds_all + [score_col])
    y = sub[score_col].values
    used, prev = [], 0.0
    print(f"\n===== {score_col}: 分层 R² (n={len(sub)} draws) =====", flush=True)
    for name, blk in blocks:
        used += blk
        Z = StandardScaler().fit_transform(sub[used].values)
        cur = r2(Z, y)
        print(f"  {name:28s} 累计R²={cur:.3f}  增量ΔR²={cur-prev:+.3f}", flush=True)
        prev = cur
    # 偏相关: 控 block1(量程)
    Zc = StandardScaler().fit_transform(sub[BLOCK1].values)
    print(f"  -- 控住量程(block1)后的偏相关 --", flush=True)
    for p in (BLOCK3 + ([N3] if has_n3 else [])):
        pc = partial_corr(sub[p].values.astype(float), y, Zc)
        print(f"     {p:14s} 偏相关={pc:+.3f}", flush=True)


def main():
    df = pd.read_csv(os.path.join(SUMM, "cohort_explainers_draws.csv"))
    hier(df, "clf_score", has_n3=True)
    hier(df, "reg_score", has_n3=False)

    # 真实 8 子集(去 teghipco 锚): 子集分数 vs 近邻可分性, 控住边界距离(量程代理)
    comp = pd.read_csv(os.path.join(SUMM, "subset_difficulty_composition.csv"))
    print("\n===== 真实纳入标准子集(去teghipco): 近邻可分性是否独立于边界/量程 =====", flush=True)
    for task in ["classification", "regression"]:
        t = comp[(comp["task"] == task) & (comp["subset"] != "teghipco_idlist")]
        y = t["score"].values
        kd = t["mean_knn_disagree"].values
        bd = t["mean_boundary_dist"].values
        raw = float(np.corrcoef(y, kd)[0, 1])
        pc = partial_corr(kd, y, bd.reshape(-1, 1))
        print(f"  [{task}] 分数~近邻异标签 原始r={raw:+.3f}  控住平均离界距离后偏相关={pc:+.3f}  (n={len(t)})", flush=True)


if __name__ == "__main__":
    main()
