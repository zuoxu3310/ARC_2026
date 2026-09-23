#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s22_granularity_spectrum.py
===========================
任务2: 图谱粒度谱。同一条固定管线(强模型 + 折内 scaler/SMOTE)跑不同粒度图谱,
看"粒度-分数曲线"是否封顶——证实并延伸 Tilwani 的"换图谱主效应不显著"。

粒度点(全队列 226, full cohort): 由 --tables 给的特征表决定。本机现成:
  1 区   = floor_volage_features.tsv (lesion_volume+age, 整脑负荷)
  33 区  = ArterialAtlas156_features_merged.tsv
  189 区 = jhu156_features_merged.tsv
(可扩展: 下 AAL108/AICHA384 重采样到本网格 -> s2 提取 -> 加进 --tables)

输出: 每粒度 × 任务的 r/bACC(逐种子均值+SD) + 相邻粒度 TOST 等价(复用思路)。
复用 s18 的单元跑法/模型工厂, 保证与任务3 同一条管线。

用法: PYENV_VERSION=data-analysis python src/s22_granularity_spectrum.py --seeds 20
"""
from __future__ import annotations
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import s18_crossed_grid as g

PKG = g.PKG
SUMM = os.path.join(PKG, "results", "summary")
DATA = os.path.join(PKG, "data")
COMP_REG = ["RandomForest", "XGBoost", "LightGBM"]
COMP_CLF = ["RandomForest", "XGBoost", "LightGBM", "LogReg"]

# 粒度点: (label, n_regions_nominal, 特征表相对 data/ 路径)。1→33→41→116→189→384 六点谱。
GRANULARITIES = [
    ("vol_1", 1, "floor_volage_features.tsv"),
    ("arterial_33", 33, "ArterialAtlas156_features_merged.tsv"),
    ("brodmann_41", 41, "brodmann41_features_merged.tsv"),
    ("aal_116", 116, "aal116_features_merged.tsv"),
    ("jhu_189", 189, "jhu156_features_merged.tsv"),
    ("aicha_384", 384, "aicha384_features_merged.tsv"),
]


def load_table(rel):
    df = pd.read_csv(os.path.join(DATA, rel), sep="\t", index_col="participant_id")
    df = df.dropna(subset=["wab_aq"]).fillna(0)
    y_aq = df["wab_aq"].copy()
    feat = df.drop(columns=["wab_aq"])
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > g.SPARSE_DROP_FRAC * n]
    # 至少保留 lesion_volume(1 区情形它本身就是唯一信息)
    if not keep:
        keep = list(feat.columns)
    return feat[keep], y_aq, len(keep)


def run_granularity(task, seeds):
    runner = g.run_cell_regression if task == "regression" else g.run_cell_classification
    models = COMP_REG if task == "regression" else COMP_CLF
    rows = []
    percell = {}   # (gran) -> list of per-(model,seed) scores, 供 TOST
    for label, nreg, rel in GRANULARITIES:
        X, y, nfeat = load_table(rel)
        scores = []
        for model in models:
            for seed in seeds:
                s = runner(X, y, model, seed)["score"]
                scores.append(s)
        percell[label] = np.array(scores)
        rows.append({"task": task, "granularity": label, "n_regions": nreg,
                     "n_features": nfeat, "mean_score": round(float(np.mean(scores)), 4),
                     "sd_score": round(float(np.std(scores)), 4),
                     "min": round(float(np.min(scores)), 4),
                     "max": round(float(np.max(scores)), 4)})
        print(f"  [{task}] {label:14s} ({nfeat:3d} feat) "
              f"score={np.mean(scores):.3f}±{np.std(scores):.3f}", flush=True)
    return pd.DataFrame(rows), percell


def tost_adjacent(percell, labels, bound=0.05):
    """相邻粒度 TOST 等价(配对 bootstrap 简化: 用两组分数差的 90%CI 是否落 ±bound)。"""
    out = []
    for a, b in zip(labels[:-1], labels[1:]):
        da, db = percell[a], percell[b]
        n = min(len(da), len(db))
        diff = db[:n] - da[:n]
        # bootstrap 90% CI of mean diff
        rng = np.random.default_rng(2026)
        boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [5, 95])
        equiv = (lo > -bound) and (hi < bound)
        out.append({"contrast": f"{a}_vs_{b}", "mean_diff": round(float(np.mean(diff)), 4),
                    "ci90_lo": round(float(lo), 4), "ci90_hi": round(float(hi), 4),
                    "equiv_bound": bound, "equivalent": bool(equiv)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--base_seed", type=int, default=3000)
    args = ap.parse_args()
    seeds = [args.base_seed + i for i in range(args.seeds)]

    all_rows, all_tost = [], []
    for task in ["regression", "classification"]:
        print(f"=== {task} 粒度谱 ===", flush=True)
        rows, percell = run_granularity(task, seeds)
        labels = [r[0] for r in GRANULARITIES]
        tost = tost_adjacent(percell, labels)
        tost.insert(0, "task", task)
        all_rows.append(rows); all_tost.append(tost)
        print(tost.to_string(index=False), flush=True)

    os.makedirs(SUMM, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(
        os.path.join(SUMM, "granularity_spectrum.csv"), index=False)
    pd.concat(all_tost, ignore_index=True).to_csv(
        os.path.join(SUMM, "granularity_tost.csv"), index=False)
    print(f"\n[write] granularity_spectrum.csv + granularity_tost.csv", flush=True)


if __name__ == "__main__":
    main()
