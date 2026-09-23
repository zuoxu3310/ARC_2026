#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s28_cohort_explainers.py
========================
机制拆解(其一): 队列级"分布 -> 分数"解释模型。
回答: 为什么"挑队列"会让分数大幅变动? 把分数对【与基准模型对错无关】的人群结构指标
做回归, 量出到底哪根杠杆在抬分数、各占多少方差。

反循环论证(硬约束): 解释变量全部是病人自带属性(标签/特征派生), 绝不使用基准模型的
预测对错来定义任何变量。
  - n              样本量
  - severe_prev    重症先验 (AQ<=50 占比)
  - aq_sd          严重度离散 (WAB-AQ 标准差)  -> 量程截断那条腿
  - aq_iqr         严重度四分位距
  - mean_bdist     平均离界距离 mean(|AQ-50|)
  - frac_near15    离 severe 界 <=15 分的占比 (边界密度)
  - mean_logvol    病灶体积(log1p)均值
  - sd_logvol      病灶体积(log1p)离散
  - n3_1nn (仅分类) 1-NN 同标签不一致率 (model-free 数据复杂度, 非基准模型)

做法: 从 226 全池抽 B 个【人数与构成各异】的随机子样(对 |AQ-50| 加随机权重以铺满指标
空间), 每个子样跑同一条固定管线(s18) 拿 RandomForest 的回归 r 与分类 balanced-acc;
再把分数对上面指标回归(标准化β + 单变量R² + RF置换重要性 + 全模型R²)。
最后把 10 个真实纳入标准子集叠加到同一张解释面上, 看随机面能否预测真实子集的分数。

用法: PYENV_VERSION=data-analysis python src/s28_cohort_explainers.py --B 200
冒烟:  PYENV_VERSION=data-analysis python src/s28_cohort_explainers.py --B 6
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

# 复用 s18 的固定管线/模型/单元评估
import s18_crossed_grid as g

PKG = g.PKG
SUMM = os.path.join(PKG, "results", "summary")
CV_SEED = 2024        # 折划分固定, 让分数变化只来自被抽到的人
DRAW_SEED = 8128
SEVERE_THR = 50.0
MODEL = "RandomForest"
MIN_CLASS = 10        # 分类要求每类 >=10 人, 否则该 draw 分类记 NaN(只留回归)

PREDICTORS = ["n", "severe_prev", "aq_sd", "aq_iqr", "mean_bdist",
              "frac_near15", "mean_logvol", "sd_logvol"]


def n3_complexity(Xstd, ylab):
    """1-NN 异标签率(留一最近邻)。model-free 数据复杂度, 不用基准模型预测。"""
    nn = NearestNeighbors(n_neighbors=2).fit(Xstd)
    _, idx = nn.kneighbors(Xstd)
    nb = idx[:, 1]   # 排除自身后的最近邻
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
    print(f"[pool] N={N}, 特征 {X.shape[1]} 列, severe={np.mean(aq_all<=SEVERE_THR)*100:.1f}%", flush=True)

    rng = np.random.default_rng(DRAW_SEED)
    rows = []
    t0 = time.time()
    for b in range(args.B):
        n = int(rng.integers(120, N + 1))
        tilt = 0.0 if (b % 7 == 0) else float(rng.uniform(-2.5, 2.5))  # 每7个放一个纯随机锚
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

    # ---- 解释模型: 分数 ~ 人群结构 ----
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
        print(f"\n===== {task}: 分数 ~ 人群结构 (n={nrow} draws) =====", flush=True)
        print(f"  全模型 R² = {r2_full:.3f}", flush=True)
        for p in sorted(preds, key=lambda q: -uni[q]):
            print(f"  {p:14s} 单变量R²={uni[p]:.3f}  标准化β={betas[p]:+.4f}  "
                  f"RF置换重要性={imp[p]:+.4f}", flush=True)
            out_rows.append({"task": task, "predictor": p, "univariate_r2": round(uni[p], 4),
                             "std_beta": round(betas[p], 4), "rf_perm_importance": round(imp[p], 4),
                             "full_model_r2": round(r2_full, 4), "n_draws": nrow})
    pd.DataFrame(out_rows).to_csv(os.path.join(SUMM, "cohort_explainers_importance.csv"), index=False)
    print(f"\n[write] cohort_explainers_importance.csv", flush=True)

    # ---- 真实 10 子集叠加: 随机解释面能否预测真实子集分数 ----
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
        # 去竞品精确同人锚(teghipco), 只留 8 个纳入标准子集 —— 与 s29/s30 一致,
        # 不把竞品的确切病人名单/分数混进我方 overlay 统计(红线)。
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
        print(f"[overlay] {task}: 真实子集 实际vs解释面预测 r={rr:.3f}, "
              f"平均|误差|={t['abs_err'].mean():.4f}", flush=True)
    print(f"[write] cohort_explainers_subset_overlay.csv", flush=True)


if __name__ == "__main__":
    main()
