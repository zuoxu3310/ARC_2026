#!/usr/bin/env python
"""
s9_chronicity_sensitivity.py  ——  慢性期敏感性分析(RF 回归 + 分类)

背景
----
ARC 官方自定义为 "chronic stroke repository"(慢性卒中库),慢性口径以他们为准。
但 participants.tsv 的 wab_days(卒中到 WAB 评估天数)在建模队列里有部分偏短
(最早 42 天)。本脚本检验:若强制只留更慢性的子集,顶端分数是否仍稳。

做什么
------
把特征表(JHU 细 / arterial 粗) join participants.tsv 的 wab_days,生成两个子集:
  * wab_days >= 180
  * wab_days >= 365
每个子集上,用与主线【完全相同的口径】重跑 RandomForest:
  * 回归: 复用 s5_nested_tuning_regression 的 run_default / run_tuned(5x10 重复分层,
          折内 StandardScaler->RF, tuned 走内层 GridSearchCV)。指标 Pearson r / MAE / RMSE / R2。
  * 分类: 复用 s3b_classification_nested_dump 的 run_outer(5x10 重复分层,
          折内 StandardScaler->SMOTE->RF)。指标 balanced-acc / AUC / F1 / MCC。
全队列(无 wab_days 过滤)同样跑一遍作锚,直接和子集对比。

防泄露(逐条)
------------
1) wab_days 过滤是【进 CV 之前的纯行选择】: 只按 "wab_days >= 阈值" 留人,不碰 target、
   不用任何特征/标签统计量,不是数据依赖的变换 -> 不引入泄露。
2) 过滤后的子集整套交给原脚本的 fold 逻辑: scaler/SMOTE 仍只在折内训练子折 fit_transform,
   外层测试折只 transform/predict。与主线口径一字不差(直接 import 复用,不复制逻辑)。
3) wab_days 仅用于选行,绝不进特征矩阵 X(过滤后立刻丢弃该列)。

用法
----
  # SMOKE(只 RF,极小网格,少 repeats,验证跑通):
  PYENV_VERSION=data-analysis python src/s9_chronicity_sensitivity.py --smoke

  # 全量(default + tuned,JHU + arterial,>=180 + >=365 + 全队列锚):
  PYENV_VERSION=data-analysis python src/s9_chronicity_sensitivity.py
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent  # literature_ieee_access_arc/
SRC = PKG / "src"
sys.path.insert(0, str(SRC))

# 直接复用主线脚本的核心函数(同一份 fold/防泄露逻辑,不复制)
import s5_nested_tuning_regression as reg   # noqa: E402
import s3b_classification_nested_dump as clf  # noqa: E402

PARTICIPANTS = ROOT / "datasets" / "arc_ds004884" / "participants.tsv"
TABLES = {
    "JHU_fine": PKG / "data" / "jhu156_features_merged.tsv",
    "arterial_coarse": PKG / "data" / "ArterialAtlas156_features_merged.tsv",
}
THRESHOLDS = [None, 180, 365]   # None = 全队列锚
SEED = reg.SEED


def load_wab_days():
    """participant_id -> wab_days(数值)。"""
    pt = pd.read_csv(PARTICIPANTS, sep="\t")
    wd = pd.to_numeric(pt["wab_days"], errors="coerce")
    return pd.Series(wd.values, index=pt["participant_id"].values, name="wab_days")


def filtered_ids(table_path, wab_days, min_days):
    """返回该表上、target 非 NA、wab_days>=min_days 的 participant_id 列表(保持表内顺序)。
    min_days=None 表示不过滤(全队列)。过滤只用 wab_days 这一列,不碰特征/标签。"""
    df = pd.read_csv(table_path, sep="\t")
    ids = df["participant_id"]
    keep_target = df["wab_aq"].notna().values   # 与主线一致: 先去掉无标签的人
    if min_days is None:
        mask = keep_target
    else:
        d = ids.map(wab_days).values
        mask = keep_target & (d >= min_days)
    return ids[mask].tolist()


# ---------------- 回归(s5 口径) ----------------
def run_regression(name, table_path, keep_ids, do_tuned, n_repeats, smoke):
    """复用 s5 的 load_xy / run_default / run_tuned / metrics_with_ci,
    在过滤后的子集上跑 RandomForest。X/y 由 reg.load_xy 加载后按 keep_ids 取子集。"""
    X, y, feat = reg.load_xy(str(table_path))   # 与主线同清洗(drop NA target->fillna0->丢稀疏列)
    keep = [i for i in keep_ids if i in X.index]
    Xs = X.loc[keep]
    ys = y.loc[keep]
    outer = reg.RepeatedStratifiedKFold(
        n_splits=reg.N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    out = {}

    t0 = time.time()
    yt_d, yp_d, _ = reg.run_default("RandomForest", Xs, ys, outer, n_repeats)
    m_d, ci_d = reg.metrics_with_ci(yt_d, yp_d)
    out["default"] = (m_d, ci_d, time.time() - t0)
    print(f"    [reg/{name}] DEFAULT r={m_d['pearson_r']:.3f} "
          f"{reg.fmt_ci(ci_d['pearson_r'])} MAE={m_d['mae']:.2f} "
          f"R2={m_d['r2']:.3f}", flush=True)

    if do_tuned:
        t1 = time.time()
        yt_t, yp_t, _, _ = reg.run_tuned("RandomForest", Xs, ys, outer, n_repeats, smoke)
        m_t, ci_t = reg.metrics_with_ci(yt_t, yp_t)
        out["tuned"] = (m_t, ci_t, time.time() - t1)
        print(f"    [reg/{name}] TUNED   r={m_t['pearson_r']:.3f} "
              f"{reg.fmt_ci(ci_t['pearson_r'])} MAE={m_t['mae']:.2f} "
              f"R2={m_t['r2']:.3f}", flush=True)
    return len(Xs), Xs.shape[1], out


# ---------------- 分类(s3b 口径) ----------------
def run_classification(name, table_path, keep_ids, do_tuned, smoke):
    """复用 s3b 的 load_xy / run_outer / bootstrap_ci,
    在过滤后的子集上跑 RandomForest(折内 SMOTE)。"""
    X, y, wab, pids, feat = clf.load_xy(str(table_path))  # numpy + pids 列表
    pid_idx = {p: k for k, p in enumerate(pids)}
    sel = [pid_idx[i] for i in keep_ids if i in pid_idx]
    sel = np.asarray(sorted(sel))
    Xs, ys, wabs = X[sel], y[sel], wab[sel]
    pids_s = [pids[k] for k in sel]
    out = {}

    for tuned in ([False, True] if do_tuned else [False]):
        tag = "tuned" if tuned else "default"
        t0 = time.time()
        summary, _, agg = clf.run_outer(
            "RandomForest", Xs, ys, wabs, pids_s, tuned, smoke=smoke, verbose=False)
        ci = clf.bootstrap_ci(agg["y_true"], agg["y_pred"], agg["y_prob"])
        out[tag] = (summary, ci, time.time() - t0)
        ba, _ = summary["balanced_acc"]
        auc, _ = summary["auc"]
        print(f"    [clf/{name}] {tag:7s} bAcc={clf.fmt_ci(ba, ci['balanced_acc'])} "
              f"AUC={auc:.3f} F1={summary['f1_severe'][0]:.3f} "
              f"MCC={summary['mcc'][0]:.3f}", flush=True)
    return len(ys), int(ys.sum()), Xs.shape[1], out


def main():
    ap = argparse.ArgumentParser(description="慢性期敏感性: RF 回归+分类,wab_days 子集")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 极小网格 + 少 repeats(分类仍 5x10,小网格)")
    ap.add_argument("--no-tuned", action="store_true", help="只跑 default")
    args = ap.parse_args()

    do_tuned = not args.no_tuned
    n_repeats_reg = 2 if args.smoke else reg.N_REPEATS
    wab_days = load_wab_days()

    rows = []
    for gran, table in TABLES.items():
        for thr in THRESHOLDS:
            subset = "full" if thr is None else f"wab_days_ge_{thr}"
            keep = filtered_ids(table, wab_days, thr)
            print(f"\n=== {gran} | {subset} | n={len(keep)} ===", flush=True)

            # 回归
            n_reg, nfeat_reg, reg_out = run_regression(
                gran, table, keep, do_tuned, n_repeats_reg, args.smoke)
            # 分类
            n_clf, n_sev, nfeat_clf, clf_out = run_classification(
                gran, table, keep, do_tuned, args.smoke)

            for cfg in (["default", "tuned"] if do_tuned else ["default"]):
                row = {
                    "granularity": gran,
                    "subset": subset,
                    "min_wab_days": (0 if thr is None else thr),
                    "config": cfg,
                    "n": n_clf,
                    "n_severe": n_sev,
                    "severe_prevalence": round(n_sev / n_clf, 4),
                    "n_features": nfeat_clf,
                }
                # 回归
                if cfg in reg_out:
                    m, ci, sec = reg_out[cfg]
                    row.update({
                        "reg_pearson_r": round(m["pearson_r"], 4),
                        "reg_r_CI": reg.fmt_ci(ci["pearson_r"]),
                        "reg_mae": round(m["mae"], 3),
                        "reg_rmse": round(m["rmse"], 3),
                        "reg_r2": round(m["r2"], 4),
                        "reg_seconds": round(sec, 1),
                    })
                # 分类
                if cfg in clf_out:
                    summ, ci, sec = clf_out[cfg]
                    row.update({
                        "clf_balanced_acc": round(summ["balanced_acc"][0], 4),
                        "clf_bAcc_CI": clf.fmt_ci(summ["balanced_acc"][0], ci["balanced_acc"]),
                        "clf_auc": round(summ["auc"][0], 4),
                        "clf_auc_CI": clf.fmt_ci(summ["auc"][0], ci["auc"]),
                        "clf_f1_severe": round(summ["f1_severe"][0], 4),
                        "clf_mcc": round(summ["mcc"][0], 4),
                        "clf_seconds": round(sec, 1),
                    })
                rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = PKG / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"chronicity_sensitivity{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsummary -> {out_path}", flush=True)
    # 简明对比打印
    cols = ["granularity", "subset", "n", "n_severe", "config",
            "reg_pearson_r", "clf_balanced_acc"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
