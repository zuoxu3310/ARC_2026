#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s20_matchedN_permutation.py
===========================
任务3 第 4 步: 匹配-N 成分对照(正式置换检验)。驳"换子集只是样本量小所以抖"。

设计(refine-logs/TASK3_DESIGN_PREREG_2026-06-13.md §5):
  对每个标准子集(大小 n_k), 从 Full 抽 B 个【按 WAB-AQ 直方图分层】的随机子样
  —— 用 50(severe 边界)和 90(天花板)当分箱边界, 一次匹配住 N + severe 先验 + 量程 SD。
  每随机子样整管线从头重训(折划分+scaler+SMOTE+模型全刷新, 不共享任何 fit/折索引)。
  建经验零分布 -> 报子集指标的双侧经验 p + 成分效应(子集分 - 零均值) + 是否落零带外。
  aq_le90 的匹配抽样自动约束到 y<=90 支撑 -> 同时控住量程截断。

判定规则(跑前定): α=0.05; 落带外(p<0.05) = 摆幅是成分不是样本量; 落带内如实报。
RNG: 抽样 RNG 与 CV RNG 独立且固定; 每抽样独立重训。

用法: PYENV_VERSION=data-analysis python src/s20_matchedN_permutation.py --task regression --B 1000
"""
from __future__ import annotations
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 复用 s18 的管线/模型/单元评估, 保证同一条固定管线
import s18_crossed_grid as g

PKG = g.PKG
SUBSET_DIR = g.SUBSET_DIR
SUMM = os.path.join(PKG, "results", "summary")
BIN_EDGES = [-0.1, 25, 50, 70, 90, 100.1]   # 50=severe 边界, 90=天花板边界
CV_SEED = 2024
DRAW_BASE = 7000


def bin_counts(y_aq, edges):
    b = pd.cut(y_aq, bins=edges, labels=False, include_lowest=True)
    return b, pd.Series(b).value_counts().sort_index()


def eval_subset(X, y_aq, model, task, seed=CV_SEED):
    runner = g.run_cell_regression if task == "regression" else g.run_cell_classification
    return runner(X, y_aq, model, seed)["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    models = args.models or (["RandomForest", "Ridge"] if args.task == "regression"
                             else ["RandomForest", "LogReg"])

    X_full, y_full, keep = g.load_full_pool()
    full_bins, _ = bin_counts(y_full, BIN_EDGES)
    full_bins = pd.Series(full_bins.values, index=X_full.index)
    # 每个 bin 在 Full 里的成员(键强制 int, 防 pd.cut 浮点码与 need 的 int 键对不上)
    bin_members = {int(b): full_bins.index[full_bins == b].tolist()
                   for b in sorted(full_bins.dropna().unique())}

    man = pd.read_csv(os.path.join(SUBSET_DIR, "subset_manifest.csv"))
    # 只对标准子集(去 full 基线)做对照; teghipco 精确同人也跑(当锚点)
    targets = [s for s in man["subset"].tolist() if s != "full"]

    rng = np.random.default_rng(DRAW_BASE)
    rows = []
    t0 = time.time()
    for sub in targets:
        ids = [i for i in g.load_subset_ids(sub) if i in X_full.index]
        Xs, ys = X_full.loc[ids], y_full.loc[ids]
        # 子集的 bin 直方图
        sub_bins, sub_hist = bin_counts(ys, BIN_EDGES)
        need = {int(b): int(c) for b, c in sub_hist.items()}
        for model in models:
            obs = eval_subset(Xs, ys, model, args.task)
            # 零分布: B 次匹配直方图随机抽样
            null = []
            for _ in range(args.B):
                draw_ids = []
                for b, c in need.items():
                    pool_b = bin_members.get(b, [])
                    if c > len(pool_b):
                        c = len(pool_b)
                    draw_ids += list(rng.choice(pool_b, size=c, replace=False))
                Xd, yd = X_full.loc[draw_ids], y_full.loc[draw_ids]
                null.append(eval_subset(Xd, yd, model, args.task))
            null = np.array(null)
            null_mean = float(null.mean())
            lo, hi = float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))
            # 双侧经验 p
            p_two = 2 * min(np.mean(null >= obs), np.mean(null <= obs))
            p_two = float(min(1.0, p_two))
            comp = obs - null_mean
            outside = not (lo <= obs <= hi)
            rows.append({"task": args.task, "subset": sub, "model": model,
                         "n": len(ids), "obs_score": round(obs, 4),
                         "null_mean": round(null_mean, 4),
                         "null_ci_lo": round(lo, 4), "null_ci_hi": round(hi, 4),
                         "composition_effect": round(comp, 4),
                         "p_two_sided": round(p_two, 4),
                         "outside_band": outside, "B": args.B})
            print(f"  [{sub:18s}|{model:13s}] obs={obs:.3f} null={null_mean:.3f}"
                  f"[{lo:.3f},{hi:.3f}] comp={comp:+.3f} p={p_two:.3f} "
                  f"{'OUT' if outside else 'in'} ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(SUMM, exist_ok=True)
    out.to_csv(os.path.join(SUMM, f"matchedN_perm_{args.task}.csv"), index=False)
    n_out = int(out["outside_band"].sum())
    print(f"\n[write] matchedN_perm_{args.task}.csv  "
          f"{n_out}/{len(out)} (子集×模型) 落零带外(成分效应显著)", flush=True)


if __name__ == "__main__":
    main()
