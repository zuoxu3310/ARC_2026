#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s18_crossed_grid.py
===================
任务3 第 2 步: 全交叉网格 (subset × model × seed) × 固定 JHU 细特征管线。
每个单元 = 一次 5 折分层 CV(种子统一驱动折划分+SMOTE+模型), 出一个分数。
供 s19 方差分解 / s20 匹配-N 置换 / s21 量程校正 用。

设计依据: refine-logs/TASK3_DESIGN_PREREG_2026-06-13.md
- 固定管线: 折内 StandardScaler(+SMOTE 分类), 默认超参, 逐折刷新, 泄漏安全。
- 稀疏列在【全池 226】上算一次冻结, 所有子集用同一套特征列(防稀疏过滤随子集漂)。
- 单元种子 = 折划分 + SMOTE + 模型 random_state 统一(seed 轴 = 完整管线随机性 = "换划分")。
- TabPFN 必须单独进程: PYENV_VERSION=... python src/s18_crossed_grid.py --task regression --models TabPFN
  且套硬超时(perl alarm)。绝不跟在别的模型后同进程。

用法(快模型):
  PYENV_VERSION=data-analysis python src/s18_crossed_grid.py --task regression --seeds 20
  PYENV_VERSION=data-analysis python src/s18_crossed_grid.py --task classification --seeds 20
TabPFN(单独, 限种子, 套超时):
  perl -e 'alarm 7200; exec @ARGV' env PYENV_VERSION=data-analysis \
      python src/s18_crossed_grid.py --task regression --models TabPFN --seeds 10
"""
from __future__ import annotations
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, mean_absolute_error, mean_squared_error)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBSET_DIR = os.path.join(PKG, "data", "subsets")
TABLE = os.path.join(PKG, "data", "jhu156_features_merged.tsv")
SEVERE_THR = 50.0
SPARSE_DROP_FRAC = 0.10
N_SPLITS = 5
N_STRAT_BINS = 4

REG_FAST = ["Ridge", "ElasticNet", "SVR_rbf", "RandomForest", "XGBoost", "LightGBM", "MLP"]
CLF_FAST = ["LogReg", "SVM_rbf", "RandomForest", "XGBoost", "LightGBM", "MLP"]


# ---------------- 模型工厂(复用 pipelines.py / s3 的确切定义) ----------------
def reg_estimator(name, seed):
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "ElasticNet":
        return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=seed, max_iter=10000)
    if name == "SVR_rbf":
        return SVR(kernel="rbf")
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, n_jobs=-1, verbosity=0)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                            max_iter=2000, early_stopping=False, random_state=seed)
    if name == "TabPFN":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device="cpu", random_state=seed)
    raise ValueError(name)


def clf_estimator(name, seed):
    if name == "LogReg":
        return LogisticRegression(max_iter=5000, random_state=seed)
    if name == "SVM_rbf":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed,
                             n_jobs=-1, verbosity=0, eval_metric="logloss")
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.8, random_state=seed,
                              n_jobs=-1, verbose=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=seed)
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu")
    raise ValueError(name)


# ---------------- 数据 ----------------
def load_full_pool():
    """全 226 池: X(DataFrame index=participant_id), y_aq。稀疏列在全池上冻结。"""
    df = pd.read_csv(TABLE, sep="\t", index_col="participant_id")
    df = df.dropna(subset=["wab_aq"]).fillna(0)
    y_aq = df["wab_aq"].copy()
    feat = df.drop(columns=["wab_aq"])
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    feat = feat[keep]
    return feat, y_aq, keep


def load_subset_ids(name):
    with open(os.path.join(SUBSET_DIR, f"{name}_ids.txt")) as f:
        return [x.strip() for x in f if x.strip()]


def strat_labels(y_aq, task):
    if task == "classification":
        return (y_aq.values <= SEVERE_THR).astype(int)
    # 回归: 分位分箱分层
    lab = pd.qcut(y_aq, q=N_STRAT_BINS, labels=False, duplicates="drop")
    return np.asarray(lab)


# ---------------- 单元: 一次 5 折 -> 一个分数 ----------------
def run_cell_regression(X, y_aq, model, seed):
    strat = strat_labels(y_aq, "regression")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    yt = np.empty(len(y_aq)); yp = np.empty(len(y_aq))
    Xv, yv = X.values, y_aq.values
    for tr, te in skf.split(Xv, strat):
        pipe = ImbPipeline([("scaler", StandardScaler()),
                            ("model", reg_estimator(model, seed))])
        pipe.fit(Xv[tr], yv[tr])
        yp[te] = pipe.predict(Xv[te])
        yt[te] = yv[te]
    r = pearsonr(yt, yp)[0] if np.std(yp) > 1e-12 else 0.0
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"score": float(r), "r": float(r), "mae": float(mae),
            "rmse": float(rmse), "r2": float(r2), "smote_skips": 0}


def run_cell_classification(X, y_aq, model, seed):
    y = (y_aq.values <= SEVERE_THR).astype(int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    Xv = X.values
    yt = np.empty(len(y), int); yp = np.empty(len(y), int); pr = np.zeros(len(y))
    skips = 0
    for tr, te in skf.split(Xv, y):
        n_min = int(min(np.bincount(y[tr])))
        if model == "TabPFN":
            steps = [("scaler", StandardScaler())]
            if n_min > 1:
                steps.append(("smote", SMOTE(random_state=seed,
                                             k_neighbors=min(5, n_min - 1))))
            else:
                skips += 1
            steps.append(("clf", clf_estimator(model, seed)))
            pipe = ImbPipeline(steps)
        else:
            steps = [("scaler", StandardScaler())]
            if n_min > 1:
                steps.append(("smote", SMOTE(random_state=seed,
                                             k_neighbors=min(5, n_min - 1))))
            else:
                skips += 1
            steps.append(("clf", clf_estimator(model, seed)))
            pipe = ImbPipeline(steps)
        pipe.fit(Xv[tr], y[tr])
        yp[te] = pipe.predict(Xv[te])
        try:
            pr[te] = pipe.predict_proba(Xv[te])[:, 1]
        except Exception:
            pr[te] = yp[te]
        yt[te] = y[te]
    return {"score": float(balanced_accuracy_score(yt, yp)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            "auc": float(roc_auc_score(yt, pr)) if len(np.unique(yt)) > 1 else np.nan,
            "f1_severe": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(yt, yp)),
            "smote_skips": skips}


# ---------------- 主 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--base_seed", type=int, default=1000)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="默认跑 manifest 全部子集")
    args = ap.parse_args()

    models = args.models or (REG_FAST if args.task == "regression" else CLF_FAST)
    man = pd.read_csv(os.path.join(SUBSET_DIR, "subset_manifest.csv"))
    subsets = args.subsets or man["subset"].tolist()
    seeds = [args.base_seed + i for i in range(args.seeds)]
    runner = run_cell_regression if args.task == "regression" else run_cell_classification

    X_full, y_full, keep = load_full_pool()
    print(f"[grid:{args.task}] 全池 N={len(X_full)} features={len(keep)} "
          f"models={models} subsets={len(subsets)} seeds={len(seeds)} "
          f"= {len(models)*len(subsets)*len(seeds)} cells", flush=True)
    man_idx = man.set_index("subset")

    rows = []
    t0 = time.time()
    for si, sub in enumerate(subsets):
        ids = [i for i in load_subset_ids(sub) if i in X_full.index]
        Xs = X_full.loc[ids]; ys = y_full.loc[ids]
        nrow = man_idx.loc[sub]
        for model in models:
            tc = time.time()
            for seed in seeds:
                res = runner(Xs, ys, model, seed)
                rows.append({"task": args.task, "subset": sub, "category": nrow["category"],
                             "model": model, "seed": seed, "n": len(ids),
                             "severe_prior": nrow["severe_prior"], "target_sd": nrow["target_sd"],
                             **{k: v for k, v in res.items()}})
            dt = time.time() - tc
            cell = [r for r in rows if r["subset"] == sub and r["model"] == model]
            sc = np.mean([r["score"] for r in cell])
            print(f"  [{si+1}/{len(subsets)} {sub:18s} | {model:13s}] "
                  f"mean_score={sc:.3f} over {len(seeds)} seeds ({dt:.1f}s)", flush=True)

    out = pd.DataFrame(rows)
    out_dir = os.path.join(PKG, "results", "runs")
    os.makedirs(out_dir, exist_ok=True)
    # 按 (task,模型集) 区分 TabPFN 单独进程的输出, 合并而非覆盖
    tag = "TabPFN" if models == ["TabPFN"] else "fast"
    path = os.path.join(out_dir, f"grid_{args.task}_{tag}.csv")
    out.to_csv(path, index=False)
    print(f"\n[write] {path}  ({len(out)} rows)  总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
