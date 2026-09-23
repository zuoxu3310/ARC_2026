#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s10_optuna_ceiling_regression.py
================================
天花板压力测试(回归 WAB-AQ): 用 Optuna 自动搜索代替手工小网格, 在与主榜
**完全相同**的 5x10 外层协议下, 给搜索最大机会去冲 Pearson r。

这张实验回答的是【我们搜得够不够】, 不是【任务到顶没到顶】:
  - "trial 涨点曲线变平" = 超参搜索收敛(我们这辆车调到头了)。它只能反驳
    "你网格太小"。**不能**单独当"病灶图信息天花板"的证据。
  - "天花板" 由 held-out 分(外层测试折)落在 ~0.70 + 地板线 + 跨家族收敛 +
    0/8 不显著/TOST 等价 这几条扛, 与本脚本相互独立。

== 防泄漏(与 s5 同口径) ==
  1) StandardScaler 永远是 Pipeline 第一步, 进搜索后只在每个【内层训练子折】fit。
  2) Optuna 只见外层训练折 X[tr]; 目标 = 外层训练折上再切 INNER 折的 CV 平均 r。
     外层测试折 X[te] 绝不参与搜索/选参, 只在选出最优参后预测一次。
  3) 报两个分: held-out 池化 OOF r(+bootstrap CI) 和逐折平均 r —— 与主榜同尺,
     可直接比较 / 配对检验。搜索内部的 best-r 只用于画收敛曲线, 不对外当成绩。

== 用法 ==
  PYENV_VERSION=data-analysis python src/s10_optuna_ceiling_regression.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine --trials 50
  # smoke(验证跑通+无泄漏): --smoke (2 外折 x 8 trials)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")    # 静音 sklearn/LightGBM 的特征名等无害告警

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import optuna

# 复用回归 harness 的数据加载 / 指标 / bootstrap CI(与 s5 同口径)
HARNESS_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "arc_wab_aq_benchmark", "src"))
sys.path.insert(0, HARNESS_SRC)
import data as data_mod          # noqa: E402  load_data
import evaluate as ev            # noqa: E402  regression_metrics / bootstrap_ci

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 42
N_SPLITS = 5
N_REPEATS = 10
N_STRAT_BINS = 4
INNER_SPLITS = 3              # 内层调参 CV 折数(与分类侧 s3b 一致, 3 折省时)
BOOTSTRAP_N = 2000
SPARSE_DROP_FRAC = 0.10

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_xy(table_path):
    abs_path = table_path if os.path.isabs(table_path) else os.path.join(PKG, table_path)
    cfg = {"data": {"path": abs_path, "sep": "\t", "id_col": "participant_id",
                    "target": "wab_aq", "sparse_drop_frac": SPARSE_DROP_FRAC}}
    X, y, feat = data_mod.load_data(cfg)
    return X, y, feat


def _strat_labels(y, n_bins):
    labels = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(labels)


def _make_estimator(trial):
    """从【放宽的连续/离散空间】采样: 先选模型类型, 再采该类型超参。
    返回 (estimator, model_type)。比 s5 的手工小网格宽得多。"""
    mt = trial.suggest_categorical(
        "model_type", ["RandomForest", "XGBoost", "LightGBM", "ElasticNet",
                       "Ridge", "SVR_rbf"])
    if mt == "RandomForest":
        est = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("rf_max_depth", 2, 20),
            max_features=trial.suggest_float("rf_max_features", 0.1, 1.0),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 8),
            random_state=SEED, n_jobs=-1)
    elif mt == "XGBoost":
        from xgboost import XGBRegressor
        est = XGBRegressor(
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 800, step=100),
            learning_rate=trial.suggest_float("xgb_lr", 1e-2, 3e-1, log=True),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 8),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
            random_state=SEED, n_jobs=-1, verbosity=0)
    elif mt == "LightGBM":
        from lightgbm import LGBMRegressor
        est = LGBMRegressor(
            n_estimators=trial.suggest_int("lgb_n_estimators", 200, 800, step=100),
            learning_rate=trial.suggest_float("lgb_lr", 1e-2, 3e-1, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 7, 127),
            subsample=trial.suggest_float("lgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
            random_state=SEED, n_jobs=-1, verbose=-1)
    elif mt == "ElasticNet":
        est = ElasticNet(
            alpha=trial.suggest_float("en_alpha", 1e-3, 10.0, log=True),
            l1_ratio=trial.suggest_float("en_l1_ratio", 0.05, 0.95),
            random_state=SEED, max_iter=10000)
    elif mt == "Ridge":
        est = Ridge(
            alpha=trial.suggest_float("ridge_alpha", 1e-3, 1e2, log=True),
            random_state=SEED)
    else:  # SVR_rbf
        est = SVR(
            kernel="rbf",
            C=trial.suggest_float("svr_C", 1e-1, 1e2, log=True),
            gamma=trial.suggest_float("svr_gamma", 1e-3, 1.0, log=True))
    return est, mt


def _inner_cv_pearson(est, Xtr, ytr, bins_tr):
    """内层 CV 平均 Pearson r。scaler 在每个内层训练子折 fit -> 无泄漏。"""
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    rs = []
    for itr, iva in inner.split(Xtr, bins_tr):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", est)])
        pipe.fit(Xtr.iloc[itr], ytr.iloc[itr])
        pred = pipe.predict(Xtr.iloc[iva])
        yv = ytr.iloc[iva].to_numpy()
        if np.std(pred) < 1e-12:        # 退化预测(常数), r 未定义 -> 记 0
            rs.append(0.0)
        else:
            rs.append(pearsonr(yv, pred)[0])
    return float(np.mean(rs))


def run_optuna_nested(name, X, y, outer, n_trials, smoke):
    """嵌套: 每个外层折跑一套 Optuna(只见训练折), 选最优参 refit -> 预测测试折一次。
    返回 held-out 池化预测 + 逐折 r + 每折的 trial 收敛史(best-so-far)。"""
    yt = np.empty(len(y), dtype=float)
    yp = np.empty(len(y), dtype=float)
    strat = _strat_labels(y, N_STRAT_BINS)
    per_fold_r, chosen_types = [], []
    trial_curves = []                    # 每折一条 best-r-so-far 曲线(长度 n_trials)
    n_folds = outer.get_n_splits()
    t0 = time.time()
    for i, (tr, te) in enumerate(outer.split(X, strat)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        bins_tr = strat[tr]              # 外层训练折的分层箱标签, 供内层分层

        def objective(trial):
            est, _mt = _make_estimator(trial)
            return _inner_cv_pearson(est, Xtr, ytr, bins_tr)

        sampler = optuna.samplers.TPESampler(seed=SEED + i)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # best-so-far 收敛曲线(画图用; 这是搜索内部目标, 不是对外成绩)
        best_so_far, cur = [], -np.inf
        for t in study.trials:
            v = t.value if t.value is not None else -np.inf
            cur = max(cur, v)
            best_so_far.append(cur)
        trial_curves.append(best_so_far)

        # 用最优参 refit 在【整个外层训练折】, 只预测外层测试折一次
        best_trial = study.best_trial
        est, mt = _make_estimator(optuna.trial.FixedTrial(best_trial.params))
        final = Pipeline([("scaler", StandardScaler()), ("model", est)])
        final.fit(Xtr, ytr)
        pred = final.predict(X.iloc[te])
        yp[te] = pred
        yt[te] = y.iloc[te].to_numpy()
        # 逐折 held-out r(外层测试折)
        if np.std(pred) > 1e-12:
            per_fold_r.append(pearsonr(yt[te], pred)[0])
        else:
            per_fold_r.append(0.0)
        chosen_types.append(mt)
        if (i + 1) % 5 == 0 or (i + 1) == n_folds:
            dt = time.time() - t0
            print(f"    [optuna:{name}] outer {i+1}/{n_folds}  "
                  f"best_inner_r={study.best_value:.3f}  pick={mt}  "
                  f"held-out fold r={per_fold_r[-1]:.3f}  ({dt:.0f}s)", flush=True)
    return yt, yp, per_fold_r, chosen_types, trial_curves


def main():
    ap = argparse.ArgumentParser(description="Optuna 天花板压力测试(回归)")
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_repeats = 2 if args.smoke else N_REPEATS
    n_trials = 8 if args.smoke else args.trials
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                    random_state=SEED)
    X, y, feat = load_xy(args.table)
    print(f"[{args.name}] n={len(y)} features={X.shape[1]} "
          f"outer={N_SPLITS}x{n_repeats} trials={n_trials} smoke={args.smoke}",
          flush=True)

    yt, yp, per_fold_r, chosen_types, trial_curves = run_optuna_nested(
        args.name, X, y, outer, n_trials, args.smoke)

    # held-out 池化 OOF: 与主榜/配对检验同尺
    m = ev.regression_metrics(yt, yp)
    ci = ev.bootstrap_ci(yt, yp, BOOTSTRAP_N, SEED)
    pooled_r = m["pearson_r"]
    fold_mean_r = float(np.mean(per_fold_r))
    types, counts = np.unique(chosen_types, return_counts=True)
    type_dist = ", ".join(f"{t}:{c}" for t, c in zip(types, counts))

    print(f"\n[{args.name}] === 天花板压力测试结果 ===", flush=True)
    print(f"  held-out 池化 OOF: r={pooled_r:.3f} {ev_ci_fmt(ci)}  "
          f"MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} R2={m['r2']:.3f}", flush=True)
    print(f"  held-out 逐折平均 r={fold_mean_r:.3f} (n_folds={len(per_fold_r)})",
          flush=True)
    print(f"  选中模型类型分布: {type_dist}", flush=True)

    out_dir = os.path.join(PKG, "results", "summary")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    # 结果表
    pd.DataFrame([{
        "dataset": args.name, "n_trials": n_trials,
        "heldout_pooled_r": round(pooled_r, 4),
        "heldout_pooled_r_ci95": f"[{ci['pearson_r'][0]:.3f}, {ci['pearson_r'][1]:.3f}]",
        "heldout_fold_mean_r": round(fold_mean_r, 4),
        "mae": round(m["mae"], 3), "rmse": round(m["rmse"], 3), "r2": round(m["r2"], 4),
        "model_type_dist": type_dist,
    }]).to_csv(os.path.join(out_dir, f"optuna_ceiling_regression_{args.name}{suffix}.csv"),
               index=False)

    # 收敛曲线(逐 trial 跨折 best-so-far 的均值 + IQR), 画图用
    curves = np.array(trial_curves)                  # [n_folds, n_trials]
    mean_c = curves.mean(axis=0)
    q25, q75 = np.percentile(curves, [25, 75], axis=0)
    pd.DataFrame({
        "trial": np.arange(1, n_trials + 1),
        "best_inner_r_mean": np.round(mean_c, 4),
        "best_inner_r_q25": np.round(q25, 4),
        "best_inner_r_q75": np.round(q75, 4),
    }).to_csv(os.path.join(out_dir, f"optuna_trials_regression_{args.name}{suffix}.csv"),
              index=False)
    print(f"  写出: optuna_ceiling_regression_{args.name}{suffix}.csv + "
          f"optuna_trials_regression_{args.name}{suffix}.csv", flush=True)


def ev_ci_fmt(ci):
    c = ci["pearson_r"]
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


if __name__ == "__main__":
    main()
