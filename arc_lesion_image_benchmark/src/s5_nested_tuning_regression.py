#!/usr/bin/env python
"""
s5_nested_tuning_regression.py  ——  泄露安全的【回归】嵌套调参 vs 默认对照
(对应 EXPERIMENT_PLAN.md Block 3 / Anti-claim A1:调到最好也破不了 WAB-AQ 天花板)

做什么
------
对同一批特征表(JHU 细 / 动脉粗)上的每个可调模型, 给两套数:
  * tuned   : 外层重复分层 K 折; 每个外层训练折上用 GridSearchCV(内层 K 折)调关键超参,
              best_estimator_ 在该外层训练折上 refit, 只对外层测试折预测。
  * default : 复用现有 harness 口径(pipelines.build_pipeline + 默认超参), 同一套外层折, 折内 fit。
两套都池化折外预测算 Pearson r / MAE / RMSE / R2, 并做 patient-level bootstrap CI(复用 evaluate.bootstrap_ci)。

防泄露(逐条对照 CLAUDE.md 铁律)
--------------------------------
1) StandardScaler 是 Pipeline 第一步, 整条 Pipeline 进 GridSearchCV ->
   scaler 只在每个 inner-train 子折 fit_transform, 从不见 inner-val, 更不见外层测试折。
2) GridSearchCV.fit 只喂 X.iloc[tr](外层训练折); 选参 + refit 都只在训练折内完成。
3) 外层测试折 X.iloc[te] 只用于 .predict, 全程没有任何 fit 碰过它。
4) 外层用 wab_aq 分位分箱做分层(连续目标), 分箱标签只用于 split, 不进特征。
5) 调参与评估分离: 内层 CV 选参, 外层折只评估, 不在同一批数据上又调又报分(嵌套)。

TabPFN
------
按设计【不进网格、不调参】(默认即卖点)。只在 --models 显式含 TabPFN 时跑一套 default,
且应当【单独进程 + 硬超时】启动(见文件尾命令), 不要和其它模型同进程。

用法
----
  # SMOKE(冒烟, 只 RandomForest 极小网格, 少 repeats):
  PYENV_VERSION=data-analysis python src/s5_nested_tuning_regression.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine \
      --models RandomForest --smoke

  # 全量(全模型全网格, 由主控后台跑):
  PYENV_VERSION=data-analysis python src/s5_nested_tuning_regression.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine \
      --models Ridge ElasticNet SVR_rbf RandomForest XGBoost LightGBM

  # TabPFN 单独进程 + 硬超时(只 default):
  perl -e 'alarm 1800; exec @ARGV' env PYENV_VERSION=data-analysis python \
      src/s5_nested_tuning_regression.py --table data/jhu156_features_merged.tsv \
      --name JHU_fine --models TabPFN
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# 复用回归 harness 的数据加载 / 指标 / bootstrap CI / 默认 Pipeline 口径
HARNESS_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "arc_wab_aq_benchmark", "src"))
sys.path.insert(0, HARNESS_SRC)
import data as data_mod          # noqa: E402  load_data / describe
import evaluate as ev            # noqa: E402  regression_metrics / bootstrap_ci
from pipelines import build_pipeline  # noqa: E402  默认口径(StandardScaler -> est)

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 42
N_SPLITS = 5
N_REPEATS = 10
N_STRAT_BINS = 4          # wab_aq 分位分箱数(连续目标分层用), 226 人 / 5 折下每箱足够
INNER_SPLITS = 5
BOOTSTRAP_N = 2000
SPARSE_DROP_FRAC = 0.10

# 可调模型 + tuned 估计器 + 网格(小而合理)。键名 "model__<param>" 对准 Pipeline 步 "model"。
TUNABLE = {
    "Ridge", "ElasticNet", "SVR_rbf", "RandomForest", "XGBoost", "LightGBM",
}
# TabPFN 不在 TUNABLE: 永远只跑 default。


def _tuned_estimator_and_grid(name, smoke):
    """返回 (fresh estimator, param_grid)。smoke 时网格压到最小。"""
    if name == "Ridge":
        est = Ridge(random_state=SEED)
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif name == "ElasticNet":
        est = ElasticNet(random_state=SEED, max_iter=10000)
        grid = {"model__alpha": [0.1, 1.0, 10.0],
                "model__l1_ratio": [0.2, 0.5, 0.8]}
    elif name == "SVR_rbf":
        est = SVR(kernel="rbf")
        grid = {"model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01, 0.1]}
    elif name == "RandomForest":
        est = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        grid = {"model__n_estimators": [300, 500],
                "model__max_depth": [None, 5, 10],
                "model__max_features": ["sqrt", 0.5]}
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        est = XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0)
        grid = {"model__n_estimators": [300, 500],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.8, 1.0]}
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        est = LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)
        grid = {"model__n_estimators": [300, 500],
                "model__learning_rate": [0.03, 0.1],
                "model__num_leaves": [15, 31],
                "model__subsample": [0.8, 1.0]}
    else:
        raise ValueError(f"{name} 不是可调模型")

    if smoke:
        # SMOKE: 极小网格(只验证嵌套逻辑跑通, 不求最优)
        if name == "RandomForest":
            grid = {"model__n_estimators": [200, 400]}
        else:
            # 取每个超参第一个候选退化成 2 个点之内
            grid = {k: v[:2] for k, v in grid.items()}
    return est, grid


def _build_tuned_pipe(name, smoke):
    """tuned 用的 Pipeline + 网格。StandardScaler 第一步 -> 进 GridSearchCV 后折内 fit。"""
    est, grid = _tuned_estimator_and_grid(name, smoke)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
    return pipe, grid


def _strat_labels(y, n_bins):
    """连续 wab_aq -> 分位箱标签, 仅供外层分层 split 使用, 不进特征矩阵。"""
    # qcut 在重复值多时可能箱数不足, duplicates='drop' 兜底
    labels = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(labels)


def run_default(name, X, y, outer, n_repeats, device="cpu"):
    """默认超参口径: 复用 build_pipeline, 同一套外层折, 折内 fit。返回池化 OOF。

    额外返回 per_rep[n_samples, n_repeats]: 每人在每个 repeat 的测试折预测,
    仅供逐人 dump(跨 repeat 取均值)用, 不参与现有评测口径。
    """
    yt = np.empty(len(y), dtype=float)
    yp = np.empty(len(y), dtype=float)
    per_rep = np.full((len(y), n_repeats), np.nan, dtype=float)  # dump 用, 不动评测
    strat = _strat_labels(y, N_STRAT_BINS)
    n_folds = outer.get_n_splits()
    for i, (tr, te) in enumerate(outer.split(X, strat)):
        pipe = build_pipeline(name, SEED, device)     # 默认超参 Pipeline(scaler->est)
        pipe.fit(X.iloc[tr], y.iloc[tr])              # 只见外层训练折
        pred = pipe.predict(X.iloc[te])               # 只预测外层测试折
        yp[te] = pred                                 # 评测口径: 原样不动(末 repeat 覆盖)
        yt[te] = y.iloc[te].to_numpy()
        per_rep[te, i // N_SPLITS] = pred             # dump: 按 repeat 列累积, 每列填满全样本
        if (i + 1) % n_folds == 0 or (i + 1) % 5 == 0:
            print(f"    [default:{name}] fold {i+1}/{outer.get_n_splits()}", flush=True)
    return yt, yp, per_rep


def run_tuned(name, X, y, outer, n_repeats, smoke):
    """嵌套调参口径: 外层每个训练折上 GridSearchCV(内层 CV)选参 + refit, 只预测外层测试折。

    额外返回 per_rep[n_samples, n_repeats]: 每人在每个 repeat 的测试折预测,
    仅供逐人 dump(跨 repeat 取均值)用, 不参与现有评测口径。
    """
    yt = np.empty(len(y), dtype=float)
    yp = np.empty(len(y), dtype=float)
    per_rep = np.full((len(y), n_repeats), np.nan, dtype=float)  # dump 用, 不动评测
    strat = _strat_labels(y, N_STRAT_BINS)
    chosen = []
    n_folds = outer.get_n_splits()
    for i, (tr, te) in enumerate(outer.split(X, strat)):
        pipe, grid = _build_tuned_pipe(name, smoke)
        gs = GridSearchCV(
            pipe, grid,
            scoring="neg_mean_absolute_error",   # 与回归口径一致(优化 MAE)
            cv=INNER_SPLITS,                     # 内层 K 折(只切外层训练折)
            n_jobs=-1, refit=True,
        )
        gs.fit(X.iloc[tr], y.iloc[tr])           # 只喂外层训练折; 选参+refit 都在训练折内
        pred = gs.predict(X.iloc[te])            # best_estimator_ 只预测外层测试折
        yp[te] = pred                            # 评测口径: 原样不动(末 repeat 覆盖)
        yt[te] = y.iloc[te].to_numpy()
        per_rep[te, i // N_SPLITS] = pred         # dump: 按 repeat 列累积, 每列填满全样本
        chosen.append(gs.best_params_)
        if (i + 1) % n_folds == 0 or (i + 1) % 5 == 0:
            print(f"    [tuned:{name}] fold {i+1}/{outer.get_n_splits()} "
                  f"best={gs.best_params_}", flush=True)
    return yt, yp, per_rep, chosen


def metrics_with_ci(yt, yp):
    m = ev.regression_metrics(yt, yp)
    ci = ev.bootstrap_ci(yt, yp, BOOTSTRAP_N, SEED)
    return m, ci


def fmt_ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def dump_patient_pred(name, model, config, ids, yt, per_rep):
    """把逐人跨-repeat 平均预测写到 results/runs/reg_{粒度}_{模型}_{config}_patient_pred.csv。

    per_rep[n_samples, n_repeats]: 每列是一个 repeat 的全样本 OOF 预测。
    每人跨 repeat 取均值(nanmean 兜底, 正常每格都有值)得到 y_pred_mean。
    列: participant_id, y_true, y_pred_mean。只加 dump, 不碰评测口径。
    """
    y_pred_mean = np.nanmean(per_rep, axis=1)
    df = pd.DataFrame({
        "participant_id": np.asarray(ids),
        "y_true": np.asarray(yt, dtype=float),
        "y_pred_mean": y_pred_mean,
    })
    out_dir = os.path.join(PKG, "results", "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"reg_{name}_{model}_{config}_patient_pred.csv")
    df.to_csv(out_path, index=False)
    print(f"    dump -> {out_path}", flush=True)
    return out_path


def load_xy(table_path):
    """复用 harness data.load_data 的清洗口径(drop NA -> fillna0 -> 丢稀疏列, 不缩放)。"""
    abs_path = table_path if os.path.isabs(table_path) else os.path.join(PKG, table_path)
    cfg = {"data": {"path": abs_path, "sep": "\t", "id_col": "participant_id",
                    "target": "wab_aq", "sparse_drop_frac": SPARSE_DROP_FRAC}}
    X, y, feat = data_mod.load_data(cfg)
    return X, y, feat


def main():
    ap = argparse.ArgumentParser(description="泄露安全的回归嵌套调参 vs 默认")
    ap.add_argument("--table", required=True, help="特征表(相对 PKG 或绝对路径)")
    ap.add_argument("--name", required=True, help="数据集标签, 如 JHU_fine / arterial_coarse")
    ap.add_argument("--models", nargs="*",
                    default=["Ridge", "ElasticNet", "SVR_rbf", "RandomForest",
                             "XGBoost", "LightGBM"],
                    help="模型子集; TabPFN 只跑 default(不调参), 建议单独进程")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 极小网格 + 少 repeats, 只验证跑通无泄露")
    ap.add_argument("--repeats", type=int, default=None, help="覆盖外层 repeats")
    ap.add_argument("--device", default="cpu", help="TabPFN device")
    args = ap.parse_args()

    n_repeats = args.repeats if args.repeats is not None else (2 if args.smoke else N_REPEATS)
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                    random_state=SEED)

    X, y, feat = load_xy(args.table)
    print(f"[{args.name}] n={len(y)}  features={X.shape[1]}  "
          f"wab_aq {y.min():.0f}-{y.max():.0f}  "
          f"outer={N_SPLITS}x{n_repeats} stratified(bins={N_STRAT_BINS})  "
          f"smoke={args.smoke}", flush=True)

    ids = list(X.index)                          # participant_id(data.load_data 把 id 设成 index)
    dump_cfg_suffix = "_smoke" if args.smoke else ""  # smoke 的 dump 单独命名, 不污染正式产物

    rows = []
    for name in args.models:
        # default
        t0 = time.time()
        yt_d, yp_d, per_rep_d = run_default(name, X, y, outer, n_repeats, args.device)
        m_d, ci_d = metrics_with_ci(yt_d, yp_d)
        dt_d = time.time() - t0
        dump_patient_pred(args.name, name, "default" + dump_cfg_suffix, ids, yt_d, per_rep_d)

        row = {
            "dataset": args.name, "model": name,
            "default_r": round(m_d["pearson_r"], 4),
            "default_r_CI": fmt_ci(ci_d["pearson_r"]),
            "default_mae": round(m_d["mae"], 3),
            "default_mae_CI": fmt_ci(ci_d["mae"]),
            "default_rmse": round(m_d["rmse"], 3),
            "default_rmse_CI": fmt_ci(ci_d["rmse"]),
            "default_r2": round(m_d["r2"], 4),
            "default_r2_CI": fmt_ci(ci_d["r2"]),
            "default_seconds": round(dt_d, 1),
        }
        print(f"  {name:13s} DEFAULT r={m_d['pearson_r']:.3f} {fmt_ci(ci_d['pearson_r'])} "
              f"MAE={m_d['mae']:.2f} RMSE={m_d['rmse']:.2f} R2={m_d['r2']:.3f} ({dt_d:.1f}s)",
              flush=True)

        # tuned(TabPFN 不调; 其余可调模型才进网格)
        if name in TUNABLE:
            t1 = time.time()
            yt_t, yp_t, per_rep_t, chosen = run_tuned(name, X, y, outer, n_repeats, args.smoke)
            m_t, ci_t = metrics_with_ci(yt_t, yp_t)
            dt_t = time.time() - t1
            dump_patient_pred(args.name, name, "tuned" + dump_cfg_suffix, ids, yt_t, per_rep_t)
            row.update({
                "tuned_r": round(m_t["pearson_r"], 4),
                "tuned_r_CI": fmt_ci(ci_t["pearson_r"]),
                "tuned_mae": round(m_t["mae"], 3),
                "tuned_mae_CI": fmt_ci(ci_t["mae"]),
                "tuned_rmse": round(m_t["rmse"], 3),
                "tuned_rmse_CI": fmt_ci(ci_t["rmse"]),
                "tuned_r2": round(m_t["r2"], 4),
                "tuned_r2_CI": fmt_ci(ci_t["r2"]),
                "tuned_seconds": round(dt_t, 1),
                "tuned_minus_default_r": round(m_t["pearson_r"] - m_d["pearson_r"], 4),
            })
            print(f"  {name:13s} TUNED   r={m_t['pearson_r']:.3f} {fmt_ci(ci_t['pearson_r'])} "
                  f"MAE={m_t['mae']:.2f} RMSE={m_t['rmse']:.2f} R2={m_t['r2']:.3f} ({dt_t:.1f}s) "
                  f"d_r={m_t['pearson_r']-m_d['pearson_r']:+.3f}", flush=True)
        else:
            print(f"  {name:13s} TUNED   <skipped: not tunable / TabPFN default-only>",
                  flush=True)
        rows.append(row)

    new = pd.DataFrame(rows)
    out_dir = os.path.join(PKG, "results", "tuning")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = os.path.join(out_dir, f"nested_tuning_{args.name}{suffix}.csv")
    if os.path.exists(out_path):                 # 合并: 删本轮重跑的模型行再拼
        old = pd.read_csv(out_path)
        old = old[~old["model"].isin(new["model"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out_path, index=False)
    print(f"\nsummary -> {out_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
