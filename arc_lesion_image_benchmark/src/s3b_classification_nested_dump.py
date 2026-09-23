#!/usr/bin/env python
"""
s3b_classification_nested_dump.py

在 s3_classification_benchmark.py 基础上扩展, 服务 EXPERIMENT_PLAN Block3/Block4:
  1) 泄露安全的【嵌套调参】: GridSearchCV 套在 imblearn Pipeline 里, SMOTE 只在内层折 fit,
     外层 = 重复分层 5x10 折(只评估、不在外层调参), 对每个模型给出 tuned vs default。
  2) 逐人外层测试折【预测概率 dump】到 results/runs/, 给后续 DeLong / 校准用。
     细 JHU 与粗动脉各 dump 一份, 文件名带模型与粒度。
  3) 患者层【bootstrap CI】(非 ±std): 对外层折外预测做按患者重抽样, 报 95% CI。

== 防泄露设计(铁律) ==
  整条 [scaler -> SMOTE -> clf] 封成一个 imblearn Pipeline。
  - tuned: GridSearchCV(pipe, inner=StratifiedKFold) 只在【外层训练折】上拟合。
           内层 CV 把外层训练折再切, SMOTE/scaler 只在每个内层【训练子折】fit_transform,
           内层验证子折与外层测试折都从不参与 SMOTE/scaler/选超参。
  - default: 同一条 Pipeline, 默认超参, 同样只在外层训练折 fit。
  外层测试折 X[te] 永远只过 transform/predict, 绝不 fit 任何 scaler/SMOTE/搜参。
  => SMOTE 在【内层训练子折(tuned)/外层训练折(default)】fit, 测试折零接触。

== TabPFN ==
  本脚本只跑可调/默认树与线性模型。TabPFN 按项目铁律【单独进程、不调参、硬超时】,
  用 s3_classification_benchmark.py --models TabPFN 单独起, 不在此处同进程跑(防假死)。
  本脚本若被传入 TabPFN 会直接拒绝并提示正确跑法。

运行(SMOKE, 只 RF 小网格):
  PYENV_VERSION=data-analysis python src/s3b_classification_nested_dump.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine \
      --models RandomForest --smoke

运行(全量示例):
  PYENV_VERSION=data-analysis python src/s3b_classification_nested_dump.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine \
      --models RandomForest XGBoost LightGBM SVM_rbf LogReg
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

PKG = Path(__file__).resolve().parents[1]
SEED = 42
THRESHOLD = 50            # severe = wab_aq <= 50 (Teghipco 口径)
SPARSE_DROP_FRAC = 0.10
N_SPLITS, N_REPEATS = 5, 10
INNER_SPLITS = 3          # 内层调参 CV 折数
N_BOOT = 2000             # 患者层 bootstrap 次数
# 本脚本负责的【可调】分类器(TabPFN 不在此处, 见 docstring)
TUNABLE_MODELS = ["RandomForest", "XGBoost", "LightGBM", "SVM_rbf", "LogReg"]


# ---------- 估计器 + 内层网格 ----------
def base_estimator(name):
    if name == "LogReg":
        # 线性(正则): 调 C 与正则类型(l1 需 saga)
        return LogisticRegression(max_iter=5000, random_state=SEED, solver="saga")
    if name == "SVM_rbf":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=1, verbosity=0, eval_metric="logloss")
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=1, verbose=-1)
    raise ValueError(f"unsupported/clf-only model: {name}")


def param_grid(name, smoke=False):
    """返回作用在 Pipeline 上的网格(键带 'clf__' 前缀)。smoke=极小网格只为跑通。"""
    if name == "RandomForest":
        if smoke:
            return {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 4]}
        return {"clf__n_estimators": [300, 500],
                "clf__max_depth": [None, 6, 12],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.5]}
    if name == "XGBoost":
        if smoke:
            return {"clf__max_depth": [3], "clf__learning_rate": [0.05]}
        return {"clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9],
                "clf__colsample_bytree": [0.7, 0.9]}
    if name == "LightGBM":
        if smoke:
            return {"clf__num_leaves": [15], "clf__learning_rate": [0.05]}
        return {"clf__num_leaves": [7, 15, 31],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9],
                "clf__min_child_samples": [5, 20]}
    if name == "SVM_rbf":
        if smoke:
            return {"clf__C": [1.0], "clf__gamma": ["scale"]}
        return {"clf__C": [0.1, 1, 10, 100],
                "clf__gamma": ["scale", 0.01, 0.1]}
    if name == "LogReg":
        if smoke:
            return {"clf__C": [1.0], "clf__penalty": ["l2"]}
        return {"clf__C": [0.01, 0.1, 1, 10],
                "clf__penalty": ["l1", "l2"]}
    raise ValueError(name)


def build_pipe(name):
    # 折内 scaler -> SMOTE -> clf。整条作为一个估计器, GridSearchCV 在其上搜参。
    return ImbPipeline([("scaler", StandardScaler()),
                        ("smote", SMOTE(random_state=SEED)),
                        ("clf", base_estimator(name))])


# ---------- 数据 ----------
def load_xy(table_path):
    df = pd.read_csv(table_path, sep="\t", index_col="participant_id")
    df = df.dropna(subset=["wab_aq"]).fillna(0)
    y = (df["wab_aq"].values <= THRESHOLD).astype(int)   # 1 = severe
    wab = df["wab_aq"].values.astype(float)
    feat = df.drop(columns=["wab_aq"])
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    feat = feat[keep]
    return feat.values, y, wab, list(df.index), list(feat.columns)


# ---------- 指标 ----------
def metrics_for(y_true, y_pred, y_prob):
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1_severe": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "prec_severe": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rec_severe": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=N_BOOT, seed=SEED):
    """患者层 bootstrap: 对【人】重抽样算各指标 95% CI。
    y_*: 形状 (n_patients,), 取每人折外预测(repeat 平均后的均值预测/概率)。"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["balanced_acc", "auc", "f1_severe", "mcc", "prec_severe", "rec_severe"]
    samp = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:    # 重抽样退化成单类, 跳过该次
            continue
        m = metrics_for(yt, yp, ypr)
        for k in keys:
            samp[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.array(samp[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            out[k] = (np.nan, np.nan)
        else:
            out[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


# ---------- 外层 CV(tuned 或 default), 含逐人 dump ----------
def run_outer(name, X, y, wab, pids, tuned, smoke=False, verbose=True):
    """外层重复分层 5x10。tuned=True 时每个外层训练折跑内层 GridSearchCV;
    default=False 用默认超参。返回 (per_repeat_metrics, long_records, agg_per_patient)。
    long_records: 逐人逐外层折一行(repeat,fold,prob) —— dump 原料。"""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    keys = ["balanced_acc", "auc", "f1_severe", "mcc", "prec_severe", "rec_severe"]
    per_repeat = {k: [] for k in keys}
    n = len(y)
    oof_pred = np.full(n, -1)
    oof_prob = np.zeros(n)
    # 累加每人在所有 repeat 的概率/预测, 供患者层聚合 + bootstrap
    prob_sum = np.zeros(n); pred_sum = np.zeros(n); seen = np.zeros(n)
    long_records = []
    n_total = N_SPLITS * N_REPEATS

    for i, (tr, te) in enumerate(rskf.split(X, y)):
        repeat_id = i // N_SPLITS
        fold_id = i % N_SPLITS
        pipe = build_pipe(name)
        if tuned:
            inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
            gs = GridSearchCV(pipe, param_grid(name, smoke=smoke),
                              scoring="balanced_accuracy", cv=inner, n_jobs=-1, refit=True)
            gs.fit(X[tr], y[tr])          # 只见外层训练折; 内层再切, SMOTE 只在内层训练子折 fit
            est = gs.best_estimator_
        else:
            pipe.fit(X[tr], y[tr])        # 默认超参, 只见外层训练折
            est = pipe

        pr = est.predict(X[te])
        try:
            prob = est.predict_proba(X[te])[:, 1]
        except Exception:
            prob = pr.astype(float)
        oof_pred[te] = pr
        oof_prob[te] = prob

        prob_sum[te] += prob; pred_sum[te] += pr; seen[te] += 1
        for j, t in enumerate(te):
            long_records.append({
                "participant_id": pids[t], "repeat": repeat_id, "fold": fold_id,
                "y_true": int(y[t]), "wab_aq": float(wab[t]),
                "y_prob": float(prob[j]), "y_pred": int(pr[j]),
            })

        if verbose and (i + 1) % 5 == 0:
            print(f"    [{name}/{'tuned' if tuned else 'default'}] "
                  f"fold {i+1}/{n_total}", flush=True)
        if (i + 1) % N_SPLITS == 0:       # 一个 repeat 跑满, 算这轮指标
            m = metrics_for(y, oof_pred, oof_prob)
            for k in keys:
                per_repeat[k].append(m[k])
            oof_pred = np.full(n, -1); oof_prob = np.zeros(n)

    # 患者层聚合: 每人取所有 repeat 的平均概率, 阈值 0.5 定标签
    mean_prob = prob_sum / np.maximum(seen, 1)
    mean_pred = (mean_prob >= 0.5).astype(int)
    agg = {"y_true": y.copy(), "y_prob": mean_prob, "y_pred": mean_pred}
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_repeat.items()}
    return summary, long_records, agg


# ---------- dump ----------
def dump_probs(long_records, agg, pids, name, gran, tuned, out_dir):
    """写两份: (1) 逐人逐折长表(给 DeLong, 每人多行); (2) 按 repeat 平均的宽表(每人一行)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "tuned" if tuned else "default"
    long_df = pd.DataFrame(long_records)[
        ["participant_id", "repeat", "fold", "y_true", "wab_aq", "y_prob", "y_pred"]]
    long_path = out_dir / f"clf_{gran}_{name}_{tag}_oof_probs.csv"
    long_df.to_csv(long_path, index=False)

    agg_df = pd.DataFrame({
        "participant_id": pids,
        "y_true": agg["y_true"].astype(int),
        "y_prob_mean": agg["y_prob"],
        "y_pred": agg["y_pred"].astype(int),
    })
    agg_path = out_dir / f"clf_{gran}_{name}_{tag}_patient_probs.csv"
    agg_df.to_csv(agg_path, index=False)
    return long_path, agg_path


def fmt_ci(point, ci):
    lo, hi = ci
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True, help="粒度标签, 如 JHU_fine / arterial_coarse")
    ap.add_argument("--models", nargs="*", default=TUNABLE_MODELS)
    ap.add_argument("--smoke", action="store_true", help="极小网格, 只为跑通验证, 不全量")
    ap.add_argument("--no-default", action="store_true", help="跳过 default, 只跑 tuned")
    ap.add_argument("--no-tuned", action="store_true", help="跳过 tuned, 只跑 default")
    args = ap.parse_args()

    if any(m == "TabPFN" for m in args.models):
        raise SystemExit(
            "TabPFN 不在本脚本跑(防同进程假死)。请单独:\n"
            "  perl -e 'alarm 1800; exec @ARGV' env PYENV_VERSION=data-analysis "
            "python src/s3_classification_benchmark.py --table <表> --name <名> --models TabPFN")

    table = PKG / args.table if not Path(args.table).is_absolute() else Path(args.table)
    X, y, wab, pids, feat = load_xy(table)
    n_sev = int(y.sum())
    print(f"[{args.name}] n={len(y)}  severe(<=≤{THRESHOLD})={n_sev} "
          f"({n_sev/len(y):.0%})  features={X.shape[1]}  models={args.models}  "
          f"smoke={args.smoke}", flush=True)

    runs_dir = PKG / "results" / "runs"
    rows = []
    for name in args.models:
        if name not in TUNABLE_MODELS:
            print(f"  跳过 {name}: 不在可调集合 {TUNABLE_MODELS}", flush=True)
            continue
        for tuned in ([False] if args.no_tuned else
                      [True] if args.no_default else [False, True]):
            tag = "tuned" if tuned else "default"
            t0 = time.time()
            summary, long_records, agg = run_outer(name, X, y, wab, pids, tuned, smoke=args.smoke)
            ci = bootstrap_ci(agg["y_true"], agg["y_pred"], agg["y_prob"])
            dt = time.time() - t0
            long_path, agg_path = dump_probs(long_records, agg, pids, name, args.name, tuned, runs_dir)

            row = {"dataset": args.name, "model": name, "config": tag}
            for k, (m, s) in summary.items():
                row[k] = f"{m:.3f}±{s:.3f}"
                row[f"{k}_ci"] = fmt_ci(m, ci[k])
            row["seconds"] = round(dt, 1)
            row["balanced_acc_sort"] = summary["balanced_acc"][0]
            rows.append(row)

            ba, _ = summary["balanced_acc"]; auc, _ = summary["auc"]; f1, _ = summary["f1_severe"]
            print(f"  {name:13s} [{tag:7s}] bAcc={fmt_ci(ba, ci['balanced_acc'])}  "
                  f"AUC={fmt_ci(auc, ci['auc'])}  F1(sev)={f1:.3f}  ({dt:.1f}s)", flush=True)
            print(f"      dump long -> {long_path.relative_to(PKG)}", flush=True)
            print(f"      dump agg  -> {agg_path.relative_to(PKG)}", flush=True)

    if not rows:
        print("无可跑模型, 退出。", flush=True)
        return
    new = pd.DataFrame(rows).drop(columns=["balanced_acc_sort"])
    summ = PKG / "results" / "summary" / f"classification_nested_{args.name}.csv"
    summ.parent.mkdir(parents=True, exist_ok=True)
    if summ.exists():
        old = pd.read_csv(summ)
        key = old["model"].astype(str) + "|" + old.get("config", "").astype(str)
        newkey = new["model"].astype(str) + "|" + new["config"].astype(str)
        old = old[~key.isin(set(newkey))]
        new = pd.concat([old, new], ignore_index=True)
    new["_s"] = new["balanced_acc"].astype(str).str.split("±").str[0].astype(float)
    new = new.sort_values("_s", ascending=False).drop(columns=["_s"])
    new.to_csv(summ, index=False)
    print(f"\nsummary -> {summ.relative_to(PKG)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
