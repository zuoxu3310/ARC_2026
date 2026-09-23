#!/usr/bin/env python
"""
s12_type_nested.py — 失语类型【二分类】的泄露安全嵌套调参 vs 默认 + 每人 dump + bootstrap CI

镜像 s3b_classification_nested_dump.py, 但目标换成类型二分(--task):
  fluent : 1=非流畅 {Broca,Global,TranscorticalMotor,TranscorticalMixed}
           0=流畅   {Wernicke,Conduction,Anomic,TranscorticalSensory}
  broca  : 1=Broca, 0=其它失语类型(去 None/NaN)

协议与 s3b 完全一致:
  - 整条 [scaler -> SMOTE -> clf] 封 imblearn Pipeline; tuned 用 GridSearchCV(内层 StratifiedKFold)
    只在外层训练折上 fit; default 同一条 Pipeline 默认超参。外层重复分层 5x10。
  - 每人跨 repeat 平均折外概率 -> 患者层 bootstrap CI(B=2000)。
  - dump: results/runs/clf_type_{task}_{gran}_{model}_{tag}_patient_probs.csv (每人一行, 供 s13/s14)
          results/runs/clf_type_{task}_{gran}_{model}_{tag}_oof_probs.csv     (每人逐折, 供 DeLong)

正类约定: pos=1 (fluent->非流畅, broca->Broca)。指标: balanced_acc/AUC/F1(pos)/MCC/精确率/召回率。

运行(全量, RF+XGB+LGBM+SVM+LogReg, fluent & broca):
  PYENV_VERSION=data-analysis python src/s12_type_nested.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine --task fluent
  (TabPFN 不在此处; 类型 TabPFN 看 s11 主榜)
"""
from __future__ import annotations
import argparse
import time
import warnings
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

warnings.filterwarnings("ignore", message="The least populated class in y has only")

PKG = Path(__file__).resolve().parents[1]
PARTICIPANTS = PKG.parent / "datasets" / "arc_ds004884" / "participants.tsv"
SEED = 42
SPARSE_DROP_FRAC = 0.10
N_SPLITS, N_REPEATS = 5, 10
INNER_SPLITS = 3
N_BOOT = 2000
TUNABLE_MODELS = ["RandomForest", "XGBoost", "LightGBM", "SVM_rbf", "LogReg"]

NONFLUENT = {"Broca", "Global", "TranscorticalMotor", "TranscorticalMixed"}
FLUENT = {"Wernicke", "Conduction", "Anomic", "TranscorticalSensory"}
NON_TYPE = {"None", "nan", "NaN", ""}


def base_estimator(name):
    if name == "LogReg":
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
    raise ValueError(f"unsupported model: {name}")


def param_grid(name, smoke=False):
    if name == "RandomForest":
        if smoke:
            return {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 4]}
        return {"clf__n_estimators": [300, 500], "clf__max_depth": [None, 6, 12],
                "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", 0.5]}
    if name == "XGBoost":
        if smoke:
            return {"clf__max_depth": [3], "clf__learning_rate": [0.05]}
        return {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9], "clf__colsample_bytree": [0.7, 0.9]}
    if name == "LightGBM":
        if smoke:
            return {"clf__num_leaves": [15], "clf__learning_rate": [0.05]}
        return {"clf__num_leaves": [7, 15, 31], "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9], "clf__min_child_samples": [5, 20]}
    if name == "SVM_rbf":
        if smoke:
            return {"clf__C": [1.0], "clf__gamma": ["scale"]}
        return {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]}
    if name == "LogReg":
        if smoke:
            return {"clf__C": [1.0], "clf__penalty": ["l2"]}
        return {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]}
    raise ValueError(name)


def build_pipe(name):
    return ImbPipeline([("scaler", StandardScaler()),
                        ("smote", SMOTE(random_state=SEED)),
                        ("clf", base_estimator(name))])


def load_xy(table_path, task):
    df = pd.read_csv(table_path, sep="\t", index_col="participant_id")
    pp = pd.read_csv(PARTICIPANTS, sep="\t", index_col="participant_id")["wab_type"]
    df = df.join(pp, how="left")
    df["wab_type"] = df["wab_type"].astype(str)
    if task == "fluent":
        sub = df[df["wab_type"].isin(NONFLUENT | FLUENT)].copy()
        y = sub["wab_type"].isin(NONFLUENT).astype(int).values
    elif task == "broca":
        sub = df[~df["wab_type"].isin(NON_TYPE) & df["wab_type"].notna()].copy()
        y = (sub["wab_type"] == "Broca").astype(int).values
    else:
        raise ValueError(task)
    feat = sub.drop(columns=[c for c in ["wab_aq", "wab_type"] if c in sub.columns]).fillna(0)
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    feat = feat[keep]
    return feat.values, y, list(sub.index), list(feat.columns)


def metrics_for(y_true, y_pred, y_prob):
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "prec_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rec_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["balanced_acc", "auc", "f1_pos", "mcc", "prec_pos", "rec_pos"]
    samp = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = metrics_for(yt, yp, ypr)
        for k in keys:
            samp[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.array(samp[k], dtype=float); arr = arr[~np.isnan(arr)]
        out[k] = (np.nan, np.nan) if len(arr) == 0 else \
            (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


def run_outer(name, X, y, pids, tuned, smoke=False):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    keys = ["balanced_acc", "auc", "f1_pos", "mcc", "prec_pos", "rec_pos"]
    per_repeat = {k: [] for k in keys}
    n = len(y)
    oof_pred = np.full(n, -1); oof_prob = np.zeros(n)
    prob_sum = np.zeros(n); seen = np.zeros(n)
    long_records = []
    n_total = N_SPLITS * N_REPEATS
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        repeat_id, fold_id = i // N_SPLITS, i % N_SPLITS
        pipe = build_pipe(name)
        if tuned:
            inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
            gs = GridSearchCV(pipe, param_grid(name, smoke=smoke),
                              scoring="balanced_accuracy", cv=inner, n_jobs=-1, refit=True)
            gs.fit(X[tr], y[tr]); est = gs.best_estimator_
        else:
            pipe.fit(X[tr], y[tr]); est = pipe
        pr = est.predict(X[te])
        try:
            prob = est.predict_proba(X[te])[:, 1]
        except Exception:
            prob = pr.astype(float)
        oof_pred[te] = pr; oof_prob[te] = prob
        prob_sum[te] += prob; seen[te] += 1
        for j, t in enumerate(te):
            long_records.append({"participant_id": pids[t], "repeat": repeat_id, "fold": fold_id,
                                 "y_true": int(y[t]), "y_prob": float(prob[j]), "y_pred": int(pr[j])})
        if (i + 1) % 5 == 0:
            print(f"    [{name}/{'tuned' if tuned else 'default'}] fold {i+1}/{n_total}", flush=True)
        if (i + 1) % N_SPLITS == 0:
            m = metrics_for(y, oof_pred, oof_prob)
            for k in keys:
                per_repeat[k].append(m[k])
            oof_pred = np.full(n, -1); oof_prob = np.zeros(n)
    mean_prob = prob_sum / np.maximum(seen, 1)
    mean_pred = (mean_prob >= 0.5).astype(int)
    agg = {"y_true": y.copy(), "y_prob": mean_prob, "y_pred": mean_pred}
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_repeat.items()}
    return summary, long_records, agg


def dump_probs(long_records, agg, pids, task, name, gran, tuned, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "tuned" if tuned else "default"
    long_df = pd.DataFrame(long_records)[
        ["participant_id", "repeat", "fold", "y_true", "y_prob", "y_pred"]]
    long_path = out_dir / f"clf_type_{task}_{gran}_{name}_{tag}_oof_probs.csv"
    long_df.to_csv(long_path, index=False)
    agg_df = pd.DataFrame({"participant_id": pids, "y_true": agg["y_true"].astype(int),
                           "y_prob_mean": agg["y_prob"], "y_pred": agg["y_pred"].astype(int)})
    agg_path = out_dir / f"clf_type_{task}_{gran}_{name}_{tag}_patient_probs.csv"
    agg_df.to_csv(agg_path, index=False)
    return long_path, agg_path


def fmt_ci(point, ci):
    lo, hi = ci
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True, help="粒度: JHU_fine / arterial_coarse")
    ap.add_argument("--task", required=True, choices=["fluent", "broca"])
    ap.add_argument("--models", nargs="*", default=TUNABLE_MODELS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if any(m == "TabPFN" for m in args.models):
        raise SystemExit("TabPFN 类型基线看 s11 主榜, 不在此处(防同进程假死)。")

    table = PKG / args.table if not Path(args.table).is_absolute() else Path(args.table)
    X, y, pids, feat = load_xy(table, args.task)
    n_pos = int(y.sum())
    print(f"[{args.name}/{args.task}] n={len(y)}  pos={n_pos} ({n_pos/len(y):.0%})  "
          f"features={X.shape[1]}  models={args.models}  smoke={args.smoke}", flush=True)

    runs_dir = PKG / "results" / "runs"
    rows = []
    for name in args.models:
        if name not in TUNABLE_MODELS:
            print(f"  跳过 {name}", flush=True); continue
        for tuned in [False, True]:
            tag = "tuned" if tuned else "default"
            t0 = time.time()
            summary, long_records, agg = run_outer(name, X, y, pids, tuned, smoke=args.smoke)
            ci = bootstrap_ci(agg["y_true"], agg["y_pred"], agg["y_prob"])
            dt = time.time() - t0
            lp, ap_ = dump_probs(long_records, agg, pids, args.task, name, args.name, tuned, runs_dir)
            row = {"task": args.task, "dataset": args.name, "model": name, "config": tag}
            for k, (m, s) in summary.items():
                row[k] = f"{m:.3f}±{s:.3f}"; row[f"{k}_ci"] = fmt_ci(m, ci[k])
            row["seconds"] = round(dt, 1); row["balanced_acc_sort"] = summary["balanced_acc"][0]
            rows.append(row)
            ba, _ = summary["balanced_acc"]; auc, _ = summary["auc"]; f1, _ = summary["f1_pos"]
            print(f"  {name:13s} [{tag:7s}] bAcc={fmt_ci(ba, ci['balanced_acc'])}  "
                  f"AUC={fmt_ci(auc, ci['auc'])}  F1={f1:.3f}  ({dt:.1f}s)", flush=True)

    new = pd.DataFrame(rows).drop(columns=["balanced_acc_sort"])
    summ = PKG / "results" / "summary" / f"type_nested_{args.task}_{args.name}.csv"
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
