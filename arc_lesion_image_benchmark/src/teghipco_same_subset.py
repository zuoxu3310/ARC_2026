#!/usr/bin/env python
"""
teghipco_same_subset.py

"便宜表格 vs 昂贵 CNN" 的同数据硬证据。
在 Teghipco 公开 173 子集里能对上我们表的 172 人上, 用我们的可解释表格特征
(细 JHU 189 区 / 粗动脉 32 区, 都加 lesion_volume + age) 跑 RandomForest + LogReg + SVM,
折内 StandardScaler -> SMOTE -> clf 的 imblearn Pipeline (防泄露), 重复分层 5x10 折,
对标 Teghipco 2024 CNN 报告值 (balanced-acc 0.77, F1(severe) 0.70, precision 0.59)。

标签: 默认用 .mat 自带 wabClassi (Teghipco 训 CNN 时的真值, 0=non-severe / 1=severe),
      另跑一套我们表 wab_aq<50 作对照 (两者一致率 99.4%)。

防泄露 (CLAUDE.md 铁律): StandardScaler / SMOTE 只在每折训练子集 fit。
  - build_pipe() 里 ImbPipeline([scaler, smote, clf]); pipe.fit(X[tr]) 只见训练折。
  - bootstrap CI 在折外 (OOF) 预测上按患者 resample, 不重训, 不碰测试折信息泄露。

运行 (单数据集单标签):
  PYENV_VERSION=data-analysis python teghipco_same_subset.py
"""
from __future__ import annotations
from pathlib import Path
import collections
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "arc_lesion_image_benchmark"
MAT = ROOT / "datasets/arc_ds004884/teghipco_figshare/DLAphasiaSeverityARCSubset.mat"
TABLES = {
    "JHU_fine": PKG / "data/jhu156_features_merged.tsv",
    "arterial_coarse": PKG / "data/ArterialAtlas156_features_merged.tsv",
}
OUT = PKG / "results/summary/teghipco_same_subset.csv"

SEED = 42
SPARSE_DROP_FRAC = 0.10        # 稀疏脑区列丢弃阈值, 与主分类口径一致
N_SPLITS, N_REPEATS = 5, 10
N_BOOT = 2000
MODELS = ["RandomForest", "LogReg", "SVM_rbf"]

# Teghipco 2024 (Comm Med) CNN 报告值 (source data 口径, 见 ARC_competitor_matrix)
TEGHIPCO_CNN = {"balanced_acc": 0.77, "f1_severe": 0.70, "prec_severe": 0.59}


def load_mat_labels():
    m = loadmat(str(MAT), squeeze_me=True)
    subs = np.asarray(m["subsOnly"]).astype(str)
    wabClassi = np.asarray(m["wabClassi"]).astype(int)   # 0 non-severe / 1 severe (Teghipco binary)
    ids = [s if str(s).startswith("sub-") else f"sub-{s}" for s in subs]
    return pd.DataFrame({"participant_id": ids, "mat_severe": wabClassi})


def estimator(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    if name == "LogReg":
        return LogisticRegression(max_iter=5000, random_state=SEED)
    if name == "SVM_rbf":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    raise ValueError(name)


def build_pipe(name):
    # 折内: scaler -> SMOTE -> clf, 全部只在训练折 fit (防泄露)
    return ImbPipeline([("scaler", StandardScaler()),
                        ("smote", SMOTE(random_state=SEED)),
                        ("clf", estimator(name))])


def metrics_for(y_true, y_pred, y_prob):
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1_severe": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "prec_severe": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rec_severe": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def patient_bootstrap_ci(y_true, oof_pred, oof_prob, n_boot=N_BOOT, seed=SEED):
    """患者级 bootstrap: 在折外预测上按患者重采样, 给 95% CI。不重训模型。"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["balanced_acc", "auc", "f1_severe", "prec_severe", "rec_severe"]
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, pr = y_true[idx], oof_pred[idx], oof_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = metrics_for(yt, yp, pr)
        for k in keys:
            boot[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}


def run_model(name, X, y):
    """重复分层 5x10 折, 每 repeat 汇总折外预测算指标 -> 跨 repeat 均值±std;
    并保留最后一个 repeat 的 OOF 预测做患者级 bootstrap CI 的代表 (口径稳定)。"""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    per_repeat = {k: [] for k in ["balanced_acc", "auc", "f1_severe", "mcc", "prec_severe", "rec_severe"]}
    oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
    # 累加每个患者跨 repeat 的平均预测概率, 用于稳定的患者级 bootstrap
    prob_sum = np.zeros(len(y)); pred_sum = np.zeros(len(y))
    n_done_repeats = 0
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        pipe = build_pipe(name)
        pipe.fit(X[tr], y[tr])
        oof_pred[te] = pipe.predict(X[te])
        try:
            oof_prob[te] = pipe.predict_proba(X[te])[:, 1]
        except Exception:
            oof_prob[te] = oof_pred[te]
        if (i + 1) % N_SPLITS == 0:
            m = metrics_for(y, oof_pred, oof_prob)
            for k in per_repeat:
                per_repeat[k].append(m[k])
            prob_sum += oof_prob; pred_sum += oof_pred
            n_done_repeats += 1
            oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
            print(f"    [{name}] repeat {n_done_repeats}/{N_REPEATS} done", flush=True)
    # 跨 repeat 平均后的患者级 OOF (概率取均值, 预测按 0.5 阈值)
    mean_prob = prob_sum / n_done_repeats
    mean_pred = (mean_prob >= 0.5).astype(int)
    ci = patient_bootstrap_ci(y, mean_pred, mean_prob)
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_repeat.items()}
    return summary, ci, mean_pred, mean_prob


def load_xy(table_path, label_df, label_mode):
    df = pd.read_csv(table_path, sep="\t")
    df = df.dropna(subset=["wab_aq"])
    # 只保留 Teghipco 子集里能对上的患者
    merged = label_df.merge(df, on="participant_id", how="inner")
    merged = merged.fillna(0)
    if label_mode == "mat_wabClassi":
        y = merged["mat_severe"].values.astype(int)
    elif label_mode == "table_aq_lt50":
        y = (merged["wab_aq"].values < 50).astype(int)
    else:
        raise ValueError(label_mode)
    feat = merged.drop(columns=["participant_id", "mat_severe", "wab_aq"])
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    feat = feat[keep]
    return feat.values, y, merged["participant_id"].tolist(), list(feat.columns)


def main():
    label_df = load_mat_labels()
    rows = []
    for label_mode in ["mat_wabClassi", "table_aq_lt50"]:
        for ds_name, table_path in TABLES.items():
            X, y, pids, feat = load_xy(table_path, label_df, label_mode)
            n_sev = int(y.sum())
            print(f"\n=== [{ds_name} | {label_mode}] n={len(y)} severe={n_sev} "
                  f"({n_sev/len(y):.0%}) features={X.shape[1]} ===", flush=True)
            for name in MODELS:
                t0 = time.time()
                summary, ci, _, _ = run_model(name, X, y)
                dt = time.time() - t0
                row = {"dataset": ds_name, "label": label_mode, "model": name,
                       "n": len(y), "n_severe": n_sev, "n_features": X.shape[1]}
                for k, (mu, sd) in summary.items():
                    row[k] = round(mu, 3)
                    row[f"{k}_sd"] = round(sd, 3)
                for k, (lo, hi) in ci.items():
                    row[f"{k}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
                row["seconds"] = round(dt, 1)
                rows.append(row)
                ba = summary["balanced_acc"][0]; f1 = summary["f1_severe"][0]
                auc = summary["auc"][0]; pr = summary["prec_severe"][0]
                print(f"  {name:13s} bAcc={ba:.3f} {row['balanced_acc_ci95']}  "
                      f"AUC={auc:.3f}  F1(sev)={f1:.3f} {row['f1_severe_ci95']}  "
                      f"prec(sev)={pr:.3f} ({dt:.1f}s)", flush=True)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(f"\nTeghipco CNN anchor: balanced_acc={TEGHIPCO_CNN['balanced_acc']}, "
          f"F1(severe)={TEGHIPCO_CNN['f1_severe']}, precision(severe)={TEGHIPCO_CNN['prec_severe']}")
    # 落一份简明对照
    print("\n=== vs Teghipco CNN (point-estimate anchor; 非其精确 CV split) ===")
    for _, r in out.iterrows():
        cover_ba = "covers" if _ci_covers(r["balanced_acc_ci95"], TEGHIPCO_CNN["balanced_acc"]) else "no"
        cover_f1 = "covers" if _ci_covers(r["f1_severe_ci95"], TEGHIPCO_CNN["f1_severe"]) else "no"
        print(f"  {r['dataset']:15s} {r['label']:14s} {r['model']:13s} "
              f"bAcc {r['balanced_acc']} CI{r['balanced_acc_ci95']} [{cover_ba} 0.77]  "
              f"F1 {r['f1_severe']} CI{r['f1_severe_ci95']} [{cover_f1} 0.70]")


def _ci_covers(ci_str, val):
    lo, hi = [float(x) for x in ci_str.strip("[]").split(",")]
    return lo <= val <= hi


if __name__ == "__main__":
    raise SystemExit(main())
