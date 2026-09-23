#!/usr/bin/env python
"""
s3c_tabpfn_nest.py — 严重度【severe 二分类】TabPFN 的 n_estimators 扫描。

目的: 验证之前 s3 主榜里 TabPFN(默认 n_estimators=8)的分数对集成遍数敏不敏感。
若 n=2..8 精度基本平 -> 之前默认 8 的数稳, 且可改用 n=2 提速; 若明显变 -> 需更新报告。

协议与 s3 完全一致: severe = wab_aq <= 50; 折内 scaler -> SMOTE -> TabPFN(n_estimators=n);
分层重复 5x10; 每 repeat 汇总折外预测算指标, 跨 repeat 均值±std。只跑 TabPFN(单进程)。

运行:
  perl -e 'alarm shift; exec @ARGV' 5400 env PYENV_VERSION=data-analysis python \
      src/s3c_tabpfn_nest.py --table data/ArterialAtlas156_features_merged.tsv \
      --name arterial_coarse --nest 2 3 4 5 6 8
"""
from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
PKG = Path(__file__).resolve().parents[1]
SEED = 42
THRESHOLD = 50
SPARSE_DROP_FRAC = 0.10
N_SPLITS, N_REPEATS = 5, 10


def load_xy(table_path):
    df = pd.read_csv(table_path, sep="\t", index_col="participant_id")
    df = df.dropna(subset=["wab_aq"]).fillna(0)
    y = (df["wab_aq"].values <= THRESHOLD).astype(int)
    feat = df.drop(columns=["wab_aq"])
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    return feat[keep].values, y


def run_one(X, y, n_est, n_repeats):
    from tabpfn import TabPFNClassifier
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    per = {k: [] for k in ["balanced_acc", "auc", "f1_severe", "mcc"]}
    oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
    t0 = time.time()
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        pipe = ImbPipeline([("scaler", StandardScaler()),
                            ("smote", SMOTE(random_state=SEED)),
                            ("clf", TabPFNClassifier(device="cpu", n_estimators=n_est))])
        pipe.fit(X[tr], y[tr])
        oof_pred[te] = pipe.predict(X[te])
        try:
            oof_prob[te] = pipe.predict_proba(X[te])[:, 1]
        except Exception:
            oof_prob[te] = oof_pred[te]
        if (i + 1) % 5 == 0:
            print(f"    [n_est={n_est}] fold {i+1}/{N_SPLITS*n_repeats} ({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % N_SPLITS == 0:
            per["balanced_acc"].append(balanced_accuracy_score(y, oof_pred))
            per["auc"].append(roc_auc_score(y, oof_prob))
            per["f1_severe"].append(f1_score(y, oof_pred, pos_label=1, zero_division=0))
            per["mcc"].append(matthews_corrcoef(y, oof_pred))
            oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
    dt = time.time() - t0
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in per.items()}, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--nest", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    tp = PKG / args.table if not Path(args.table).is_absolute() else Path(args.table)
    X, y = load_xy(tp)
    n_sev = int(y.sum())
    print(f"[{args.name}/severe] n={len(y)}  severe={n_sev} ({n_sev/len(y):.0%})  "
          f"features={X.shape[1]}  protocol=5x{args.repeats}  nest={args.nest}", flush=True)

    rows = []
    for n_est in args.nest:
        summ, dt = run_one(X, y, n_est, args.repeats)
        ba, bas = summ["balanced_acc"]; auc, _ = summ["auc"]; f1, _ = summ["f1_severe"]
        rows.append({"dataset": args.name, "target": "severe", "n_estimators": n_est,
                     "balanced_acc": f"{ba:.3f}±{bas:.3f}", "auc": round(auc, 3),
                     "f1_severe": round(f1, 3), "seconds": round(dt, 1)})
        print(f"  n_est={n_est}: bAcc={ba:.3f}±{bas:.3f}  AUC={auc:.3f}  F1={f1:.3f}  "
              f"[{dt:.0f}s={dt/60:.1f}min]", flush=True)

    out = PKG / "results" / "summary" / f"tabpfn_nest_severe_{args.name}.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out); old = old[~old["n_estimators"].isin(df["n_estimators"])]
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values("n_estimators").to_csv(out, index=False)
    print(f"\nsummary -> {out.relative_to(PKG)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
