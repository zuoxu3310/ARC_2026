#!/usr/bin/env python
"""
s3_classification_benchmark.py

Mirrors the WAB-AQ regression task as a Teghipco-style binary classification:
severe = WAB-AQ <= 50 vs the rest. The same classifiers are run on the same feature
tables (coarse arterial / fine JHU) to test whether fine-grained features help under
the classification framing, with the Teghipco CNN (F1 ~0.70, precision ~0.59) as a
reference anchor only (different protocol).

Leakage guard + class imbalance: within each fold StandardScaler -> SMOTE -> classifier,
wrapped in an imblearn Pipeline so the scaler and SMOTE are fit on the training fold only.
Protocol: repeated stratified K-fold (default 5x10); out-of-fold predictions are pooled
per repeat to compute metrics, then mean +/- SD is taken across repeats.
Primary metric: balanced accuracy; also reports AUC / F1(severe) / MCC / severe-class
precision and recall.

Run:
  python src/s3_classification_benchmark.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine [--models RandomForest XGBoost ...]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

PKG = Path(__file__).resolve().parents[1]
SEED = 42
THRESHOLD = 50          # severe = wab_aq <= 50 (Teghipco)
SPARSE_DROP_FRAC = 0.10
N_SPLITS, N_REPEATS = 5, 10
ALL_MODELS = ["Dummy", "LogReg", "SVM_rbf", "RandomForest", "XGBoost", "LightGBM", "MLP", "TabPFN"]


def estimator(name):
    if name == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "LogReg":
        return LogisticRegression(max_iter=5000, random_state=SEED)
    if name == "SVM_rbf":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=-1, verbosity=0, eval_metric="logloss")
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=-1, verbose=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=SEED)
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu")
    raise ValueError(name)


def build_pipe(name):
    # Dummy needs no scaler/SMOTE; all others use in-fold scaler -> SMOTE -> clf
    if name == "Dummy":
        return ImbPipeline([("clf", estimator(name))])
    return ImbPipeline([("scaler", StandardScaler()),
                        ("smote", SMOTE(random_state=SEED)),
                        ("clf", estimator(name))])


def load_xy(table_path):
    df = pd.read_csv(table_path, sep="\t", index_col="participant_id")
    df = df.dropna(subset=["wab_aq"]).fillna(0)
    y = (df["wab_aq"].values <= THRESHOLD).astype(int)   # 1 = severe
    feat = df.drop(columns=["wab_aq"])
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    feat = feat[keep]
    return feat.values, y, list(feat.columns)


def metrics_for(y_true, y_pred, y_prob):
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1_severe": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "prec_severe": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rec_severe": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def run_model(name, X, y):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    per_repeat = {k: [] for k in ["balanced_acc", "auc", "f1_severe", "mcc", "prec_severe", "rec_severe"]}
    # Pool out-of-fold predictions per repeat
    oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        pipe = build_pipe(name)
        pipe.fit(X[tr], y[tr])
        oof_pred[te] = pipe.predict(X[te])
        try:
            oof_prob[te] = pipe.predict_proba(X[te])[:, 1]
        except Exception:
            oof_prob[te] = oof_pred[te]
        if (i + 1) % 5 == 0:          # progress: print every 5 folds
            print(f"    [{name}] fold {i+1}/{N_SPLITS*N_REPEATS}", flush=True)
        if (i + 1) % N_SPLITS == 0:   # one full repeat completed
            m = metrics_for(y, oof_pred, oof_prob)
            for k in per_repeat:
                per_repeat[k].append(m[k])
            oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_repeat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True, help="dataset label, e.g. JHU_fine / arterial_coarse")
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    args = ap.parse_args()

    X, y, feat = load_xy(PKG / args.table if not Path(args.table).is_absolute() else args.table)
    n_sev = int(y.sum())
    print(f"[{args.name}] n={len(y)}  severe(<={THRESHOLD})={n_sev} ({n_sev/len(y):.0%})  "
          f"features={X.shape[1]}")

    rows = []
    import time
    for name in args.models:
        t0 = time.time()
        res = run_model(name, X, y)
        dt = time.time() - t0
        rows.append({"dataset": args.name, "model": name,
                     **{f"{k}": f"{m:.3f}±{s:.3f}" for k, (m, s) in res.items()},
                     "balanced_acc_sort": res["balanced_acc"][0], "seconds": round(dt, 1)})
        ba, _ = res["balanced_acc"]; f1, _ = res["f1_severe"]; auc, _ = res["auc"]
        print(f"  {name:14s} bAcc={ba:.3f}  AUC={auc:.3f}  F1(sev)={f1:.3f}  ({dt:.1f}s)")

    new = pd.DataFrame(rows).drop(columns=["balanced_acc_sort"])
    summ = PKG / "results" / "summary" / f"classification_{args.name}.csv"
    if summ.exists():                       # merge, not overwrite: drop rows for re-run models, then append
        old = pd.read_csv(summ)
        old = old[~old["model"].isin(new["model"])]
        new = pd.concat([old, new], ignore_index=True)
    new["_s"] = new["balanced_acc"].astype(str).str.split("±").str[0].astype(float)
    new = new.sort_values("_s", ascending=False).drop(columns=["_s"])
    new.to_csv(summ, index=False)
    print(f"\nsummary -> {summ.relative_to(PKG)}")


if __name__ == "__main__":
    raise SystemExit(main())
