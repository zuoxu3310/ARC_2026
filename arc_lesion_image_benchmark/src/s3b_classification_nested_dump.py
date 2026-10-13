#!/usr/bin/env python
"""
s3b_classification_nested_dump.py

Extends s3_classification_benchmark.py with:
  1) Leakage-safe nested tuning: GridSearchCV wraps an imblearn Pipeline, so SMOTE is fit
     only on inner training folds. Outer loop = repeated stratified 5x10 CV (evaluation
     only, no tuning on the outer folds). Each model is reported as tuned vs default.
  2) Per-patient outer-test-fold predicted probabilities dumped to results/runs/ for
     downstream DeLong tests / calibration. One dump each for fine JHU and coarse
     arterial features; file names include model and granularity.
  3) Patient-level bootstrap CI (instead of +/- std): resample patients from the outer
     out-of-fold predictions and report 95% CIs.

== Leakage guard ==
  The whole [scaler -> SMOTE -> clf] chain is a single imblearn Pipeline.
  - tuned:   GridSearchCV(pipe, inner=StratifiedKFold) is fit on the outer training fold only.
             The inner CV re-splits the outer training fold; SMOTE/scaler are fit_transform'ed
             on each inner training sub-fold only. Inner validation sub-folds and the outer
             test fold never take part in SMOTE/scaler fitting or hyperparameter selection.
  - default: the same Pipeline with default hyperparameters, also fit on the outer training
             fold only.
  The outer test fold X[te] only goes through transform/predict; it never fits any
  scaler/SMOTE/search.
  => SMOTE is fit on the inner training sub-fold (tuned) / outer training fold (default);
     the test fold is never touched.

== TabPFN ==
  This script runs only the tunable/default tree and linear models. TabPFN is run in a
  separate process, untuned, with a hard timeout, via
  s3_classification_benchmark.py --models TabPFN, not in this process (avoids hangs).
  If TabPFN is passed to this script, it refuses and prints the correct command.

Usage (smoke test, RF only, small grid):
  python src/s3b_classification_nested_dump.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine \
      --models RandomForest --smoke

Usage (full example):
  python src/s3b_classification_nested_dump.py \
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
THRESHOLD = 50            # severe = wab_aq <= 50 (Teghipco et al. definition)
SPARSE_DROP_FRAC = 0.10
N_SPLITS, N_REPEATS = 5, 10
INNER_SPLITS = 3          # inner tuning CV folds
N_BOOT = 2000             # patient-level bootstrap resamples
# tunable classifiers handled by this script (TabPFN excluded, see docstring)
TUNABLE_MODELS = ["RandomForest", "XGBoost", "LightGBM", "SVM_rbf", "LogReg"]


# ---------- estimators + inner grids ----------
def base_estimator(name):
    if name == "LogReg":
        # regularised linear model: tune C and penalty type (l1 needs saga)
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
    """Return the grid applied to the Pipeline (keys prefixed with 'clf__'). smoke = tiny grid, only to verify the run."""
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
    # in-fold scaler -> SMOTE -> clf as one estimator; GridSearchCV searches over it.
    return ImbPipeline([("scaler", StandardScaler()),
                        ("smote", SMOTE(random_state=SEED)),
                        ("clf", base_estimator(name))])


# ---------- data ----------
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


# ---------- metrics ----------
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
    """Patient-level bootstrap: resample patients to get a 95% CI for each metric.
    y_*: shape (n_patients,), each patient's out-of-fold prediction (prediction/probability averaged over repeats)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["balanced_acc", "auc", "f1_severe", "mcc", "prec_severe", "rec_severe"]
    samp = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:    # resample has a single class; skip it
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


# ---------- outer CV (tuned or default), with per-patient dump ----------
def run_outer(name, X, y, wab, pids, tuned, smoke=False, verbose=True):
    """Outer repeated stratified 5x10 CV. tuned=True runs an inner GridSearchCV on each
    outer training fold; tuned=False uses default hyperparameters.
    Returns (per_repeat_metrics, long_records, agg_per_patient).
    long_records: one row per patient per outer fold (repeat, fold, prob) - input for the dump."""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    keys = ["balanced_acc", "auc", "f1_severe", "mcc", "prec_severe", "rec_severe"]
    per_repeat = {k: [] for k in keys}
    n = len(y)
    oof_pred = np.full(n, -1)
    oof_prob = np.zeros(n)
    # accumulate each patient's probability/prediction over all repeats (patient-level aggregation + bootstrap)
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
            gs.fit(X[tr], y[tr])          # outer training fold only; inner CV re-splits it, SMOTE fit on inner training sub-folds only
            est = gs.best_estimator_
        else:
            pipe.fit(X[tr], y[tr])        # default hyperparameters, outer training fold only
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
        if (i + 1) % N_SPLITS == 0:       # one repeat complete: compute its metrics
            m = metrics_for(y, oof_pred, oof_prob)
            for k in keys:
                per_repeat[k].append(m[k])
            oof_pred = np.full(n, -1); oof_prob = np.zeros(n)

    # patient-level aggregation: mean probability over all repeats, label by threshold 0.5
    mean_prob = prob_sum / np.maximum(seen, 1)
    mean_pred = (mean_prob >= 0.5).astype(int)
    agg = {"y_true": y.copy(), "y_prob": mean_prob, "y_pred": mean_pred}
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_repeat.items()}
    return summary, long_records, agg


# ---------- dump ----------
def dump_probs(long_records, agg, pids, name, gran, tuned, out_dir):
    """Write two files: (1) long table, one row per patient per fold (for DeLong, several rows per patient);
    (2) wide table averaged over repeats (one row per patient)."""
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
    ap.add_argument("--name", required=True, help="granularity label, e.g. JHU_fine / arterial_coarse")
    ap.add_argument("--models", nargs="*", default=TUNABLE_MODELS)
    ap.add_argument("--smoke", action="store_true", help="tiny grid, only to verify the run (not a full run)")
    ap.add_argument("--no-default", action="store_true", help="skip default, run tuned only")
    ap.add_argument("--no-tuned", action="store_true", help="skip tuned, run default only")
    args = ap.parse_args()

    if any(m == "TabPFN" for m in args.models):
        raise SystemExit(
            "TabPFN is not run by this script (avoids in-process hangs). Run it separately:\n"
            "  perl -e 'alarm 1800; exec @ARGV' "
            "python src/s3_classification_benchmark.py --table <table> --name <name> --models TabPFN")

    table = PKG / args.table if not Path(args.table).is_absolute() else Path(args.table)
    X, y, wab, pids, feat = load_xy(table)
    n_sev = int(y.sum())
    print(f"[{args.name}] n={len(y)}  severe(<={THRESHOLD})={n_sev} "
          f"({n_sev/len(y):.0%})  features={X.shape[1]}  models={args.models}  "
          f"smoke={args.smoke}", flush=True)

    runs_dir = PKG / "results" / "runs"
    rows = []
    for name in args.models:
        if name not in TUNABLE_MODELS:
            print(f"  skip {name}: not in tunable set {TUNABLE_MODELS}", flush=True)
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
        print("No runnable models; exiting.", flush=True)
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
