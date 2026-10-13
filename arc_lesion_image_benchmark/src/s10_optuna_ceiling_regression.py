#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s10_optuna_ceiling_regression.py
================================
Ceiling stress test (WAB-AQ regression): Optuna search replaces the small hand-written
grid, under **exactly the same** 5x10 outer protocol as the main leaderboard, to give the
search the best possible chance of raising Pearson r.

This experiment answers "did we search enough?", not "has the task hit its ceiling?":
  - A flattening best-so-far curve over trials = the hyperparameter search has converged
    (this model family is tuned out). It only rebuts "your grid was too small" and is
    **not** on its own evidence of an information ceiling in the lesion maps.
  - The ceiling claim rests on independent evidence: held-out (outer test fold) scores
    around ~0.70, the floor baseline, convergence across model families, and 0/8
    significant differences / TOST equivalence.

== Leakage guards (same protocol as s5) ==
  1) StandardScaler is always the first Pipeline step; inside the search it is fit only
     on each inner training fold.
  2) Optuna only sees the outer training fold X[tr]; the objective is the mean CV r over
     INNER folds cut from the outer training fold. The outer test fold X[te] never takes
     part in search or selection and is predicted exactly once with the chosen params.
  3) Two scores are reported: held-out pooled OOF r (+ bootstrap CI) and per-fold mean r,
     on the same scale as the main leaderboard so they can be compared / paired-tested
     directly. The inner best-r is used only for the convergence curve, never as a result.

== Usage ==
  python src/s10_optuna_ceiling_regression.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine --trials 50
  # smoke (checks it runs and does not leak): --smoke (2 outer repeats x 8 trials)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")    # silence harmless sklearn/LightGBM warnings (feature names etc.)

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

# Reuse data loading / metrics / bootstrap CI from the regression harness (same protocol as s5)
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
INNER_SPLITS = 3              # inner tuning CV folds (same as classification s3b; 3 folds to save time)
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
    """Sample from a widened continuous/discrete space: first the model type, then its
    hyperparameters. Returns (estimator, model_type). Much wider than the small s5 grid."""
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
    """Mean inner-CV Pearson r. The scaler is fit on each inner training fold -> no leakage."""
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    rs = []
    for itr, iva in inner.split(Xtr, bins_tr):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", est)])
        pipe.fit(Xtr.iloc[itr], ytr.iloc[itr])
        pred = pipe.predict(Xtr.iloc[iva])
        yv = ytr.iloc[iva].to_numpy()
        if np.std(pred) < 1e-12:        # degenerate (constant) prediction, r undefined -> record 0
            rs.append(0.0)
        else:
            rs.append(pearsonr(yv, pred)[0])
    return float(np.mean(rs))


def run_optuna_nested(name, X, y, outer, n_trials, smoke):
    """Nested: one Optuna study per outer fold (sees the training fold only); refit with the
    best params -> predict the test fold once. Returns held-out pooled predictions, per-fold r
    and each fold's trial convergence history (best-so-far)."""
    yt = np.empty(len(y), dtype=float)
    yp = np.empty(len(y), dtype=float)
    strat = _strat_labels(y, N_STRAT_BINS)
    per_fold_r, chosen_types = [], []
    trial_curves = []                    # one best-r-so-far curve per fold (length n_trials)
    n_folds = outer.get_n_splits()
    t0 = time.time()
    for i, (tr, te) in enumerate(outer.split(X, strat)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        bins_tr = strat[tr]              # stratification bins of the outer training fold, for inner stratification

        def objective(trial):
            est, _mt = _make_estimator(trial)
            return _inner_cv_pearson(est, Xtr, ytr, bins_tr)

        sampler = optuna.samplers.TPESampler(seed=SEED + i)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # best-so-far convergence curve (for plotting; this is the inner search objective, not a reported result)
        best_so_far, cur = [], -np.inf
        for t in study.trials:
            v = t.value if t.value is not None else -np.inf
            cur = max(cur, v)
            best_so_far.append(cur)
        trial_curves.append(best_so_far)

        # Refit with the best params on the whole outer training fold; predict the outer test fold once
        best_trial = study.best_trial
        est, mt = _make_estimator(optuna.trial.FixedTrial(best_trial.params))
        final = Pipeline([("scaler", StandardScaler()), ("model", est)])
        final.fit(Xtr, ytr)
        pred = final.predict(X.iloc[te])
        yp[te] = pred
        yt[te] = y.iloc[te].to_numpy()
        # per-fold held-out r (outer test fold)
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
    ap = argparse.ArgumentParser(description="Optuna ceiling stress test (regression)")
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

    # held-out pooled OOF: same scale as the main leaderboard / paired tests
    m = ev.regression_metrics(yt, yp)
    ci = ev.bootstrap_ci(yt, yp, BOOTSTRAP_N, SEED)
    pooled_r = m["pearson_r"]
    fold_mean_r = float(np.mean(per_fold_r))
    types, counts = np.unique(chosen_types, return_counts=True)
    type_dist = ", ".join(f"{t}:{c}" for t, c in zip(types, counts))

    print(f"\n[{args.name}] === ceiling stress-test results ===", flush=True)
    print(f"  held-out pooled OOF: r={pooled_r:.3f} {ev_ci_fmt(ci)}  "
          f"MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} R2={m['r2']:.3f}", flush=True)
    print(f"  held-out per-fold mean r={fold_mean_r:.3f} (n_folds={len(per_fold_r)})",
          flush=True)
    print(f"  selected model-type distribution: {type_dist}", flush=True)

    out_dir = os.path.join(PKG, "results", "summary")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    # Results table
    pd.DataFrame([{
        "dataset": args.name, "n_trials": n_trials,
        "heldout_pooled_r": round(pooled_r, 4),
        "heldout_pooled_r_ci95": f"[{ci['pearson_r'][0]:.3f}, {ci['pearson_r'][1]:.3f}]",
        "heldout_fold_mean_r": round(fold_mean_r, 4),
        "mae": round(m["mae"], 3), "rmse": round(m["rmse"], 3), "r2": round(m["r2"], 4),
        "model_type_dist": type_dist,
    }]).to_csv(os.path.join(out_dir, f"optuna_ceiling_regression_{args.name}{suffix}.csv"),
               index=False)

    # Convergence curve (per-trial mean + IQR of best-so-far across folds), for plotting
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
    print(f"  written: optuna_ceiling_regression_{args.name}{suffix}.csv + "
          f"optuna_trials_regression_{args.name}{suffix}.csv", flush=True)


def ev_ci_fmt(ci):
    c = ci["pearson_r"]
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


if __name__ == "__main__":
    main()
