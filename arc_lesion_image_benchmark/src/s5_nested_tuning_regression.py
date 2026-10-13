#!/usr/bin/env python
"""
s5_nested_tuning_regression.py  -  leakage-safe nested tuning vs default for regression
(tests whether hyperparameter tuning can move the WAB-AQ performance ceiling)

What it does
------------
For each tunable model on the same feature tables (fine JHU / coarse arterial), report two sets:
  * tuned   : outer repeated stratified K-fold; on each outer training fold, GridSearchCV
              (inner K-fold) tunes key hyperparameters, best_estimator_ is refit on that outer
              training fold and predicts only the outer test fold.
  * default : same setup as the existing harness (pipelines.build_pipeline + default
              hyperparameters), same outer folds, fit within fold.
Both pool out-of-fold predictions to compute Pearson r / MAE / RMSE / R2, with patient-level
bootstrap CIs (reusing evaluate.bootstrap_ci).

Leakage guard
-------------
1) StandardScaler is the first Pipeline step and the whole Pipeline goes into GridSearchCV ->
   the scaler is fit_transform'ed only on each inner-train sub-fold; it never sees inner-val,
   let alone the outer test fold.
2) GridSearchCV.fit only receives X.iloc[tr] (outer training fold); hyperparameter selection
   and refit both happen inside the training fold.
3) The outer test fold X.iloc[te] is only used for .predict; nothing is ever fit on it.
4) Outer stratification uses quantile bins of wab_aq (continuous target); bin labels are used
   only for splitting and never enter the features.
5) Tuning and evaluation are separated: the inner CV selects hyperparameters, the outer folds
   only evaluate (nested), so the same data is never used both to tune and to report.

TabPFN
------
By design TabPFN has no grid and is not tuned (its default configuration is the point).
A single default set is run only when --models explicitly includes TabPFN, and it should be
launched in a separate process with a hard timeout (see the last command below), not in the
same process as other models.

Usage
-----
  # SMOKE test (RandomForest only, tiny grid, few repeats):
  python src/s5_nested_tuning_regression.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine \
      --models RandomForest --smoke

  # Full run (all models, full grids):
  python src/s5_nested_tuning_regression.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine \
      --models Ridge ElasticNet SVR_rbf RandomForest XGBoost LightGBM

  # TabPFN in a separate process with a hard timeout (default only):
  perl -e 'alarm 1800; exec @ARGV' python \
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

# reuse the regression harness's data loading / metrics / bootstrap CI / default Pipeline setup
HARNESS_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "arc_wab_aq_benchmark", "src"))
sys.path.insert(0, HARNESS_SRC)
import data as data_mod          # noqa: E402  load_data / describe
import evaluate as ev            # noqa: E402  regression_metrics / bootstrap_ci
from pipelines import build_pipeline  # noqa: E402  default setup (StandardScaler -> est)

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 42
N_SPLITS = 5
N_REPEATS = 10
N_STRAT_BINS = 4          # wab_aq quantile bins for stratifying the continuous target; enough per bin with 226 patients / 5 folds
INNER_SPLITS = 5
BOOTSTRAP_N = 2000
SPARSE_DROP_FRAC = 0.10

# tunable models + tuned estimators + grids (small and reasonable). Keys "model__<param>" target Pipeline step "model".
TUNABLE = {
    "Ridge", "ElasticNet", "SVR_rbf", "RandomForest", "XGBoost", "LightGBM",
}
# TabPFN is not in TUNABLE: always default only.


def _tuned_estimator_and_grid(name, smoke):
    """Return (fresh estimator, param_grid). With smoke, the grid is reduced to a minimum."""
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
        raise ValueError(f"{name} is not a tunable model")

    if smoke:
        # SMOKE: tiny grid (only checks that the nested logic runs, not optimal)
        if name == "RandomForest":
            grid = {"model__n_estimators": [200, 400]}
        else:
            # keep at most the first 2 candidates per hyperparameter
            grid = {k: v[:2] for k, v in grid.items()}
    return est, grid


def _build_tuned_pipe(name, smoke):
    """Pipeline + grid for the tuned run. StandardScaler is the first step -> fit within fold inside GridSearchCV."""
    est, grid = _tuned_estimator_and_grid(name, smoke)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
    return pipe, grid


def _strat_labels(y, n_bins):
    """Continuous wab_aq -> quantile bin labels, used only for the outer stratified split, never in the feature matrix."""
    # qcut may produce fewer bins when there are many ties; duplicates='drop' as a fallback
    labels = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(labels)


def run_default(name, X, y, outer, n_repeats, device="cpu"):
    """Default hyperparameters: reuse build_pipeline, same outer folds, fit within fold. Returns pooled OOF.

    Also returns per_rep[n_samples, n_repeats]: each patient's test-fold prediction in each repeat,
    used only for the per-patient dump (mean across repeats); not part of the existing evaluation.
    """
    yt = np.empty(len(y), dtype=float)
    yp = np.empty(len(y), dtype=float)
    per_rep = np.full((len(y), n_repeats), np.nan, dtype=float)  # for the dump only; evaluation unchanged
    strat = _strat_labels(y, N_STRAT_BINS)
    n_folds = outer.get_n_splits()
    for i, (tr, te) in enumerate(outer.split(X, strat)):
        pipe = build_pipeline(name, SEED, device)     # default-hyperparameter Pipeline (scaler->est)
        pipe.fit(X.iloc[tr], y.iloc[tr])              # outer training fold only
        pred = pipe.predict(X.iloc[te])               # predicts the outer test fold only
        yp[te] = pred                                 # evaluation: unchanged (last repeat overwrites)
        yt[te] = y.iloc[te].to_numpy()
        per_rep[te, i // N_SPLITS] = pred             # dump: one column per repeat, each column covers all samples
        if (i + 1) % n_folds == 0 or (i + 1) % 5 == 0:
            print(f"    [default:{name}] fold {i+1}/{outer.get_n_splits()}", flush=True)
    return yt, yp, per_rep


def run_tuned(name, X, y, outer, n_repeats, smoke):
    """Nested tuning: on each outer training fold, GridSearchCV (inner CV) selects hyperparameters
    and refits; predicts the outer test fold only.

    Also returns per_rep[n_samples, n_repeats]: each patient's test-fold prediction in each repeat,
    used only for the per-patient dump (mean across repeats); not part of the existing evaluation.
    """
    yt = np.empty(len(y), dtype=float)
    yp = np.empty(len(y), dtype=float)
    per_rep = np.full((len(y), n_repeats), np.nan, dtype=float)  # for the dump only; evaluation unchanged
    strat = _strat_labels(y, N_STRAT_BINS)
    chosen = []
    n_folds = outer.get_n_splits()
    for i, (tr, te) in enumerate(outer.split(X, strat)):
        pipe, grid = _build_tuned_pipe(name, smoke)
        gs = GridSearchCV(
            pipe, grid,
            scoring="neg_mean_absolute_error",   # consistent with the regression setup (optimise MAE)
            cv=INNER_SPLITS,                     # inner K-fold (splits the outer training fold only)
            n_jobs=-1, refit=True,
        )
        gs.fit(X.iloc[tr], y.iloc[tr])           # outer training fold only; selection + refit inside it
        pred = gs.predict(X.iloc[te])            # best_estimator_ predicts the outer test fold only
        yp[te] = pred                            # evaluation: unchanged (last repeat overwrites)
        yt[te] = y.iloc[te].to_numpy()
        per_rep[te, i // N_SPLITS] = pred         # dump: one column per repeat, each column covers all samples
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
    """Write each patient's prediction averaged across repeats to
    results/runs/reg_{granularity}_{model}_{config}_patient_pred.csv.

    per_rep[n_samples, n_repeats]: each column is one repeat's full-sample OOF prediction.
    The per-patient mean across repeats (nanmean as a safeguard; normally every cell is filled)
    gives y_pred_mean.
    Columns: participant_id, y_true, y_pred_mean. Dump only; does not affect evaluation.
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
    """Reuse the cleaning rules of harness data.load_data (drop NA -> fillna 0 -> drop sparse columns, no scaling)."""
    abs_path = table_path if os.path.isabs(table_path) else os.path.join(PKG, table_path)
    cfg = {"data": {"path": abs_path, "sep": "\t", "id_col": "participant_id",
                    "target": "wab_aq", "sparse_drop_frac": SPARSE_DROP_FRAC}}
    X, y, feat = data_mod.load_data(cfg)
    return X, y, feat


def main():
    ap = argparse.ArgumentParser(description="Leakage-safe nested tuning vs default for regression")
    ap.add_argument("--table", required=True, help="feature table (relative to PKG, or absolute path)")
    ap.add_argument("--name", required=True, help="dataset label, e.g. JHU_fine / arterial_coarse")
    ap.add_argument("--models", nargs="*",
                    default=["Ridge", "ElasticNet", "SVR_rbf", "RandomForest",
                             "XGBoost", "LightGBM"],
                    help="model subset; TabPFN runs default only (no tuning), preferably in a separate process")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: tiny grid + few repeats, only checks that it runs without leakage")
    ap.add_argument("--repeats", type=int, default=None, help="override the number of outer repeats")
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

    ids = list(X.index)                          # participant_id (data.load_data sets the id as index)
    dump_cfg_suffix = "_smoke" if args.smoke else ""  # smoke dumps get their own name so they never overwrite real outputs

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

        # tuned (TabPFN is not tuned; only tunable models enter the grid search)
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
    if os.path.exists(out_path):                 # merge: drop rows for models re-run now, then concatenate
        old = pd.read_csv(out_path)
        old = old[~old["model"].isin(new["model"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out_path, index=False)
    print(f"\nsummary -> {out_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
