"""Dedicated, observable TabPFN leave-one-out runner.

Why separate: instantiating a fresh TabPFNRegressor for all 226 LOO folds was
unstable on this machine (repeated thread-pool creation -> intermittent 0%-CPU
hang). Here we instantiate the regressor ONCE and re-fit it per fold. This is
still leakage-free: a fresh StandardScaler is fit on each training fold, and
TabPFN.fit() fully replaces its stored training context each call (it never
sees the held-out patient). Progress is flushed per chunk so a stall is visible.

Run with a hard timeout so it can never zombie:
    PYENV_VERSION=data-analysis python src/run_tabpfn_loo.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as config_mod
import data as data_mod
import evaluate as ev

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut


def main():
    cfg = config_mod.load_config()
    np.random.seed(cfg["seed"])
    X, y, _ = data_mod.load_data(cfg)
    n = len(y)
    print(f"TabPFN LOO on {n} patients (reuse-one-estimator)", flush=True)

    from tabpfn import TabPFNRegressor
    reg = TabPFNRegressor(device=cfg["device"], random_state=cfg["seed"])

    loo = LeaveOneOut()
    y_true = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)
    t0 = time.time()
    for i, (tr, te) in enumerate(loo.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X.iloc[tr])
        Xte = scaler.transform(X.iloc[te])
        reg.fit(Xtr, y.iloc[tr])
        y_pred[te] = reg.predict(Xte)
        y_true[te] = y.iloc[te].to_numpy()
        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"  fold {i+1}/{n}  elapsed {time.time()-t0:.1f}s", flush=True)

    m = ev.regression_metrics(y_true, y_pred)
    ci = ev.bootstrap_ci(y_true, y_pred, cfg["bootstrap"]["n"], cfg["seed"])
    pred_path = os.path.join(cfg["output"]["runs_dir"], "TabPFN_loo_predictions.csv")
    pd.DataFrame({"participant_id": X.index, "y_true": y_true, "y_pred": y_pred}).to_csv(pred_path, index=False)

    print(f"TabPFN  r={m['pearson_r']:.3f} [{ci['pearson_r'][0]:.3f}, {ci['pearson_r'][1]:.3f}]  "
          f"MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}", flush=True)
    print(f"predictions -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()
