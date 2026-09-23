"""Cross-validation runners, regression metrics, and bootstrap CIs.

Two protocols:
  - leave_one_out: refit a fresh pipeline per held-out patient, then pool the
    226 single-point predictions and compute one set of metrics. This matches
    the Gibson demo and is the headline comparison.
  - repeated_kfold: RepeatedKFold; pool out-of-fold predictions per repeat then
    report mean +/- std across repeats (robustness check).

All metrics are computed on pooled predictions. Bootstrap CIs resample whole
patients (their (y_true, y_pred) pairs) with replacement.
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold


def regression_metrics(y_true, y_pred):
    """Pearson r (+p), MAE, RMSE, R^2 on pooled predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r, p = pearsonr(y_true, y_pred)
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def leave_one_out(build_fn, X, y):
    """Pooled leave-one-out predictions. `build_fn()` returns a fresh pipeline."""
    loo = LeaveOneOut()
    y_true = np.empty(len(y), dtype=float)
    y_pred = np.empty(len(y), dtype=float)
    for tr, te in loo.split(X):
        pipe = build_fn()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        y_pred[te] = pipe.predict(X.iloc[te])
        y_true[te] = y.iloc[te].to_numpy()
    return y_true, y_pred


def repeated_kfold(build_fn, X, y, n_splits, n_repeats, seed):
    """Per-repeat pooled metrics under RepeatedKFold; returns list of metric dicts."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds_per_repeat = n_splits
    yt = np.empty(len(y), dtype=float)
    yp = np.empty(len(y), dtype=float)
    repeat_metrics = []
    for i, (tr, te) in enumerate(rkf.split(X)):
        pipe = build_fn()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        yp[te] = pipe.predict(X.iloc[te])
        yt[te] = y.iloc[te].to_numpy()
        if (i + 1) % folds_per_repeat == 0:
            repeat_metrics.append(regression_metrics(yt, yp))
    return repeat_metrics


def bootstrap_ci(y_true, y_pred, n_boot, seed, alpha=0.05):
    """Patient-level percentile bootstrap CIs for each pooled metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    keys = ["pearson_r", "mae", "rmse", "r2"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # need >=2 distinct points and nonzero variance for pearsonr
        if np.ptp(y_true[idx]) == 0 or np.ptp(y_pred[idx]) == 0:
            continue
        m = regression_metrics(y_true[idx], y_pred[idx])
        for k in keys:
            samples[k].append(m[k])
    ci = {}
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        ci[k] = (float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q)))
    return ci
