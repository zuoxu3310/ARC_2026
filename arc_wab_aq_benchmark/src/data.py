"""Data loading and Gibson-matched preprocessing.

Replicates the deterministic, leakage-free preprocessing used by the official
ARC demo (deep_learn.py):

  1. read tab-separated table, drop participant id to the index
  2. drop rows where the target (wab_aq) is missing
  3. fillna(0) on remaining cells
  4. drop sparse feature columns: keep a column only if its nonzero count
     exceeds frac * n_rows (default 0.10)

Feature scaling is NOT done here: it lives inside the per-fold Pipeline so the
scaler only ever sees training-fold data (see pipelines.py).
"""
import pandas as pd


def load_data(cfg):
    """Return (X, y, feature_names) for the configured ARC table.

    X is a DataFrame of features (index = participant_id), y is the target
    Series. No scaling is applied here.
    """
    d = cfg["data"]
    df = pd.read_csv(d["path"], sep=d["sep"], index_col=d["id_col"])

    target = d["target"]
    df = df.dropna(subset=[target])
    df = df.fillna(0)

    n = df.shape[0]
    frac = d["sparse_drop_frac"]
    keep_cols = [
        c for c in df.columns
        if c == target or (df[c] != 0).sum() > frac * n
    ]
    df = df[keep_cols]

    X = df.drop(columns=[target])
    y = df[target]
    return X, y, list(X.columns)


def describe(X, y):
    """Short human-readable summary of the loaded dataset."""
    return {
        "n_patients": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std(ddof=1)),
    }
