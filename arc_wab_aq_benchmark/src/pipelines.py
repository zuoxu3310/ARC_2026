"""Leakage-free model pipelines.

Every model is wrapped in an sklearn Pipeline whose first step is a
StandardScaler. Because the whole Pipeline is fit() inside each CV fold, the
scaler learns its mean/std ONLY from the training fold and then transforms the
held-out fold — no statistic ever crosses from test to train. This mirrors the
official ARC demo's manual `scaler.fit_transform(X_train)` /
`scaler.transform(X_test)` split.

`build_pipeline` returns a FRESH, unfitted Pipeline on every call. The evaluator
calls it once per fold so no fitted state is ever reused across folds.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def _estimator(name, seed, device):
    """Instantiate a fresh regressor by name with a fixed seed."""
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "ElasticNet":
        return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=seed, max_iter=10000)
    if name == "SVR_rbf":
        # Matches the Gibson demo SVR(kernel='rbf') with library defaults.
        return SVR(kernel="rbf")
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=500, random_state=seed, n_jobs=-1
        )
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    if name == "MLP":
        # sklearn stand-in for the Gibson Keras net (Dense 64 -> 32 -> 1).
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=2000, early_stopping=False, random_state=seed,
        )
    if name == "TabPFN":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device=device, random_state=seed)
    raise ValueError(f"Unknown model name: {name}")


def build_pipeline(name, seed, device="cpu"):
    """Return a fresh leakage-free Pipeline(StandardScaler -> estimator)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", _estimator(name, seed, device)),
    ])
