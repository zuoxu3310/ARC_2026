"""Versioned ARC reference populations, recorded predictions, and paired comparison.

The historical s18 score files are read-only anchors. New prediction banks have
their own identities and preserve any difference from those anchors.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

_warning_filters = warnings.filters[:]
import s18_crossed_grid as legacy
warnings.filters[:] = _warning_filters

PKG = Path(legacy.PKG)
IMPORTED_SOURCE_BYTES = Path(__file__).read_bytes()
IMPORTED_SOURCE_SHA256 = hashlib.sha256(IMPORTED_SOURCE_BYTES).hexdigest()
TRACK = "ARC-v1.0.2-lesion71-AQ50-s18"
ENGINE_VERSION = 1
MODELS = {"regression": legacy.REG_FAST + ["TabPFN"],
          "classification": legacy.CLF_FAST + ["TabPFN"]}
METRICS = {"regression": ["r", "mae", "rmse", "r2"],
           "classification": ["balanced_acc", "auc", "f1_severe", "mcc"]}
PRIMARY = {"regression": "r", "classification": "balanced_acc"}
TASK_OUTCOME = {"regression": "WAB-AQ at the recorded assessment",
                "classification": "Severe aphasia: WAB-AQ <= 50"}
VERSIONS = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
            "xgboost", "lightgbm", "tabpfn", "torch", "threadpoolctl"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{now()}] {message}", flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def versions():
    result = {"python": platform.python_version()}
    for name in VERSIONS:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


@lru_cache(maxsize=8)
def checkpoint_digest(path, mtime_ns, size):
    return sha(path)


def tabpfn_artifact(task):
    from tabpfn.model_loading import get_cache_dir
    name = f"tabpfn-v2.5-{'regressor' if task == 'regression' else 'classifier'}-v2.5_default.ckpt"
    path = get_cache_dir() / name
    require(path.is_file(), "Expected pinned TabPFN 6.3.2 default checkpoint was not found")
    stat = path.stat()
    return {"filename": name, "sha256": checkpoint_digest(str(path), stat.st_mtime_ns, stat.st_size),
            "bytes": stat.st_size, "library_version": importlib.metadata.version("tabpfn")}


def read_frame(path, strings=()):
    return pd.read_csv(path, dtype={name: str for name in strings}, keep_default_na=False)


def read_ids(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = read_frame(path, ["patient_id", "participant_id"])
        key = "patient_id" if "patient_id" in frame else "participant_id"
        require(key in frame, "ID CSV needs a patient_id column")
        ids = frame[key].tolist()
    else:
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    require(ids and len(ids) == len(set(ids)), "Patient IDs must be nonempty and unique")
    require(all(isinstance(p, str) and p and p == p.strip() for p in ids), "Invalid patient ID")
    return ids


def all_anchors():
    frames = []
    for task in MODELS:
        for kind in ["fast", "TabPFN"]:
            path = PKG / f"results/runs/grid_{task}_{kind}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                frame["source_file"] = path.name
                frames.append(frame)
    require(frames, "No manuscript grid scores found; supply the reference results or reproduce the grid")
    result = pd.concat(frames, ignore_index=True)
    require(not result.duplicated(["task", "subset", "model", "seed"]).any(), "Duplicate manuscript grid cells")
    return result


def build_catalog(output):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Catalogue output must be empty")
    anchors = all_anchors()
    definitions = pd.read_csv(Path(legacy.SUBSET_DIR) / "subset_manifest.csv").set_index("subset")
    rows = []
    for (task, cohort, model), part in anchors.groupby(["task", "subset", "model"], sort=True):
        require(sorted(part.seed.tolist()) == list(range(1000, 1020)), "Historical entry lacks 20 expected repeats")
        definition = definitions.loc[cohort]
        ids = read_ids(Path(legacy.SUBSET_DIR) / f"{cohort}_ids.txt")
        require(len(ids) == int(definition.n) == int(part.n.iloc[0]), "Cohort size mismatch")
        row = {"track": TRACK, "cohort": cohort, "category": definition.category,
               "n": len(ids), "task": task, "outcome": TASK_OUTCOME[task], "model": model,
               "primary_metric": PRIMARY[task], "primary_mean": float(part.score.mean()),
               "primary_repeat_sd": float(part.score.std(ddof=1)), "repeats": len(part),
               "folds": 5, "input_features": 71, "cohort_ids_sha256": digest(ids),
               "criterion": definition.criterion, "result_kind": "historical_manuscript_scores",
               "model_configuration": "fixed s18 configuration; not nested hyperparameter search"}
        for metric in METRICS[task]:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_repeat_sd"] = float(part[metric].std(ddof=1))
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "catalog.csv", pd.DataFrame(rows))
    write_csv(output / "scores_by_repeat.csv", anchors.drop(columns="source_file"))
    sources = {str(p.relative_to(PKG)): sha(p) for p in
               sorted((PKG / "results/runs").glob("grid_*.csv"))}
    write_json(output / "catalog.json", {"schema_version": 1, "track": TRACK,
               "created_utc": now(), "entries": len(rows), "repeat_scores": len(anchors),
               "source_sha256": sources,
               "cohort_meaning": "Explicit ARC reference populations; availability rules do not assert identity with unpublished study cohorts.",
               "repeat_sd_meaning": "Variation across repeated splits, not a confidence interval.",
               "direct_use": "Match exact patients, task, feature track and outer folds; use a recorded prediction bank for paired comparisons."})
    log(f"Catalogue: {len(rows)} entries from {len(anchors)} repeat scores -> {output}")
    return rows


def validate_assignments(frame, ids, y, task):
    require(set(["patient_id", "repeat", "fold"]) <= set(frame), "Folds need patient_id,repeat,fold")
    frame = frame[["patient_id", "repeat", "fold"]].astype(str)
    require(not (frame == "").any().any(), "Empty fold value")
    require(not frame.duplicated(["patient_id", "repeat"]).any(), "Duplicate patient/repeat assignment")
    require(set(frame.patient_id) == set(ids), "Fold patients do not match the requested cohort")
    for repeat, part in frame.groupby("repeat", sort=False):
        require(len(part) == len(ids) and set(part.patient_id) == set(ids), f"Incomplete repeat {repeat}")
        require(part.fold.nunique() >= 2, "Each repeat needs at least two nonempty test folds")
        for fold in part.fold.unique():
            train = part.loc[part.fold != fold, "patient_id"].tolist()
            require(len(train) >= 2, f"Insufficient training patients in {repeat}/{fold}")
            if task == "classification":
                require(len(set(y.loc[train].tolist())) == 2, f"One-class training fold in {repeat}/{fold}")
    return frame


def model_parameters(task, model, seed):
    factory = legacy.reg_estimator if task == "regression" else legacy.clf_estimator
    estimator = factory(model, seed)
    params = estimator.get_params(deep=False)
    # XGBoost's documented missing-value sentinel is NaN. Preserve it as an
    # explicit metadata token instead of writing non-standard JSON numbers.
    return json.loads(json.dumps(params, default=str),
                      parse_constant=lambda value: {"nonfinite_parameter": value})


def prepare_bank(output, task, cohort=None, ids_path=None, folds_path=None,
                 repeats=20, base_seed=1000, models=None, table=None):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Bank output must be empty; train resumes an already prepared bank")
    require(bool(cohort) != bool(ids_path), "Choose exactly one of cohort or ids")
    require(task in MODELS and repeats > 0, "Invalid task or repeat count")
    models = list(models or MODELS[task])
    require(models and len(models) == len(set(models)) and set(models) <= set(MODELS[task]), "Invalid model selection")
    table = Path(table or legacy.TABLE)
    raw = pd.read_csv(table, sep="\t", dtype={"participant_id": str}).set_index("participant_id")
    require(raw.index.is_unique and "wab_aq" in raw, "Feature table needs unique participant_id and wab_aq")
    raw = raw.dropna(subset=["wab_aq"]).fillna(0)
    require(len(raw) == 226, "This feature track requires the frozen full 226-patient ARC dictionary")
    columns = [c for c in raw if c != "wab_aq" and (raw[c] != 0).sum() > .10 * len(raw)]
    require(len(columns) == 71, "The ARC reference track requires the original 71 input columns")
    if cohort:
        require(cohort in pd.read_csv(Path(legacy.SUBSET_DIR) / "subset_manifest.csv").subset.tolist(), "Unknown standard cohort")
        ids = read_ids(Path(legacy.SUBSET_DIR) / f"{cohort}_ids.txt")
    else:
        ids = read_ids(ids_path)
        cohort = "custom-" + digest(ids)[:12]
    require(set(ids) <= set(raw.index), "Requested IDs are absent from the ARC feature pool")
    X = raw.loc[ids, columns]
    aq = raw.loc[ids, "wab_aq"]
    y = aq if task == "regression" else (aq <= 50).astype(int)
    require(np.isfinite(X.to_numpy()).all() and np.isfinite(y).all(), "Nonfinite input values")
    if task == "classification":
        require(y.nunique() == 2, "The classification cohort must contain both outcome classes")
    if folds_path:
        assignments = read_frame(folds_path, ["patient_id", "repeat", "fold"])
    else:
        labels = legacy.strat_labels(aq, task)
        require(len(ids) >= 5 and np.min(np.unique(labels, return_counts=True)[1]) >= 5,
                "Generated five-fold stratification needs five patients per stratum; supply valid custom folds for smaller cohorts")
        rows = []
        for seed in range(base_seed, base_seed + repeats):
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (_, test) in enumerate(splitter.split(X, labels), 1):
                rows.extend({"patient_id": ids[i], "repeat": str(seed), "fold": str(fold)} for i in test)
        assignments = pd.DataFrame(rows)
    assignments = validate_assignments(assignments, ids, y, task)
    repeat_names = assignments.repeat.drop_duplicates().tolist()
    seed_map = {r: int(r) if r.isdecimal() else base_seed + i for i, r in enumerate(repeat_names)}
    patients = pd.DataFrame({"patient_id": ids, "y_aq": aq.to_numpy(), "y_true": y.to_numpy()})
    feature_frame = X.reset_index().rename(columns={X.index.name or "index": "patient_id"})
    setup = {"schema_version": 1, "engine_version": ENGINE_VERSION, "track": TRACK,
             "cohort": cohort, "cohort_source": "custom_ids" if ids_path else "standard_rule",
             "task": task, "outcome": TASK_OUTCOME[task], "n": len(ids), "features": columns,
             "models": models, "repeat_seeds": seed_map, "repeats": len(repeat_names),
             "fold_origin": "provided" if folds_path else "within_cohort_stratified_s18",
             "table_sha256": sha(table), "legacy_code_sha256": sha(legacy.__file__),
             "versions": versions(), "patient_order_sha256": digest(ids),
             "preprocessing": "Training-fold StandardScaler; classification uses training-fold SMOTE(k=min(5,minority_n-1)) when possible",
             "feature_dictionary": "Nonzero in >10% of the frozen full 226-patient pool, independent of labels; not reselected per cohort",
             "parameters_seed_example": {m: model_parameters(task, m, next(iter(seed_map.values()))) for m in models}}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "patients.csv", patients)
    write_csv(output / "features.csv", feature_frame)
    write_csv(output / "assignments.csv", assignments)
    setup["input_sha256"] = {name: sha(output / name) for name in ["patients.csv", "features.csv", "assignments.csv"]}
    setup["benchmark_id"] = digest(setup)
    setup["created_utc"] = now()
    if setup["cohort_source"] == "standard_rule" and not folds_path:
        try:
            old = all_anchors()
            old = old[(old.task == task) & (old.subset == cohort) & (old.model.isin(models)) & (old.seed.isin(seed_map.values()))]
            if len(old):
                write_csv(output / "historical_scores.csv", old.drop(columns="source_file"))
                setup["historical_scores_sha256"] = sha(output / "historical_scores.csv")
        except ValueError:
            pass
    write_json(output / "benchmark.json", setup)
    log(f"Prepared {cohort}/{task}: {len(ids)} patients, {len(repeat_names)} repeats, {len(models)} models")
    return setup


def load_bank(path, check_versions=False, require_features=False):
    path = Path(path)
    info = json.loads((path / "benchmark.json").read_text())
    require(info["schema_version"] == 1 and info["engine_version"] == ENGINE_VERSION, "Unsupported benchmark version")
    identity = {k: v for k, v in info.items() if k not in ["benchmark_id", "created_utc", "historical_scores_sha256"]}
    require(digest(identity) == info["benchmark_id"], "Benchmark metadata fingerprint differs")
    for name, value in info["input_sha256"].items():
        if name == "features.csv" and not (path / name).exists() and not require_features:
            continue
        require(sha(path / name) == value, f"Prepared input changed: {name}")
    require(sha(legacy.__file__) == info["legacy_code_sha256"], "The frozen estimator definitions changed")
    if check_versions:
        require(versions() == info["versions"], "Training environment differs from the prepared bank")
    if "historical_scores_sha256" in info:
        require(sha(path / "historical_scores.csv") == info["historical_scores_sha256"], "Historical anchors changed")
    patients = read_frame(path / "patients.csv", ["patient_id"])
    features = read_frame(path / "features.csv", ["patient_id"]) if (path / "features.csv").exists() else None
    assignments = read_frame(path / "assignments.csv", ["patient_id", "repeat", "fold"])
    if features is not None:
        require(patients.patient_id.tolist() == features.patient_id.tolist(), "Feature/patient order mismatch")
    y = pd.Series(patients.y_true.to_numpy(), index=patients.patient_id)
    assignments = validate_assignments(assignments, patients.patient_id.tolist(), y, info["task"])
    return info, patients, features, assignments


def hydrate_bank(path, table=None):
    """Regenerate the exact input matrix; result archives need not bundle features."""
    path = Path(path)
    info, patients, features, _ = load_bank(path)
    if features is not None:
        log(f"Features already match the bank: {path}")
        return
    table = Path(table or legacy.TABLE)
    require(sha(table) == info["table_sha256"], "Feature source differs from this bank's frozen source")
    raw = pd.read_csv(table, sep="\t", dtype={"participant_id": str}).set_index("participant_id")
    raw = raw.dropna(subset=["wab_aq"]).fillna(0)
    frame = raw.loc[patients.patient_id.tolist(), info["features"]].reset_index().rename(columns={"participant_id": "patient_id"})
    content = frame.to_csv(index=False).encode()
    require(hashlib.sha256(content).hexdigest() == info["input_sha256"]["features.csv"], "Regenerated feature serialization differs")
    (path / "features.csv").write_bytes(content)
    log(f"Restored exact feature matrix from verified public inputs: {path}")


def expected_folds(info, patients, assignments, repeat):
    ids = patients.patient_id.tolist()
    part = assignments[assignments.repeat == str(repeat)].set_index("patient_id").loc[ids]
    records = []
    fold_lookup = dict(zip(ids, part.fold.tolist()))
    for fold in part.fold.drop_duplicates():
        test = [p for p in ids if fold_lookup[p] == fold]
        test_set = set(test)
        records.append({"repeat": str(repeat), "fold": str(fold),
                        "train_ids": [p for p in ids if p not in test_set], "test_ids": test})
    return sorted(records, key=lambda r: r["fold"])


def scores(task, truth, prediction, probability=None):
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    require(np.isfinite(y).all() and np.isfinite(p).all(), "Nonfinite scoring input")
    if task == "regression":
        r = float(pearsonr(y, p).statistic) if np.any(p != p[0]) and np.any(y != y[0]) else None
        return {"r": r, "mae": float(mean_absolute_error(y, p)),
                "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p))}
    require(set(y) == {0., 1.} and set(p) <= {0., 1.}, "Classification needs binary labels and both truth classes")
    auc = None
    if probability is not None:
        q = np.asarray(probability, float)
        require(np.isfinite(q).all(), "Nonfinite probability/decision score")
        auc = float(roc_auc_score(y, q))
    return {"balanced_acc": float(balanced_accuracy_score(y, p)), "auc": auc,
            "f1_severe": float(f1_score(y, p, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, p))}


def cell_dir(path, model, repeat):
    require(all(c.isalnum() or c in "_-" for c in str(repeat)), "Repeat names must use letters, digits, underscores or hyphens")
    return Path(path) / "cells" / model / str(repeat)


def check_cell(path, model, repeat):
    info, patients, features, assignments = load_bank(path)
    dest = cell_dir(path, model, repeat)
    result = json.loads((dest / "result.json").read_text())
    require(result["benchmark_id"] == info["benchmark_id"] and result["model"] == model and result["repeat"] == str(repeat), "Cell identity mismatch")
    for name, value in result["artifact_sha256"].items():
        require(sha(dest / name) == value, f"Cell artifact changed: {dest}/{name}")
    frame = read_frame(dest / "predictions.csv", ["patient_id", "repeat", "fold"])
    require(frame.patient_id.tolist() == patients.patient_id.tolist(), "Cell patient coverage/order mismatch")
    require(frame.repeat.unique().tolist() == [str(repeat)], "Wrong prediction repeat")
    require(np.array_equal(frame.y_true.to_numpy(), patients.y_true.to_numpy()), "Wrong prediction outcome")
    target = assignments[assignments.repeat == str(repeat)].set_index("patient_id").loc[frame.patient_id]
    require(frame.fold.tolist() == target.fold.tolist(), "Wrong prediction test folds")
    records = json.loads((dest / "folds.json").read_text())
    expected = expected_folds(info, patients, assignments, repeat)
    require(len(records) == len(expected), "Wrong fold record count")
    for record, wanted in zip(sorted(records, key=lambda r: r["fold"]), expected):
        for key in ["repeat", "fold", "train_ids", "test_ids"]:
            require(record[key] == wanted[key], f"Wrong {key} in cell fold record")
    metric = scores(info["task"], frame.y_true, frame.prediction,
                    frame.score if info["task"] == "classification" else None)
    for name, value in metric.items():
        old = result["metrics"][name]
        require((value is None and old is None) or (value is not None and old is not None and abs(value-old) < 1e-10), f"Saved metric mismatch: {name}")
    return result


def record_cell(path, info, model, repeat, frame, records, origin, seconds, warning_rows=None, extra=None):
    dest = cell_dir(path, model, repeat)
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "predictions.csv", frame)
    write_json(dest / "folds.json", records)
    metric = scores(info["task"], frame.y_true, frame.prediction,
                    frame.score if info["task"] == "classification" else None)
    result = {"benchmark_id": info["benchmark_id"], "model": model, "repeat": str(repeat),
              "origin": origin, "seconds": seconds, "metrics": metric, "created_utc": now(),
              "training_source_sha256": IMPORTED_SOURCE_SHA256, "warnings": warning_rows or [],
              "artifact_sha256": {n: sha(dest / n) for n in ["predictions.csv", "folds.json"]}}
    anchor = Path(path) / "historical_scores.csv"
    if anchor.exists():
        old = pd.read_csv(anchor)
        old = old[(old.model == model) & (old.seed == info["repeat_seeds"][str(repeat)])]
        if len(old) == 1:
            value = metric[PRIMARY[info["task"]]]
            result["historical_primary"] = float(old.score.iloc[0])
            result["historical_difference"] = None if value is None else float(value-old.score.iloc[0])
            result["historical_match_1e_6"] = value is not None and abs(result["historical_difference"]) <= 1e-6
    if extra:
        result.update(extra)
    snapshot = Path(path) / "source_snapshots" / (IMPORTED_SOURCE_SHA256 + ".py")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(IMPORTED_SOURCE_BYTES)
    write_json(dest / "result.json", result)
    return check_cell(path, model, repeat)


def import_full_pool_cell(path, info, patients, assignments, model, repeat, source):
    if info["cohort"] != "full" or info["fold_origin"] != "within_cohort_stratified_s18" or model not in ["RandomForest", "XGBoost"]:
        return None
    source = Path(source)
    filename = f"{info['task']}_{model}_{info['repeat_seeds'][repeat]}"
    pred_path, fold_path = source / "predictions" / (filename + ".csv"), source / "folds" / (filename + ".json")
    if not pred_path.exists() or not fold_path.exists():
        return None
    provenance = json.loads((source / "provenance.json").read_text())
    require(provenance["sha256"]["data/jhu156_features_merged.tsv"] == info["table_sha256"], "Imported features have different provenance")
    require(provenance["features"] == info["features"], "Imported feature dictionary differs")
    old = pd.read_csv(pred_path, dtype={"participant_id": str})
    require(old.participant_id.tolist() == patients.patient_id.tolist(), "Imported patient order differs")
    require(np.array_equal(old.y_true, patients.y_true), "Imported outcomes differ")
    expected = expected_folds(info, patients, assignments, repeat)
    previous = json.loads(fold_path.read_text())
    records = []
    for wanted in expected:
        match = [r for r in previous if str(r["fold"]) == wanted["fold"]]
        require(len(match) == 1, "Imported fold missing")
        require(match[0]["train_ids"] == wanted["train_ids"] and match[0]["test_ids"] == wanted["test_ids"], "Imported memberships differ")
        records.append({**wanted, "scaler_n": match[0]["scaler_n"],
                        "scaler_train_mean_max_error": match[0]["scaler_train_mean_max_error"]})
    frame = pd.DataFrame({"patient_id": old.participant_id, "repeat": repeat, "fold": old.fold.astype(str),
                          "y_true": old.y_true, "prediction": old.prediction})
    if info["task"] == "classification":
        frame["score"] = old.probability_severe
    return record_cell(path, info, model, repeat, frame, records, "imported_verified_full_pool", 0.,
                       extra={"imported_source_sha256": {pred_path.name: sha(pred_path), fold_path.name: sha(fold_path)}})


def train_cell(job):
    path, model, repeat, threads, import_source = job
    path = Path(path)
    info, patients, features, assignments = load_bank(path, check_versions=True, require_features=True)
    require(model in info["models"] and repeat in info["repeat_seeds"], "Unknown requested cell")
    dest = cell_dir(path, model, repeat)
    if (dest / "result.json").exists():
        result = check_cell(path, model, repeat)
        return {**result, "resumed": True}
    if import_source:
        imported = import_full_pool_cell(path, info, patients, assignments, model, repeat, import_source)
        if imported:
            log(f"Imported {info['cohort']}/{info['task']}/{model}/{repeat}")
            return imported
    records = expected_folds(info, patients, assignments, repeat)
    ids = patients.patient_id.tolist()
    lookup = {p: i for i, p in enumerate(ids)}
    X, y = features[info["features"]].to_numpy(float), patients.y_true.to_numpy()
    pred, prob = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    fold_ids = np.empty(len(ids), object)
    seed = info["repeat_seeds"][repeat]
    started = time.monotonic()
    warnings_seen = []
    runtime_models = []
    if model == "TabPFN":
        import torch
        torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        for rec in records:
            begin = time.monotonic()
            tr = np.array([lookup[p] for p in rec["train_ids"]])
            te = np.array([lookup[p] for p in rec["test_ids"]])
            steps = [("scaler", StandardScaler())]
            if info["task"] == "classification":
                minority = int(np.bincount(y[tr].astype(int), minlength=2).min())
                require(minority > 0, "Cannot fit classifier on a single class")
                if minority > 1:
                    steps.append(("smote", SMOTE(random_state=seed, k_neighbors=min(5, minority-1))))
                estimator = legacy.clf_estimator(model, seed)
            else:
                estimator = legacy.reg_estimator(model, seed)
            if model != "TabPFN" and "n_jobs" in estimator.get_params():
                estimator.set_params(n_jobs=threads)
            steps.append(("estimator", estimator))
            pipe = Pipeline(steps)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipe.fit(X[tr], y[tr])
                if model == "TabPFN" and info["task"] == "classification":
                    # In pinned TabPFN 6.3.2, predict() is argmax(_predict_proba)
                    # followed by inverse label encoding. Reuse the same forward
                    # output instead of invoking the network twice per test fold.
                    probabilities = pipe.predict_proba(X[te])
                    encoded = probabilities.argmax(axis=1)
                    encoder = getattr(estimator, "label_encoder_", None)
                    pred[te] = encoder.inverse_transform(encoded) if encoder is not None else encoded
                    prob[te] = probabilities[:, 1]
                else:
                    pred[te] = pipe.predict(X[te])
                if info["task"] == "classification" and model != "TabPFN":
                    prob[te] = pipe.predict_proba(X[te])[:, 1]
            warnings_seen.extend({"fold": rec["fold"], "category": w.category.__name__, "message": str(w.message)} for w in caught)
            fold_ids[te] = rec["fold"]
            scaler = pipe.named_steps["scaler"]
            rec["scaler_n"] = int(scaler.n_samples_seen_)
            rec["scaler_train_mean_max_error"] = float(np.max(np.abs(scaler.mean_ - X[tr].mean(axis=0))))
            require(rec["scaler_n"] == len(tr) and rec["scaler_train_mean_max_error"] < 1e-10, "Scaler used unexpected rows")
            rec["smote_minority_n"] = int(pipe.named_steps["smote"].nn_k_._fit_X.shape[0]) if "smote" in pipe.named_steps else None
            if model == "TabPFN":
                runtime_models.append({"fold": rec["fold"], "model_artifact": tabpfn_artifact(info["task"])})
            rec["seconds"] = time.monotonic()-begin
            dest.mkdir(parents=True, exist_ok=True)
            write_json(dest / "heartbeat.json", {"updated_utc": now(), "pid": os.getpid(), "fold": rec["fold"],
                                                 "folds_total": len(records), "elapsed_seconds": time.monotonic()-started})
            log(f"{info['cohort']}/{info['task']}/{model}/{repeat} fold {rec['fold']}/{len(records)} {rec['seconds']:.2f}s")
    require(np.isfinite(pred).all(), "Incomplete held-out predictions")
    frame = pd.DataFrame({"patient_id": ids, "repeat": repeat, "fold": fold_ids,
                          "y_true": y, "prediction": pred})
    if info["task"] == "classification":
        frame["score"] = prob
    return record_cell(path, info, model, repeat, frame, records, "replayed_s18_configuration",
                       time.monotonic()-started, warnings_seen,
                       {"estimator_threads": threads, "runtime_models": runtime_models})


def train_bank(path, models=None, repeats=None, workers=1, threads=2, import_source=None):
    info, _, _, _ = load_bank(path, check_versions=True, require_features=True)
    models = list(models or info["models"])
    repeats = [str(x) for x in (repeats or info["repeat_seeds"])]
    require(set(models) <= set(info["models"]) and set(repeats) <= set(info["repeat_seeds"]), "Requested model/repeat is not in this bank")
    require(workers >= 1 and threads >= 1, "Worker/thread counts must be positive")
    require("TabPFN" not in models or models == ["TabPFN"], "TabPFN workers must be isolated from the conventional learner pool")
    jobs = [(str(Path(path).resolve()), m, r, threads, str(import_source) if import_source else None)
            for r in repeats for m in models]
    started = time.monotonic()
    done = []
    status_file = Path(path) / ("status_TabPFN.json" if models == ["TabPFN"] else "status_fast.json")
    write_json(status_file, {"status": "running", "started_utc": now(), "pid": os.getpid(), "total": len(jobs), "completed": 0})
    try:
        if workers == 1:
            iterator = map(train_cell, jobs)
            for result in iterator:
                done.append(result)
                write_json(status_file, {"status": "running", "updated_utc": now(), "pid": os.getpid(),
                                        "total": len(jobs), "completed": len(done), "elapsed_seconds": time.monotonic()-started})
        else:
            # sklearn/joblib estimators manage nested worker resources. Using
            # joblib's own outer executor avoids a second multiprocessing
            # resource-tracker layer that can hang on macOS process teardown.
            from joblib import Parallel, delayed, parallel_config
            with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=threads):
                iterator = Parallel(return_as="generator_unordered")(
                    delayed(train_cell)(job) for job in jobs)
                for result in iterator:
                    done.append(result)
                    write_json(status_file, {"status": "running", "updated_utc": now(), "pid": os.getpid(),
                                            "total": len(jobs), "completed": len(done), "elapsed_seconds": time.monotonic()-started})
    except BaseException as exc:
        write_json(status_file, {"status": "failed", "updated_utc": now(), "completed": len(done),
                                "total": len(jobs), "error": repr(exc)})
        raise
    write_json(status_file, {"status": "complete", "updated_utc": now(), "completed": len(done),
                            "total": len(jobs), "elapsed_seconds": time.monotonic()-started})
    return done


def verify_bank(path, allow_partial=False):
    info, _, _, _ = load_bank(path)
    done, missing, rows = [], [], []
    for model in info["models"]:
        for repeat in info["repeat_seeds"]:
            if not (cell_dir(path, model, repeat) / "result.json").exists():
                missing.append({"model": model, "repeat": repeat})
                continue
            result = check_cell(path, model, repeat)
            done.append(result)
            rows.append({"track": info["track"], "cohort": info["cohort"], "task": info["task"],
                         "model": model, "repeat": repeat, "n": info["n"],
                         **result["metrics"], "origin": result["origin"],
                         "historical_primary": result.get("historical_primary"),
                         "historical_difference": result.get("historical_difference"),
                         "historical_match_1e_6": result.get("historical_match_1e_6")})
    frame = pd.DataFrame(rows)
    summary = []
    if len(frame):
        for model, part in frame.groupby("model", sort=False):
            row = {"track": info["track"], "cohort": info["cohort"], "task": info["task"],
                   "model": model, "n": info["n"], "completed_repeats": len(part),
                   "expected_repeats": info["repeats"], "complete": len(part) == info["repeats"],
                   "benchmark_id": info["benchmark_id"]}
            for metric in METRICS[info["task"]]:
                row[metric + "_mean"] = part[metric].mean()
                row[metric + "_repeat_sd"] = part[metric].std(ddof=1)
            summary.append(row)
        write_csv(Path(path) / "scores_by_repeat.csv", frame)
        write_csv(Path(path) / "summary.csv", pd.DataFrame(summary))
    mismatches = [{"model": r["model"], "repeat": r["repeat"], "difference": r.get("historical_difference")}
                  for r in done if r.get("historical_match_1e_6") is False]
    result = {"status": "complete" if not missing else "partial", "verified_utc": now(),
              "benchmark_id": info["benchmark_id"], "completed_cells": len(done),
              "expected_cells": len(info["models"])*info["repeats"], "missing": missing,
              "historical_mismatches": mismatches, "prediction_rows": len(done)*info["n"],
              "warnings": sum(len(r["warnings"]) for r in done)}
    write_json(Path(path) / "verification.json", result)
    require(allow_partial or not missing, f"Bank incomplete: {len(missing)} cells still missing")
    log(f"Verified {info['cohort']}/{info['task']}: {len(done)} cells, {len(missing)} missing, {len(mismatches)} historical differences")
    return result


def export_reference(path, model, output):
    info, patients, _, assignments = load_bank(path)
    require(model in info["models"], "Reference model is absent from this bank")
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Export output must be empty")
    predictions, folds = [], []
    for repeat in info["repeat_seeds"]:
        check_cell(path, model, repeat)
        directory = cell_dir(path, model, repeat)
        predictions.append(read_frame(directory / "predictions.csv", ["patient_id", "repeat", "fold"]))
        folds += json.loads((directory / "folds.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", pd.concat(predictions, ignore_index=True))
    write_json(output / "folds.json", folds)
    write_json(output / "reference.json", {"benchmark_id": info["benchmark_id"], "model": model,
               "outcome": info["outcome"], "track": info["track"], "source": "recorded_reference_bank"})
    return output


def aligned_candidate(info, patients, assignments, predictions, records):
    required = {"patient_id", "repeat", "fold", "prediction"}
    require(required <= set(predictions), f"Candidate predictions need {sorted(required)}")
    predictions = predictions.copy()
    for col in ["patient_id", "repeat", "fold"]:
        predictions[col] = predictions[col].astype(str)
    require(not predictions.duplicated(["patient_id", "repeat"]).any(), "Duplicate candidate patient/repeat")
    require(set(predictions.repeat) == set(info["repeat_seeds"]), "Candidate repeats do not match the reference")
    require(isinstance(records, list), "Candidate folds must be a JSON array")
    keys = [(str(r.get("repeat", "")), str(r.get("fold", ""))) for r in records]
    require(len(keys) == len(set(keys)), "Duplicate candidate fold record")
    lookup = dict(zip(keys, records))
    expected_keys, frames = set(), []
    ids = patients.patient_id.tolist()
    for repeat in info["repeat_seeds"]:
        part = predictions[predictions.repeat == repeat]
        require(len(part) == len(ids) and set(part.patient_id) == set(ids), f"Candidate coverage differs in repeat {repeat}")
        part = part.set_index("patient_id").loc[ids].reset_index()
        target = assignments[assignments.repeat == repeat].set_index("patient_id").loc[ids]
        require(part.fold.tolist() == target.fold.tolist(), f"Candidate test folds differ in repeat {repeat}")
        if "y_true" in part:
            require(np.array_equal(part.y_true.to_numpy(float), patients.y_true.to_numpy(float)), "Candidate outcomes disagree with the reference")
        part["y_true"] = patients.y_true.to_numpy()
        require(np.isfinite(part.prediction.to_numpy(float)).all(), "Candidate predictions are nonfinite")
        if info["task"] == "classification":
            require(set(part.prediction.astype(float)) <= {0., 1.}, "Classification prediction must be 0/1; use score for probabilities")
        if "score" in part:
            require(np.isfinite(part.score.to_numpy(float)).all(), "Candidate scores are nonfinite")
        for wanted in expected_folds(info, patients, assignments, repeat):
            key = (repeat, wanted["fold"])
            expected_keys.add(key)
            require(key in lookup, f"Candidate training record missing: {key}")
            for role in ["train_ids", "test_ids"]:
                actual = lookup[key].get(role)
                require(isinstance(actual, list) and all(isinstance(p, str) for p in actual), f"Candidate {role} must be an ID array")
                require(len(actual) == len(set(actual)) and set(actual) == set(wanted[role]), f"Candidate {role} differs in {key}")
        frames.append(part)
    require(set(lookup) == expected_keys, "Candidate has extra or missing fold records")
    return frames


def weighted_metrics(task, y, pred, weights, score=None):
    """Patient weights are shared across models and all repeated predictions."""
    W = np.asarray(weights, float)
    P = np.asarray(pred, float)
    y = np.asarray(y, float)
    require(W.ndim == 2 and P.ndim == 2 and W.shape[1] == P.shape[1] == len(y), "Metric shape mismatch")
    mass = W.sum(axis=1)
    require((W >= 0).all() and (mass > 0).all(), "Invalid bootstrap weights")
    result = {}
    if task == "regression":
        # Reuse the independently reviewed centered/scaled correlation routine.
        # A raw second-moment shortcut is unstable near constant predictions.
        from s35_compare_predictions import weighted_scores
        stable = weighted_scores(W, y, P, task)
        yc = y-y[0]
        sy = W @ yc
        vy = W @ (yc*yc) - sy*sy/mass
        with np.errstate(divide="ignore", invalid="ignore"):
            result["r"] = stable["pearson_r"]
            result["mae"] = stable["mae"]
            mse = (W @ ((P-y)**2).T)/mass[:, None]
            result["rmse"] = np.sqrt(mse)
            result["r2"] = np.where(vy[:, None] > 0, 1-mse*mass[:, None]/vy[:, None], np.nan)
    else:
        positive, negative = y == 1, y == 0
        p_mass, n_mass = W[:, positive].sum(1), W[:, negative].sum(1)
        tp = W @ ((P == 1)*positive).T
        fp = W @ ((P == 1)*negative).T
        fn, tn = p_mass[:, None]-tp, n_mass[:, None]-fp
        with np.errstate(divide="ignore", invalid="ignore"):
            valid = (p_mass[:, None] > 0) & (n_mass[:, None] > 0)
            result["balanced_acc"] = np.where(valid, (tp/p_mass[:, None]+tn/n_mass[:, None])/2, np.nan)
            f1_den = 2*tp+fp+fn
            result["f1_severe"] = np.where(valid, np.divide(2*tp, f1_den, out=np.zeros_like(tp), where=f1_den > 0), np.nan)
            denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            result["mcc"] = np.where(valid, np.divide(tp*tn-fp*fn, denominator, out=np.zeros_like(tp), where=denominator > 0), np.nan)
        if score is not None:
            values = []
            for q in np.asarray(score, float):
                # Ties receive half credit. Grouping equal scores avoids dependence
                # on patient order and matches the weighted Mann-Whitney statistic.
                unique = np.unique(q)
                group = np.searchsorted(unique, q)
                pg = np.zeros((len(W), len(unique)))
                ng = np.zeros_like(pg)
                for g in range(len(unique)):
                    pg[:, g] = W[:, (group == g) & positive].sum(1)
                    ng[:, g] = W[:, (group == g) & negative].sum(1)
                below = np.cumsum(ng, axis=1)-ng
                numerator = (pg*(below+.5*ng)).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    values.append(np.where((p_mass > 0) & (n_mass > 0), numerator/(p_mass*n_mass), np.nan))
            result["auc"] = np.asarray(values).T
    return result


def compare_candidate(path, model, predictions_path, folds_path, output, draws=2000, seed=20261004):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Comparison output must be empty")
    require(draws >= 100, "Use at least 100 bootstrap draws")
    info, patients, _, assignments = load_bank(path)
    require(model in info["models"], "Unknown baseline model")
    candidate = read_frame(predictions_path, ["patient_id", "repeat", "fold"])
    records = json.loads(Path(folds_path).read_text())
    candidates = aligned_candidate(info, patients, assignments, candidate, records)
    refs, repeat_rows = [], []
    for repeat, part in zip(info["repeat_seeds"], candidates):
        check_cell(path, model, repeat)
        ref = read_frame(cell_dir(path, model, repeat) / "predictions.csv", ["patient_id", "repeat", "fold"])
        refs.append(ref)
        for name, frame in [("reference", ref), ("candidate", part)]:
            metric = scores(info["task"], frame.y_true, frame.prediction,
                            frame.score if info["task"] == "classification" and "score" in frame else None)
            repeat_rows.extend({"repeat": repeat, "method": name, "metric": key, "value": value} for key, value in metric.items() if value is not None)
    y = patients.y_true.to_numpy(float)
    ref_p, cand_p = np.array([f.prediction for f in refs]), np.array([f.prediction for f in candidates])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(y), np.full(len(y), 1/len(y)), size=draws)
    score_pair = info["task"] == "classification" and all("score" in f for f in refs+candidates)
    ref_s = np.array([f.score for f in refs]) if score_pair else None
    cand_s = np.array([f.score for f in candidates]) if score_pair else None
    ref_boot = weighted_metrics(info["task"], y, ref_p, weights, ref_s)
    cand_boot = weighted_metrics(info["task"], y, cand_p, weights, cand_s)
    repeats = pd.DataFrame(repeat_rows)
    rows = []
    for metric in ref_boot.keys() & cand_boot.keys():
        # Require every repeated-fit metric to exist in a resample; do not quietly
        # change the repeat denominator for an undefined correlation.
        valid = np.isfinite(ref_boot[metric]).all(axis=1) & np.isfinite(cand_boot[metric]).all(axis=1)
        difference = (cand_boot[metric][valid]-ref_boot[metric][valid]).mean(axis=1)
        point_ref = repeats[(repeats.method == "reference") & (repeats.metric == metric)].value.mean()
        point_cand = repeats[(repeats.method == "candidate") & (repeats.metric == metric)].value.mean()
        require(np.isfinite(point_ref) and np.isfinite(point_cand), f"Undefined point metric {metric}")
        require(valid.sum() >= 100, f"Too few valid paired resamples for {metric}")
        low, high = np.quantile(difference, [.025, .975])
        rows.append({"metric": metric, "reference": float(point_ref), "candidate": float(point_cand),
                     "candidate_minus_reference": float(point_cand-point_ref), "ci95_low": float(low), "ci95_high": float(high),
                     "valid_draws": int(valid.sum()), "invalid_draws": int(draws-valid.sum()),
                     "better_direction": "lower" if metric in ["mae", "rmse"] else "higher"})
    rows.sort(key=lambda row: METRICS[info["task"]].index(row["metric"]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", pd.DataFrame(rows))
    write_csv(output / "scores_by_repeat.csv", repeats)
    write_json(output / "report.json", {"status": "passed", "benchmark_id": info["benchmark_id"],
               "reference_model": model, "n": len(y), "repeats": info["repeats"],
               "candidate_predictions_sha256": sha(predictions_path), "candidate_folds_sha256": sha(folds_path),
               "bootstrap_draws": draws, "bootstrap_seed": seed,
               "inference": "Paired patient bootstrap, same weights across models/repeats, conditional on recorded fits and folds; pointwise 95% intervals, not an equivalence test.",
               "validation": "Exact patient/outcome/test-fold agreement and recorded training membership. Preprocessing/tuning integrity must be checked in the candidate implementation.",
               "metrics": rows})
    log(f"Paired comparison: {model}, {len(y)} patients, {info['repeats']} repeats -> {output}")
    return rows


def query_catalog(path, cohort=None, task=None, model=None):
    path = Path(path)
    frame = pd.read_csv(path / "catalog.csv" if path.is_dir() else path)
    for name, value in [("cohort", cohort), ("task", task), ("model", model)]:
        if value:
            frame = frame[frame[name] == value]
    require(len(frame), "No matching reference entry")
    columns = ["cohort", "n", "task", "model", "primary_metric"]
    columns += (["reference_primary_mean", "historical_primary_mean", "bank_complete"]
                if "reference_primary_mean" in frame else ["primary_mean", "primary_repeat_sd"])
    columns += ["repeats", "folds"]
    print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="assemble the manuscript's immutable reference score catalogue")
    catalog.add_argument("--output", type=Path, required=True)
    query = commands.add_parser("query", help="look up reference scores")
    query.add_argument("--catalog", type=Path, required=True)
    query.add_argument("--cohort")
    query.add_argument("--task", choices=MODELS)
    query.add_argument("--model")
    prepare = commands.add_parser("prepare", help="freeze standard/custom ARC cohort and evaluation assignments")
    select = prepare.add_mutually_exclusive_group(required=True)
    select.add_argument("--cohort")
    select.add_argument("--ids", type=Path)
    prepare.add_argument("--task", required=True, choices=MODELS)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--folds", type=Path)
    prepare.add_argument("--table", type=Path)
    prepare.add_argument("--repeats", type=int, default=20)
    prepare.add_argument("--base-seed", type=int, default=1000)
    prepare.add_argument("--models", nargs="+")
    hydrate = commands.add_parser("hydrate", help="restore a saved bank's input features from verified public inputs")
    hydrate.add_argument("--bank", type=Path, required=True)
    hydrate.add_argument("--table", type=Path)
    train = commands.add_parser("train", help="train/resume recorded baselines; TabPFN uses isolated workers")
    train.add_argument("--bank", type=Path, required=True)
    train.add_argument("--models", nargs="+")
    train.add_argument("--repeats", nargs="+")
    train.add_argument("--workers", type=int, default=1)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--import-full-pool", type=Path)
    verify = commands.add_parser("verify", help="recompute metrics and check all memberships and fingerprints")
    verify.add_argument("--bank", type=Path, required=True)
    verify.add_argument("--allow-partial", action="store_true")
    export = commands.add_parser("export", help="export one complete model as candidate-compatible files")
    export.add_argument("--bank", type=Path, required=True)
    export.add_argument("--model", required=True)
    export.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="compare an actual candidate against recorded baseline predictions")
    compare.add_argument("--bank", type=Path, required=True)
    compare.add_argument("--model", required=True)
    compare.add_argument("--predictions", type=Path, required=True)
    compare.add_argument("--folds", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--draws", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            build_catalog(args.output)
        elif args.command == "query":
            query_catalog(args.catalog, args.cohort, args.task, args.model)
        elif args.command == "prepare":
            prepare_bank(args.output, args.task, args.cohort, args.ids, args.folds, args.repeats, args.base_seed, args.models, args.table)
        elif args.command == "hydrate":
            hydrate_bank(args.bank, args.table)
        elif args.command == "train":
            info, _, _, _ = load_bank(args.bank)
            selected = args.models or info["models"]
            if "TabPFN" in selected and selected != ["TabPFN"]:
                import subprocess
                train_bank(args.bank, [m for m in selected if m != "TabPFN"], args.repeats,
                           args.workers, args.threads, args.import_full_pool)
                command = [sys.executable, str(Path(__file__).with_name("s37_reference_benchmark.py")),
                           "train", "--bank", str(args.bank.resolve()), "--models", "TabPFN",
                           "--workers", str(min(args.workers, 2)), "--threads", str(args.threads)]
                if args.repeats:
                    command += ["--repeats", *args.repeats]
                subprocess.run(command, check=True)
            else:
                train_bank(args.bank, selected, args.repeats, args.workers, args.threads, args.import_full_pool)
        elif args.command == "verify":
            verify_bank(args.bank, args.allow_partial)
        elif args.command == "export":
            export_reference(args.bank, args.model, args.output)
        else:
            compare_candidate(args.bank, args.model, args.predictions, args.folds, args.output, args.draws, args.seed)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
