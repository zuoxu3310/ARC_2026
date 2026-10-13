"""Fit and audit a population-controlled comparison from one recorded outer design.

The source reference bank defines the full pool and its outer folds. References
trained on eligible patients still predict every original test patient. Model
selection uses inner held-out predictions from the current outer training set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import arc_reference as ref
import s35_compare_predictions as protocol

IMPLEMENTATION = Path(__file__).resolve()
SUPPORTED = {"RandomForest", "XGBoost"}
ROLES = ("baseline", "restricted_baseline", "candidate")


def checked_scores(frame):
    """A supplied score column is numeric and finite; absence is explicit."""
    if "score" not in frame:
        return None
    try:
        values = pd.to_numeric(frame.score, errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Supplied candidate/model scores must be numeric and finite") from exc
    ref.require(np.isfinite(values).all(), "Supplied candidate/model scores must be numeric and finite")
    return values


class TrainingNonzeroSelector(BaseEstimator, TransformerMixin):
    """Select the lesion dictionary using the current fitting patients only."""
    def fit(self, X, y=None):
        self.support_ = (np.asarray(X) != 0).sum(axis=0) > .10 * len(X)
        ref.require(self.support_.any(), "No features survive training-only selection")
        self.fit_n_ = len(X)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def validate_policy(policy, task):
    ref.require(policy.get("schema_version") == 1, "Policy schema_version must be 1")
    ref.require(policy.get("objective") in ref.METRICS[task], "Choose a supported selection objective")
    ref.require(policy.get("feature_policy") in {"frozen_dictionary", "training_nonzero"},
                "Declare frozen_dictionary or training_nonzero features")
    ref.require(type(policy.get("inner_folds")) is int and policy["inner_folds"] >= 2,
                "inner_folds must be an integer >= 2")
    for name in ("reference", "candidate"):
        item = policy[name]
        if name == "candidate" and item.get("mode") == "external":
            ref.require(bool(item.get("description", "").strip()), "Describe the external candidate selection policy")
            continue
        ref.require(item.get("model") in SUPPORTED, "Built-in fitting supports RandomForest and XGBoost")
        ref.require(item.get("mode") in {"fixed", "nested"}, "Select fixed or nested policy")
        grid = item.get("parameter_grid")
        ref.require(isinstance(grid, list) and grid and all(isinstance(p, dict) for p in grid),
                    "parameter_grid must explicitly list parameter dictionaries")
        ref.require(len({ref.digest(p) for p in grid}) == len(grid), "Duplicate parameter configurations")
        thresholds = item.get("thresholds", ["estimator"])
        ref.require(isinstance(thresholds, list) and thresholds, "Threshold list must be nonempty")
        ref.require(all(t == "estimator" or (type(t) in {int, float} and 0 <= t <= 1) for t in thresholds),
                    "Invalid threshold")
        ref.require(len({str(t) for t in thresholds}) == len(thresholds), "Duplicate thresholds")
        ref.require(task == "classification" or thresholds == ["estimator"], "Regression has no decision threshold")
        if item["mode"] == "fixed":
            ref.require(len(grid) == len(thresholds) == 1, "Fixed policy requires one configuration and threshold")
        for params in grid:
            ref.require(not ({"random_state", "seed", "n_jobs", "nthread"} & params.keys()),
                        "Random seeds and worker counts are controlled by the recorded run")
            factory = ref.legacy.reg_estimator if task == "regression" else ref.legacy.clf_estimator
            estimator = factory(item["model"], 1000)
            ref.require(set(params) <= set(estimator.get_params()), "Unknown model parameter")


def planned_folds(info, patients, assignments, eligible):
    selected = set(eligible)
    records = []
    for repeat in info["repeat_seeds"]:
        for fold in ref.expected_folds(info, patients, assignments, repeat):
            for role in ROLES:
                records.append({"model": role, "repeat": repeat, "fold": fold["fold"],
                                "train_ids": [p for p in fold["train_ids"] if role == "baseline" or p in selected],
                                "test_ids": [p for p in fold["test_ids"] if role != "candidate" or p in selected]})
    return records


def prepare(pool_bank, eligible_path, policy_path, output, repeats=None, table=None,
            candidate_predictions=None, candidate_folds=None, draws=2000, bootstrap_seed=20261005):
    output = Path(output)
    ref.require(not output.exists() or not any(output.iterdir()), "Output must be empty; use run to resume")
    info, patients, features, assignments = ref.load_bank(pool_bank)
    ref.require(info["cohort"] == "full" and len(patients) == 226,
                "Start with a full-pool bank; a within-subset bank cannot define the full-pool controls")
    if repeats is not None:
        repeats = [str(r) for r in repeats]
        ref.require(repeats and len(repeats) == len(set(repeats)) and set(repeats) <= set(info["repeat_seeds"]),
                    "Requested repeats must be unique and present in the full-pool bank")
        info = {**info, "repeat_seeds": {r: info["repeat_seeds"][r] for r in repeats}}
        assignments = assignments[assignments.repeat.isin(repeats)].copy()
    eligible = ref.read_ids(eligible_path)
    ref.require(len(eligible) >= 2 and set(eligible) <= set(patients.patient_id), "Invalid eligible membership")
    policy = json.loads(Path(policy_path).read_text())
    validate_policy(policy, info["task"])
    ref.require(type(draws) is int and draws >= 100 and type(bootstrap_seed) is int and bootstrap_seed >= 0,
                "Invalid bootstrap plan")
    if features is None or policy["feature_policy"] == "training_nonzero":
        table = Path(table or ref.legacy.TABLE)
        ref.require(ref.sha(table) == info["table_sha256"], "Feature source does not match the recorded bank")
        raw = pd.read_csv(table, sep="\t", dtype={"participant_id": str}).set_index("participant_id")
        raw = raw.dropna(subset=["wab_aq"]).fillna(0)
        columns = info["features"] if policy["feature_policy"] == "frozen_dictionary" else [c for c in raw if c != "wab_aq"]
        features = raw.loc[patients.patient_id, columns].reset_index().rename(columns={"participant_id": "patient_id"})
    else:
        columns = info["features"]
    ref.require(features.patient_id.tolist() == patients.patient_id.tolist(), "Feature row order differs")
    ref.require(np.isfinite(features[columns].to_numpy(float)).all(), "Nonfinite features")
    patients = patients.assign(eligible=patients.patient_id.isin(eligible).astype(int))
    records = planned_folds(info, patients, assignments, eligible)
    labels = patients.set_index("patient_id").y_true
    if info["task"] == "classification":
        ref.require(set(labels.loc[eligible]) == {0, 1}, "Eligible population must contain both outcome classes")
    for rec in records:
        item = policy["reference" if rec["model"] != "candidate" else "candidate"]
        y = labels.loc[rec["train_ids"]].to_numpy()
        ref.require(len(y) >= 2, "Insufficient training patients")
        if info["task"] == "classification":
            ref.require(set(y) == {0, 1}, "An eligible training fold has only one class")
        if item["mode"] == "nested":
            minimum = np.bincount(y.astype(int)).min() if info["task"] == "classification" else len(y)
            ref.require(minimum >= policy["inner_folds"], "Too few training patients for the declared inner folds")
    external = policy["candidate"]["mode"] == "external"
    ref.require(external == bool(candidate_predictions and candidate_folds),
                "External policy needs both candidate predictions and folds; built-in policy needs neither")
    ref.require(external or not (candidate_predictions or candidate_folds), "Unexpected external candidate file")
    output.mkdir(parents=True, exist_ok=True)
    ref.write_csv(output / "patients.csv", patients)
    ref.write_csv(output / "features.csv", features)
    ref.write_csv(output / "assignments.csv", assignments)
    ref.write_json(output / "planned-folds.json", records)
    ref.write_json(output / "policy.json", policy)
    input_names = ["patients.csv", "features.csv", "assignments.csv", "planned-folds.json", "policy.json"]
    if external:
        frame = ref.read_frame(candidate_predictions, ["patient_id", "repeat", "fold"])
        ref.require(set(["patient_id", "repeat", "fold", "prediction"]) <= set(frame), "Candidate columns missing")
        provided_scores = checked_scores(frame)
        ref.require(policy["objective"] != "auc" or provided_scores is not None,
                    "The declared AUC comparison needs candidate scores")
        if "y_true" in frame:
            ref.require(set(frame.patient_id) <= set(labels.index), "Unknown candidate patient")
            ref.require(np.array_equal(frame.y_true.to_numpy(), labels.loc[frame.patient_id].to_numpy()), "Candidate targets differ")
        frame["y_true"] = labels.reindex(frame.patient_id).to_numpy()
        frame["model"] = "candidate"
        external_records = [{**r, "model": "candidate", "repeat": str(r["repeat"]), "fold": str(r["fold"])}
                            for r in json.loads(Path(candidate_folds).read_text())]
        ref.write_csv(output / "external-predictions.csv", frame)
        ref.write_json(output / "external-folds.json", external_records)
        input_names += ["external-predictions.csv", "external-folds.json"]
        # Validate external memberships before fitting a single reference.
        placeholder = []
        for rec in records:
            if rec["model"] != "candidate":
                placeholder.extend({"patient_id": p, "repeat": rec["repeat"], "fold": rec["fold"],
                                    "model": rec["model"], "y_true": labels[p], "prediction": 0.0} for p in rec["test_ids"])
        validation = output / "preflight"
        validation.mkdir()
        ref.write_csv(validation / "predictions.csv", pd.concat([pd.DataFrame(placeholder), frame], ignore_index=True))
        ref.write_json(validation / "folds.json", [r for r in records if r["model"] != "candidate"] + external_records)
        preflight = comparison_spec(info["task"], "../patients.csv", draws, bootstrap_seed)
        ref.write_json(validation / "comparison.json", preflight)
        protocol.load_inputs(validation / "comparison.json")
    source_paths = [IMPLEMENTATION, Path(ref.__file__), Path(protocol.__file__), Path(ref.legacy.__file__)]
    setup = {"schema_version": 1, "source_bank_id": info["benchmark_id"], "task": info["task"],
             "outcome": info["outcome"], "feature_columns": columns, "repeat_seeds": info["repeat_seeds"],
             "full_n": len(patients), "eligible_n": len(eligible), "outer_fold_origin": info["fold_origin"],
             "objective": policy["objective"], "versions": ref.versions(),
             "draws": draws, "bootstrap_seed": bootstrap_seed,
             "source_sha256": {p.name: ref.sha(p) for p in source_paths},
             "input_sha256": {n: ref.sha(output / n) for n in input_names}}
    setup["run_id"] = ref.digest(setup)
    ref.write_json(output / "run.json", setup)
    ref.log(f"Prepared controlled run: {len(patients)} full / {len(eligible)} eligible, {len(info['repeat_seeds'])} repeats")
    return setup


def load_run(directory, require_environment=True):
    directory = Path(directory)
    info = json.loads((directory / "run.json").read_text())
    ref.require(info["schema_version"] == 1 and ref.digest({k: v for k, v in info.items() if k != "run_id"}) == info["run_id"],
                "Run metadata fingerprint differs")
    for name, value in info["input_sha256"].items():
        ref.require(ref.sha(directory / name) == value, f"Prepared input changed: {name}")
    for source in [IMPLEMENTATION, Path(ref.__file__), Path(protocol.__file__), Path(ref.legacy.__file__)]:
        ref.require(ref.sha(source) == info["source_sha256"][source.name], f"Run implementation changed: {source.name}")
    if require_environment:
        ref.require(ref.versions() == info["versions"], "Recorded training environment changed")
    return info, json.loads((directory / "policy.json").read_text()), ref.read_frame(directory / "patients.csv", ["patient_id"]), ref.read_frame(directory / "features.csv", ["patient_id"])


def pipeline(task, item, params, seed, threads, y, feature_policy):
    steps = []
    if feature_policy == "training_nonzero":
        steps.append(("features", TrainingNonzeroSelector()))
    steps.append(("scaler", StandardScaler()))
    if task == "classification":
        minority = int(np.bincount(np.asarray(y, int), minlength=2).min())
        ref.require(minority > 0, "Cannot fit a single-class training set")
        if minority > 1:
            steps.append(("smote", SMOTE(random_state=seed, k_neighbors=min(5, minority - 1))))
    factory = ref.legacy.reg_estimator if task == "regression" else ref.legacy.clf_estimator
    estimator = factory(item["model"], seed)
    estimator.set_params(**params, n_jobs=threads)
    steps.append(("estimator", estimator))
    return Pipeline(steps)


def predict(pipe, X, task, threshold):
    if not len(X):
        return np.array([]), np.array([])
    prediction = pipe.predict(X)
    score = pipe.predict_proba(X)[:, 1] if task == "classification" else None
    if task == "classification" and threshold != "estimator":
        prediction = (score >= threshold).astype(int)
    return prediction, score


def select_policy(X, y, ids, task, item, objective, folds, feature_policy, seed, threads):
    thresholds = item.get("thresholds", ["estimator"])
    if item["mode"] == "fixed":
        return {"configuration": 0, "parameters": item["parameter_grid"][0], "threshold": thresholds[0]}, [], [], [], 0
    splitter = (StratifiedKFold(folds, shuffle=True, random_state=seed) if task == "classification"
                else KFold(folds, shuffle=True, random_state=seed))
    splits = list(splitter.split(X, y))
    memberships = [{"inner_fold": i, "train_ids": [ids[j] for j in tr], "validation_ids": [ids[j] for j in va]}
                   for i, (tr, va) in enumerate(splits, 1)]
    trials, predictions = [], []
    for configuration, params in enumerate(item["parameter_grid"]):
        pred, score = np.full(len(y), np.nan), np.full(len(y), np.nan)
        inner_assignment = np.zeros(len(y), int)
        for inner, (tr, va) in enumerate(splits, 1):
            pipe = pipeline(task, item, params, seed, threads, y[tr], feature_policy)
            pipe.fit(X[tr], y[tr])
            p, q = predict(pipe, X[va], task, "estimator")
            pred[va] = p
            if q is not None:
                score[va] = q
            inner_assignment[va] = inner
        for j, patient in enumerate(ids):
            row = {"patient_id": patient, "configuration": configuration, "inner_fold": int(inner_assignment[j]),
                   "y_true": float(y[j]), "prediction": float(pred[j])}
            if task == "classification":
                row["score"] = float(score[j])
            predictions.append(row)
        for threshold in thresholds:
            p = (score >= threshold).astype(int) if threshold != "estimator" else pred
            metrics = ref.scores(task, y, p, score if task == "classification" else None)
            value = metrics[objective]
            ref.require(value is not None and np.isfinite(value), "Undefined inner selection objective")
            trials.append({"configuration": configuration, "parameters": params, "threshold": threshold,
                           "objective_value": float(value), "inner_metrics": metrics})
    direction = -1 if objective in {"mae", "rmse"} else 1
    # First declared configuration/threshold wins a numerical tie.
    selected = max(trials, key=lambda t: direction * t["objective_value"])
    return selected, trials, predictions, memberships, len(splits) * len(item["parameter_grid"])


def unit_path(directory, index):
    return Path(directory) / "units" / f"{index:05d}"


def verify_unit(path, run_id, record):
    result = json.loads((path / "result.json").read_text())
    ref.require(result["run_id"] == run_id and result["membership"] == record, "Unit identity or membership changed")
    for name, value in result["files_sha256"].items():
        ref.require(ref.sha(path / name) == value, f"Saved unit changed: {path}/{name}")
    return result


def run(directory, threads=2):
    directory = Path(directory)
    info, policy, patients, features = load_run(directory)
    ref.require(type(threads) is int and threads >= 1, "threads must be positive")
    lookup = {p: i for i, p in enumerate(patients.patient_id)}
    X = features[info["feature_columns"]].to_numpy(float)
    y = patients.y_true.to_numpy()
    records = json.loads((directory / "planned-folds.json").read_text())
    total_units = sum(policy["reference" if record["model"] != "candidate" else "candidate"]["mode"] != "external"
                      for record in records)
    units = []
    with threadpool_limits(limits=threads):
        for index, record in enumerate(records):
            item = policy["reference" if record["model"] != "candidate" else "candidate"]
            if item["mode"] == "external":
                continue
            destination = unit_path(directory, index)
            if (destination / "result.json").exists():
                units.append(verify_unit(destination, info["run_id"], record))
                continue
            started = time.monotonic()
            seed = info["repeat_seeds"][record["repeat"]]
            tr = np.array([lookup[p] for p in record["train_ids"]], int)
            te = np.array([lookup[p] for p in record["test_ids"]], int)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                selected, trials, inner, inner_folds, fits = select_policy(
                    X[tr], y[tr], record["train_ids"], info["task"], item, policy["objective"],
                    policy["inner_folds"], policy["feature_policy"], seed, threads)
                pipe = pipeline(info["task"], item, selected["parameters"], seed, threads, y[tr], policy["feature_policy"])
                pipe.fit(X[tr], y[tr])
                p, q = predict(pipe, X[te], info["task"], selected["threshold"])
            prediction = pd.DataFrame({"patient_id": record["test_ids"], "repeat": record["repeat"],
                                       "fold": record["fold"], "model": record["model"], "y_true": y[te], "prediction": p})
            if q is not None:
                prediction["score"] = q
            transformed = pipe.named_steps["features"].transform(X[tr]) if "features" in pipe.named_steps else X[tr]
            scaler = pipe.named_steps["scaler"]
            ref.require(int(scaler.n_samples_seen_) == len(tr), "Scaler training count differs")
            error = float(np.max(np.abs(scaler.mean_ - transformed.mean(axis=0))))
            ref.require(error < 1e-10, "Scaler used unexpected data")
            names = [n for n, keep in zip(info["feature_columns"], pipe.named_steps["features"].support_) if keep] if "features" in pipe.named_steps else info["feature_columns"]
            ref.write_csv(destination / "predictions.csv", prediction)
            ref.write_json(destination / "selection.json", {"selected": selected, "trials": trials,
                           "inner_folds": inner_folds, "objective": policy["objective"], "mode": item["mode"]})
            ref.write_csv(destination / "inner-predictions.csv", pd.DataFrame(inner, columns=
                          ["patient_id", "configuration", "inner_fold", "y_true", "prediction"] + (["score"] if info["task"] == "classification" else [])))
            result = {"run_id": info["run_id"], "membership": record, "policy": item,
                      "fits": fits + 1, "threads": threads, "feature_columns": names,
                      "scaler_n": int(scaler.n_samples_seen_), "scaler_mean_error": error,
                      "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                      "seconds": time.monotonic() - started,
                      "files_sha256": {n: ref.sha(destination / n) for n in ["predictions.csv", "selection.json", "inner-predictions.csv"]}}
            ref.write_json(destination / "result.json", result)
            units.append(verify_unit(destination, info["run_id"], record))
            ref.write_json(directory / "progress.json", {"complete_units": len(units), "latest_index": index,
                           "total_units": total_units, "status": "running", "updated_utc": ref.now()})
            ref.log(f"{record['model']} repeat={record['repeat']} fold={record['fold']} fits={fits+1} train={len(tr)} test={len(te)} seconds={result['seconds']:.2f}")
    frames, exported_folds = [], []
    for index, record in enumerate(records):
        item = policy["reference" if record["model"] != "candidate" else "candidate"]
        if item["mode"] != "external":
            verify_unit(unit_path(directory, index), info["run_id"], record)
            frames.append(ref.read_frame(unit_path(directory, index) / "predictions.csv", ["patient_id", "repeat", "fold"]))
            exported_folds.append(record)
    if policy["candidate"]["mode"] == "external":
        frames.append(ref.read_frame(directory / "external-predictions.csv", ["patient_id", "repeat", "fold"]))
        exported_folds += json.loads((directory / "external-folds.json").read_text())
    inputs = directory / "comparison-inputs"
    ref.write_csv(inputs / "patients.csv", patients[["patient_id", "y_true", "eligible"]])
    ref.write_csv(inputs / "predictions.csv", pd.concat(frames, ignore_index=True))
    ref.write_json(inputs / "folds.json", exported_folds)
    spec = comparison_spec(info["task"], "patients.csv", info["draws"], info["bootstrap_seed"])
    spec["protocol"] = f"Run {info['run_id']}; recorded full-pool outer folds; {policy['feature_policy']}; policy.json declares {policy['objective']} selection."
    ref.write_json(inputs / "comparison.json", spec)
    result = protocol.compare(inputs / "comparison.json", directory / "comparison")
    all_metrics(inputs / "comparison.json", directory / "comparison")
    ref.write_json(directory / "COMPLETE.json", {"run_id": info["run_id"], "status": "complete",
                   "units": len(units), "training_fits": sum(u["fits"] for u in units),
                   "fold_records_checked": result["fold_records_checked"],
                   "output_sha256": {str(p.relative_to(directory)): ref.sha(p) for p in sorted(directory.glob("comparison*/*")) if p.is_file()}})
    ref.write_json(directory / "progress.json", {"complete_units": len(units), "total_units": total_units,
                   "status": "complete", "updated_utc": ref.now()})
    return result


def comparison_spec(task, patients, draws, seed):
    return {"schema_version": 1, "task": task, "outcome": ref.TASK_OUTCOME[task],
            "protocol": "Recorded full-pool outer folds; declared reference/candidate policies; selection uses training patients.",
            "patients": patients, "predictions": "predictions.csv", "folds": "folds.json",
            "baseline": "baseline", "restricted_baseline": "restricted_baseline", "candidate": "candidate",
            "candidate_reference": "restricted_baseline", "bootstrap": {"draws": draws, "seed": seed}}


def all_metrics(spec_path, output):
    """Use shared patient resamples for the complete metric and contrast path."""
    spec, paths, ids, y, eligible, repeats, roles, banks, _ = protocol.load_inputs(spec_path)
    frame = ref.read_frame(paths["predictions"], ["patient_id", "repeat", "fold"])
    W, _ = protocol.load_bootstrap(spec, Path(spec_path).parent, ids)
    groups = [("S00", "baseline", np.ones(len(ids), bool)), ("S01", "baseline", eligible),
              ("S10", "restricted_baseline", np.ones(len(ids), bool)), ("S11", "restricted_baseline", eligible),
              ("candidate", "candidate", eligible)]
    points, draws, score_rows, availability = {}, {}, [], {}
    for cell, role, mask in groups:
        pred = banks[role][:, mask]
        values = frame[frame.model == roles[role]]
        score = None
        if spec["task"] == "classification":
            # Combined CSVs have empty score cells for a model that supplied no
            # score column. Partial missingness or malformed values are errors.
            has_scores = "score" in values and not values.score.astype(str).eq("").all()
            availability[role] = "provided" if has_scores else "not_supplied"
            if has_scores:
                checked_scores(values)
                score = np.array([values[values.repeat == repeat].set_index("patient_id").loc[np.asarray(ids)[mask], "score"].to_numpy(float) for repeat in repeats])
        observed = ref.weighted_metrics(spec["task"], y[mask], pred, np.ones((1, mask.sum())), score)
        valid_mass = W[:, mask].sum(axis=1) > 0
        sampled = ref.weighted_metrics(spec["task"], y[mask], pred, W[valid_mass][:, mask], score)
        for metric, value in observed.items():
            points[cell, metric] = float(value.mean())
            distribution = np.full(len(W), np.nan)
            distribution[valid_mass] = sampled[metric].mean(axis=1)
            draws[cell, metric] = distribution
            score_rows.append({"cell": cell, "metric": metric, **protocol.interval(points[cell, metric], distribution)})
    contrasts = {"scoring": {"S01": 1, "S00": -1}, "training": {"S11": 1, "S01": -1},
                 "method": {"candidate": 1, "S11": -1}, "apparent_total": {"candidate": 1, "S00": -1},
                 "training_at_full_scoring": {"S10": 1, "S00": -1},
                 "scoring_at_restricted_training": {"S11": 1, "S10": -1},
                 "interaction": {"S11": 1, "S10": -1, "S01": -1, "S00": 1}}
    metrics = set.intersection(*[{m for c, m in points if c == cell} for cell, _, _ in groups])
    rows, identities = [], []
    for metric in sorted(metrics):
        values, distributions = {}, {}
        for name, coefficients in contrasts.items():
            values[name] = sum(v * points[c, metric] for c, v in coefficients.items())
            distributions[name] = sum(v * draws[c, metric] for c, v in coefficients.items())
            rows.append({"contrast": name, "metric": metric, **protocol.interval(values[name], distributions[name])})
        residual = distributions["apparent_total"] - distributions["scoring"] - distributions["training"] - distributions["method"]
        finite = np.isfinite(residual)
        error = float(np.max(np.abs(residual[finite]))) if finite.any() else None
        ref.require(error is None or error < 1e-10, "Paired contrast path failed")
        point_residual = values["apparent_total"] - values["scoring"] - values["training"] - values["method"]
        identities.append({"metric": metric, "max_shared_draw_identity_error": error,
                           "valid_identity_draws": int(finite.sum()),
                           "status": "verified" if error is not None else "undefined_metric",
                           "point_residual": float(point_residual) if np.isfinite(point_residual) else None})
    ref.write_csv(Path(output) / "all-metric-scores.csv", pd.DataFrame(score_rows))
    ref.write_csv(Path(output) / "all-metric-contrasts.csv", pd.DataFrame(rows))
    ref.write_json(Path(output) / "path-verification.json", {"metrics": identities, "bootstrap_draws": len(W),
                   "intervals": "Each contrast uses the same patient draws; interval endpoints are not added."})
    ref.write_json(Path(output) / "metric-availability.json", {"score_columns": availability,
                   "paired_metrics": sorted(metrics), "undefined_metrics": [m["metric"] for m in identities if m["status"] == "undefined_metric"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--pool-bank", type=Path, required=True)
    p.add_argument("--eligible-ids", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", nargs="+")
    p.add_argument("--table", type=Path)
    p.add_argument("--candidate-predictions", type=Path)
    p.add_argument("--candidate-folds", type=Path)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261005)
    r = commands.add_parser("run")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.pool_bank, args.eligible_ids, args.policy, args.output, args.repeats, args.table,
                    args.candidate_predictions, args.candidate_folds, args.draws, args.bootstrap_seed)
        else:
            run(args.run, args.threads)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Comparison rejected: {exc}\n")


if __name__ == "__main__":
    main()
