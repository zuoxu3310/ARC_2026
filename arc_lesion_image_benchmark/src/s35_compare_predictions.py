#!/usr/bin/env python3
"""Compare recorded out-of-fold predictions with explicit population controls.

No estimator is fitted. The input contract is documented in COMPARISON_PROTOCOL.md.
Membership checks verify supplied records, not unobserved preprocessing or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class InputError(ValueError):
    """The inputs do not support the requested controlled comparison."""


def require(condition, message):
    if not condition:
        raise InputError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifiers(values, label):
    require(isinstance(values, list), f"{label}: expected an array of string IDs")
    require(all(isinstance(v, str) and v and v == v.strip() for v in values),
            f"{label}: IDs must be nonempty strings without surrounding whitespace")
    require(len(values) == len(set(values)), f"{label}: duplicate IDs")
    return set(values)


def read_csv(path, columns):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(set(columns).issubset(frame.columns), f"{path.name}: required columns {columns}")
    require(len(frame) > 0, f"{path.name}: empty table")
    return frame


def numeric(series, label):
    try:
        value = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{label}: values must be numeric") from exc
    require(np.isfinite(value).all(), f"{label}: nonfinite or missing value")
    return value


def load_inputs(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    require(isinstance(spec, dict), "spec must be a JSON object")
    require(spec.get("schema_version") == 1, "schema_version must be 1")
    task = spec.get("task")
    require(task in {"regression", "classification"}, "task must be regression or classification")
    for key in ("outcome", "protocol"):
        require(isinstance(spec.get(key), str) and spec[key].strip(), f"{key}: description required")
    paths = {key: (spec_path.parent / spec[key]).resolve()
             for key in ("patients", "predictions", "folds")}
    patients = read_csv(paths["patients"], ["patient_id", "y_true", "eligible"])
    ids = patients.patient_id.tolist()
    universe = identifiers(ids, "patients")
    y = numeric(patients.y_true, "patients.y_true")
    eligible = numeric(patients.eligible, "patients.eligible")
    require(set(eligible).issubset({0, 1}), "eligible must contain only 0 or 1")
    eligible = eligible.astype(bool)
    require(eligible.sum() >= 2, "at least two eligible patients are required")
    selected = set(np.asarray(ids)[eligible])
    require(len(ids) >= 3, "at least three patients are required")
    if task == "classification":
        require(set(y) == {0, 1}, "classification y_true must include both 0 and 1")
        require(set(y[eligible]) == {0, 1}, "eligible patients must include both outcome classes")
    roles = {role: spec[role] for role in ("baseline", "restricted_baseline", "candidate") if role in spec}
    require("baseline" in roles, "baseline model is required")
    for role, model in roles.items():
        require(isinstance(model, str) and model and model == model.strip(), f"{role}: invalid model name")
    require(len(roles) == len(set(roles.values())), "each role must have a distinct model name")
    ref_role = spec.get("candidate_reference")
    if "candidate" in roles:
        require(ref_role in {"baseline", "restricted_baseline"} and ref_role in roles,
                "candidate_reference must name an available baseline role")
    else:
        require(ref_role is None, "candidate_reference requires a candidate")

    pred = read_csv(paths["predictions"], ["patient_id", "repeat", "fold", "model", "y_true", "prediction"])
    for col in ("patient_id", "repeat", "fold", "model"):
        require(pred[col].map(lambda x: bool(x) and x == x.strip()).all(), f"predictions.{col}: invalid value")
    require(set(pred.model) == set(roles.values()), "prediction model names must exactly match the configured roles")
    require(set(pred.patient_id).issubset(universe), "unknown patient in predictions")
    require(not pred.duplicated(["model", "repeat", "patient_id"]).any(), "duplicate patient prediction within model/repeat")
    pred["y_true"] = numeric(pred.y_true, "predictions.y_true")
    pred["prediction"] = numeric(pred.prediction, "predictions.prediction")
    expected_y = pd.Series(y, index=ids).loc[pred.patient_id].to_numpy()
    require(np.allclose(pred.y_true, expected_y, rtol=0, atol=1e-12), "patient outcome differs between input tables")
    if task == "classification":
        require(set(pred.prediction).issubset({0, 1}), "classification prediction must be a 0/1 label, not a probability")
    base = pred.loc[pred.model == roles["baseline"]]
    repeats = sorted(set(base.repeat))
    require(len(repeats) >= 1, "baseline repeats missing")
    fold_assignments, fold_keys = {}, set()
    for rep in repeats:
        one = base.loc[base.repeat == rep]
        require(set(one.patient_id) == universe, f"baseline repeat {rep}: missing patients")
        require(one.fold.nunique() >= 2, f"baseline repeat {rep}: at least two outer folds required")
        fold_assignments[rep] = one.set_index("patient_id").fold.to_dict()
        fold_keys.update((rep, fold) for fold in one.fold.unique())

    raw_folds = json.loads(paths["folds"].read_text())
    require(isinstance(raw_folds, list), "folds must be a JSON array")
    records = {}
    for entry in raw_folds:
        require(isinstance(entry, dict) and all(k in entry for k in ["model", "repeat", "fold", "train_ids", "test_ids"]),
                "fold record is missing required fields")
        key = (str(entry["model"]), str(entry["repeat"]), str(entry["fold"]))
        require(key not in records, "duplicate fold record")
        train = identifiers(entry["train_ids"], f"{key} training")
        test = identifiers(entry["test_ids"], f"{key} testing")
        require(train and train.issubset(universe) and test.issubset(universe), f"{key}: empty or unknown training IDs, or unknown test IDs")
        require(not (train & test), f"{key}: train/test overlap")
        records[key] = (train, test)
    expected_keys = {(model, rep, fold) for model in roles.values() for rep, fold in fold_keys}
    require(set(records) == expected_keys, "fold records must cover exactly every configured model/repeat/fold")

    banks = {}
    for role, model in roles.items():
        frame = pred.loc[pred.model == model]
        scope = selected if role == "candidate" else universe
        require(set(frame.repeat) == set(repeats), f"{model}: repeat set differs from baseline")
        bank = np.zeros((len(repeats), len(ids)), dtype=float)
        for i, rep in enumerate(repeats):
            one = frame.loc[frame.repeat == rep].set_index("patient_id")
            require(set(one.index) == scope, f"{model} repeat {rep}: predictions do not exactly cover declared scoring population")
            require(all(one.loc[p, "fold"] == fold_assignments[rep][p] for p in scope),
                    f"{model} repeat {rep}: patient folds differ from baseline")
            for j, patient in enumerate(ids):
                if patient in scope:
                    bank[i, j] = one.loc[patient, "prediction"]
        banks[role] = bank
        for rep, fold in fold_keys:
            original_test = {p for p, f in fold_assignments[rep].items() if f == fold}
            original_train = universe - original_test
            training_scope = ref_role if role == "candidate" else role
            expected_train = original_train & selected if training_scope == "restricted_baseline" else original_train
            expected_test = original_test & scope
            train, test = records[(model, rep, fold)]
            require(train == expected_train, f"{model} repeat {rep} fold {fold}: training membership differs from declared scope")
            require(test == expected_test, f"{model} repeat {rep} fold {fold}: test membership differs from predictions")
    return spec, paths, ids, y, eligible, repeats, roles, banks, records


def load_bootstrap(spec, directory, ids):
    config = spec.get("bootstrap", {})
    require(isinstance(config, dict), "bootstrap must be an object")
    n = len(ids)
    if "file" in config:
        path = (directory / config["file"]).resolve()
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == {"weights", "patient_ids"}, "bootstrap archive requires weights and patient_ids")
            order = saved["patient_ids"].tolist()
            require(identifiers(order, "bootstrap") == set(ids), "bootstrap patient IDs differ")
            weights = saved["weights"].copy()
        require(weights.ndim == 2 and weights.shape[1] == n, "bootstrap weight shape differs from population")
        lookup = {patient: i for i, patient in enumerate(order)}
        weights = weights[:, [lookup[p] for p in ids]]
        provenance = {"source": "supplied_patient_counts", "sha256": sha(path)}
    else:
        draws, seed = config.get("draws", 2000), config.get("seed", 20261003)
        require(type(draws) is int and draws >= 100, "bootstrap draws must be an integer >= 100")
        require(type(seed) is int and seed >= 0, "bootstrap seed must be a nonnegative integer")
        weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=draws)
        provenance = {"source": "generated_patient_counts", "seed": seed}
    require(weights.shape[0] >= 100, "at least 100 bootstrap draws required")
    require(np.issubdtype(weights.dtype, np.number) and np.isfinite(weights).all(), "bootstrap weights must be finite numbers")
    require((weights >= 0).all() and (weights == np.floor(weights)).all(), "bootstrap weights must be nonnegative integer counts")
    require((weights.sum(axis=1) == n).all(), "each bootstrap draw must sample the full patient count")
    return weights.astype(float), {**provenance, "draws": len(weights)}


def weighted_scores(weights, y, predictions, task):
    """Return one metric per bootstrap draw and repeat; undefined r stays NaN."""
    w = np.atleast_2d(weights).astype(float)
    total = w.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        if task == "regression":
            corr = np.full((len(w), len(predictions)), np.nan)
            for j, row in enumerate(w):
                used = row > 0
                if not used.any():
                    continue
                mass = row[used] / row[used].sum()
                # Subtract an observed anchor before the weighted mean. Raw
                # second moments cancel for nearly constant predictions.
                centered_y = y[used] - y[used][0]
                centered_y -= mass @ centered_y
                centered_p = predictions[:, used] - predictions[:, used][:, :1]
                centered_p -= (centered_p @ mass)[:, None]
                yscale = np.max(np.abs(centered_y))
                pscale = np.max(np.abs(centered_p), axis=1)
                valid = (pscale > 0) & np.isfinite(pscale)
                if yscale == 0 or not np.isfinite(yscale) or not valid.any():
                    continue
                # Scaling before taking norms also avoids squaring extremely
                # small or large centered values. Constants remain undefined.
                yn = centered_y / yscale * np.sqrt(mass)
                pn = centered_p[valid] / pscale[valid, None] * np.sqrt(mass)
                yn /= np.linalg.norm(yn)
                pn /= np.linalg.norm(pn, axis=1, keepdims=True)
                corr[j, valid] = pn @ yn
            return {"pearson_r": np.clip(corr, -1, 1),
                    "mae": (w @ np.abs(predictions - y).T) / total}
        pos, neg = w * (y == 1), w * (y == 0)
        sensitivity = (pos @ (predictions == 1).T) / pos.sum(axis=1, keepdims=True)
        specificity = (neg @ (predictions == 0).T) / neg.sum(axis=1, keepdims=True)
        return {"balanced_accuracy": (sensitivity + specificity) / 2}


def interval(point, draws):
    finite = np.isfinite(draws)
    # A missing class or constant outcome makes that resample undefined.
    # Keep its count; do not replace it with zero or drop repeats selectively.
    low, high = (np.quantile(draws[finite], [.025, .975])
                 if np.isfinite(point) and finite.sum() >= 2 else (np.nan, np.nan))
    return {"estimate": float(point), "ci_low": float(low), "ci_high": float(high),
            "valid_bootstrap_draws": int(finite.sum()), "bootstrap_draws": len(draws)}


def compare(spec_path, output):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    spec, paths, ids, y, eligible, repeats, roles, banks, records = load_inputs(spec_path)
    input_paths = {spec_path, *paths.values()}
    if "file" in spec.get("bootstrap", {}):
        input_paths.add((spec_path.parent / spec["bootstrap"]["file"]).resolve())
    output_paths = {output / name for name in ["scores.csv", "contrasts.csv", "scores_by_repeat.csv", "report.json", "comparison.txt"]}
    require(not (input_paths & output_paths), "output would overwrite an input file")
    weights, boot_info = load_bootstrap(spec, spec_path.parent, ids)
    masks = {"full": np.ones(len(ids), dtype=bool), "eligible": eligible}
    groups = [("S00", "baseline", "full"), ("S01", "baseline", "eligible")]
    if "restricted_baseline" in roles:
        groups += [("S10", "restricted_baseline", "full"), ("S11", "restricted_baseline", "eligible")]
    if "candidate" in roles:
        groups += [("candidate", "candidate", "eligible")]
    points, draws, score_rows, repeat_rows = {}, {}, [], []
    for cell, role, scope in groups:
        mask = masks[scope]
        observed = weighted_scores(mask, y, banks[role], spec["task"])
        sampled = weighted_scores(weights * mask, y, banks[role], spec["task"])
        for metric, repeat_values in observed.items():
            key = cell, metric
            points[key] = float(repeat_values.mean())
            draws[key] = sampled[metric].mean(axis=1)
            score_rows.append({"cell": cell, "model": roles[role], "scoring_population": scope,
                               "n_patients": int(mask.sum()), "coverage": float(mask.mean()), "metric": metric,
                               **interval(points[key], draws[key])})
            repeat_rows += [{"cell": cell, "model": roles[role], "repeat": rep, "metric": metric,
                             "n_patients": int(mask.sum()), "score": float(value)}
                            for rep, value in zip(repeats, repeat_values[0])]
    contrasts = {"scoring_at_full_training": {"S01": 1, "S00": -1}}
    if "restricted_baseline" in roles:
        contrasts.update({"training_at_full_scoring": {"S10": 1, "S00": -1},
                          "training_at_eligible_scoring": {"S11": 1, "S01": -1},
                          "scoring_at_restricted_training": {"S11": 1, "S10": -1},
                          "joint_change": {"S11": 1, "S00": -1},
                          "interaction": {"S11": 1, "S10": -1, "S01": -1, "S00": 1}})
    if "candidate" in roles:
        ref_cell = "S11" if spec["candidate_reference"] == "restricted_baseline" else "S01"
        contrasts["method_difference_on_eligible"] = {"candidate": 1, ref_cell: -1}
    contrast_rows = []
    for name, coeffs in contrasts.items():
        for metric in observed:
            value = sum(coef * points[(cell, metric)] for cell, coef in coeffs.items())
            distribution = sum(coef * draws[(cell, metric)] for cell, coef in coeffs.items())
            contrast_rows.append({"contrast": name, "metric": metric,
                                  "difference_direction": "candidate_minus_reference" if name.startswith("method_") else "as_defined_in_protocol",
                                  "higher_is_better": metric != "mae", **interval(value, distribution)})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(output / "scores.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output / "contrasts.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(output / "scores_by_repeat.csv", index=False)
    report = {"schema_version": 1, "task": spec["task"], "outcome": spec["outcome"],
              "protocol_declared_by_user": spec["protocol"],
              "status": "controlled_method_comparison" if "candidate" in roles else "population_controls_only",
              "full_n": len(ids), "eligible_n": int(eligible.sum()), "eligible_fraction": float(eligible.mean()),
              "repeats": len(repeats), "fold_records_checked": len(records), "models": roles,
              "candidate_reference_role": spec.get("candidate_reference"),
              "bootstrap": boot_info, "training_fits_performed": 0,
              "interval_scope": "Pointwise percentile 95% intervals, conditional on supplied predictions and splits; undefined draws are counted per result.",
              "validation_scope": "Recorded identities, outcomes, fold coverage and training membership; preprocessing, tuning and record authenticity require the original training implementation.",
              "inputs_sha256": {key: sha(path) for key, path in paths.items()} | {"spec": sha(spec_path), "implementation": sha(__file__)},
              "scores_file": "scores.csv", "contrasts_file": "contrasts.csv",
              "data_identifiers_exported": False}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["Controlled comparison of aphasia predictions", "",
             f"Eligible patients: {report['eligible_n']} / {report['full_n']} ({report['eligible_fraction']:.1%})",
             f"Repeated outer evaluation: {len(repeats)} repeats; {len(records)} supplied fold records checked.",
             f"Outcome: {spec['outcome']}", "",
             "Differences below follow the indicated contrast; lower MAE is better.",
             "Balanced accuracy is a proportion (multiply differences by 100 for percentage points)."]
    for row in contrast_rows:
        lines.append(f"{row['contrast']} / {row['metric']}: {row['estimate']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]; "
                     f"valid bootstrap draws {row['valid_bootstrap_draws']}/{row['bootstrap_draws']}")
    lines += ["", report["interval_scope"], report["validation_scope"],
              "Patient-selection contrasts are design-specific; they are not correction factors for published scores from other populations."]
    (output / "comparison.txt").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON input specification")
    parser.add_argument("--output", type=Path, required=True, help="directory for aggregate comparison outputs")
    args = parser.parse_args()
    try:
        report = compare(args.spec, args.output)
    except (InputError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Input rejected: {exc}\n")
    print(json.dumps({key: report[key] for key in ["status", "full_n", "eligible_n", "repeats", "fold_records_checked"]}, indent=2))


if __name__ == "__main__":
    main()
