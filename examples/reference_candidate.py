#!/usr/bin/env python
"""Train a real example candidate on an ARC bank's exact outer partitions.

Replace make_model() with your method. Fit all preprocessing and any parameter
selection using only that fold's training rows. This example does no tuning.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arc_lesion_image_benchmark/src"))
import arc_reference as reference


def make_model(task, seed):
    if task == "regression":
        return ElasticNet(alpha=0.5, l1_ratio=0.25, max_iter=10000, random_state=seed)
    return LogisticRegression(C=2.0, max_iter=5000, random_state=seed)


def run(bank, output):
    output = Path(output)
    reference.require(not output.exists() or not any(output.iterdir()), "Candidate output must be empty")
    info, patients, features, assignments = reference.load_bank(bank, require_features=True)
    ids = patients.patient_id.tolist()
    lookup = {name: i for i, name in enumerate(ids)}
    X = features[info["features"]].to_numpy(float)
    y = patients.y_true.to_numpy()
    frames, all_records = [], []
    for repeat, seed in info["repeat_seeds"].items():
        pred, probability = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
        folds = np.empty(len(ids), object)
        records = reference.expected_folds(info, patients, assignments, repeat)
        for rec in records:
            tr = np.array([lookup[p] for p in rec["train_ids"]])
            te = np.array([lookup[p] for p in rec["test_ids"]])
            steps = [("scaler", StandardScaler())]
            if info["task"] == "classification":
                minority = int(np.bincount(y[tr].astype(int), minlength=2).min())
                if minority > 1:
                    steps.append(("smote", SMOTE(random_state=seed, k_neighbors=min(5, minority-1))))
            steps.append(("model", make_model(info["task"], seed)))
            pipeline = Pipeline(steps)
            pipeline.fit(X[tr], y[tr])
            pred[te] = pipeline.predict(X[te])
            if info["task"] == "classification":
                probability[te] = pipeline.predict_proba(X[te])[:, 1]
            folds[te] = rec["fold"]
        frame = pd.DataFrame({"patient_id": ids, "repeat": repeat, "fold": folds,
                              "y_true": y, "prediction": pred})
        if info["task"] == "classification":
            frame["score"] = probability
        frames.append(frame)
        all_records.extend(records)
    output.mkdir(parents=True, exist_ok=True)
    reference.write_csv(output / "predictions.csv", pd.concat(frames, ignore_index=True))
    reference.write_json(output / "folds.json", all_records)
    reference.write_json(output / "candidate.json", {"benchmark_id": info["benchmark_id"],
                         "task": info["task"], "outcome": info["outcome"],
                         "description": "Illustrative fixed ElasticNet/LogisticRegression, not a newly proposed prediction method",
                         "configuration": make_model(info["task"], next(iter(info["repeat_seeds"].values()))).get_params()})
    print(f"Trained example on {len(ids)} patients x {info['repeats']} repeats: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.bank, args.output)
