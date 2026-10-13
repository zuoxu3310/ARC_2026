#!/usr/bin/env python
"""Restrict training membership while preserving every original outer test fold.

Reuses s31's full-training prediction bank. New predictions cover all 226
patients for every eligibility-restricted training condition. See --stage.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import s18_crossed_grid as base
import s31_frozen_prediction_control as frozen

OUT = frozen.ROOT / "results/fixed_test_training_2026-10-02"


def provenance():
    X, aq, features, manifest, masks = frozen.load_data()
    source_files = [Path(__file__), Path(base.__file__), Path(frozen.__file__), Path(base.TABLE),
                    frozen.OUT / "resampling_patient_weights.npz", frozen.OUT / "provenance.json"]
    source_files += sorted((Path(base.SUBSET_DIR)).glob("*_ids.txt"))
    source_files += sorted((frozen.OUT / "predictions").glob("*.csv"))
    source_files += sorted((frozen.OUT / "folds").glob("*.json"))
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "joblib", "threadpoolctl"]
    versions = {p: importlib.metadata.version(p) for p in packages}
    old = json.loads((frozen.OUT / "provenance.json").read_text())["versions"]
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost"]:
        assert versions[name] == old[name], (name, versions[name], old[name])
    return {"files_sha256": {str(p): frozen.sha(p) for p in source_files},
            "versions": versions, "python": sys.version, "features": features,
            "tasks": frozen.TASKS, "models": frozen.MODELS, "seeds": frozen.SEEDS,
            "restricted_sets": [s for s in manifest.subset if s != "full"],
            "n_patients": len(X), "folds": 5, "estimator_threads": 2}


def one_cell(job):
    subset, task, model, seed, smoke = job
    X, aq, _, _, masks = frozen.load_data()
    ids = X.index.to_numpy(); id_to_i = {p: i for i, p in enumerate(ids)}
    values = X.to_numpy()
    y = aq.to_numpy() if task == "regression" else (aq.to_numpy() <= 50).astype(int)
    records = json.loads((frozen.OUT / "folds" / f"{task}_{model}_{seed}.json").read_text())
    old = pd.read_csv(frozen.OUT / "predictions" / f"{task}_{model}_{seed}.csv")
    assert old.participant_id.tolist() == ids.tolist()
    pred = np.full(len(ids), np.nan); prob = np.full(len(ids), np.nan); fold_ids = np.zeros(len(ids), dtype=int)
    saved_folds = []
    begin = time.monotonic()
    with threadpool_limits(limits=2):
        for rec in records:
            fold_start = time.monotonic()
            original_train = np.array([id_to_i[p] for p in rec["train_ids"]])
            te = np.array([id_to_i[p] for p in rec["test_ids"]])
            tr = original_train[masks[subset][original_train]]
            assert len(tr) > 1 and not set(tr) & set(te)
            assert set(tr) <= set(original_train)
            np.testing.assert_array_equal(old.fold.to_numpy()[te], np.repeat(rec["fold"], len(te)))
            steps = [("scaler", StandardScaler())]
            if task == "classification":
                counts = np.bincount(y[tr], minlength=2)
                minority = int(counts.min())
                assert minority > 1, (subset, seed, rec["fold"], counts)
                steps.append(("smote", SMOTE(random_state=seed, k_neighbors=min(5, minority - 1))))
                estimator = base.clf_estimator(model, seed)
            else:
                minority = None
                estimator = base.reg_estimator(model, seed)
            estimator.set_params(n_jobs=2)
            steps.append(("estimator", estimator))
            pipe = Pipeline(steps)
            pipe.fit(values[tr], y[tr])
            pred[te] = pipe.predict(values[te])
            if task == "classification":
                prob[te] = pipe.predict_proba(values[te])[:, 1]
                assert pipe.named_steps["smote"].nn_k_._fit_X.shape[0] == minority
            fold_ids[te] = rec["fold"]
            scaler = pipe.named_steps["scaler"]
            assert scaler.n_samples_seen_ == len(tr)
            mean_error = float(np.max(np.abs(scaler.mean_ - values[tr].mean(axis=0))))
            assert mean_error < 1e-10
            saved_folds.append({"fold": rec["fold"], "train_ids": ids[tr].tolist(),
                                "test_ids": ids[te].tolist(), "original_train_n": len(original_train),
                                "restricted_train_n": len(tr), "scaler_n": int(scaler.n_samples_seen_),
                                "scaler_train_mean_max_error": mean_error,
                                "smote_original_minority_n": minority})
            frozen.log(f"{subset} {task} {model} seed={seed} fold={rec['fold']}/5 train={len(tr)} test={len(te)} seconds={time.monotonic()-fold_start:.2f}")
    assert (fold_ids > 0).all() and np.isfinite(pred).all()
    frame = pd.DataFrame({"participant_id": ids, "task": task, "model": model, "seed": seed,
                          "train_subset": subset, "fold": fold_ids, "y_aq": aq.to_numpy(),
                          "y_true": y, "prediction": pred, "probability_severe": prob})
    if smoke:
        error = float(np.max(np.abs(pred - old.prediction.to_numpy())))
        assert error < 1e-6, (task, model, error)
        if task == "classification":
            assert np.max(np.abs(prob - old.probability_severe.to_numpy())) < 1e-6
        directory = OUT / "anchors"
    else:
        error = None
        directory = OUT / "predictions"
    stem = f"{subset}__{task}__{model}__{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / (stem + ".csv")
    temp = directory / (stem + ".tmp")
    frame.to_csv(temp, index=False); temp.replace(dest)
    frozen.write_json(OUT / "folds" / (stem + ".json"), saved_folds)
    return {"subset": subset, "task": task, "model": model, "seed": seed,
            "prediction_sha256": frozen.sha(dest), "anchor_max_abs_error": error,
            "seconds": time.monotonic() - begin}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "folds").mkdir(exist_ok=True)
    current = provenance()
    path = OUT / "provenance.json"
    if path.exists():
        assert json.loads(path.read_text()) == current, "Provenance changed; do not mix runs."
    else:
        frozen.write_json(path, current)
    return current


def smoke():
    prepare()
    checks = [one_cell(("full", task, model, 1000, True)) for task in frozen.TASKS for model in frozen.MODELS]
    frozen.write_json(OUT / "ANCHORS_PASS.json", {"completed_utc": frozen.stamp(), "checks": checks})
    frozen.log("All four full-training anchor cells reproduced.")


def train(workers):
    p = prepare()
    assert (OUT / "ANCHORS_PASS.json").exists()
    alljobs = [(subset, task, model, seed, False) for seed in frozen.SEEDS
               for subset in p["restricted_sets"] for task in frozen.TASKS for model in frozen.MODELS]
    jobs = []
    completed = []
    for j in alljobs:
        subset, task, model, seed, _ = j
        stem = f"{subset}__{task}__{model}__{seed}"
        path = OUT / "predictions" / (stem + ".csv")
        folds = OUT / "folds" / (stem + ".json")
        if path.exists() and folds.exists():
            f = pd.read_csv(path)
            assert len(f) == 226 and f.participant_id.nunique() == 226 and np.isfinite(f.prediction).all()
            completed.append({"subset": subset, "task": task, "model": model, "seed": seed,
                              "prediction_sha256": frozen.sha(path), "resumed": True})
        else:
            jobs.append(j)
    frozen.log(f"Training {len(jobs)} cells; {len(completed)} already complete; {workers} workers x 2 threads.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one_cell, j): j for j in jobs}
        for future in concurrent.futures.as_completed(pending):
            result = future.result()
            completed.append(result)
            frozen.write_json(OUT / "progress.json", {"updated_utc": frozen.stamp(), "cells_complete": len(completed),
                              "cells_total": len(alljobs), "latest": result})
            frozen.log(f"CELL_COMPLETE {len(completed)}/{len(alljobs)} {result['subset']} {result['task']} {result['model']} {result['seed']}")
    assert len(completed) == 720
    frozen.write_json(OUT / "TRAINING_COMPLETE.json", {"completed_utc": frozen.stamp(), "cells": completed,
                      "new_fits": len(completed)*5, "prediction_rows": len(completed)*226})


def analyze():
    prepare()
    assert (OUT / "TRAINING_COMPLETE.json").exists()
    X, aq, _, manifest, masks = frozen.load_data()
    boot = np.load(frozen.OUT / "resampling_patient_weights.npz")["patient_bootstrap"]
    summary, repeats, cells = [], [], []
    for task in frozen.TASKS:
        for model in frozen.MODELS:
            old = [pd.read_csv(frozen.OUT / "predictions" / f"{task}_{model}_{seed}.csv") for seed in frozen.SEEDS]
            y = old[0].y_true.to_numpy()
            p0 = np.stack([f.prediction.to_numpy() for f in old])
            for sub in manifest.itertuples():
                if sub.subset == "full": continue
                new = [pd.read_csv(OUT / "predictions" / f"{sub.subset}__{task}__{model}__{seed}.csv") for seed in frozen.SEEDS]
                for i, frame in enumerate(new):
                    assert frame.participant_id.tolist() == X.index.tolist()
                    np.testing.assert_array_equal(frame.fold, old[i].fold)
                    np.testing.assert_allclose(frame.y_true, y)
                p1 = np.stack([f.prediction.to_numpy() for f in new])
                scope = "secondary" if sub.category in ["target_defined", "exact_persons"] else "primary"
                point, resamples = {}, {}
                for name, pred, mask in [("S00", p0, masks['full']), ("S01", p0, masks[sub.subset]),
                                         ("S10", p1, masks['full']), ("S11", p1, masks[sub.subset])]:
                    point[name] = frozen.weighted_metrics(mask, y, pred, task)
                    resamples[name] = frozen.weighted_metrics(boot*mask, y, pred, task)
                    for metric in point[name]:
                        bs = resamples[name][metric].mean(axis=1)
                        lo, hi = np.quantile(bs, [.025, .975])
                        cells.append({"task": task, "model": model, "subset": sub.subset, "scope": scope,
                                      "cell": name, "metric": metric, "mean": float(point[name][metric].mean()),
                                      "ci_low": lo, "ci_high": hi})
                        for i, seed in enumerate(frozen.SEEDS):
                            repeats.append({"task": task, "model": model, "subset": sub.subset,
                                            "cell": name, "metric": metric, "seed": seed,
                                            "score": float(point[name][metric][0,i])})
                contrasts = {"training_at_full_scoring": {"S10":1,"S00":-1},
                             "training_at_selected_scoring": {"S11":1,"S01":-1},
                             "scoring_at_full_training": {"S01":1,"S00":-1},
                             "scoring_at_selected_training": {"S11":1,"S10":-1},
                             "joint_change": {"S11":1,"S00":-1},
                             "interaction": {"S11":1,"S10":-1,"S01":-1,"S00":1}}
                for metric in point['S00']:
                    for contrast, coef in contrasts.items():
                        value = sum(w*point[c][metric].mean() for c,w in coef.items())
                        draws = sum(w*resamples[c][metric].mean(axis=1) for c,w in coef.items())
                        lo, hi = np.quantile(draws, [.025,.975])
                        summary.append({"task":task,"model":model,"subset":sub.subset,"scope":scope,
                                        "metric":metric,"contrast":contrast,"delta":float(value),
                                        "ci_low":float(lo),"ci_high":float(hi)})
                frozen.log(f"Analysis complete: {task} {model} {sub.subset}")
    pd.DataFrame(cells).to_csv(OUT / "factorial_cells.csv",index=False)
    pd.DataFrame(repeats).to_csv(OUT / "scores_by_seed.csv",index=False)
    pd.DataFrame(summary).to_csv(OUT / "factorial_contrasts.csv",index=False)
    frozen.write_json(OUT / "ANALYSIS_COMPLETE.json", {"completed_utc":frozen.stamp(),"cells":len(cells),
                      "contrasts":len(summary),"bootstrap_draws":len(boot),"conditional_pointwise_intervals":True})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",required=True,choices=["smoke","train","analyze"])
    ap.add_argument("--workers",type=int,default=2)
    args=ap.parse_args()
    assert 1 <= args.workers <= 4
    if args.stage=='smoke': smoke()
    elif args.stage=='train': train(args.workers)
    else: analyze()


if __name__ == '__main__':
    main()
