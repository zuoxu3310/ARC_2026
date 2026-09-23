#!/usr/bin/env python
"""Secondary cohort analysis with s18-compatible, frozen OOF predictions.

Train once on the full-pool folds; the analysis stage never fits an estimator.
See the timestamped ANALYSIS_PLAN.md for estimands and inference limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import s18_crossed_grid as base

ROOT = Path(base.PKG)
OUT = ROOT / "results/frozen_prediction_control_2026-09-22"
MODELS = ["RandomForest", "XGBoost"]
TASKS = ["regression", "classification"]
SEEDS = list(range(1000, 1020))
B = 2000
TOL = 1e-6


def stamp():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{stamp()}] {message}", flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_data():
    X, y, features = base.load_full_pool()
    ids = base.load_subset_ids("full")
    assert len(ids) == len(set(ids)) == len(X) == 226
    assert set(ids) == set(X.index)
    X, y = X.loc[ids], y.loc[ids]
    assert X.index.equals(y.index) and np.isfinite(X.to_numpy()).all()
    manifest = pd.read_csv(Path(base.SUBSET_DIR) / "subset_manifest.csv")
    masks = {}
    for row in manifest.itertuples():
        subids = base.load_subset_ids(row.subset)
        assert len(subids) == len(set(subids)) == row.n
        assert set(subids) <= set(ids)
        masks[row.subset] = np.isin(ids, subids)
        assert int((y.to_numpy()[masks[row.subset]] <= 50).sum()) == row.n_severe
    return X, y, features, manifest, masks


def make_provenance(features):
    files = [Path(__file__), Path(base.__file__), Path(base.TABLE),
             OUT / "ANALYSIS_PLAN.md", Path(base.SUBSET_DIR) / "subset_manifest.csv"]
    files += sorted(Path(base.SUBSET_DIR).glob("*_ids.txt"))
    files += [ROOT / f"results/runs/grid_{task}_{kind}.csv"
              for task in TASKS for kind in ["fast", "TabPFN"]]
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                "xgboost", "lightgbm", "matplotlib", "joblib", "threadpoolctl"]
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(),
            "versions": {p: importlib.metadata.version(p) for p in packages},
            "sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT)
                       else str(p): sha(p) for p in files},
            "features": features, "models": MODELS, "seeds": SEEDS,
            "folds": 5, "resamples": B, "anchor_tolerance": TOL}


def verify_provenance():
    _, _, features, _, _ = load_data()
    old = json.loads((OUT / "provenance.json").read_text())
    current = make_provenance(features)
    assert current == old, "Code, data, protocol, or environment differs from frozen provenance"


def one_cell(X, aq, task, model, seed):
    y = aq.to_numpy() if task == "regression" else (aq.to_numpy() <= 50).astype(int)
    strat = base.strat_labels(aq, task)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred, prob = np.empty(len(y)), np.full(len(y), np.nan)
    fold_ids = np.zeros(len(y), dtype=int)
    split_records = []
    values = X.to_numpy()
    for fold, (tr, te) in enumerate(splitter.split(values, strat), 1):
        started = time.monotonic()
        assert not set(tr).intersection(te)
        steps = [("scaler", StandardScaler())]
        if task == "classification":
            minority = int(min(np.bincount(y[tr])))
            assert minority > 1
            steps.append(("smote", SMOTE(random_state=seed,
                                        k_neighbors=min(5, minority - 1))))
            steps.append(("clf", base.clf_estimator(model, seed)))
        else:
            steps.append(("model", base.reg_estimator(model, seed)))
        pipe = Pipeline(steps)
        pipe.fit(values[tr], y[tr])
        pred[te] = pipe.predict(values[te])
        if task == "classification":
            prob[te] = pipe.predict_proba(values[te])[:, 1]
        fold_ids[te] = fold
        scaler = pipe.named_steps["scaler"]
        assert int(scaler.n_samples_seen_) == len(tr)
        mean_error = float(np.max(np.abs(scaler.mean_ - values[tr].mean(axis=0))))
        assert mean_error < 1e-10
        smote_n = (int(pipe.named_steps["smote"].nn_k_._fit_X.shape[0])
                   if task == "classification" else None)
        if task == "classification":
            assert smote_n == minority
        split_records.append({"task": task, "model": model, "seed": seed, "fold": fold,
                              "train_ids": X.index[tr].tolist(),
                              "test_ids": X.index[te].tolist(),
                              "scaler_n": len(tr), "scaler_train_mean_max_error": mean_error,
                              "smote_original_minority_n": smote_n})
        log(f"{task} {model} seed={seed} fold={fold}/5 finished in {time.monotonic()-started:.1f}s")
    assert (fold_ids > 0).all() and np.isfinite(pred).all()
    result = pd.DataFrame({"participant_id": X.index, "task": task, "model": model,
                           "seed": seed, "fold": fold_ids, "y_aq": aq.to_numpy(),
                           "y_true": y, "prediction": pred, "probability_severe": prob})
    score = (float(pearsonr(y, pred).statistic) if task == "regression"
             else float(balanced_accuracy_score(y, pred)))
    return result, split_records, score


def train():
    X, aq, features, manifest, masks = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    pred_dir, fold_dir = OUT / "predictions", OUT / "folds"
    pred_dir.mkdir(exist_ok=True)
    fold_dir.mkdir(exist_ok=True)
    provenance = make_provenance(features)
    if (OUT / "provenance.json").exists():
        verify_provenance()
    else:
        write_json(OUT / "provenance.json", provenance)
        (OUT / "requirements-lock.txt").write_text(
            "# Python " + platform.python_version() + "\n" +
            "\n".join(f"{p}=={v}" for p, v in provenance["versions"].items()) + "\n")
    manifest.to_csv(OUT / "subset_manifest.csv", index=False)
    pd.DataFrame({"participant_id": X.index, "y_aq": aq.to_numpy(),
                  **masks}).to_csv(OUT / "patient_cohort_membership.csv", index=False)
    anchors = {t: pd.read_csv(ROOT / f"results/runs/grid_{t}_fast.csv") for t in TASKS}
    checks = []
    # Seed first: all four M0 cells finish before any later seed is trained.
    for seed in SEEDS:
        for task in TASKS:
            for model in MODELS:
                stem = f"{task}_{model}_{seed}"
                path, fold_path = pred_dir / f"{stem}.csv", fold_dir / f"{stem}.json"
                if path.exists() and fold_path.exists():
                    frame = pd.read_csv(path)
                    assert frame.participant_id.tolist() == X.index.tolist()
                    assert len(frame) == 226 and frame.participant_id.nunique() == 226
                    score = (pearsonr(frame.y_true, frame.prediction).statistic
                             if task == "regression" else
                             balanced_accuracy_score(frame.y_true, frame.prediction))
                    log(f"resume saved cell {stem}")
                else:
                    frame, fold_records, score = one_cell(X, aq, task, model, seed)
                    write_json(fold_path, fold_records)
                    frame.to_csv(path, index=False)
                    check_saved = pd.read_csv(path)
                    assert check_saved.participant_id.nunique() == len(X)
                    assert check_saved.fold.nunique() == 5
                    assert np.array_equal(check_saved.y_true.to_numpy(), frame.y_true.to_numpy())
                ref = anchors[task].query("subset == 'full' and model == @model and seed == @seed")
                assert len(ref) == 1
                reference = float(ref.score.iloc[0])
                delta = float(score - reference)
                checks.append({"task": task, "model": model, "seed": seed,
                               "stored_score": reference, "reproduced_score": float(score),
                               "difference": delta, "passed": abs(delta) <= TOL})
                pd.DataFrame(checks).to_csv(OUT / "anchor_reconciliation.csv", index=False)
                assert abs(delta) <= TOL, f"M0 anchor mismatch {stem}: {score} != {reference}"
                log(f"anchor PASS {stem}: {score:.10f}; difference={delta:.3g}")
        if seed == SEEDS[0]:
            write_json(OUT / "M0_PASS.json", {"at": stamp(), "cells": checks})
            log("M0 PASS: all four first-seed cells reproduced; continuing remaining 19 seeds")
    write_json(OUT / "TRAINING_COMPLETE.json", {"at": stamp(), "cells": len(checks),
                "prediction_rows": len(checks) * len(X), "folds": len(checks) * 5,
                "max_anchor_error": max(abs(c["difference"]) for c in checks)})
    log("TRAINING_COMPLETE")


def weighted_metrics(weights, y, predictions, task):
    """Return B x repeat matrices; shared patient weights preserve repeat dependence."""
    W = np.atleast_2d(weights).astype(float)
    P = np.asarray(predictions, dtype=float)  # repeat x patient
    mass = W.sum(axis=1)[:, None]
    assert (mass > 0).all()
    if task == "regression":
        sy = (W @ y)[:, None]
        sp = W @ P.T
        cross = W @ (P * y).T - sy * sp / mass
        vy = (W @ (y * y))[:, None] - sy * sy / mass
        vp = W @ (P * P).T - sp * sp / mass
        denominator = np.sqrt(np.maximum(vy * vp, 0))
        assert (denominator > 1e-10).all(), "Degenerate weighted Pearson sample"
        return {"score": cross / denominator, "mae": (W @ np.abs(P - y).T) / mass}
    pos, neg = (y == 1).astype(float), (y == 0).astype(float)
    npos, nneg = (W @ pos)[:, None], (W @ neg)[:, None]
    assert (npos > 0).all() and (nneg > 0).all()
    return {"score": .5 * ((W @ ((P == 1) * pos).T) / npos +
                           (W @ ((P == 0) * neg).T) / nneg)}


def reference_weights(mask, aq, seed, matched):
    rng = np.random.default_rng(seed)
    W = np.zeros((B, len(mask)), dtype=np.int16)
    if not matched:
        for b in range(B):
            W[b, rng.choice(len(mask), int(mask.sum()), replace=False)] = 1
    else:
        bins = pd.cut(aq, [-.1, 25, 50, 70, 90, 100.1], labels=False).to_numpy()
        assert np.isfinite(bins).all()
        for value in np.unique(bins):
            pool = np.flatnonzero(bins == value)
            n = int(np.sum(mask & (bins == value)))
            for b in range(B):
                W[b, rng.choice(pool, n, replace=False)] = 1
    assert (W.sum(axis=1) == mask.sum()).all()
    return W


def tail_rank(observed, draws):
    return min(1., 2 * min((1 + np.sum(draws <= observed)) / (len(draws) + 1),
                           (1 + np.sum(draws >= observed)) / (len(draws) + 1)))


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    out = np.empty_like(values)
    out[order] = adjusted
    return out


def reanalyze_grid(manifest):
    tables = []
    for task in TASKS:
        frame = pd.concat([pd.read_csv(ROOT / f"results/runs/grid_{task}_{tag}.csv")
                           for tag in ["fast", "TabPFN"]], ignore_index=True)
        assert not frame.duplicated(["task", "subset", "model", "seed"]).any()
        assert (frame.groupby(["subset", "model"]).size() == 20).all()
        tab = frame.groupby(["task", "subset", "category", "model"], as_index=False).agg(
            n=("n", "first"), mean_score=("score", "mean"), seed_sd=("score", "std"))
        ref = tab.query("subset == 'full'")[["task", "model", "mean_score"]].rename(
            columns={"mean_score": "full_score"})
        tab = tab.merge(ref, on=["task", "model"], validate="many_to_one")
        tab["delta_from_full"] = tab.mean_score - tab.full_score
        tab["scope"] = np.where(tab.category.isin(["target_defined", "exact_persons"]),
                                 "secondary", "primary")
        tables.append(tab)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(OUT / "existing_grid_all_cohort_differences.csv", index=False)
    ranges = []
    for (task, model), frame in result.groupby(["task", "model"]):
        for scope in ["primary", "all_including_secondary"]:
            use = frame.query("scope == 'primary'") if scope == "primary" else frame
            lo, hi = use.loc[use.mean_score.idxmin()], use.loc[use.mean_score.idxmax()]
            ranges.append({"task": task, "model": model, "scope": scope,
                           "low_subset": lo.subset, "high_subset": hi.subset,
                           "low_score": lo.mean_score, "high_score": hi.mean_score,
                           "range": hi.mean_score - lo.mean_score})
    pd.DataFrame(ranges).to_csv(OUT / "existing_grid_ranges.csv", index=False)
    return result


def analyze():
    verify_provenance()
    assert (OUT / "TRAINING_COMPLETE.json").exists()
    X, aq, features, manifest, masks = load_data()
    existing = reanalyze_grid(manifest)
    rng = np.random.default_rng(31092026)
    boot = rng.multinomial(len(X), np.full(len(X), 1 / len(X)), size=B).astype(np.int16)
    weights = {"patient_bootstrap": boot}
    for i, row in enumerate(manifest.itertuples()):
        if row.subset == "full":
            continue
        weights[f"size__{row.subset}"] = reference_weights(masks[row.subset], aq, 31093026+i, False)
        weights[f"aqbin__{row.subset}"] = reference_weights(masks[row.subset], aq, 31094026+i, True)
    np.savez_compressed(OUT / "resampling_patient_weights.npz", **weights)
    summaries, seed_rows, distributions = [], [], {}
    for task in TASKS:
        for model in MODELS:
            frames = [pd.read_csv(OUT / "predictions" / f"{task}_{model}_{seed}.csv") for seed in SEEDS]
            for frame in frames:
                assert frame.participant_id.tolist() == X.index.tolist()
                assert np.array_equal(frame.y_aq.to_numpy(), aq.to_numpy())
            y = frames[0].y_true.to_numpy()
            pred = np.stack([f.prediction.to_numpy() for f in frames])
            full = weighted_metrics(np.ones(len(X)), y, pred, task)
            full_boot = weighted_metrics(boot, y, pred, task)
            for row in manifest.itertuples():
                mask = masks[row.subset]
                scores = weighted_metrics(mask, y, pred, task)
                boots = weighted_metrics(boot * mask, y, pred, task)
                for j, seed in enumerate(SEEDS):
                    seed_rows.append({"task": task, "model": model, "subset": row.subset,
                                      "seed": seed, "n": row.n,
                                      **{key: float(value[0, j]) for key, value in scores.items()}})
                refs = {kind: weighted_metrics(weights[f"{kind}__{row.subset}"], y, pred, task)
                        for kind in ["size", "aqbin"]} if row.subset != "full" else {}
                for metric in scores:
                    observed = float(scores[metric].mean())
                    reference = float(full[metric].mean())
                    bscore = boots[metric].mean(axis=1)
                    bdelta = (boots[metric] - full_boot[metric]).mean(axis=1)
                    rec = {"task": task, "model": model, "subset": row.subset,
                           "category": row.category, "n": row.n, "metric": metric,
                           "scope": "secondary" if row.category in ["target_defined", "exact_persons"] else "primary",
                           "observed": observed, "full_reference": reference,
                           "delta_from_full": observed-reference,
                           "conditional_ci_low": float(np.quantile(bscore, .025)),
                           "conditional_ci_high": float(np.quantile(bscore, .975)),
                           "delta_ci_low": float(np.quantile(bdelta, .025)),
                           "delta_ci_high": float(np.quantile(bdelta, .975))}
                    prefix = f"{task}__{model}__{row.subset}__{metric}"
                    distributions[prefix+"__bootstrap_score"] = bscore
                    distributions[prefix+"__bootstrap_delta"] = bdelta
                    for kind, values in refs.items():
                        draws = values[metric].mean(axis=1)
                        rec.update({f"{kind}_mean": float(draws.mean()),
                                    f"{kind}_low": float(np.quantile(draws, .025)),
                                    f"{kind}_high": float(np.quantile(draws, .975)),
                                    f"{kind}_observed_minus_mean": observed-float(draws.mean()),
                                    f"{kind}_tail_rank": tail_rank(observed, draws)})
                        distributions[prefix+f"__{kind}"] = draws
                    summaries.append(rec)
                log(f"analysis {task} {model} {row.subset}: score={scores['score'].mean():.6f}, n={row.n}")
    summary = pd.DataFrame(summaries)
    for _, group in summary.query("scope == 'primary' and subset != 'full'").groupby(["task", "model", "metric"]):
        assert len(group) == 7
        for kind in ["size", "aqbin"]:
            summary.loc[group.index, f"{kind}_holm_tail_rank"] = holm(group[f"{kind}_tail_rank"])
    summary.to_csv(OUT / "frozen_prediction_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "frozen_scores_by_seed.csv", index=False)
    np.savez_compressed(OUT / "resampling_metric_distributions.npz", **distributions)
    comparison = summary.query("metric == 'score'").merge(
        existing[["task", "model", "subset", "mean_score", "delta_from_full"]].rename(
            columns={"mean_score": "original_retrained_score", "delta_from_full": "original_retrained_delta"}),
        on=["task", "model", "subset"], validate="one_to_one")
    comparison.to_csv(OUT / "original_vs_frozen.csv", index=False)
    write_json(OUT / "ANALYSIS_COMPLETE.json", {"at": stamp(), "summary_rows": len(summary),
                                               "seed_score_rows": len(seed_rows), "resamples": B})
    log("ANALYSIS_COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "analyze"])
    args = parser.parse_args()
    train() if args.stage == "train" else analyze()
