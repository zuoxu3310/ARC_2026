#!/usr/bin/env python
"""Exhaustive descriptive RF/XGBoost comparisons under fixed predictions.

The patient bootstrap is paired across models, scoring sets, and repeats.
Intervals condition on the existing prediction bank; there is no retraining.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import s31_frozen_prediction_control as frozen

DEFAULT_OUT = frozen.ROOT / "results/comparison_population_2026-10-02"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    X, aq, _, manifest, masks = frozen.load_data()
    weights_path = frozen.OUT / "resampling_patient_weights.npz"
    boot = np.load(weights_path)["patient_bootstrap"]
    assert boot.shape == (2000, len(X))
    reference = pd.read_csv(frozen.OUT / "frozen_prediction_summary.csv")
    primary = manifest.loc[~manifest.category.isin(["target_defined", "exact_persons"]), "subset"].tolist()
    allsets = manifest.subset.tolist()
    input_hashes = {str(weights_path): frozen.sha(weights_path), str(Path(__file__)): frozen.sha(__file__)}
    values, draws, rows, summaries = {}, {}, [], []
    for task in frozen.TASKS:
        for model in frozen.MODELS:
            frames = []
            for seed in frozen.SEEDS:
                p = frozen.OUT / "predictions" / f"{task}_{model}_{seed}.csv"
                frame = pd.read_csv(p)
                assert frame.participant_id.tolist() == X.index.tolist()
                np.testing.assert_allclose(frame.y_aq, aq)
                frames.append(frame)
                input_hashes[str(p)] = frozen.sha(p)
            pred = np.stack([f.prediction.to_numpy() for f in frames])
            y = frames[0].y_true.to_numpy()
            for subset in allsets:
                observed = frozen.weighted_metrics(masks[subset], y, pred, task)
                b = frozen.weighted_metrics(boot * masks[subset], y, pred, task)
                for metric in observed:
                    key = (task, model, subset, metric)
                    value = float(observed[metric].mean())
                    expected = reference.query("task == @task and model == @model and subset == @subset and metric == @metric").observed.item()
                    np.testing.assert_allclose(value, expected, atol=1e-12)
                    values[key] = value
                    draws[key] = b[metric].mean(axis=1)
        metrics = ["score", "mae"] if task == "regression" else ["score"]
        for metric in metrics:
            utility_sign = -1 if metric == "mae" else 1
            for a, b in itertools.product(allsets, repeat=2):
                differences = {}
                uncertainty = {}
                for name, rfset, xgset in [("cross", a, b), ("same_rf_set", a, a), ("same_xgb_set", b, b)]:
                    kx, kr = (task, "XGBoost", xgset, metric), (task, "RandomForest", rfset, metric)
                    differences[name] = values[kx] - values[kr]
                    uncertainty[name] = draws[kx] - draws[kr]
                signs = {k: np.sign(utility_sign * v) for k, v in differences.items()}
                reversal = signs["cross"] != 0 and signs["same_rf_set"] == signs["same_xgb_set"] == -signs["cross"]
                rec = {"task": task, "metric": metric, "rf_set": a, "xgb_set": b,
                       "scope": "primary" if a in primary and b in primary else "includes_secondary",
                       "different_scoring_sets": a != b,
                       "rf_score": values[(task, "RandomForest", a, metric)],
                       "xgb_score": values[(task, "XGBoost", b, metric)],
                       "strict_point_ordering_reversal": bool(reversal)}
                for name in differences:
                    rec[name + "_xgb_minus_rf"] = differences[name]
                    rec[name + "_ci_low"], rec[name + "_ci_high"] = np.quantile(uncertainty[name], [.025, .975])
                rows.append(rec)
            current = pd.DataFrame(rows).query("task == @task and metric == @metric and scope == 'primary' and different_scoring_sets")
            assert len(current) == len(primary) * (len(primary) - 1) == 56
            summaries.append({"task": task, "metric": metric, "primary_sets": len(primary),
                              "ordered_distinct_set_pairs": len(current),
                              "strict_point_ordering_reversals": int(current.strict_point_ordering_reversal.sum()),
                              "interpretation": "Descriptive design count, not a population probability or independent tests."})
    pd.DataFrame(rows).to_csv(out / "all_model_pair_contrasts.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "primary_ordering_counts.csv", index=False)
    frozen.write_json(out / "verification.json", {"completed_utc": frozen.stamp(), "inputs_sha256": input_hashes,
                      "all_existing_score_anchors_match": True, "contrast_rows": len(rows),
                      "bootstrap_draws": len(boot), "pointwise_conditional_intervals": True,
                      "training_performed": False, "summaries": summaries})
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
