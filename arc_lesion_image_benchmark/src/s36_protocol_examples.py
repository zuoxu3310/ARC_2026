#!/usr/bin/env python3
"""Run the reusable comparison protocol on existing ARC and MRRI predictions.

Creates local input contracts from saved prediction/fold banks; never fits a model.
Individual-level inputs stay in the local output directory, outside release archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import s35_compare_predictions as protocol

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "arc_lesion_image_benchmark/results"
DEFAULT_OUT = RESULTS / "comparison_protocol_2026-10-03"
SEEDS = range(1000, 1020)


def export_example(cohort, task, output):
    if cohort == "ARC":
        full_dir = RESULTS / "frozen_prediction_control_2026-09-22"
        restricted_dir = RESULTS / "fixed_test_training_2026-10-02"
        rule, description = "has_flair", "FLAIR available"
        boot_path, boot_key = full_dir / "resampling_patient_weights.npz", "patient_bootstrap"
    else:
        full_dir = restricted_dir = RESULTS / "mrri_replication_2026-10-02"
        rule, description = "has_nwr", "nonword repetition available"
        boot_path, boot_key = full_dir / "patient_bootstrap.npz", "weights"
    membership_path = full_dir / "patient_cohort_membership.csv"
    membership = pd.read_csv(membership_path, dtype={"participant_id": str})
    ids = membership.participant_id.tolist()
    eligible = membership[rule].astype(bool).to_numpy()
    eligible_ids = set(np.asarray(ids)[eligible])
    source_hashes = {str(p.relative_to(ROOT)): protocol.sha(p) for p in [membership_path, boot_path]}
    frames, folds, truth = [], [], None
    for role, model in [("RF_full", "RandomForest"), ("RF_eligible", "RandomForest"), ("XGB_eligible", "XGBoost")]:
        for seed in SEEDS:
            is_full = role == "RF_full"
            directory = full_dir if is_full else restricted_dir
            stem = (f"{task}_{model}_{seed}" if cohort == "ARC" and is_full
                    else f"{'full' if is_full else rule}__{task}__{model}__{seed}")
            csv_path = directory / "predictions" / (stem + ".csv")
            fold_path = directory / "folds" / (stem + ".json")
            frame = pd.read_csv(csv_path, dtype={"participant_id": str})
            assert frame.participant_id.tolist() == ids
            if truth is None:
                truth = frame.y_true.to_numpy()
            np.testing.assert_allclose(truth, frame.y_true, rtol=0, atol=1e-12)
            exported = frame.rename(columns={"participant_id": "patient_id", "seed": "repeat"})
            exported["model"] = role
            if role == "XGB_eligible":
                exported = exported.loc[exported.patient_id.isin(eligible_ids)]
            frames.append(exported[["patient_id", "repeat", "fold", "model", "y_true", "prediction"]])
            for record in json.loads(fold_path.read_text()):
                folds.append({"model": role, "repeat": str(seed), "fold": str(record["fold"]),
                              "train_ids": record["train_ids"],
                              "test_ids": [p for p in record["test_ids"] if role != "XGB_eligible" or p in eligible_ids]})
            source_hashes[str(csv_path.relative_to(ROOT))] = protocol.sha(csv_path)
            source_hashes[str(fold_path.relative_to(ROOT))] = protocol.sha(fold_path)
    directory = output / cohort / task / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": ids, "y_true": truth, "eligible": eligible.astype(int)}).to_csv(directory / "patients.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
    (directory / "folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    with np.load(boot_path) as stored:
        np.savez_compressed(directory / "bootstrap.npz", weights=stored[boot_key], patient_ids=np.asarray(ids))
    spec = {"schema_version": 1, "task": task,
            "outcome": "WAB-AQ" if task == "regression" else "Severe aphasia = WAB-AQ <= 50 (label 1)",
            "protocol": f"{cohort}; {description}; 20 repeated stratified five-fold partitions; shared original test folds; fixed published RF/XGBoost settings; scaler and classification SMOTE fitted inside each eligible training fold.",
            "patients": "patients.csv", "predictions": "predictions.csv", "folds": "folds.json",
            "baseline": "RF_full", "restricted_baseline": "RF_eligible", "candidate": "XGB_eligible",
            "candidate_reference": "restricted_baseline", "bootstrap": {"file": "bootstrap.npz"}}
    path = directory / "comparison.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    (directory.parent / "source-manifest.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    return path, rule


def run_examples(output):
    rows, validations = [], []
    for cohort in ["ARC", "MRRI"]:
        for task in ["regression", "classification"]:
            print(f"Protocol example: {cohort} / {task}", flush=True)
            spec, rule = export_example(cohort, task, output)
            destination = spec.parent.parent / "comparison"
            report = protocol.compare(spec, destination)
            comparisons = pd.read_csv(destination / "contrasts.csv")
            for row in comparisons.to_dict("records"):
                rows.append({"cohort": cohort, "eligibility_rule": rule, "task": task,
                             "full_n": report["full_n"], "eligible_n": report["eligible_n"], **row})
            validations.append({"cohort": cohort, "task": task, "input_validation_passed": True,
                                "fold_records_checked": report["fold_records_checked"],
                                "report_sha256": protocol.sha(destination / "report.json")})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "worked_example_summary.csv", index=False)
    record = {"examples": validations, "models_fitted": 0, "rows": len(summary),
              "roles": "RF is the reference; XGBoost is the comparator; both use the recorded original settings.",
              "individual_inputs_are_local_only": True,
              "summary_sha256": protocol.sha(output / "worked_example_summary.csv")}
    (output / "EXAMPLES_COMPLETE.json").write_text(json.dumps(record, indent=2) + "\n")
    return summary


def write_table(summary, target):
    headers = [r"\begin{table*}[!t]",
               r"\caption{Worked comparison protocol. ARC uses FLAIR availability; MRRI uses nonword-repetition availability. RF is the reference and XGBoost the comparator, retaining their fixed configurations. Scoring changes eligibility with full-training RF predictions fixed. Training compares restricted with full RF training on eligible scoring patients. Method compares XGBoost with RF under shared eligible training and scoring. Values are differences in mean scores over 20 repeats, with paired conditional 95\% intervals. BA denotes balanced accuracy; its differences are percentage points. Coverage is relative to the analyzed pool.}",
               r"\label{tab:protocol-example}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
               r"\begin{tabular}{llccc}", r"\toprule",
               r"Population (coverage) & Metric & Scoring change & Training change & Method difference \\", r"\midrule"]
    names = ["scoring_at_full_training", "training_at_eligible_scoring", "method_difference_on_eligible"]
    for cohort, label in [("ARC", "ARC (135/226)"), ("MRRI", "MRRI (128/183)")]:
        for i, (metric, shown) in enumerate([("pearson_r", "$r$"), ("mae", "MAE"), ("balanced_accuracy", "BA (pp)")]):
            current = summary.query("cohort == @cohort and metric == @metric").set_index("contrast")
            scale, decimals = (100, 2) if metric == "balanced_accuracy" else (1, 3 if metric == "pearson_r" else 2)
            values = []
            for name in names:
                row = current.loc[name]
                values.append(f"${row.estimate*scale:+.{decimals}f}\\;[{row.ci_low*scale:.{decimals}f},{row.ci_high*scale:.{decimals}f}]$")
            headers.append(" & ".join([label if i == 0 else "", shown, *values]) + r" \\")
        if cohort == "ARC":
            headers.append(r"\midrule")
    headers += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(headers) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--install-table", action="store_true")
    args = parser.parse_args()
    summary = run_examples(args.output)
    table = (ROOT / "paper/tables/protocol_worked_example.tex" if args.install_table
             else args.output / "protocol_worked_example.tex")
    write_table(summary, table)
    print(summary.loc[summary.contrast == "method_difference_on_eligible"].to_string(index=False))


if __name__ == "__main__":
    main()
