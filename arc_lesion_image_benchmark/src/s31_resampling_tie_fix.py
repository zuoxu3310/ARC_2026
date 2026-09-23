#!/usr/bin/env python
"""Numerical tie correction for saved s31 reference distributions.

The AQ<=90 bin-matched draws all contain the identical patient set. BLAS roundoff
can nevertheless put a computed Pearson correlation ~1e-16 to one side of its
identical reference. Preserve frozen training code/provenance, and explicitly
apply tolerance-aware inclusive tail comparisons in this downstream step.
Scores, predictions, intervals, and sampling weights do not change.
"""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/frozen_prediction_control_2026-09-22"
TIE_TOL = 1e-12
summary_path = OUT / "frozen_prediction_summary.csv"
compare_path = OUT / "original_vs_frozen.csv"
backup = OUT / "audit_original_tail_ranks"
backup.mkdir(exist_ok=True)
for path in [summary_path, compare_path]:
    if not (backup / path.name).exists():
        shutil.copyfile(path, backup / path.name)
summary = pd.read_csv(summary_path)
draws = np.load(OUT / "resampling_metric_distributions.npz")
weights = np.load(OUT / "resampling_patient_weights.npz")
changes = []
for idx, row in summary.iterrows():
    if row.subset == "full":
        continue
    key = f"{row.task}__{row.model}__{row.subset}__{row.metric}"
    for kind in ["size", "aqbin"]:
        dist = draws[f"{key}__{kind}"]
        W = weights[f"{kind}__{row.subset}"]
        degenerate = bool(np.all(W == W[0]))
        observed = float(row.observed)
        rank = min(1., 2 * min((1 + np.sum(dist <= observed + TIE_TOL)) / (len(dist)+1),
                               (1 + np.sum(dist >= observed - TIE_TOL)) / (len(dist)+1)))
        if degenerate:
            assert np.max(np.abs(dist - observed)) <= TIE_TOL
            rank = 1.
        old = float(row[f"{kind}_tail_rank"])
        if not np.isclose(old, rank, rtol=0, atol=1e-15):
            changes.append({"task": row.task, "model": row.model, "subset": row.subset,
                            "metric": row.metric, "reference": kind,
                            "old_rank": old, "new_rank": rank})
        summary.loc[idx, f"{kind}_tail_rank"] = rank
        summary.loc[idx, f"{kind}_reference_degenerate"] = degenerate
for _, group in summary.query("scope == 'primary' and subset != 'full'").groupby(["task", "model", "metric"]):
    for kind in ["size", "aqbin"]:
        values = group[f"{kind}_tail_rank"].to_numpy()
        order = np.argsort(values)
        adjusted = np.minimum(1., np.maximum.accumulate(values[order]*np.arange(len(values), 0, -1)))
        restored = np.empty_like(adjusted)
        restored[order] = adjusted
        summary.loc[group.index, f"{kind}_holm_tail_rank"] = restored
summary.to_csv(summary_path, index=False)
old = pd.read_csv(OUT / "existing_grid_all_cohort_differences.csv")
comparison = summary.query("metric == 'score'").merge(
    old[["task", "model", "subset", "mean_score", "delta_from_full"]].rename(columns={
        "mean_score": "original_retrained_score", "delta_from_full": "original_retrained_delta"}),
    on=["task", "model", "subset"], validate="one_to_one")
comparison.to_csv(compare_path, index=False)
record = {"at": datetime.now(timezone.utc).isoformat(), "absolute_tie_tolerance": TIE_TOL,
          "reason": "Numerical ties and a structurally degenerate AQ<=90 AQ-bin reference",
          "scores_and_intervals_unchanged": True, "changed_tail_ranks": changes,
          "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), summary_path, compare_path,
                               OUT / "resampling_metric_distributions.npz",
                               OUT / "resampling_patient_weights.npz"]}}
(OUT / "NUMERICAL_TIE_CORRECTION.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({"corrected_tail_ranks": len(changes), "changes": changes}, indent=2))
