#!/usr/bin/env python
"""Editable figures for the prospective secondary frozen-prediction analysis."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nature_style import apply_publication_style, finalize, PALETTE, COL_DOUBLE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/frozen_prediction_control_2026-09-22"
FIG = OUT / "figures"
LABELS = {"has_dwi": "DWI available", "has_rsfmri": "Resting fMRI available",
          "has_taskfmri": "Task fMRI available", "has_flair": "FLAIR available",
          "multimodal_complete": "All three modalities", "chronic_365": "At least 365 days",
          "chronic_180": "At least 180 days", "aq_le90": "AQ at most 90",
          "teghipco_idlist": "Published patient list", "full": "All patients"}
apply_publication_style(font_size=8., axes_linewidth=.8)
plt.rcParams["axes.labelsize"] = 8.5
plt.rcParams["axes.titlesize"] = 9.
summary = pd.read_csv(OUT / "frozen_prediction_summary.csv")
comparison = pd.read_csv(OUT / "original_vs_frozen.csv")
manifest = pd.read_csv(OUT / "subset_manifest.csv")
order = manifest.loc[manifest.category.isin(["availability", "clinical_threshold"]), "subset"].tolist()
models = [("RandomForest", "Random Forest", PALETTE["focal"], "o", -.13),
          ("XGBoost", "XGBoost", PALETTE["flag"], "s", .13)]


def layout_axis(ax, task, include_labels=True):
    ax.axvline(0, color=PALETTE["grey_text"], lw=.8, zorder=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LABELS[s] for s in order] if include_labels else [])
    ax.set_ylim(len(order)-.45, -.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Change in Pearson r" if task == "regression" else
                  "Change in balanced accuracy (percentage points)")


# Main new control: all seven primary rules and both models, no extrema selection.
fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 3.5), layout="constrained")
for ax, task, panel in zip(axes, ["regression", "classification"], ["A", "B"]):
    scale = 1 if task == "regression" else 100
    layout_axis(ax, task, include_labels=(task == "regression"))
    for model, label, color, marker, offset in models:
        rows = summary.query("task == @task and model == @model and metric == 'score'").set_index("subset").loc[order]
        ypos = np.arange(len(order)) + offset
        ax.hlines(ypos, rows.delta_ci_low*scale, rows.delta_ci_high*scale, color=color, lw=1.1)
        ax.scatter(rows.delta_from_full*scale, ypos, s=21, color=color, marker=marker, label=label, zorder=3)
    ax.set_title(f"{panel}  {'Severity regression' if task == 'regression' else 'Severe / non-severe'}", loc="left", fontweight="bold")
axes[0].legend(loc="lower left", fontsize=7.3, handlelength=1., borderpad=0)
fig.get_layout_engine().set(rect=(0, .13, 1, .87))
fig.text(.01, .055, "Fixed held-out predictions; selected patients minus all 226 patients.", fontsize=7.3)
fig.text(.01, .012, "Lines: 95% patient-bootstrap intervals conditional on the saved predictions.", fontsize=7.3)
finalize(fig, str(FIG / "frozen_prediction_primary"), tight=False, dpi=220)

# Existing and new designs next to each other. This is not a causal decomposition.
fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 6.0), layout="constrained")
for row_i, task in enumerate(["regression", "classification"]):
    scale = 1 if task == "regression" else 100
    for col_i, (model, label, _, _, _) in enumerate(models):
        ax = axes[row_i, col_i]
        layout_axis(ax, task, include_labels=(col_i == 0))
        rows = comparison.query("task == @task and model == @model").set_index("subset").loc[order]
        ypos = np.arange(len(order))
        for j, (_, r) in enumerate(rows.iterrows()):
            ax.plot([r.original_retrained_delta*scale, r.delta_from_full*scale], [j-.12, j+.12],
                    color=PALETTE["mute"], lw=.8, zorder=0)
        ax.scatter(rows.original_retrained_delta*scale, ypos-.12, color=PALETTE["grey_paper"],
                   marker="s", s=21, label="Train and score within each cohort")
        ax.scatter(rows.delta_from_full*scale, ypos+.12, color=PALETTE["focal"], s=21,
                   label="Freeze predictions; select scored patients")
        ax.set_title(f"{'ABCD'[row_i*2+col_i]}  {label} | {'regression' if task == 'regression' else 'classification'}",
                     loc="left", fontweight="bold")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(.5, .005), fontsize=7.5)
fig.get_layout_engine().set(rect=(0, .105, 1, .895))
finalize(fig, str(FIG / "original_and_frozen_primary"), tight=False, dpi=220)

# Secondary examples remain visibly separate from the primary criteria.
fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.8), layout="constrained")
secondary = ["aq_le90", "teghipco_idlist"]
for ax, task, panel in zip(axes, ["regression", "classification"], ["A", "B"]):
    scale = 1 if task == "regression" else 100
    ax.axvline(0, color=PALETTE["grey_text"], lw=.8)
    for model, label, color, marker, offset in models:
        rows = summary.query("task == @task and model == @model and metric == 'score'").set_index("subset").loc[secondary]
        ypos = np.arange(2)+offset
        ax.hlines(ypos, rows.delta_ci_low*scale, rows.delta_ci_high*scale, color=color, lw=1.1)
        ax.scatter(rows.delta_from_full*scale, ypos, color=color, s=22, marker=marker, label=label)
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels([LABELS[s] for s in secondary] if task == "regression" else [])
    ax.set_ylim(1.55, -.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{panel}  {'Regression' if task == 'regression' else 'Classification'}", loc="left", fontweight="bold")
    ax.set_xlabel("Change in Pearson r" if task == "regression" else "Change in balanced accuracy (pp)")
axes[0].legend(loc="lower left", fontsize=7.3)
fig.get_layout_engine().set(rect=(0, .2, 1, .8))
fig.text(.01, .09, "Secondary analyses: outcome restriction and a published patient list.", fontsize=7.3)
fig.text(.01, .025, "They are excluded from the primary cohort range; the published model is not reproduced.", fontsize=7.3)
finalize(fig, str(FIG / "frozen_prediction_secondary"), tight=False, dpi=220)

files = [Path(__file__), OUT / "frozen_prediction_summary.csv", OUT / "original_vs_frozen.csv"]
files += sorted(FIG.glob("*"))
(OUT / "figure_provenance.json").write_text(json.dumps(
    {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2)+"\n")
print(f"Saved 3 editable figures (SVG, PDF, PNG) to {FIG}")
