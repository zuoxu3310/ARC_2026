#!/usr/bin/env python3
"""Three manuscript data figures from the audited frozen-prediction results.

This is a plotting-only script: it does not fit models or change analysis outputs.
Running it writes frozen_scores, composition_controls and refit_vs_frozen to
paper/images. All numerical panels read the audited CSV/NPZ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "arc_lesion_image_benchmark/results/frozen_prediction_control_2026-09-22"
DEST = ROOT / "paper/images"
CONTRACT = ROOT / "outputs/manuscript-rewrite-2026-09-22/FIGURE_CONTRACTS.md"
WIDTH = 7.16
INK = "#20262C"
MUTED = "#59626A"
LIGHT = "#DCE0E3"
RF = "#246B8E"
XGB = "#AA5B2A"
MODEL_CONFIG = [("RandomForest", "Random forest", RF, "o", -0.14),
                ("XGBoost", "XGBoost", XGB, "s", 0.14)]
PRIMARY = ["has_dwi", "has_rsfmri", "has_taskfmri", "has_flair",
           "multimodal_complete", "chronic_365", "chronic_180"]
SECONDARY = ["aq_le90", "teghipco_idlist"]
# Row names follow the manuscript result tables.
LABELS = {"has_dwi": "Diffusion MRI", "has_rsfmri": "Resting-state fMRI",
          "has_taskfmri": "Task fMRI", "has_flair": "FLAIR",
          "multimodal_complete": "Multimodal complete",
          "chronic_365": "≥365 days", "chronic_180": "≥180 days",
          "aq_le90": "WAB-AQ ≤90 (secondary)",
          "teghipco_idlist": "Published-list intersection\n(secondary)"}

# Figures are drawn at 7.16 in and printed at the 6.99 in text width (97.6%):
# text sizes below print at >= 9 pt and the 8.2 pt panel letters at 8 pt.
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5, "axes.labelsize": 9.5,
    "axes.titlesize": 10.0, "xtick.labelsize": 9.4, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "axes.linewidth": 0.65,
    "xtick.major.width": 0.65, "ytick.major.width": 0.65,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "mathtext.fontset": "dejavusans", "legend.frameon": False,
})

SUMMARY = pd.read_csv(DATA / "frozen_prediction_summary.csv")
COMPARE = pd.read_csv(DATA / "original_vs_frozen.csv")
MANIFEST = pd.read_csv(DATA / "subset_manifest.csv").set_index("subset")
DRAWS = np.load(DATA / "resampling_metric_distributions.npz")
RECORDS: dict[str, dict] = {}


def signed(x: float, _pos=None) -> str:
    if abs(x) < 1e-10:
        return "0"
    s = f"{x:+.2f}".rstrip("0").rstrip(".")
    return s.replace("-", "−")


def signed_int(x: float, _pos=None) -> str:
    return "0" if abs(x) < 1e-10 else f"{x:+.0f}".replace("-", "−")


def rows(task: str, model: str, order: list[str], source=SUMMARY) -> pd.DataFrame:
    result = source.query("task == @task and model == @model and metric == 'score'")
    return result.set_index("subset").loc[order]


def export(fig, stem: str, sources: list[str], data_description: str):
    fig.canvas.draw()
    sizes = sorted({float(t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text)
                    if t.get_visible() and t.get_text()})
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(DEST / f"{stem}.{extension}", dpi=300)
    media = PdfReader(DEST / f"{stem}.pdf").pages[0].mediabox
    RECORDS[stem] = {
        "width_in": float(media.width) / 72,
        "height_in": float(media.height) / 72,
        "font_sizes_pt": sizes,
        "sources": sources,
        "description": data_description,
        "files_sha256": {f"{stem}.{ext}": hashlib.sha256((DEST / f"{stem}.{ext}").read_bytes()).hexdigest()
                         for ext in ["pdf", "svg", "png"]},
    }
    assert abs(float(media.width) / 72 - WIDTH) < 1e-6
    assert min(sizes) >= 8.0
    plt.close(fig)


def panel_labels(fig, axes, letters, gap_pt=5.0):
    """Centre (a), (b), ... under each axes, a fixed gap below the lowest text of the row."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes)
    y = (bottom - gap_pt * fig.dpi / 72) / fig.bbox.height
    for ax, letter in zip(axes, letters):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, y, f"({letter})", ha="center", va="top",
                 fontsize=8.2, fontfamily="Times New Roman")


def plot_axis(ax, task: str, order: list[str], ypos, labels=False, include_n=True):
    ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABELS[s] for s in order] if labels else [])
    if labels and include_n:
        # Patient counts form their own right-aligned column between names and axis.
        ax.tick_params(axis="y", pad=34)
        counts = offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=-8, units="points")
        for s, y in zip(order, ypos):
            ax.text(0, y, f"{int(MANIFEST.loc[s, 'n'])}", transform=counts, ha="right",
                    va="center_baseline", fontsize=plt.rcParams["ytick.labelsize"], clip_on=False)
        ax.text(0, -0.85, "n", transform=counts, ha="right", va="center_baseline",
                fontsize=9.5, fontweight="bold", clip_on=False)
    ax.xaxis.set_major_formatter(FuncFormatter(signed if task == "regression" else signed_int))
    ax.grid(axis="x", color="#ECEEEF", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def make_frozen_scores():
    fig = plt.figure(figsize=(WIDTH, 4.6))
    positions = [[0.345, 0.17, 0.29, 0.775], [0.685, 0.17, 0.29, 0.775]]
    axes = [fig.add_axes(p) for p in positions]
    order = PRIMARY + SECONDARY
    ys = np.r_[np.arange(7), [7.85, 8.85]]
    for ax, task in zip(axes, ["regression", "classification"]):
        scale = 1 if task == "regression" else 100
        plot_axis(ax, task, order, ys, labels=task == "regression")
        ax.set_ylim(9.4, -1.05)
        # A visible gap and divider keep the two secondary rules distinct.
        ax.axhline(6.925, color="#BBC2C7", linewidth=0.65)
        for model, name, color, marker, offset in MODEL_CONFIG:
            tab = rows(task, model, order)
            y = ys + offset
            ax.hlines(y, tab.delta_ci_low * scale, tab.delta_ci_high * scale,
                      color=color, linewidth=1.35, zorder=2)
            ax.scatter(tab.delta_from_full * scale, y, color=color, marker=marker,
                       s=21, linewidths=0.55, edgecolors="white", label=name, zorder=3)
        if task == "regression":
            ax.set_xlim(-0.25, 0.105)
            ax.set_xticks([-0.2, -0.1, 0, 0.1])
            ax.set_xlabel("Change in Pearson r", labelpad=6)
            ax.set_title("Severity regression", loc="left", fontweight="bold", pad=6)
        else:
            ax.set_xlim(-12.5, 6.5)
            ax.set_xticks([-10, -5, 0, 5])
            ax.set_xlabel("Change in balanced accuracy\n(percentage points)", labelpad=6)
            ax.set_title("Severe / non-severe", loc="left", fontweight="bold", pad=6)
    ytrans = axes[0].get_yaxis_transform()
    axes[0].text(-1.12, -0.85, "Primary eligibility rules", transform=ytrans,
                 fontsize=9.5, fontweight="bold", va="center_baseline", clip_on=False)
    # The legend sits in the free space under the row names.
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=1, loc="lower left",
               bbox_to_anchor=(0.02, 0.02), handletextpad=0.3, labelspacing=0.45,
               borderaxespad=0.2)
    panel_labels(fig, axes, "ab")
    export(fig, "frozen_scores", ["frozen_prediction_summary.csv", "subset_manifest.csv"],
           "All seven primary rules and two separate secondary rules; paired patient-bootstrap "
           "95% intervals for change from the full 226-person pool, conditional on saved predictions.")


def make_composition_controls():
    fig = plt.figure(figsize=(WIDTH, 4.75))
    grid = fig.add_gridspec(2, 2, left=0.255, right=0.975, top=0.835,
                           bottom=0.145, hspace=1.02, wspace=0.26)
    x = np.linspace(0.54, 0.84, 500)
    for row_i, subset in enumerate(["has_flair", "has_taskfmri"]):
        row_axes = []
        for col_i, (model, name, color, _, _) in enumerate(MODEL_CONFIG):
            ax = fig.add_subplot(grid[row_i, col_i])
            row_axes.append(ax)
            rec = rows("regression", model, [subset]).iloc[0]
            for kind, y, fill, outline, linestyle in [
                ("size", 1.0, "#D8DDE0", "#7B858C", "-"),
                ("aqbin", 0.0, "#AFBAC2", "#4C5963", "--"),
            ]:
                arr = DRAWS[f"regression__{model}__{subset}__score__{kind}"]
                assert len(arr) == 2000 and np.all(np.isfinite(arr))
                assert np.isclose(arr.mean(), rec[f"{kind}_mean"], atol=1e-12)
                lo, hi = np.quantile(arr, [0.025, 0.975])
                assert np.isclose(lo, rec[f"{kind}_low"], atol=1e-12)
                assert np.isclose(hi, rec[f"{kind}_high"], atol=1e-12)
                density = gaussian_kde(arr)(x)
                density = density / density.max() * 0.31
                ax.fill_between(x, y-density, y+density, color=fill,
                                edgecolor=outline, linewidth=0.6, linestyle=linestyle, zorder=1)
                ax.plot([lo, hi], [y, y], color=outline, linewidth=2.2, zorder=2)
                ax.scatter([arr.mean()], [y], marker="o", s=16, color="white",
                           edgecolor=outline, linewidth=0.8, zorder=3)
            ax.axvline(rec.observed, color=INK, linestyle=(0, (3, 2)), linewidth=1.05, zorder=4)
            ax.text(0.02, 1.07, f"Observed r = {rec.observed:.3f}",
                    transform=ax.transAxes, fontsize=9.5)
            ax.set_xlim(0.54, 0.84)
            ax.set_xticks([0.55, 0.65, 0.75])
            ax.set_ylim(-0.5, 1.5)
            ax.set_yticks([1, 0])
            ax.set_yticklabels(["Size-only\n(same n)", "Severity-matched\n(same n + WAB-AQ bins)"]
                               if col_i == 0 else [])
            ax.tick_params(axis="y", length=0, pad=8)
            ax.tick_params(axis="x", length=3)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel("Pearson r", labelpad=4)
            if row_i == 0:
                ax.set_title(name, color=color, fontsize=10, fontweight="bold", pad=31)
        panel_labels(fig, row_axes, "abcd"[row_i*2:row_i*2+2])
        y = 0.965 if row_i == 0 else 0.475
        fig.text(0.02, y, f"{LABELS[subset]}\nn = {int(MANIFEST.loc[subset, 'n'])}",
                 fontsize=9.5, fontweight="bold", va="top")
    export(fig, "composition_controls", ["frozen_prediction_summary.csv", "resampling_metric_distributions.npz"],
           "Actual 2,000-draw regression score distributions for FLAIR and task-fMRI availability; "
           "size-only and severity-matched reference selection. Smoothed violin outlines, central 95% "
           "reference intervals, reference means and observed score.")


def make_refit_vs_frozen():
    fig = plt.figure(figsize=(WIDTH, 4.85))
    grid = fig.add_gridspec(2, 2, left=0.265, right=0.975, top=0.875,
                           bottom=0.165, wspace=0.26, hspace=0.84)
    for row_i, task in enumerate(["regression", "classification"]):
        scale = 1 if task == "regression" else 100
        row_axes = []
        for col_i, (model, name, color, marker, _) in enumerate(MODEL_CONFIG):
            ax = fig.add_subplot(grid[row_i, col_i])
            row_axes.append(ax)
            ys = np.arange(len(PRIMARY))
            plot_axis(ax, task, PRIMARY, ys, labels=col_i == 0, include_n=False)
            ax.set_ylim(6.5, -0.5)
            tab = rows(task, model, PRIMARY, source=COMPARE)
            for j, rec in enumerate(tab.itertuples()):
                ax.plot([rec.original_retrained_delta*scale, rec.delta_from_full*scale],
                        [j-0.13, j+0.13], color=color, linewidth=1.0, alpha=0.75, zorder=2)
            ax.scatter(tab.original_retrained_delta*scale, ys-0.13, facecolors="white",
                       edgecolors=color, marker=marker, linewidths=1.0, s=24, zorder=3)
            ax.scatter(tab.delta_from_full*scale, ys+0.13, color=color,
                       marker=marker, edgecolors="white", linewidths=0.55, s=24, zorder=4)
            if task == "regression":
                ax.set_xlim(-0.11, 0.035)
                ax.set_xticks([-0.10, -0.05, 0])
                ax.set_xlabel("Change in Pearson r", labelpad=4)
            else:
                ax.set_xlim(-13.0, 3.5)
                ax.set_xticks([-10, -5, 0])
                ax.set_xlabel("Change in balanced accuracy\n(percentage points)", labelpad=4)
            ax.set_title(f"{name} | {'regression' if task == 'regression' else 'classification'}",
                         loc="left", fontsize=9.5, fontweight="bold", pad=6)
        panel_labels(fig, row_axes, "abcd"[row_i*2:row_i*2+2])
    pair = {"linestyle": "none", "markersize": 5.2}
    handles = [(Line2D([], [], marker="o", color=RF, markerfacecolor="white", markeredgewidth=1.0, **pair),
                Line2D([], [], marker="s", color=XGB, markerfacecolor="white", markeredgewidth=1.0, **pair)),
               (Line2D([], [], marker="o", color=RF, markeredgecolor="white", markeredgewidth=0.55, **pair),
                Line2D([], [], marker="s", color=XGB, markeredgecolor="white", markeredgewidth=0.55, **pair))]
    fig.legend(handles, ["Refit within each selected set", "Fixed full-pool predictions"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, loc="upper center",
               ncol=2, bbox_to_anchor=(0.6, 0.995), handlelength=2.2, handletextpad=0.5,
               columnspacing=1.6)
    export(fig, "refit_vs_frozen", ["original_vs_frozen.csv"],
           "All seven primary rules in two tasks and two models; line segments pair the original "
           "cohort-refitted score change with the fixed-prediction score change. These are descriptive "
           "comparisons, not subtraction-based estimates of a training-cohort causal effect.")


def make_study_design():
    """Use the current first-figure builder so a full rebuild cannot restore the old diagram."""
    script = ROOT / "arc_lesion_image_benchmark/src/fig_first_story_20260923.py"
    subprocess.run([sys.executable, str(script), "--install"], check=True,
                   capture_output=True, text=True)
    from fig_first_story_20260923 import OUT as output
    qa = json.loads((output / "qa/figure-verification.json").read_text())
    scene = json.loads((output / "scene.json").read_text())
    RECORDS["study_design"] = {
        "width_in": qa["width_in"], "height_in": qa["height_in"],
        "font_sizes_pt": sorted({c["fs"] for c in scene["cells"] if c["kind"] == "text"}),
        "sources": list(qa["data_sources"]),
        "description": qa["claim"],
        "source_script": qa["source_script"],
        "drawio_vertices": qa["native_vertices"] + qa["native_groups"],
        "drawio_edges": qa["native_connectors"],
        "files_sha256": qa["files_sha256"],
    }


def write_contracts():
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    text = """# Figure contracts: manuscript figures

The figures are organised as design -> fixed-prediction results -> composition
controls -> link between the refitting and fixed-prediction designs. All values are
drawn from the verified CSV/NPZ outputs; no new model fits, data exclusions or
imputed observations are involved.

## Figure 1 - study_design

**Question:** With every patient's prediction held fixed and only the scored patients changed, how much can the score change?

**Argument:** An input requirement changes both the predictor and the patients who can be scored, which raises the question of how to attribute a method's increment. The 226 patient-level predictions are frozen, and the actual inclusion matrices and scores for the full pool, FLAIR availability and the published-list intersection are shown directly; the bottom row links the magnitude, fixed-prediction and composition evidence.

**Data:** The mean RF Pearson r is 0.712, 0.625 and 0.756; the latter two differ from the full pool by -0.087 and +0.044. FLAIR availability is the primary rule; the published-list intersection is labelled as a secondary analysis. Each small square is one actual patient, and all four matrices share the same severity order. The two examples in the first figure explain the control; the complete rules and their uncertainty are shown in Figure 2.

**Design references:** the conceptual layering of TabPFN Fig. 1 and the object and data-flow style of SAM 3 Fig. 4.

**Editability:** The `.drawio` file uses native text, shapes, groups and connectors; the PDF/SVG share its scene specification. Native rendering was checked with the official Draw.io viewer; an edit-and-export round trip inside the editor is not claimed. Save and export manual Draw.io edits separately; rerunning this script restores the script-defined figure.

**Reproduction:** This script calls `fig_first_story_20260923.py --install` to build the first figure; the other three figures use their own plotting functions. The current Figure 1 caption is in `paper/sec_intro.tex`.

## Figure 2 - frozen_scores

**Question:** Without updating predictions, how much does each inclusion rule change the score?

**Argument:** The primary rules are shown in full with conditional intervals; the two secondary analyses are kept separate. RF and XGBoost use colour plus distinct markers, so they remain distinguishable in greyscale. The zero line on the horizontal axis is the reference of all 226 patients; regression and classification keep their own units.

**Design references:** consistent encoding across multi-panel figures and layered multi-dimensional comparisons; no rankings, stars or highlighted extremes are used.

**Caption draft:** Score changes produced by changing only the scored patients. Points show the mean change from the full 226-person pool across 20 repeated five-fold prediction banks, for Random Forest (circles) and XGBoost (squares). Lines are paired 95% patient-bootstrap intervals conditional on the saved prediction banks, based on 2,000 resamples. (a) Change in Pearson correlation for aphasia-severity regression. (b) Change in balanced accuracy for severe/non-severe classification, in percentage points. The seven primary eligibility rules are shown above the divider. Outcome restriction (AQ at most 90) and the published-list intersection are separate secondary analyses. The latter contains 172 patients shared by the 173-person published release and the current modelable pool. Parentheses give patient counts. Imaging availability selects patients; the corresponding scans are not additional model inputs.

**Interpretation constraints:** The intervals describe score differences under paired patient resampling and do not include retraining uncertainty; secondary analyses are outside the primary scope. "DWI + both fMRI types" refers only to the three named modalities and is not described as "all modalities complete".

## Figure 3 - composition_controls

**Question:** When patient count and severity composition are matched, where does the observed score sit in the reference distribution?

**Argument:** FLAIR and task-fMRI regression results are the main sampling-reference differences; both models are shown. The N-only and N-plus-AQ-bin references share the same coordinates; the observed line does not move with the reference.

**Design references:** paired control figures that change the control constraint and ask how the same observation is explained. All outlines are smoothed densities of the actual sampled results, not schematic distributions.

**Caption draft:** Score distributions under patient-count and severity-composition matching. For each model, 2,000 subsets are drawn from the full pool with either the observed cohort size (Same n) or both its size and counts in the prespecified AQ bins (Same n + AQ-bin counts). Violin outlines show smoothed densities of the stored reference scores; horizontal segments mark their central 95% ranges and white points their means. Dashed vertical lines mark the observed fixed-prediction score, which is identical for both references within each panel. (a, b) FLAIR availability, n = 135. (c, d) Task-fMRI availability, n = 189. The matched-severity reference shifts toward the observed scores; the degree of remaining separation varies by cohort and model. All predictions are fixed. These sampling references describe composition dependence within the observed patient pool.

**Interpretation constraints:** The middle segment is the 95% range of the reference sampling distribution, not a confidence interval for an effect. An adjusted non-significant result is not drawn as equivalence or as a complete explanation; tail ranks for all seven rules are reported in the text and supplementary tables. Matched sampling degenerates for AQ <= 90, so that rule is not used as a composition-mechanism panel.

## Figure 4 - refit_vs_frozen

**Question:** How do the original refitting score differences correspond to the fixed-prediction score differences?

**Argument:** Each segment connects only the two descriptive results for the same model, task and inclusion rule. Open markers are within-cohort refits; filled markers are fixed predictions. All primary rules are kept, and the two procedures are not subtracted to obtain a training effect.

**Placement:** If the main text already reports this link in prose, the figure can be placed in the supplement to avoid duplicating Figure 2.

**Caption draft:** Connecting the original cohort-refitting grid with the fixed-prediction control. Open markers show changes from the full-pool score when models are refitted within each selected cohort; filled markers show changes when the full-pool out-of-fold predictions are retained and only the scored patients are selected. Segments pair the same eligibility rule. Panels show Random Forest and XGBoost for regression (a, b) and classification (c, d). All seven primary rules are included. Differences between the two designs are descriptive and are not estimates of a separate causal training-population effect.

## Production and verification

- All figures are exactly 7.16 in wide in the PDF MediaBox; heights adapt to readable layout.
- Arial is used for the figure text. Main labels in the data figures use 9-10 pt; the first figure uses 8.5-10 pt body text, 8.2 pt notes and larger headings/scores; centered subfigure letters use 8 pt Times New Roman, matching the Access template convention.
- Numeric figures preserve zero references, task-specific units, all seven primary rules, two model encodings, and distinct secondary analyses.
- PDF outputs are vector; SVG preserves text objects. PNG is a 300 dpi viewing copy, not the preferred submission source.
- Per-reference checks recompute the means and 2.5/97.5% quantiles from stored NPZ draws and compare them with the summary CSV before drawing.
- The code checks exact PDF width, native Draw.io vertex/edge presence and minimum text size. Source and output fingerprints are below.
- All four PNG renderings were inspected for overlapping or clipped labels.
- `pdffonts` on all four final PDFs confirms Arial, Arial Bold and Times New Roman are embedded TrueType subsets; no Type 3 fonts are present.

Reproduce from the project root:

```bash
MPLCONFIGDIR=/tmp/arc-manuscript-mpl python arc_lesion_image_benchmark/src/fig_manuscript_rewrite_20260922.py
```

## Artifact measurements and provenance

```json
"""
    provenance = {"source_sha256": {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                                    for name in ["frozen_prediction_summary.csv", "original_vs_frozen.csv",
                                                 "subset_manifest.csv", "resampling_metric_distributions.npz"]},
                  "figures": RECORDS}
    text += json.dumps(provenance, ensure_ascii=False, indent=2) + "\n```\n"
    CONTRACT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    assert len(PRIMARY) == 7
    for task in ["regression", "classification"]:
        for model, *_ in MODEL_CONFIG:
            assert len(rows(task, model, PRIMARY)) == 7
            assert len(rows(task, model, SECONDARY)) == 2
    # Figure 1 is not built here; make_study_design() and write_contracts() are not called.
    make_frozen_scores()
    make_composition_controls()
    make_refit_vs_frozen()
    print(json.dumps(RECORDS, indent=2))
