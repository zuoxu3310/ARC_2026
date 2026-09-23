#!/usr/bin/env python3
"""Four manuscript figures from the audited frozen-prediction results.

This is a plotting-only script: it does not fit models or change analysis outputs.
The design diagram is rendered from the same object list to Matplotlib and to
native Draw.io vertices/edges. All numerical panels read the audited CSV/NPZ.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
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
MODEL_CONFIG = [("RandomForest", "Random Forest", RF, "o", -0.14),
                ("XGBoost", "XGBoost", XGB, "s", 0.14)]
PRIMARY = ["has_dwi", "has_rsfmri", "has_taskfmri", "has_flair",
           "multimodal_complete", "chronic_365", "chronic_180"]
SECONDARY = ["aq_le90", "teghipco_idlist"]
LABELS = {"has_dwi": "DWI available", "has_rsfmri": "Resting fMRI available",
          "has_taskfmri": "Task fMRI available", "has_flair": "FLAIR available",
          "multimodal_complete": "DWI + both fMRI types",
          "chronic_365": "≥365 days after stroke", "chronic_180": "≥180 days after stroke",
          "aq_le90": "AQ at most 90", "teghipco_idlist": "Published-list intersection"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5, "axes.labelsize": 9.5,
    "axes.titlesize": 10.0, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0, "axes.linewidth": 0.65,
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


def panel_label(fig, x, y, letter):
    fig.text(x, y, f"({letter})", ha="center", va="center",
             fontsize=8, fontfamily="Times New Roman")


def plot_axis(ax, task: str, order: list[str], ypos, labels=False, include_n=True):
    ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{LABELS[s]}  ({int(MANIFEST.loc[s, 'n'])})" if include_n else LABELS[s]
                        for s in order] if labels else [])
    ax.xaxis.set_major_formatter(FuncFormatter(signed if task == "regression" else signed_int))
    ax.grid(axis="x", color="#ECEEEF", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def make_frozen_scores():
    fig = plt.figure(figsize=(WIDTH, 5.5))
    positions = [[0.325, 0.16, 0.294, 0.68], [0.68, 0.16, 0.294, 0.68]]
    axes = [fig.add_axes(p) for p in positions]
    order = PRIMARY + SECONDARY
    ys = np.r_[np.arange(7), [8.55, 9.55]]
    for ax, task in zip(axes, ["regression", "classification"]):
        scale = 1 if task == "regression" else 100
        plot_axis(ax, task, order, ys, labels=task == "regression")
        ax.set_ylim(10.15, -1.0)
        # A visible gap and separate label keep the two secondary rules distinct.
        ax.axhline(7.18, color="#BBC2C7", linewidth=0.65)
        for model, name, color, marker, offset in MODEL_CONFIG:
            tab = rows(task, model, order)
            y = ys + offset
            ax.hlines(y, tab.delta_ci_low * scale, tab.delta_ci_high * scale,
                      color=color, linewidth=1.35, zorder=2)
            ax.scatter(tab.delta_from_full * scale, y, color=color, marker=marker,
                       s=21, linewidths=0.4, edgecolors="white", label=name, zorder=3)
        if task == "regression":
            ax.set_xlim(-0.25, 0.105)
            ax.set_xticks([-0.2, -0.1, 0, 0.1])
            ax.set_xlabel("Change in Pearson r", labelpad=6)
            ax.set_title("Severity regression", loc="left", fontweight="bold", pad=15)
        else:
            ax.set_xlim(-12.5, 6.5)
            ax.set_xticks([-10, -5, 0, 5])
            ax.set_xlabel("Change in balanced accuracy\n(percentage points)", labelpad=6)
            ax.set_title("Severe / non-severe", loc="left", fontweight="bold", pad=15)
    ytrans = axes[0].get_yaxis_transform()
    axes[0].text(-1.04, -0.85, "Primary eligibility rules  (n)", transform=ytrans,
                 fontsize=9.5, fontweight="bold", va="center", clip_on=False)
    axes[0].text(-1.04, 7.83, "Secondary analyses", transform=ytrans,
                 fontsize=9.5, fontweight="bold", va="center", clip_on=False)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="upper center",
               bbox_to_anchor=(0.655, 0.98), handletextpad=0.4, columnspacing=1.4)
    panel_label(fig, 0.472, 0.025, "a")
    panel_label(fig, 0.827, 0.025, "b")
    export(fig, "frozen_scores", ["frozen_prediction_summary.csv", "subset_manifest.csv"],
           "All seven primary rules and two separate secondary rules; paired patient-bootstrap "
           "95% intervals for change from the full 226-person pool, conditional on saved predictions.")


def make_composition_controls():
    fig = plt.figure(figsize=(WIDTH, 4.45))
    grid = fig.add_gridspec(2, 2, left=0.205, right=0.975, top=0.845,
                           bottom=0.15, hspace=0.9, wspace=0.30)
    x = np.linspace(0.54, 0.84, 500)
    for row_i, subset in enumerate(["has_flair", "has_taskfmri"]):
        for col_i, (model, name, color, _, _) in enumerate(MODEL_CONFIG):
            ax = fig.add_subplot(grid[row_i, col_i])
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
                                edgecolor=outline, linewidth=0.55, linestyle=linestyle, zorder=1)
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
            ax.set_yticklabels(["Same n", "Same n +\nAQ-bin counts"] if col_i == 0 else [])
            ax.tick_params(axis="y", length=0, pad=8)
            ax.tick_params(axis="x", length=3)
            ax.spines["left"].set_visible(False)
            if row_i == 1:
                ax.set_xlabel("Pearson r", labelpad=5)
            if row_i == 0:
                ax.set_title(name, color=color, fontsize=10, fontweight="bold", pad=31)
            bbox = ax.get_position()
            panel_label(fig, (bbox.x0+bbox.x1)/2, bbox.y0-0.105 if row_i == 1 else bbox.y0-0.07,
                        "abcd"[row_i*2+col_i])
        y = 0.965 if row_i == 0 else 0.48
        fig.text(0.025, y, f"{'FLAIR available' if row_i == 0 else 'Task fMRI available'}\nn = {int(MANIFEST.loc[subset, 'n'])}",
                 fontsize=9.5, fontweight="bold", va="top")
    export(fig, "composition_controls", ["frozen_prediction_summary.csv", "resampling_metric_distributions.npz"],
           "Actual 2,000-draw regression score distributions for FLAIR and task-fMRI availability; "
           "same-n and same-n-plus-AQ-bin reference selection. Smoothed violin outlines, central 95% "
           "reference intervals, reference means and observed score.")


def make_refit_vs_frozen():
    fig = plt.figure(figsize=(WIDTH, 6.5))
    grid = fig.add_gridspec(2, 2, left=0.295, right=0.975, top=0.86,
                           bottom=0.15, wspace=0.3, hspace=0.69)
    for row_i, task in enumerate(["regression", "classification"]):
        scale = 1 if task == "regression" else 100
        for col_i, (model, name, color, marker, _) in enumerate(MODEL_CONFIG):
            ax = fig.add_subplot(grid[row_i, col_i])
            ys = np.arange(len(PRIMARY))
            plot_axis(ax, task, PRIMARY, ys, labels=col_i == 0, include_n=False)
            ax.set_ylim(6.6, -0.6)
            tab = rows(task, model, PRIMARY, source=COMPARE)
            for j, rec in enumerate(tab.itertuples()):
                ax.plot([rec.original_retrained_delta*scale, rec.delta_from_full*scale],
                        [j-0.10, j+0.10], color=color, linewidth=1.0, alpha=0.75, zorder=2)
            ax.scatter(tab.original_retrained_delta*scale, ys-0.10, facecolors="white",
                       edgecolors=color, marker=marker, linewidths=1.0, s=26, zorder=3)
            ax.scatter(tab.delta_from_full*scale, ys+0.10, color=color,
                       marker=marker, edgecolors="white", linewidths=0.4, s=26, zorder=4)
            if task == "regression":
                ax.set_xlim(-0.11, 0.035)
                ax.set_xticks([-0.10, -0.05, 0])
                ax.set_xlabel("Change in Pearson r", labelpad=5)
            else:
                ax.set_xlim(-13.0, 3.5)
                ax.set_xticks([-10, -5, 0])
                ax.set_xlabel("Change in balanced accuracy\n(percentage points)", labelpad=5)
            ax.set_title(f"{name} | {'regression' if task == 'regression' else 'classification'}",
                         loc="left", fontsize=9.5, fontweight="bold", pad=10)
            bbox = ax.get_position()
            panel_label(fig, (bbox.x0+bbox.x1)/2, bbox.y0-0.073 if row_i == 0 else 0.025,
                        "abcd"[row_i*2+col_i])
    handles = [Line2D([], [], marker="o", linestyle="none", color=MUTED,
                      markerfacecolor="white", markersize=5.4,
                      label="Refit within each cohort"),
               Line2D([], [], marker="o", linestyle="none", color=MUTED,
                      markerfacecolor=MUTED, markersize=5.4,
                      label="Fixed prediction bank")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.59, 0.985),
               handletextpad=0.5, columnspacing=1.1)
    export(fig, "refit_vs_frozen", ["original_vs_frozen.csv"],
           "All seven primary rules in two tasks and two models; line segments pair the original "
           "cohort-refitted score change with the fixed-prediction score change. These are descriptive "
           "comparisons, not subtraction-based estimates of a training-cohort causal effect.")


def make_study_design():
    """Use the current first-figure builder so a full rebuild cannot restore the old diagram."""
    script = ROOT / "arc_lesion_image_benchmark/src/fig_first_story_20260923.py"
    subprocess.run([sys.executable, str(script), "--install"], check=True,
                   capture_output=True, text=True)
    output = ROOT / "outputs/first-figure-redesign-2026-09-23"
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
    text = """# Figure contracts — manuscript rewrite, 2026-09-22

Master ZX，本组图按“设计 → 固定预测的结果 → 构成对照 → 新旧设计连接”组织。所有数值由已审核 CSV/NPZ 绘制；没有新拟合、数据筛除或补造观测值。

## Figure 1 — study_design（2026-09-23 首图重做）

**唯一问题：** 每个人的预测都不变，只改变计分患者，成绩能改变多少？

**承担的论证：** 输入要求同时改变预测器与可计分患者，提出方法增量归因问题；冻结 226 人的逐人预测，直接展示全池、FLAIR 和公开名单交集的真实纳入矩阵及分数；底部连接量级、固定预测、构成解释三项证据。

**数据：** RF 的平均 Pearson r 为 0.712、0.625、0.756，后两者相对全池分别为 −0.087、+0.044。FLAIR 是主规则，公开名单交集明确标为次要分析。每个小方格对应一位实际患者，四个矩阵保持同一严重度顺序。首图中的两例用于解释控制，完整规则和不确定性由 Figure 2 承担。

**借鉴：** TabPFN Fig. 1 的概念层次、SAM 3 Fig. 4 的对象与数据流表达，以及已实读的小红书技术路线复现/绘图规格。借鉴来源、实际读取范围与取舍见 `outputs/first-figure-redesign-2026-09-23/REPORT.zh-CN.md`。

**可编辑性：** `.drawio` 使用原生文本、形状、分组和连接线；PDF/SVG 与其共享场景规格。已通过官方 Draw.io 查看器的本地原生渲染核看，但未宣称编辑器内修改导出的往返测试。Draw.io 手工编辑后应另存并导出；再次运行脚本会恢复脚本定义的图。

**复现：** 本脚本委托 `fig_first_story_20260923.py --install` 生成首图，其余三图仍使用各自原有绘图函数。首图最新图注以 `paper/sec_intro.tex` 为准。

## Figure 2 — frozen_scores

**唯一问题：** 不更新预测，全部纳入规则分别让分数改变多少？

**承担的论证：** 主规则完整展示且含条件区间，两个次要分析分开。RF 与 XGBoost 使用颜色加不同标记，即使黑白阅读也可区分。横坐标的零线是全体 226 人的参照；回归与分类保持原有不同单位。

**借鉴：** Access 多面板的统一编码，以及 Nature 多维比较的层次；本图不使用排名、星号或任意极值突出。

**Caption draft:** Score changes produced by changing only the scored patients. Points show the mean change from the full 226-person pool across 20 repeated five-fold prediction banks, for Random Forest (circles) and XGBoost (squares). Lines are paired 95% patient-bootstrap intervals conditional on the saved prediction banks, based on 2,000 resamples. (a) Change in Pearson correlation for aphasia-severity regression. (b) Change in balanced accuracy for severe/non-severe classification, in percentage points. The seven primary eligibility rules are shown above the divider. Outcome restriction (AQ at most 90) and the published-list intersection are separate secondary analyses. The latter contains 172 patients shared by the 173-person published release and the current modelable pool. Parentheses give patient counts. Imaging availability selects patients; the corresponding scans are not additional model inputs.

**解释约束：** 区间为配对患者重抽样下的分数差区间，并不包括重训流程的不确定性；次要分析不进入主要范围。DWI + both fMRI types 只是三个指定模态，不能写作“所有模态完整”。

## Figure 3 — composition_controls

**唯一问题：** 人数相同及严重度构成相同时，观察分数位于什么样的参照分布中？

**承担的论证：** FLAIR 与 task-fMRI 回归结果是主要的抽样参照差异，两模型均展示。N-only 与 N-plus-AQ-bin 参照以相同坐标排列；观察线不因参照变化而移动。

**借鉴：** NeurIPS 成对控制图的逻辑：改变对照约束，追问同一观察的解释。所有轮廓来自实际抽样结果的平滑密度，不是示意分布。

**Caption draft:** Score distributions under patient-count and severity-composition matching. For each model, 2,000 subsets are drawn from the full pool with either the observed cohort size (Same n) or both its size and counts in the prespecified AQ bins (Same n + AQ-bin counts). Violin outlines show smoothed densities of the stored reference scores; horizontal segments mark their central 95% ranges and white points their means. Dashed vertical lines mark the observed fixed-prediction score, which is identical for both references within each panel. (a, b) FLAIR availability, n = 135. (c, d) Task-fMRI availability, n = 189. The matched-severity reference shifts toward the observed scores; the degree of remaining separation varies by cohort and model. All predictions are fixed. These sampling references describe composition dependence within the observed patient pool.

**解释约束：** 中间线段为参照抽样分布的 95% 范围，不是效应的置信区间。没有把校正后的“不显著”图形化为等价或全部解释；完整七条规则尾秩由正文/补充表报告。AQ≤90 的匹配抽样退化，因此不作为构成机制面板。

## Figure 4 — refit_vs_frozen

**唯一问题：** 原有重新拟合的分差与冻结预测的分差怎样对应？

**承担的论证：** 每一条线仅连接同一模型、任务和纳入规则下的两个描述性结果。空心为队列内重拟合、实心为固定预测。完整主规则保留，不把两条流程相减转为训练效应。

**位置建议：** 主文若已用段落报告新旧连接，本图可移补充，避免与 Figure 2 重复占据证据重心。它是完整可用的连接图，不必因“已经做了”强行塞入主文。

**Caption draft:** Connecting the original cohort-refitting grid with the fixed-prediction control. Open markers show changes from the full-pool score when models are refitted within each selected cohort; filled markers show changes when the full-pool out-of-fold predictions are retained and only the scored patients are selected. Segments pair the same eligibility rule. Panels show Random Forest and XGBoost for regression (a, b) and classification (c, d). All seven primary rules are included. Differences between the two designs are descriptive and are not estimates of a separate causal training-population effect.

## Production and verification

- All figures are exactly 7.16 in wide in the PDF MediaBox; heights adapt to readable layout.
- Arial is used for the figure text. Main labels in the data figures use 9–10 pt; the redesigned first figure uses 8.5–10 pt body text, 8.2 pt notes and larger headings/scores; centered subfigure letters use 8 pt Times New Roman, matching the Access template convention.
- Numeric figures preserve zero references, task-specific units, all seven primary rules, two model encodings, and distinct secondary analyses.
- PDF outputs are vector; SVG preserves text objects. PNG is a 300 dpi viewing copy, not the preferred submission source.
- Per-reference checks recompute the means and 2.5/97.5% quantiles from stored NPZ draws and compare them with the summary CSV before drawing.
- The code checks exact PDF width, native Draw.io vertex/edge presence and minimum text size. Source and output fingerprints are below.
- Local visual inspection: all four PNG renderings were opened and inspected. A clipped original XGBoost/FLAIR classification point in an earlier preview was corrected by widening the comparison axis; the updated point and bottom panel labels were rechecked. The three other figures showed no overlapping or cut-off labels in their full-width previews.
- `pdffonts` on all four final PDFs confirms Arial, Arial Bold and Times New Roman are embedded TrueType subsets; no Type 3 fonts are present.
- Final acceptance still requires the parent session to review the figures in the newly compiled two-column manuscript. Human acceptance is not claimed.

Reproduce from the project root:

```bash
python arc_lesion_image_benchmark/src/fig_manuscript_rewrite_20260922.py
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
    make_study_design()
    make_frozen_scores()
    make_composition_controls()
    make_refit_vs_frozen()
    write_contracts()
    print(json.dumps(RECORDS, indent=2))
