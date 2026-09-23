#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nature_style.py —— shared figure style for every paper figure.

House style = **figures4papers** (Chen Liu, ChenLiu-1996/figures4papers,
`scientific-figure-making/` skill). The look:
  - open axes: top/right spines OFF;
  - frameless legend; minimal / no grid (value labels carry the numbers);
  - bold sans (Helvetica/Arial), editable SVG/PDF text;
  - semantic blue / green / red / neutral PALETTE (key=blue, contrast=red);
  - black-edged bars; values printed on bars.

What stays from IEEE Access: the **physical figure size** only. Canvases use the
IEEE column widths (COL_SINGLE / COL_DOUBLE) and finalize() exports at that exact
size WITHOUT bbox_inches="tight", so check_figure_sizes.py keeps passing. Only
the *visual style* switched to figures4papers (per Master ZX, 2026-06-13).

The filename is kept (every fig_*.py imports `nature_style`); the module now
implements the figures4papers conventions instead of the earlier IEEE-plain look.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- PALETTE: full figures4papers semantic palette (api.md / design-theory) ----
# blue = key/proposed; green = improvements; red/pink = baselines/contrasts;
# neutral = background. Existing keys (blue_main/secondary, red_strong, teal,
# neutral_*, band) are preserved so all fig_*.py keep working.
PALETTE = {
    "blue_main": "#0F4D92",       # key signal / hero
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",  # contrast / reference
    "teal": "#42949E", "violet": "#9A4D8E", "highlight": "#FFD700",
    "neutral_light": "#CFCECE",   # neutral band / secondary category
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "band": "#EAEFF5",            # light-blue neutral band (ceiling region)
    # Unified single-focus language shared by the main-text Fig 1-3 and the
    # supplementary figures: one focal blue, one flag orange for the anomalous /
    # non-equivalent point, one mute grey for everything else.
    "focal": "#1B5E8C",           # focal signal (main + supp figures)
    "flag": "#C0603A",            # anomaly / non-equivalent / hard case
    "mute": "#BFC4C9",            # de-emphasized category / null band
    "ink": "#1A1A1A",             # near-black ink for value labels / axis text
    "grey_text": "#8A9099",       # secondary annotation / reference line
    "grey_paper": "#6B7178",      # published-literature lane (Fig 1)
    "band_cool": "#E4E9EF",       # cool equivalence band (Fig 2/3)
    "box_fill": "#F6F7F9",         # near-white neutral box fill (pipeline)
}

# Ordered default colors when a script does not pass its own (api.md DEFAULT_COLORS).
DEFAULT_COLORS = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                  PALETTE["teal"], PALETTE["violet"], PALETTE["neutral_light"]]

# figures4papers comparison ramps: a blue hero + a graded family for the "other
# methods" in a sorted ladder (CellSpliceNet / ImmunoStruct look). Warm = warm
# pink/orange for alternatives; green = incremental improvements.
WARM_RAMP = ["#F09F97", "#F1B3AC", "#EFBEB8", "#F0CDC8", "#F3D9D8", "#FCEEED"]
GREEN_RAMP = ["#8BCF8B", "#AADCA9", "#C7E8C6", "#DDF3DE"]


def is_dark(hex_color, threshold=140):
    """True if a fill is dark enough to want white text on top (else use black)."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


def ladder_colors(n_others, hero_first=True):
    """Hero (blue_main) + n_others warm-ramp shades, for sorted bar ladders."""
    others = [WARM_RAMP[min(i, len(WARM_RAMP) - 1)] for i in range(n_others)]
    return [PALETTE["blue_main"], *others] if hero_first else others

# IEEE Access canvas widths (inches): single column 3.5, double (full page) 7.16.
# Result figures default to single column; only genuinely wide figures (3-panel,
# pipeline) use double — matching the journal's column-width habit.
COL_SINGLE = 3.5
COL_DOUBLE = 7.16


# Model display names (consistent across all figures + the main-text tables;
# avoid code-style RandomForest / SVR_rbf inside figures).
DISPLAY = {
    "RandomForest": "Random Forest",
    "SVR_rbf": "SVR (RBF)",
    "SVM_rbf": "SVM (RBF)",
    "LogReg": "LogReg",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "TabPFN": "TabPFN",
    "ElasticNet": "ElasticNet",
    "Ridge": "Ridge",
    "MLP": "MLP",
    "Dummy": "Dummy",
}


def apply_publication_style(font_size=8.5, axes_linewidth=1.3):
    """Set figures4papers-style rcParams at IEEE column sizes. Call before plotting.

    figures4papers font sizes (15-24pt) target very wide canvases; at a 3.5 in
    IEEE column those would overflow, so sizing is scaled down while the *visual
    grammar* (open axes, frameless legend, no grid, bold sans, black-edged bars,
    value labels) is kept faithful.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Nimbus Sans",
                                       "Liberation Sans", "DejaVu Sans"]
    # Math/italic (e.g. $r$) would otherwise fall back to DejaVu. Force Arial so
    # italic r, sub/superscripts render in a journal-accepted sans, not DejaVu.
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"
    plt.rcParams["svg.fonttype"] = "none"     # keep SVG text editable <text>
    plt.rcParams["pdf.fonttype"] = 42          # editable TrueType in PDF
    plt.rcParams["font.size"] = font_size
    # figures4papers: open axes — top/right spines OFF (the signature look).
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["axes.labelsize"] = font_size + 1
    plt.rcParams["axes.titlesize"] = font_size + 1
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["xtick.major.width"] = axes_linewidth
    plt.rcParams["ytick.major.width"] = axes_linewidth
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5
    # figures4papers: minimal/no grid — rely on value labels and axis ticks.
    plt.rcParams["axes.grid"] = False
    # figures4papers: frameless legend.
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = font_size - 0.5
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.axisbelow"] = True


def add_panel_label(ax, label, x=-0.08, y=1.04, fontsize=11):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom")


def finalize(fig, out_base, dpi=600, formats=("svg", "pdf", "png"), tight=True):
    """Export at the figure's physical size. out_base has no extension.

    Do not use bbox_inches="tight": it changes the saved PDF's physical
    dimensions, which breaks IEEE column-width sizing (check_figure_sizes.py).
    """
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    if tight:
        fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{out_base}.{fmt}", dpi=dpi)
    plt.close(fig)
