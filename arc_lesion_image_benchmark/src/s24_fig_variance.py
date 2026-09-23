#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s24_fig_variance.py
===================
IEEE Access single-column figure: variance decomposition (headline).
One panel of grouped bars: the share of score variance ($\\eta^2$) carried by the
subset, model, and split factors, for regression and classification, in the
competitive-model scope. The subset factor dominates; the split is a noise floor.
The all-model contrast lives in the main-text table, not here.

Reads results/summary/variance_decomposition_*.csv (scope primary_competitive_models).
Uses the shared nature_style helper so size/style match every other paper figure.
Usage: PYENV_VERSION=data-analysis python src/s24_fig_variance.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import nature_style as ns
import matplotlib.pyplot as plt

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMM = os.path.join(PKG, "results", "summary")


# Headline competitive scope: regression INCLUDES TabPFN (separate CSV, its own
# scope tag); classification stays WITHOUT TabPFN (primary_competitive_models).
ROW = {
    "regression": ("variance_decomposition_regression_competitive_with_tabpfn.csv",
                   "primary_competitive_models_WITH_TabPFN"),
    "classification": ("variance_decomposition_classification.csv",
                       "primary_competitive_models"),
}


def eta_row(task):
    fname, scope = ROW[task]
    d = pd.read_csv(os.path.join(SUMM, fname))
    return d[d.scope == scope].iloc[0]


def main():
    ns.apply_publication_style(font_size=8.5, axes_linewidth=1.3)
    P = ns.PALETTE
    tasks = ["regression", "classification"]
    xlab = ["Regression", "Classification"]
    comps = [("subset", "Subset", P["blue_main"]),
             ("model", "Model", P["red_strong"]),
             ("seed", "Split", P["neutral_light"])]

    fig, ax = plt.subplots(figsize=(ns.COL_SINGLE, 2.45))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.90)
    x = np.arange(len(tasks))
    w = 0.26
    # Faint anchor at the regression Subset value so the eye reads the hero bar
    # against a reference line (number already printed on the bar). Behind bars.
    ax.axhline(0.62, color=P["neutral_light"], ls=":", lw=0.9, zorder=0)
    for i, (key, lab, col) in enumerate(comps):
        vals = [float(eta_row(t)[f"eta2_{key}"]) for t in tasks]
        hero = (key == "subset")
        # Hero (Subset) bars get a thicker edge so they pop; Model/Split stay thin.
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col,
                      edgecolor=P["neutral_black"],
                      linewidth=1.3 if hero else 0.7, zorder=3)
        if hero:
            # Hero value sits INSIDE the dark blue bar; is_dark -> white text so
            # the large bold number is legible on the deep fill.
            lbl_color = "white" if ns.is_dark(col) else P["neutral_black"]
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals],
                         label_type="center", fontsize=8.5,
                         fontweight="bold", color=lbl_color)
        else:
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2,
                         fontsize=7.0, color=P["neutral_black"])
    ax.set_xticks(x)
    ax.set_xticklabels(xlab)
    ax.set_ylabel(r"Variance share $\eta^2$")
    ax.set_ylim(0, 0.78)
    ax.set_title("Cohort subset drives the variance", fontsize=8.0,
                 loc="left", pad=3)
    ax.legend(loc="upper right", fontsize=7.0, handlelength=1.2,
              labelspacing=0.3, borderpad=0.4)
    ns.finalize(fig, os.path.join(PKG, "results", "figures",
                                  "variance_decomposition"), tight=False)
    print("saved -> results/figures/variance_decomposition.{svg,pdf,png}")
    for t in tasks:
        r = eta_row(t)
        print(f"  {t:14s} subset={r.eta2_subset} model={r.eta2_model} "
              f"split={r.eta2_seed}")


if __name__ == "__main__":
    main()
