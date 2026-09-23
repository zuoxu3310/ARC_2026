#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s25_fig_granularity.py
======================
IEEE Access single-column figure: atlas granularity spectrum.
Mean competitive-model score (regression Pearson r, classification balanced
accuracy) versus number of atlas regions on a log axis, from 1 to 384 regions.
The adjacent-step TOST equivalence result is stated in the caption, not drawn on
the figure (no explanatory text annotations in result figures).

Reads results/summary/granularity_spectrum.csv.
Uses the shared nature_style helper so size/style match every other paper figure.
Usage: PYENV_VERSION=data-analysis python src/s25_fig_granularity.py
"""
from __future__ import annotations
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import nature_style as ns
import matplotlib.pyplot as plt

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMM = os.path.join(PKG, "results", "summary")


def main():
    spec = pd.read_csv(os.path.join(SUMM, "granularity_spectrum.csv"))
    ns.apply_publication_style(font_size=8.5, axes_linewidth=1.3)
    P = ns.PALETTE
    C_REG, C_CLF = P["blue_main"], P["red_strong"]

    fig, ax = plt.subplots(figsize=(ns.COL_SINGLE, 2.45))
    fig.subplots_adjust(left=0.16, right=0.84, bottom=0.20, top=0.91)
    nreg = sorted(spec["n_regions"].unique())
    xpos = {n: i for i, n in enumerate(nreg)}

    # Dual y-axis: Pearson r and balanced accuracy are different quantities, so a
    # single shared "Score" axis would invite a false reg-vs-clf comparison. Left
    # spine = regression r (blue); twin right spine = classification bal-acc (red).
    ax2 = ax.twinx()
    dr = spec[spec.task == "regression"].sort_values("n_regions")
    xr = [xpos[n] for n in dr["n_regions"]]
    h1 = ax.errorbar(xr, dr["mean_score"], yerr=dr["sd_score"], marker="o",
                     color=C_REG, capsize=2.5, lw=1.3, ms=4.5,
                     label="Regression ($r$)", zorder=3)
    dc = spec[spec.task == "classification"].sort_values("n_regions")
    xc = [xpos[n] for n in dc["n_regions"]]
    h2 = ax2.errorbar(xc, dc["mean_score"], yerr=dc["sd_score"], marker="s",
                      color=C_CLF, capsize=2.5, lw=1.3, ms=4.5,
                      label="Classification (bal. acc.)", zorder=3)

    ax.set_xticks(range(len(nreg)))
    ax.set_xticklabels([str(int(v)) for v in nreg])
    ax.set_xlabel("Atlas regions")
    # Each metric on its own data-tightened range (whiskers included) so neither
    # curve dominates; the ranges are offset so the two series do not overplot.
    ax.set_ylim(0.57, 0.73)
    ax2.set_ylim(0.69, 0.80)

    # Color each spine / axis to its series so the reader maps curve -> scale.
    ax.set_ylabel("Pearson $r$", color=C_REG)
    ax.tick_params(axis="y", colors=C_REG)
    ax.spines["left"].set_color(C_REG)
    ax2.spines.right.set_visible(True)          # twin axis needs its right spine
    ax2.spines.right.set_color(C_CLF)
    ax2.set_ylabel("Balanced accuracy", color=C_CLF)
    ax2.tick_params(axis="y", colors=C_CLF)

    ax.set_title("Finer atlases do not help", fontsize=8.0, loc="left", pad=3)
    # twinx splits legends; merge both handles into one frameless legend.
    ax.legend([h1, h2], ["Regression ($r$)", "Classification (bal. acc.)"],
              loc="lower right", fontsize=7.0, handlelength=1.6,
              labelspacing=0.3, borderpad=0.4)
    ns.finalize(fig, os.path.join(PKG, "results", "figures",
                                  "granularity_spectrum"), tight=False)
    print("saved -> results/figures/granularity_spectrum.{svg,pdf,png}")
    print(spec.to_string(index=False))


if __name__ == "__main__":
    main()
