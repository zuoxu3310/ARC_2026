#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IEEE Access-style compact forest plot for equivalence-effect sizes."""
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
BOUND = 0.05

LABELS = {
    ("fine_vs_coarse__RF_default", "pearson_r"): "Atlas ($r$)",
    ("tuned_vs_default__RF_JHU_fine", "pearson_r"): "Tuning ($r$)",
    ("fine_vs_coarse__RF_default", "AUC"): "Atlas (AUC)",
    ("tuned_vs_default__RF_JHU_fine", "AUC"): "Tuning (AUC)",
}


def main():
    df = pd.read_csv(os.path.join(SUMM, "equivalence_power.csv"))
    rows = []
    for _, r in df.iterrows():
        key = (r["contrast"], r["metric"])
        rows.append((LABELS.get(key, f"{r['contrast']}\n{r['metric']}"),
                     float(r["delta_A_minus_B"]), float(r["ci95_lo"]),
                     float(r["ci95_hi"]), bool(r["equivalent_at_0.05"])))

    ns.apply_publication_style(font_size=8.0, axes_linewidth=1.3)
    P = ns.PALETTE
    fig, ax = plt.subplots(figsize=(ns.COL_SINGLE, 2.25))
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.22, top=0.86)

    y = list(range(len(rows)))[::-1]
    ax.axvline(0, color=P["neutral_mid"], ls="--", lw=1.0, zorder=1)
    ax.axvline(-BOUND, color=P["neutral_light"], lw=0.9, zorder=1)
    ax.axvline(BOUND, color=P["neutral_light"], lw=0.9, zorder=1)

    for yi, (lab, d, lo, hi, equ) in zip(y, rows):
        xerr = [[d - lo], [hi - d]]
        ax.errorbar(d, yi, xerr=xerr, fmt="s", ms=4.2,
                    mfc=P["blue_main"], mec=P["blue_main"],
                    ecolor=P["neutral_black"], elinewidth=1.1,
                    capsize=3, capthick=1.0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Difference (A-B)")
    ax.set_title("Method contrasts near zero", fontsize=8.0,
                 loc="left", pad=3)
    ax.set_xlim(-0.06, 0.06)
    ax.set_ylim(-0.7, y[0] + 0.7)
    ns.finalize(fig, os.path.join(PKG, "results", "figures", "equivalence_forest"),
                tight=False)
    print("saved -> results/figures/equivalence_forest.{svg,pdf,png}")
    for lab, d, lo, hi, equ in rows:
        print(f"  {lab.replace(chr(10),' '):40s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  equiv={equ}")


if __name__ == "__main__":
    main()
