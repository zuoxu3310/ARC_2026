#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s22_granularity_spectrum.py
===========================
Atlas granularity spectrum. One fixed pipeline (strong models + in-fold scaler/SMOTE)
is run on atlases of different granularity to check whether the granularity-vs-score
curve plateaus - confirming and extending Tilwani et al.'s finding that the atlas
main effect is not significant.

Granularity points (full cohort, n=226) are defined by the feature tables in
GRANULARITIES, e.g.:
  1 region     = floor_volage_features.tsv (lesion_volume + age, whole-brain load)
  33 regions   = ArterialAtlas156_features_merged.tsv
  189 regions  = jhu156_features_merged.tsv
(Extensible: resample another atlas, e.g. AAL/AICHA, to this grid -> extract with s2
 -> add its table to GRANULARITIES.)

Output: r / bACC per granularity x task (mean + SD over seeds) and TOST equivalence
between adjacent granularities.
Reuses the s18 cell runner / model factory so the pipeline is identical to the
crossed-grid analysis.

Usage: python src/s22_granularity_spectrum.py --seeds 20
"""
from __future__ import annotations
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import s18_crossed_grid as g

PKG = g.PKG
SUMM = os.path.join(PKG, "results", "summary")
DATA = os.path.join(PKG, "data")
COMP_REG = ["RandomForest", "XGBoost", "LightGBM"]
COMP_CLF = ["RandomForest", "XGBoost", "LightGBM", "LogReg"]

# Granularity points: (label, n_regions_nominal, feature table path relative to data/).
# Six-point spectrum: 1 -> 33 -> 41 -> 116 -> 189 -> 384 regions.
GRANULARITIES = [
    ("vol_1", 1, "floor_volage_features.tsv"),
    ("arterial_33", 33, "ArterialAtlas156_features_merged.tsv"),
    ("brodmann_41", 41, "brodmann41_features_merged.tsv"),
    ("aal_116", 116, "aal116_features_merged.tsv"),
    ("jhu_189", 189, "jhu156_features_merged.tsv"),
    ("aicha_384", 384, "aicha384_features_merged.tsv"),
]


def load_table(rel):
    df = pd.read_csv(os.path.join(DATA, rel), sep="\t", index_col="participant_id")
    df = df.dropna(subset=["wab_aq"]).fillna(0)
    y_aq = df["wab_aq"].copy()
    feat = df.drop(columns=["wab_aq"])
    n = len(feat)
    keep = [c for c in feat.columns if (feat[c] != 0).sum() > g.SPARSE_DROP_FRAC * n]
    # Always keep at least lesion_volume (in the 1-region case it is the only information)
    if not keep:
        keep = list(feat.columns)
    return feat[keep], y_aq, len(keep)


def run_granularity(task, seeds):
    runner = g.run_cell_regression if task == "regression" else g.run_cell_classification
    models = COMP_REG if task == "regression" else COMP_CLF
    rows = []
    percell = {}   # (gran) -> list of per-(model,seed) scores, used for TOST
    for label, nreg, rel in GRANULARITIES:
        X, y, nfeat = load_table(rel)
        scores = []
        for model in models:
            for seed in seeds:
                s = runner(X, y, model, seed)["score"]
                scores.append(s)
        percell[label] = np.array(scores)
        rows.append({"task": task, "granularity": label, "n_regions": nreg,
                     "n_features": nfeat, "mean_score": round(float(np.mean(scores)), 4),
                     "sd_score": round(float(np.std(scores)), 4),
                     "min": round(float(np.min(scores)), 4),
                     "max": round(float(np.max(scores)), 4)})
        print(f"  [{task}] {label:14s} ({nfeat:3d} feat) "
              f"score={np.mean(scores):.3f}±{np.std(scores):.3f}", flush=True)
    return pd.DataFrame(rows), percell


def tost_adjacent(percell, labels, bound=0.05):
    """TOST equivalence between adjacent granularities (simplified paired bootstrap:
    equivalent if the 90% CI of the score difference lies within +/-bound)."""
    out = []
    for a, b in zip(labels[:-1], labels[1:]):
        da, db = percell[a], percell[b]
        n = min(len(da), len(db))
        diff = db[:n] - da[:n]
        # bootstrap 90% CI of mean diff
        rng = np.random.default_rng(2026)
        boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [5, 95])
        equiv = (lo > -bound) and (hi < bound)
        out.append({"contrast": f"{a}_vs_{b}", "mean_diff": round(float(np.mean(diff)), 4),
                    "ci90_lo": round(float(lo), 4), "ci90_hi": round(float(hi), 4),
                    "equiv_bound": bound, "equivalent": bool(equiv)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--base_seed", type=int, default=3000)
    args = ap.parse_args()
    seeds = [args.base_seed + i for i in range(args.seeds)]

    all_rows, all_tost = [], []
    for task in ["regression", "classification"]:
        print(f"=== {task} granularity spectrum ===", flush=True)
        rows, percell = run_granularity(task, seeds)
        labels = [r[0] for r in GRANULARITIES]
        tost = tost_adjacent(percell, labels)
        tost.insert(0, "task", task)
        all_rows.append(rows); all_tost.append(tost)
        print(tost.to_string(index=False), flush=True)

    os.makedirs(SUMM, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(
        os.path.join(SUMM, "granularity_spectrum.csv"), index=False)
    pd.concat(all_tost, ignore_index=True).to_csv(
        os.path.join(SUMM, "granularity_tost.csv"), index=False)
    print(f"\n[write] granularity_spectrum.csv + granularity_tost.csv", flush=True)


if __name__ == "__main__":
    main()
