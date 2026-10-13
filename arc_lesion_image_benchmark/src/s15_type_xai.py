#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s15_type_xai.py - region contributions for fluent vs. non-fluent classification
(leakage-safe), reusing the s8 machinery.

Question: which brain regions does the model use to separate "non-fluent"
(Broca/Global/TCM; classic anterior/frontal damage) from "fluent"
(Wernicke/Conduction/Anomic; posterior/temporo-parietal damage)? Checks whether they fall
in left frontal language areas, consistent with the existing s8 severity XAI
(left dorsal language stream).

Interpretation: predictive association, not causal localisation (same as s8).
Method: out-of-fold permutation importance (column level + collinear-cluster group level)
        + in-fold volume correction (region-specific signal vs. total-lesion-volume proxy),
        in-fold scaler->SMOTE->RF, outer repeated stratified 5x10. Directly reuses s8's
        _oof_perm / collinearity / volume_correction (task='clf', with y/strat set to the
        fluency label).

Writes: results/summary/type_xai_perm_{gran}.csv
        results/summary/type_xai_group_{gran}.csv
        results/summary/type_xai_volume_correction_{gran}.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import s8_xai_region_contribution as s8  # noqa: E402  reuse OOF permutation / collinearity / volume correction

PKG = os.path.abspath(os.path.join(SRC, ".."))
SUMMARY = os.path.join(PKG, "results", "summary")
PARTICIPANTS = os.path.join(PKG, "..", "datasets", "arc_ds004884", "participants.tsv")
SPARSE_DROP_FRAC = 0.10
N_REPEATS = 10

NONFLUENT = {"Broca", "Global", "TranscorticalMotor", "TranscorticalMixed"}
FLUENT = {"Wernicke", "Conduction", "Anomic", "TranscorticalSensory"}
TABLES = {
    "arterial_coarse": "data/ArterialAtlas156_features_merged.tsv",
    "JHU_fine": "data/jhu156_features_merged.tsv",
}


def load_fluent(table_rel):
    """X (drop wab_aq/wab_type, fillna(0), drop sparse columns) + y (1 = non-fluent) + feat."""
    df = pd.read_csv(os.path.join(PKG, table_rel), sep="\t", index_col="participant_id")
    pp = pd.read_csv(PARTICIPANTS, sep="\t", index_col="participant_id")["wab_type"].astype(str)
    df = df.join(pp, how="left")
    sub = df[df["wab_type"].isin(NONFLUENT | FLUENT)].copy()
    y = sub["wab_type"].isin(NONFLUENT).astype(int).values
    feat_df = sub.drop(columns=[c for c in ["wab_aq", "wab_type"] if c in sub.columns]).fillna(0)
    n = len(feat_df)
    keep = [c for c in feat_df.columns if (feat_df[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    feat_df = feat_df[keep]
    return feat_df, y, list(feat_df.columns)


def run(gran, table_rel, n_rep):
    print(f"\n========== fluent XAI: {gran} ({table_rel}) ==========", flush=True)
    X, y, feat = load_fluent(table_rel)
    n_pos = int(y.sum())
    print(f"n={len(X)}  nonfluent={n_pos} fluent={len(X)-n_pos}  features={len(feat)}", flush=True)

    # Collinear clusters (target-independent; feature correlation structure only)
    cluster_df, group_cols, corr_df = s8.collinearity_clusters(X, feat)
    cluster_df.to_csv(os.path.join(SUMMARY, f"type_xai_collinearity_{gran}.csv"), index=False)

    # 1) Column-level OOF permutation importance (task='clf' -> scaler->SMOTE->RF; strat/y use the fluency label)
    perm = s8._oof_perm("clf", X, y, y, feat, n_rep)
    perm = s8.annotate_collinear(perm, feat, corr_df, top_n=10)
    perm.to_csv(os.path.join(SUMMARY, f"type_xai_perm_{gran}.csv"), index=False)

    # 2) Group-level (collinear cluster) permutation importance
    grp = s8._oof_perm("clf", X, y, y, feat, n_rep, group_cols=group_cols)
    grp = grp.rename(columns={"unit": "cluster"})
    grp["members"] = grp["cluster"].map(lambda g: "|".join(group_cols.get(g, [g])))
    grp.to_csv(os.path.join(SUMMARY, f"type_xai_group_{gran}.csv"), index=False)

    # 3) Volume correction (region-specific signal vs. total-lesion-volume proxy)
    volc = s8.volume_correction("clf", X, y, y, feat, n_rep)
    volc.to_csv(os.path.join(SUMMARY, f"type_xai_volume_correction_{gran}.csv"), index=False)

    print(f"  top8 OOF permutation importance ({gran}):", flush=True)
    for _, r in perm.head(8).iterrows():
        print(f"    {r['unit']:18s} imp={r['importance_mean']:+.4f}±{r['importance_std']:.4f} "
              f"pos_folds={r['frac_folds_positive']:.0%} collin=[{r['collinear_with_top']}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quick check with 5x2 CV")
    ap.add_argument("--coarse-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(SUMMARY, exist_ok=True)
    n_rep = 2 if args.quick else N_REPEATS
    print(f"fluent/non-fluent XAI  outer=5x{n_rep}  interpretation=predictive association (not causal)", flush=True)
    t0 = time.time()
    run("arterial_coarse", TABLES["arterial_coarse"], n_rep)
    if not args.coarse_only:
        run("JHU_fine", TABLES["JHU_fine"], n_rep)
    print(f"\nDONE in {time.time()-t0:.0f}s -> results/summary/type_xai_*.csv", flush=True)
    print("[interpretation limits] predictive association, not causal; single regions within a collinear cluster should not be interpreted alone; volume correction removes only the linear component.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
