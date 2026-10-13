#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s17_freeze_pool_subsets.py
==========================
Step 1 of the inclusion-criterion subset analysis: freeze the 226-patient modelling pool and
mechanically compute, for each "inclusion-criterion subset", the exact participant_id list + N +
severe prior + target SD; write them to data/subsets/.

Mechanical selection rule (no hand-picking): the subsets are all inclusion criteria from the
surveyed ARC papers that can be faithfully rebuilt on the 226-patient public pool. Whether a
criterion can be rebuilt is decided by the available columns. Each subset maps to a published
study criterion or to a natural data-availability criterion.

The subset definitions were fixed before any modelling.
Naming rule: subsets are named after the criterion, not after a study; study names are given
only as the source. No subset claims to reproduce any study's cohort.

Usage: python src/s17_freeze_pool_subsets.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(PKG, ".."))
DS = os.path.join(ROOT, "datasets", "arc_ds004884")
OUT_DIR = os.path.join(PKG, "data", "subsets")
SEVERE_THR = 50.0      # severe = wab_aq <= 50
CEILING_THR = 90.0     # Challa-433: drop wab_aq > 90
CHRONIC_DAYS = 365     # chronic >= 1 year


def load_pool():
    """Frozen pool = jhu156 feature table (226) = has features and mask and WAB-AQ. Merges clinical/modality columns."""
    jhu = pd.read_csv(os.path.join(PKG, "data", "jhu156_features_merged.tsv"), sep="\t")
    modal = pd.read_csv(os.path.join(PKG, "data", "arc_modality_availability.tsv"), sep="\t")
    part = pd.read_csv(os.path.join(DS, "participants.tsv"), sep="\t")
    pool = jhu[["participant_id", "lesion_volume", "age_at_stroke", "wab_aq"]].copy()
    pool = pool.merge(modal[["participant_id", "anat", "dwi", "rest_fmri", "task_fmri", "flair"]],
                      on="participant_id", how="left")
    pool = pool.merge(part[["participant_id", "wab_days", "wab_type", "sex"]],
                      on="participant_id", how="left")
    # left/right lesion load (diagnostic only; no laterality subset is built from it)
    lcols = [c for c in jhu.columns if c.endswith("_L")]
    rcols = [c for c in jhu.columns if c.endswith("_R")]
    pool["_left_load"] = jhu[lcols].sum(axis=1).values
    pool["_right_load"] = jhu[rcols].sum(axis=1).values
    return pool, part


def teghipco_ids():
    p = os.path.join(PKG, "results", "summary", "teghipco_label_crosscheck.csv")
    t = pd.read_csv(p)  # comma-separated
    return set(t["participant_id"].astype(str))


def subset_defs(pool):
    """Return [(name, category, criterion_text, source_study, mask_or_idset)].
    category: baseline | availability | clinical_threshold | target_defined | exact_persons
    """
    tids = teghipco_ids()
    aq = pool["wab_aq"]
    defs = [
        ("full", "baseline", "all modelable patients", "(baseline)",
         pd.Series(True, index=pool.index)),
        # availability criteria (used in the main headline analysis)
        ("has_dwi", "availability", "dwi==1 (requires diffusion MRI)", "GenBrain",
         pool["dwi"] == 1),
        ("has_rsfmri", "availability", "rest_fmri==1 (rsfMRI is a REWIRED prerequisite; their curated N=129 NOT reproducible, no public list)", "REWIRED",
         pool["rest_fmri"] == 1),
        ("has_taskfmri", "availability", "task_fmri==1 (requires task fMRI / naming task)", "task-fMRI studies",
         pool["task_fmri"] == 1),
        ("has_flair", "availability", "flair==1 (requires FLAIR)", "FLAIR lesion studies",
         pool["flair"] == 1),
        ("multimodal_complete", "availability", "dwi==1 & rest_fmri==1 & task_fmri==1 (multimodal-complete)", "multimodal studies (e.g. REWIRED-style)",
         (pool["dwi"] == 1) & (pool["rest_fmri"] == 1) & (pool["task_fmri"] == 1)),
        # clinical thresholds (used in the main headline analysis)
        ("chronic_365", "clinical_threshold", "wab_days>=365 (chronic >= 1 year)", "chronic-stage studies",
         pool["wab_days"] >= 365),
        ("chronic_180", "clinical_threshold", "wab_days>=180 (chronic >= 6 months)", "chronic-stage studies",
         pool["wab_days"] >= 180),
        # target-defined (kept separate: it truncates the target range)
        ("aq_le90", "target_defined", "wab_aq<=90 (drop near-ceiling cases)", "Challa-433",
         aq <= CEILING_THR),
        # exact same persons (anchor; excluded from the main random-effects analysis)
        ("teghipco_idlist", "exact_persons", "exact published participant_id list (same persons, NOT same result/pipeline)", "Teghipco",
         pool["participant_id"].astype(str).isin(tids)),
    ]
    return defs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pool, part = load_pool()
    n_pool = len(pool)
    print(f"[pool] frozen modelling pool N={n_pool}  (uniq id={pool['participant_id'].nunique()})")

    # explicit reconciliation of the 245 vs 226 gap
    feat_ids = set(pool["participant_id"].astype(str))
    all_ids = set(part["participant_id"].astype(str))
    dropped = sorted(all_ids - feat_ids)
    print(f"[reconcile] participants.tsv={len(all_ids)} clinical rows, pool={len(feat_ids)}, "
          f"dropped (no features/mask)={len(dropped)} patients")
    with open(os.path.join(OUT_DIR, "_dropped_no_feature.txt"), "w") as f:
        f.write("\n".join(dropped) + "\n")

    # check for missing modality/wab_days after the merge
    miss_mod = pool["dwi"].isna().sum()
    miss_days = pool["wab_days"].isna().sum()
    print(f"[merge] missing modality={miss_mod}, missing wab_days={miss_days}")
    assert miss_mod == 0 and miss_days == 0, "missing values after merge would corrupt the subset definitions; fix first"

    rows = []
    for name, cat, crit, src, mask in subset_defs(pool):
        sub = pool[mask.values]
        ids = sub["participant_id"].astype(str).tolist()
        n = len(ids)
        sev_prior = float((sub["wab_aq"] <= SEVERE_THR).mean())
        n_sev = int((sub["wab_aq"] <= SEVERE_THR).sum())
        tgt_sd = float(sub["wab_aq"].std(ddof=1))
        tgt_min = float(sub["wab_aq"].min())
        tgt_max = float(sub["wab_aq"].max())
        # write the ID list
        with open(os.path.join(OUT_DIR, f"{name}_ids.txt"), "w") as f:
            f.write("\n".join(ids) + "\n")
        rows.append({
            "subset": name, "category": cat, "n": n,
            "n_severe": n_sev, "severe_prior": round(sev_prior, 4),
            "target_sd": round(tgt_sd, 3), "target_min": tgt_min, "target_max": tgt_max,
            "criterion": crit, "source_study": src,
        })
        print(f"  {name:20s} [{cat:17s}] N={n:3d}  severe={sev_prior*100:4.1f}%({n_sev})  "
              f"AQ_SD={tgt_sd:5.2f}  AQ=[{tgt_min:.1f},{tgt_max:.1f}]  <- {src}")

    man = pd.DataFrame(rows)
    man_path = os.path.join(OUT_DIR, "subset_manifest.csv")
    man.to_csv(man_path, index=False)
    print(f"\n[write] {man_path}  ({len(man)} subsets)")

    # overlap between subsets (Jaccard): the subsets are not independent, which is disclosed
    names = [r["subset"] for r in rows if r["subset"] not in ("full",)]
    idsets = {}
    for nm in names:
        with open(os.path.join(OUT_DIR, f"{nm}_ids.txt")) as f:
            idsets[nm] = set(x.strip() for x in f if x.strip())
    jac = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            inter = len(idsets[a] & idsets[b])
            union = len(idsets[a] | idsets[b])
            jac.loc[a, b] = round(inter / union, 3) if union else np.nan
    jac.to_csv(os.path.join(OUT_DIR, "subset_jaccard.csv"))
    print(f"[write] subset_jaccard.csv (subset overlap; documents non-independence)")

    # left/right lesion-load diagnostic (why no laterality subset is used)
    lr = pool[["_left_load", "_right_load"]]
    n_left_gt = int((pool["_left_load"] > pool["_right_load"]).sum())
    print(f"\n[laterality] sum(_L)>sum(_R) holds for {n_left_gt}/{n_pool} patients; "
          f"mean L={lr['_left_load'].mean():.3f} R={lr['_right_load'].mean():.4f} "
          f"-> left-hemisphere proxy covers the whole pool; not discriminative, dropped; the dataset has no laterality label")


if __name__ == "__main__":
    main()
