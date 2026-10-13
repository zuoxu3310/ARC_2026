#!/usr/bin/env python
"""
s2_extract_atlas_features.py

Reimplements the method of Gibson's lesion2artery.py (per-region proportional damage =
voxels in lesion AND region / voxels in region), but with the atlas as a parameter,
defaulting to the fine jhu156 atlas (189 regions). Vectorised with np.bincount, faster
than the original per-region loop.

Relative to the Gibson 0.71 table, the only change is atlas granularity
(33 arterial regions -> 189 anatomical regions); extraction method, cleaning (left to
the harness) and leave-one-out protocol are unchanged -> fair comparison.

Outputs (written to this package's data/):
  features_<atlas>.tsv          participant_id, lesion_volume, <per-region proportional damage>
  <atlas>_features_merged.tsv   the table above + age_at_stroke + wab_aq (inner join with
                                merged_artery, 226 rows); same column layout as
                                merged_artery, so it can be fed directly to the benchmark harness

Run: python src/s2_extract_atlas_features.py [--atlas jhu156|ArterialAtlas156]
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
MASK_DIR = PKG / "data" / "raw" / "lesion_masks_mni"
ATLAS_DIR = PKG / "data" / "raw" / "atlases"
LABELS_TSV = (PKG.parent / "datasets" / "arc_ds004884" /
              "derived_features_arc_demo" / "merged_artery_participants.tsv")


def load_labels(txt_path: Path) -> dict[int, str]:
    """jhu156.txt / ArterialAtlas156.txt: 'idx|abbrev|full|...'; uses Gibson's 'idx_abbrev' naming."""
    out = {}
    for line in txt_path.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("|")
        k = int(cols[0].strip())
        out[k] = f"{k}_{cols[1].strip()}"
    return out


def subj_from_mask(fname: str) -> str:
    return fname.split("_")[0][1:]   # wsub-M2001_... -> sub-M2001


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="jhu156",
                    choices=["jhu156", "ArterialAtlas156",
                             "aal116", "brodmann41", "aicha384"])
    args = ap.parse_args()

    atlas_img = nib.load(ATLAS_DIR / f"{args.atlas}.nii.gz")
    atlas = np.asanyarray(atlas_img.dataobj).astype(np.int32)
    labels = load_labels(ATLAS_DIR / f"{args.atlas}.txt")
    max_lab = int(atlas.max())
    region_sizes = np.bincount(atlas.ravel(), minlength=max_lab + 1).astype(float)
    cols = [labels.get(r, f"{r}_R{r}") for r in range(1, max_lab + 1)]
    print(f"atlas={args.atlas}: {max_lab} labels, {(region_sizes[1:]>0).sum()} non-empty")

    masks = sorted(p.name for p in MASK_DIR.glob("wsub-*_lesion.nii.gz"))
    assert len(masks) == 228, f"expected 228 masks, got {len(masks)}"
    assert atlas.shape == nib.load(MASK_DIR / masks[0]).shape, "atlas/mask grid mismatch"

    rows = []
    for m in masks:
        d = np.asanyarray(nib.load(MASK_DIR / m).dataobj)
        lesioned = d > 0
        counts = np.bincount(atlas[lesioned].ravel(), minlength=max_lab + 1).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            injury = np.where(region_sizes > 0, counts / region_sizes, 0.0)
        row = {"participant_id": subj_from_mask(m),
               "lesion_volume": int(lesioned.sum())}
        for r in range(1, max_lab + 1):
            row[cols[r - 1]] = injury[r]
        rows.append(row)

    feats = pd.DataFrame(rows)
    feat_path = PKG / "data" / f"features_{args.atlas}.tsv"
    feats.to_csv(feat_path, sep="\t", index=False)

    # Attach age + wab_aq (from merged_artery), inner join -> 226-row modelling table
    lab = pd.read_csv(LABELS_TSV, sep="\t")[["participant_id", "age_at_stroke", "wab_aq"]]
    merged = feats.merge(lab, on="participant_id", how="inner")
    merged_path = PKG / "data" / f"{args.atlas}_features_merged.tsv"
    merged.to_csv(merged_path, sep="\t", index=False)

    print(f"features  -> {feat_path.relative_to(PKG)}  ({feats.shape[0]} x {feats.shape[1]})")
    print(f"merged    -> {merged_path.relative_to(PKG)}  ({merged.shape[0]} x {merged.shape[1]})")
    print(f"modelling rows (with wab_aq): {merged.shape[0]}; feature columns (incl. lesion_volume+age): {merged.shape[1]-2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
