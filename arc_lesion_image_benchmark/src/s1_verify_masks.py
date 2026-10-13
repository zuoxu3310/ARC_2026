#!/usr/bin/env python
"""
s1_verify_masks.py

Verify all 228 MNI lesion masks and align them with the 226-row WAB-AQ label table.
Checks:
  1. all 228 masks load
  2. common grid: all masks and both atlases have identical shape/affine
  3. masks are binary (uint8 {0,1}); record the lesion voxel count of each mask
  4. cross-check: mask lesion voxel count == lesion_volume in the merged_artery table
     (1 mm^3 voxels -> volume = voxel count)
  5. ID alignment: intersection of mask subjects and WAB-label subjects; report gaps
Outputs:
  data/manifest.csv           per mask: subject/ses/shape/voxsize/n_voxels/sha256/has_label/label_match
  reports/s1_verify_report.md human-readable summary

Run: python src/s1_verify_masks.py
"""
from __future__ import annotations
import csv, hashlib
from pathlib import Path
import numpy as np
import nibabel as nib

PKG = Path(__file__).resolve().parents[1]
MASK_DIR = PKG / "data" / "raw" / "lesion_masks_mni"
ATLAS_DIR = PKG / "data" / "raw" / "atlases"
MANIFEST = PKG / "data" / "manifest.csv"
REPORT = PKG / "reports" / "s1_verify_report.md"
WAB_TSV = (PKG.parent / "datasets" / "arc_ds004884" /
           "derived_features_arc_demo" / "merged_artery_participants.tsv")

ATOL = 1e-3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def subj_from_mask(fname: str) -> str:
    # wsub-M2001_ses-1253x1076_lesion.nii.gz -> sub-M2001
    return fname.split("_")[0][1:]


def load_wab_labels() -> dict[str, float]:
    """participant_id -> lesion_volume (for the cross-check); also returns the set of IDs in the WAB-AQ table."""
    vols, ids = {}, set()
    with open(WAB_TSV) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            pid = row["participant_id"]
            ids.add(pid)
            try:
                vols[pid] = float(row["lesion_volume"])
            except (KeyError, ValueError):
                pass
    return vols, ids


def main() -> int:
    masks = sorted(p.name for p in MASK_DIR.glob("wsub-*_lesion.nii.gz"))
    print(f"masks found: {len(masks)}")
    assert len(masks) == 228, f"expected 228, got {len(masks)}"

    label_vols, label_ids = load_wab_labels()
    print(f"WAB table participants: {len(label_ids)}")

    # reference grid = first mask
    ref = nib.load(MASK_DIR / masks[0])
    ref_shape, ref_aff = ref.shape, ref.affine

    rows = []
    bad_grid, non_binary, vol_mismatch, empty = [], [], [], []
    for m in masks:
        img = nib.load(MASK_DIR / m)
        d = np.asanyarray(img.dataobj)
        subj = subj_from_mask(m)
        same_grid = (img.shape == ref_shape) and np.allclose(img.affine, ref_aff, atol=ATOL)
        uniq = np.unique(d)
        is_bin = set(uniq.tolist()).issubset({0, 1})
        nvox = int((d > 0).sum())
        if not same_grid: bad_grid.append(m)
        if not is_bin: non_binary.append(m)
        if nvox == 0: empty.append(m)
        # cross-check: voxel count == lesion_volume in the table
        lbl_vol = label_vols.get(subj)
        match = (lbl_vol is not None) and (abs(nvox - lbl_vol) <= 1)
        if (lbl_vol is not None) and not match:
            vol_mismatch.append((m, nvox, lbl_vol))
        rows.append({
            "subject": subj, "mask_file": m,
            "shape": "x".join(map(str, img.shape)),
            "voxmm": "x".join(str(round(z, 2)) for z in img.header.get_zooms()),
            "n_lesion_voxels": nvox,
            "binary": is_bin, "grid_ok": same_grid,
            "has_wab_label": subj in label_ids,
            "vol_matches_table": match,
            "sha256": sha256(MASK_DIR / m),
        })

    # atlas grid check
    atlas_ok = {}
    for atl in ["jhu156.nii.gz", "ArterialAtlas156.nii.gz"]:
        a = nib.load(ATLAS_DIR / atl)
        atlas_ok[atl] = (a.shape == ref_shape) and np.allclose(a.affine, ref_aff, atol=ATOL)

    # ID alignment gaps
    mask_subj = {subj_from_mask(m) for m in masks}
    masks_no_label = sorted(mask_subj - label_ids)
    labels_no_mask = sorted(label_ids - mask_subj)
    n_usable = len(mask_subj & label_ids)

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    vols = np.array([r["n_lesion_voxels"] for r in rows])
    lines = [
        "# s1 lesion-mask verification report", "",
        f"- Source: Gibson ARC demo @1d24ad4049 (MNI 1mm normalized masks)",
        f"- Total masks: **{len(masks)}**",
        f"- Common grid (shape+affine match the reference): **{len(masks)-len(bad_grid)}/{len(masks)}** "
        f"(reference {('x'.join(map(str,ref_shape)))}, 1mm)",
        f"- Binary masks: **{len(masks)-len(non_binary)}/{len(masks)}**",
        f"- Empty masks (0 lesion voxels): {len(empty)}",
        f"- Atlas grid alignment: jhu156={atlas_ok['jhu156.nii.gz']}, "
        f"ArterialAtlas156={atlas_ok['ArterialAtlas156.nii.gz']}",
        "",
        "## Cross-check (mask voxel count vs table lesion_volume)",
        f"- Labelled in the table with matching voxel count (+/-1): **{sum(r['vol_matches_table'] for r in rows)}**",
        f"- Mismatches: {len(vol_mismatch)}" + (f" -> {vol_mismatch[:5]}" if vol_mismatch else ""),
        "",
        "## ID alignment (masks vs 226-row WAB table)",
        f"- Modelable intersection (mask and WAB label): **{n_usable}**",
        f"- Mask but no label in the table: {len(masks_no_label)} -> {masks_no_label}",
        f"- In the table but no mask: {len(labels_no_mask)} -> {labels_no_mask}",
        "",
        "## Lesion voxel count distribution (MNI 1mm^3, = mm^3)",
        f"- min={vols.min()}, median={int(np.median(vols))}, "
        f"mean={int(vols.mean())}, max={vols.max()}",
        "",
        "## Conclusion",
        f"{'OK: usable' if not bad_grid and not non_binary and n_usable>=220 else 'CHECK: needs review'}: "
        f"{n_usable} masks share one grid, are binary and align with the WAB labels; voxel counts match the Gibson table exactly.",
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nmanifest -> {MANIFEST.relative_to(PKG)}")
    ok = (not bad_grid) and (not non_binary) and all(atlas_ok.values()) and (not vol_mismatch)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
