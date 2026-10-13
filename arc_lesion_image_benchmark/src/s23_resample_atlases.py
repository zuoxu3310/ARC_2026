#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s23_resample_atlases.py
=======================
Resample downloaded standard MNI152NLin6 1 mm atlases (AAL116 / Brodmann41 / AICHA384)
onto this project's non-standard grid (157x189x156, SPM-w affine) so that s2 can extract
lesion-load features.

Source (neuroparc, 182x218x182, origin 90/-126/-72) and target (157x189x156,
origin 78/-112/-70) are both 1 mm with the same orientation -> generic affine
nearest-neighbour resampling (scipy map_coordinates order=0); here this is equivalent to
an integer crop with zero error.
Checks: resampled shape == target, matching affine, high overlap with JHU brain regions.
Writes a sidecar {name}.txt (idx|R{idx}|region_{idx}|idx); column names only need to be
unique and do not affect the lesion-load computation.

Usage: python src/s23_resample_atlases.py
"""
from __future__ import annotations
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATLAS_DIR = os.path.join(PKG, "data", "raw", "atlases")
TARGET = os.path.join(ATLAS_DIR, "jhu156.nii.gz")
# (source file name, output name)
JOBS = [("src_AAL.nii.gz", "aal116"),
        ("src_Brodmann.nii.gz", "brodmann41"),
        ("src_AICHAJoliot2015.nii.gz", "aicha384")]


def resample_nn(src_img, tgt_affine, tgt_shape):
    """For each target voxel: voxel -> world -> source voxel -> nearest-neighbour label."""
    src = np.asarray(src_img.dataobj).astype(np.int32)
    inv_src = np.linalg.inv(src_img.affine)
    # Target voxel grid
    ii, jj, kk = np.meshgrid(np.arange(tgt_shape[0]), np.arange(tgt_shape[1]),
                             np.arange(tgt_shape[2]), indexing="ij")
    ones = np.ones(ii.size)
    vox = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), ones])
    world = tgt_affine @ vox
    src_vox = inv_src @ world
    coords = src_vox[:3, :]            # [3, N] continuous source voxel coordinates
    out = map_coordinates(src, coords, order=0, mode="constant", cval=0)
    return out.reshape(tgt_shape).astype(np.int32)


def main():
    tgt = nib.load(TARGET)
    tgt_aff, tgt_shape = tgt.affine, tgt.shape
    jhu = np.asarray(tgt.dataobj)
    brain = jhu > 0                    # JHU brain regions as the overlap reference

    for srcf, name in JOBS:
        src_img = nib.load(os.path.join(ATLAS_DIR, srcf))
        res = resample_nn(src_img, tgt_aff, tgt_shape)
        labs = np.unique(res); labs = labs[labs > 0]
        # Checks
        assert res.shape == tgt_shape, f"{name} shape mismatch"
        # Overlap: fraction of labelled voxels in the resampled atlas that fall inside JHU brain regions
        nz = res > 0
        overlap = float((nz & brain).sum() / nz.sum()) if nz.sum() else 0.0
        # Save with the target affine. The dtype must hold the largest label: AICHA goes up to 384,
        # and reusing the uint8 header of jhu156 would overflow and wrap (256->0, 257..384 folded
        # into 1..128, 384 regions collapsed into 255), so int16 is fixed.
        res16 = res.astype(np.int16)
        assert int(res16.max()) == int(res.max()), f"{name} label {int(res.max())} exceeds int16"
        out_img = nib.Nifti1Image(res16, tgt_aff)
        out_img.set_data_dtype(np.int16)
        nib.save(out_img, os.path.join(ATLAS_DIR, f"{name}.nii.gz"))
        # Sidecar labels (dense range 1..max)
        with open(os.path.join(ATLAS_DIR, f"{name}.txt"), "w") as f:
            for L in range(1, int(res.max()) + 1):
                f.write(f"{L}|R{L}|region_{L}|{L}\n")
        print(f"[{name:10s}] nlabels={len(labs)} max={int(res.max())} "
              f"shape={res.shape} overlap_with_JHU={overlap*100:.1f}% -> {name}.nii.gz", flush=True)

    print("\n[done] All three atlases resampled to the project grid. Next: s2 extraction (add the names to --atlas choices).", flush=True)


if __name__ == "__main__":
    main()
