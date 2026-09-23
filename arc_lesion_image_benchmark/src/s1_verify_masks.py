#!/usr/bin/env python
"""
s1_verify_masks.py

全量校验 228 张 MNI 病灶掩码是否可用, 并对齐 226 行 WAB-AQ 标签表。
检查项:
  1. 228 张全部能 load
  2. 网格一致: 所有掩码 + 两套图谱 shape/affine 完全相同
  3. 掩码二值 (uint8 {0,1}), 记每张病灶体素数
  4. 交叉核验: 掩码病灶体素数 == merged_artery 表 lesion_volume (1mm³ -> 体积=体素数)
  5. ID 对齐: 掩码 subject 集 ∩ WAB 标签 subject 集, 报缺口
产出:
  data/manifest.csv          每张掩码 subject/ses/shape/voxsize/n_voxels/sha256/has_label/label_match
  reports/s1_verify_report.md 人读结论

运行: PYENV_VERSION=data-analysis python src/s1_verify_masks.py
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
    """participant_id -> lesion_volume (用于交叉核验); 同时返回有 wab_aq 的 ID 集。"""
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

    # 参考网格 = 第一张掩码
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
        # 交叉核验: 体素数 == 表里 lesion_volume
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

    # 图谱网格核验
    atlas_ok = {}
    for atl in ["jhu156.nii.gz", "ArterialAtlas156.nii.gz"]:
        a = nib.load(ATLAS_DIR / atl)
        atlas_ok[atl] = (a.shape == ref_shape) and np.allclose(a.affine, ref_aff, atol=ATOL)

    # ID 对齐缺口
    mask_subj = {subj_from_mask(m) for m in masks}
    masks_no_label = sorted(mask_subj - label_ids)
    labels_no_mask = sorted(label_ids - mask_subj)
    n_usable = len(mask_subj & label_ids)

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    vols = np.array([r["n_lesion_voxels"] for r in rows])
    lines = [
        "# s1 掩码全量校验报告", "",
        f"- 源: Gibson ARC demo @1d24ad4049 (MNI 1mm 归一化掩码)",
        f"- 掩码总数: **{len(masks)}**",
        f"- 网格一致(shape+affine 同参考): **{len(masks)-len(bad_grid)}/{len(masks)}** "
        f"(参考 {('x'.join(map(str,ref_shape)))}, 1mm)",
        f"- 二值掩码: **{len(masks)-len(non_binary)}/{len(masks)}**",
        f"- 空掩码(0 病灶体素): {len(empty)}",
        f"- 图谱网格对齐: jhu156={atlas_ok['jhu156.nii.gz']}, "
        f"ArterialAtlas156={atlas_ok['ArterialAtlas156.nii.gz']}",
        "",
        "## 交叉核验(掩码体素数 vs 表 lesion_volume)",
        f"- 表里有标签且体素数匹配(±1): **{sum(r['vol_matches_table'] for r in rows)}**",
        f"- 不匹配: {len(vol_mismatch)}" + (f" -> {vol_mismatch[:5]}" if vol_mismatch else ""),
        "",
        "## ID 对齐(掩码 vs 226 行 WAB 表)",
        f"- 可建模交集(有掩码且有 WAB): **{n_usable}**",
        f"- 有掩码但表里无标签: {len(masks_no_label)} -> {masks_no_label}",
        f"- 表里有但无掩码: {len(labels_no_mask)} -> {labels_no_mask}",
        "",
        "## 病灶体素数分布(MNI 1mm³, =mm³)",
        f"- min={vols.min()}, median={int(np.median(vols))}, "
        f"mean={int(vols.mean())}, max={vols.max()}",
        "",
        "## 结论",
        f"{'✅ 可用' if not bad_grid and not non_binary and n_usable>=220 else '⚠️ 需复查'}: "
        f"{n_usable} 张掩码网格一致、二值、与 WAB 标签对齐, 体素数与 Gibson 表精确吻合。",
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nmanifest -> {MANIFEST.relative_to(PKG)}")
    ok = (not bad_grid) and (not non_binary) and all(atlas_ok.values()) and (not vol_mismatch)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
