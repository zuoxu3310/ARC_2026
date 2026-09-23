#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s23_resample_atlases.py
=======================
任务2: 把下载的标准 MNI152NLin6 1mm 图谱(AAL116/Brodmann41/AICHA384)重采样到本项目
非标准网格(157×189×156, SPM-w 仿射), 供 s2 提取病灶负荷特征。

源(neuroparc, 182×218×182, origin 90/-126/-72)与目标(157×189×156, origin 78/-112/-70)
都是 1mm 同朝向 -> 通用仿射最近邻(scipy map_coordinates order=0), 此情形等价整数裁剪、零误差。
验证: 重采样后 shape==目标、affine 一致、与 JHU 脑区重合度高。
写 sidecar {name}.txt(idx|R{idx}|region_{idx}|idx), 列名只需唯一, 不影响负荷计算。

用法: PYENV_VERSION=data-analysis python src/s23_resample_atlases.py
"""
from __future__ import annotations
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATLAS_DIR = os.path.join(PKG, "data", "raw", "atlases")
TARGET = os.path.join(ATLAS_DIR, "jhu156.nii.gz")
# (源文件名, 输出名)
JOBS = [("src_AAL.nii.gz", "aal116"),
        ("src_Brodmann.nii.gz", "brodmann41"),
        ("src_AICHAJoliot2015.nii.gz", "aicha384")]


def resample_nn(src_img, tgt_affine, tgt_shape):
    """目标每体素 -> 世界 -> 源体素 -> 最近邻取标签。"""
    src = np.asarray(src_img.dataobj).astype(np.int32)
    inv_src = np.linalg.inv(src_img.affine)
    # 目标体素网格
    ii, jj, kk = np.meshgrid(np.arange(tgt_shape[0]), np.arange(tgt_shape[1]),
                             np.arange(tgt_shape[2]), indexing="ij")
    ones = np.ones(ii.size)
    vox = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), ones])
    world = tgt_affine @ vox
    src_vox = inv_src @ world
    coords = src_vox[:3, :]            # [3, N] 源连续体素坐标
    out = map_coordinates(src, coords, order=0, mode="constant", cval=0)
    return out.reshape(tgt_shape).astype(np.int32)


def main():
    tgt = nib.load(TARGET)
    tgt_aff, tgt_shape = tgt.affine, tgt.shape
    jhu = np.asarray(tgt.dataobj)
    brain = jhu > 0                    # JHU 脑区当重合参照

    for srcf, name in JOBS:
        src_img = nib.load(os.path.join(ATLAS_DIR, srcf))
        res = resample_nn(src_img, tgt_aff, tgt_shape)
        labs = np.unique(res); labs = labs[labs > 0]
        # 验证
        assert res.shape == tgt_shape, f"{name} shape 不符"
        # 重合度: 重采样图谱有标签的体素里, 落在 JHU 脑区内的比例
        nz = res > 0
        overlap = float((nz & brain).sum() / nz.sum()) if nz.sum() else 0.0
        # 保存(用目标仿射)。dtype 必须容得下最大标签:AICHA 到 384, 复用 jhu156 的
        # uint8 头会溢出回绕(256->0、257..384 并入 1..128, 384 区塌成 255), 故固定 int16。
        res16 = res.astype(np.int16)
        assert int(res16.max()) == int(res.max()), f"{name} 标签 {int(res.max())} 超出 int16"
        out_img = nib.Nifti1Image(res16, tgt_aff)
        out_img.set_data_dtype(np.int16)
        nib.save(out_img, os.path.join(ATLAS_DIR, f"{name}.nii.gz"))
        # sidecar labels(dense 1..max 里实际出现的)
        with open(os.path.join(ATLAS_DIR, f"{name}.txt"), "w") as f:
            for L in range(1, int(res.max()) + 1):
                f.write(f"{L}|R{L}|region_{L}|{L}\n")
        print(f"[{name:10s}] nlabels={len(labs)} max={int(res.max())} "
              f"shape={res.shape} 与JHU脑区重合={overlap*100:.1f}% -> {name}.nii.gz", flush=True)

    print("\n[done] 三图谱已重采样到本网格。下一步: s2 提取(把名字加进 --atlas choices)。", flush=True)


if __name__ == "__main__":
    main()
