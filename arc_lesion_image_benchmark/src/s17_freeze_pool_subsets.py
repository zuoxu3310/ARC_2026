#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s17_freeze_pool_subsets.py
==========================
任务3 第 1 步(预注册数字来源): 冻结 226 建模池, 机械算出每个"纳入标准定义子集"
的确切 participant_id 名单 + N + severe 先验 + target SD, 写 data/subsets/。

机械选择规则(非手挑): 子集 = 在 226 公开池上可忠实重建的、出现在被调研 ARC 论文里
的纳入标准全集。可重建性由列决定。每个子集映射到某研究成文标准或自然可得性标准。

设计依据: refine-logs/TASK3_DESIGN_PREREG_2026-06-13.md
红线: 子集按"标准"命名, 不按研究名; 研究名只当出处。不冒充重现谁的队列。

用法: PYENV_VERSION=data-analysis python src/s17_freeze_pool_subsets.py
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
CHRONIC_DAYS = 365     # 慢性 >= 1 年


def load_pool():
    """冻结池 = jhu156 特征表(226) = 有特征∩有掩码∩有 WAB-AQ。合并临床/模态。"""
    jhu = pd.read_csv(os.path.join(PKG, "data", "jhu156_features_merged.tsv"), sep="\t")
    modal = pd.read_csv(os.path.join(PKG, "data", "arc_modality_availability.tsv"), sep="\t")
    part = pd.read_csv(os.path.join(DS, "participants.tsv"), sep="\t")
    pool = jhu[["participant_id", "lesion_volume", "age_at_stroke", "wab_aq"]].copy()
    pool = pool.merge(modal[["participant_id", "anat", "dwi", "rest_fmri", "task_fmri", "flair"]],
                      on="participant_id", how="left")
    pool = pool.merge(part[["participant_id", "wab_days", "wab_type", "sex"]],
                      on="participant_id", how="left")
    # 左/右病灶负荷(供诊断, 不做子集——S5 已砍)
    lcols = [c for c in jhu.columns if c.endswith("_L")]
    rcols = [c for c in jhu.columns if c.endswith("_R")]
    pool["_left_load"] = jhu[lcols].sum(axis=1).values
    pool["_right_load"] = jhu[rcols].sum(axis=1).values
    return pool, part


def teghipco_ids():
    p = os.path.join(PKG, "results", "summary", "teghipco_label_crosscheck.csv")
    t = pd.read_csv(p)  # 逗号分隔
    return set(t["participant_id"].astype(str))


def subset_defs(pool):
    """返回 [(name, category, criterion_text, source_study, mask_or_idset)]。
    category: baseline | availability | clinical_threshold | target_defined | exact_persons
    """
    tids = teghipco_ids()
    aq = pool["wab_aq"]
    defs = [
        ("full", "baseline", "all modelable patients", "(baseline)",
         pd.Series(True, index=pool.index)),
        # 可得性定义(主 headline 用)
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
        # 临床阈值(主 headline 用)
        ("chronic_365", "clinical_threshold", "wab_days>=365 (chronic >= 1 year)", "chronic-stage studies",
         pool["wab_days"] >= 365),
        ("chronic_180", "clinical_threshold", "wab_days>=180 (chronic >= 6 months)", "chronic-stage studies",
         pool["wab_days"] >= 180),
        # 靶定义(单列, 量程截断单独处理)
        ("aq_le90", "target_defined", "wab_aq<=90 (drop near-ceiling cases)", "Challa-433",
         aq <= CEILING_THR),
        # 精确同人(锚点, 不进主随机效应)
        ("teghipco_idlist", "exact_persons", "exact published participant_id list (same persons, NOT same result/pipeline)", "Teghipco",
         pool["participant_id"].astype(str).isin(tids)),
    ]
    return defs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pool, part = load_pool()
    n_pool = len(pool)
    print(f"[pool] 冻结建模池 N={n_pool}  (uniq id={pool['participant_id'].nunique()})")

    # 245 vs 226 缺口对账(评审要求显式)
    feat_ids = set(pool["participant_id"].astype(str))
    all_ids = set(part["participant_id"].astype(str))
    dropped = sorted(all_ids - feat_ids)
    print(f"[reconcile] participants.tsv={len(all_ids)} 临床行, 池={len(feat_ids)}, "
          f"无特征/掩码丢弃={len(dropped)} 人")
    with open(os.path.join(OUT_DIR, "_dropped_no_feature.txt"), "w") as f:
        f.write("\n".join(dropped) + "\n")

    # modality/wab_days 合并缺失检查
    miss_mod = pool["dwi"].isna().sum()
    miss_days = pool["wab_days"].isna().sum()
    print(f"[merge] 缺 modality={miss_mod}, 缺 wab_days={miss_days}")
    assert miss_mod == 0 and miss_days == 0, "合并有缺失, 子集定义会错, 先修"

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
        # 写 ID 名单
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
    print(f"\n[write] {man_path}  ({len(man)} 子集)")

    # 子集间重叠(Jaccard)——非独立 level 要披露
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
    print(f"[write] subset_jaccard.csv (子集重叠, 披露非独立性)")

    # 左右脑负荷诊断(S5 死因留证)
    lr = pool[["_left_load", "_right_load"]]
    n_left_gt = int((pool["_left_load"] > pool["_right_load"]).sum())
    print(f"\n[S5-dead] sum(_L)>sum(_R) 命中 {n_left_gt}/{n_pool} 人; "
          f"mean L={lr['_left_load'].mean():.3f} R={lr['_right_load'].mean():.4f} "
          f"-> 左脑代理=整池, 无区分度, 已砍; '库无侧别标签'写进 limitations")


if __name__ == "__main__":
    main()
