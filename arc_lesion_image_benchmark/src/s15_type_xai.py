#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s15_type_xai.py — fluent/非流畅 分类的脑区贡献(泄露安全), 复用 s8 机制。

问题: 模型靠哪些脑区把"非流畅"(Broca/Global/TCM, 经典前部/额叶损伤)和"流畅"
(Wernicke/Conduction/Anomic, 后部/颞顶损伤)分开? 验证是否落在左额叶语言区,
呼应 s8 已有的严重度 XAI(左背侧语言通路)。

口径: 预测关联 predictive association, 非因果定位(同 s8)。
方法: 折外置换重要度(列级 + 共线簇组级) + 折内体积校正(区分脑区特异 vs 总病灶量代理),
      折内 scaler->SMOTE->RF, 外层重复分层 5x10。直接复用 s8 的 _oof_perm / collinearity /
      volume_correction(task='clf', 但 y/strat 传 fluent 标签)。

写: results/summary/type_xai_perm_{gran}.csv
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
import s8_xai_region_contribution as s8  # noqa: E402  复用折外置换/共线/体积校正

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
    """X(去 wab_aq/wab_type, fillna0, 丢稀疏列) + y(1=非流畅) + feat。"""
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
    print(f"n={len(X)}  非流畅={n_pos} 流畅={len(X)-n_pos}  features={len(feat)}", flush=True)

    # 共线簇(目标无关, 仅特征相关结构)
    cluster_df, group_cols, corr_df = s8.collinearity_clusters(X, feat)
    cluster_df.to_csv(os.path.join(SUMMARY, f"type_xai_collinearity_{gran}.csv"), index=False)

    # 1) 列级折外置换重要度(task='clf' -> scaler->SMOTE->RF; strat/y 用 fluent 标签)
    perm = s8._oof_perm("clf", X, y, y, feat, n_rep)
    perm = s8.annotate_collinear(perm, feat, corr_df, top_n=10)
    perm.to_csv(os.path.join(SUMMARY, f"type_xai_perm_{gran}.csv"), index=False)

    # 2) 组级(共线簇)置换重要度
    grp = s8._oof_perm("clf", X, y, y, feat, n_rep, group_cols=group_cols)
    grp = grp.rename(columns={"unit": "cluster"})
    grp["members"] = grp["cluster"].map(lambda g: "|".join(group_cols.get(g, [g])))
    grp.to_csv(os.path.join(SUMMARY, f"type_xai_group_{gran}.csv"), index=False)

    # 3) 体积校正(区分脑区特异 vs 总病灶量代理)
    volc = s8.volume_correction("clf", X, y, y, feat, n_rep)
    volc.to_csv(os.path.join(SUMMARY, f"type_xai_volume_correction_{gran}.csv"), index=False)

    print(f"  top8 折外置换重要度({gran}):", flush=True)
    for _, r in perm.head(8).iterrows():
        print(f"    {r['unit']:18s} imp={r['importance_mean']:+.4f}±{r['importance_std']:.4f} "
              f"pos_folds={r['frac_folds_positive']:.0%} collin=[{r['collinear_with_top']}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5x2 折快验")
    ap.add_argument("--coarse-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(SUMMARY, exist_ok=True)
    n_rep = 2 if args.quick else N_REPEATS
    print(f"fluent/非流畅 XAI  outer=5x{n_rep}  口径=预测关联(非因果)", flush=True)
    t0 = time.time()
    run("arterial_coarse", TABLES["arterial_coarse"], n_rep)
    if not args.coarse_only:
        run("JHU_fine", TABLES["JHU_fine"], n_rep)
    print(f"\nDONE in {time.time()-t0:.0f}s -> results/summary/type_xai_*.csv", flush=True)
    print("[解释边界] 预测关联, 非因果; 共线簇内单区不可单独解读; 体积校正只去线性成分。", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
