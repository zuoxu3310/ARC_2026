#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s19b_variance_decomp_with_tabpfn.py
===================================
一次性配套脚本(不改 s19 主逻辑): 把 TabPFN 也算进【竞争模型 scope C】, 报方差分解。

s19 的 COMP 列表里没有 TabPFN, 所以 s19 重跑后 TabPFN 只进 scope A(全子集)/B(主集),
不进 scope C(竞争模型)。本脚本把 TabPFN 加进 COMP, 专门产出"竞争 scope WITH TabPFN"
那一行, 写到单独文件, 不覆盖 s19 的 canonical 输出。

直接复用 s19 的 load_grid / eta_sq / var_components / boot_eta_diff / leave_one_out / run_one。

用法: PYENV_VERSION=data-analysis python src/s19b_variance_decomp_with_tabpfn.py --task regression
"""
from __future__ import annotations
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import s19_variance_decomp as s19

SUMM = s19.SUMM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    args = ap.parse_args()
    df = s19.load_grid(args.task)

    has_tabpfn = "TabPFN" in set(df["model"].unique())
    print(f"[s19b:{args.task}] grid models = {sorted(df['model'].unique())}  TabPFN present? {has_tabpfn}", flush=True)

    # 主集 = baseline + availability + clinical_threshold (与 s19 scope B 同口径)
    primary = df[df["category"].isin(["baseline", "availability", "clinical_threshold"])]

    # 竞争模型 + TabPFN
    COMP_BASE = {"regression": ["RandomForest", "XGBoost", "LightGBM"],
                 "classification": ["RandomForest", "XGBoost", "LightGBM", "LogReg"]}[args.task]
    COMP_WITH = COMP_BASE + ["TabPFN"]

    comp_df = primary[primary["model"].isin(COMP_WITH)]
    print(f"  competitive-with-TabPFN models = {sorted(comp_df['model'].unique())}", flush=True)

    rowC, loo_sC, loo_mC = s19.run_one(comp_df, args.task, "primary_competitive_models_WITH_TabPFN")

    out = pd.DataFrame([rowC])
    os.makedirs(SUMM, exist_ok=True)
    out_path = os.path.join(SUMM, f"variance_decomposition_{args.task}_competitive_with_tabpfn.csv")
    out.to_csv(out_path, index=False)
    loo_sC.to_csv(os.path.join(SUMM, f"variance_loo_subset_competitive_with_tabpfn_{args.task}.csv"), index=False)
    loo_mC.to_csv(os.path.join(SUMM, f"variance_loo_model_competitive_with_tabpfn_{args.task}.csv"), index=False)
    print(f"\n[write] {out_path}", flush=True)


if __name__ == "__main__":
    main()
