#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s19b_variance_decomp_with_tabpfn.py
===================================
One-off companion script (does not change the s19 main logic): adds TabPFN to the
competitive-model scope C and reports the variance decomposition.

The s19 COMP list does not include TabPFN, so after an s19 run TabPFN only enters scope A
(all subsets) and scope B (primary subsets), not scope C (competitive models). This script adds
TabPFN to COMP and produces only the "competitive scope WITH TabPFN" row, written to a separate
file so the canonical s19 outputs are not overwritten.

Reuses s19's load_grid / eta_sq / var_components / boot_eta_diff / leave_one_out / run_one directly.

Usage: python src/s19b_variance_decomp_with_tabpfn.py --task regression
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

    # primary set = baseline + availability + clinical_threshold (same definition as s19 scope B)
    primary = df[df["category"].isin(["baseline", "availability", "clinical_threshold"])]

    # competitive models + TabPFN
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
