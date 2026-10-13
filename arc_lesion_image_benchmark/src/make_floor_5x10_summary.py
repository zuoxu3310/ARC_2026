#!/usr/bin/env python
"""Summarise the same-protocol 5x10 floor baseline (floor = lesion_volume + age_at_stroke) for reporting.

Reads results/tuning/nested_tuning_floor.csv (produced by s5_nested_tuning_regression.py under
repeated stratified 5x10 CV, same protocol as the main table, not leave-one-out). For each model,
takes the default / tuned Pearson r and divides by the 5x10 ceiling (JHU_fine RandomForest:
default 0.704 / tuned 0.7152) to express the floor as a percentage of the ceiling.

Output results/summary/floor_5x10.csv:
  one row per model with the floor default/tuned r and its percentage of each of the two
  ceiling definitions; plus one BEST_FLOOR row (highest floor r across all models).
"""
from __future__ import annotations
import os
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PKG, "results", "tuning", "nested_tuning_floor.csv")
OUT = os.path.join(PKG, "results", "summary", "floor_5x10.csv")

# 5x10 same-protocol ceiling (denominator): JHU_fine RandomForest, as in the main table.
CEIL_DEFAULT = 0.704      # RF default, 5x10, JHU
CEIL_TUNED = 0.7152       # RF tuned,   5x10, JHU
# For reference: arterial_coarse RF 5x10 default 0.6852 / tuned 0.6954


def pct(num, den):
    return round(100.0 * num / den, 1)


def main():
    df = pd.read_csv(SRC)
    rows = []
    for _, r in df.iterrows():
        model = r["model"]
        d_r = r.get("default_r")
        t_r = r.get("tuned_r")
        # best floor r for this model (max of default and tuned; TabPFN has no tuned run)
        cands = [v for v in (d_r, t_r) if pd.notna(v)]
        best_r = max(cands)
        rows.append({
            "model": model,
            "floor_default_r": d_r,
            "floor_tuned_r": t_r if pd.notna(t_r) else "",
            "floor_best_r": round(best_r, 4),
            # percentage of the 5x10 ceiling (same protocol)
            "pct_of_ceiling_RFdefault_0.704": pct(best_r, CEIL_DEFAULT),
            "pct_of_ceiling_RFtuned_0.7152": pct(best_r, CEIL_TUNED),
        })

    out = pd.DataFrame(rows)

    # BEST_FLOOR summary row: highest floor r across all models
    best_idx = out["floor_best_r"].idxmax()
    best_model = out.loc[best_idx, "model"]
    best_floor = out.loc[best_idx, "floor_best_r"]
    summary_row = {
        "model": f"BEST_FLOOR ({best_model})",
        "floor_default_r": "",
        "floor_tuned_r": "",
        "floor_best_r": best_floor,
        "pct_of_ceiling_RFdefault_0.704": pct(best_floor, CEIL_DEFAULT),
        "pct_of_ceiling_RFtuned_0.7152": pct(best_floor, CEIL_TUNED),
    }
    out = pd.concat([out, pd.DataFrame([summary_row])], ignore_index=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"ceiling (5x10, JHU RF): default={CEIL_DEFAULT}  tuned={CEIL_TUNED}")
    print(out.to_string(index=False))
    print(f"\nsummary -> {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
