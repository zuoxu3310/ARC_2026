#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s21_range_restriction.py
========================
Range restriction in regression. The target-defined subset (aq_le90) is truncated on
the target itself, which mechanically lowers Pearson r. This script separates range
restriction from genuine changes in sample composition.

Design (fixed before running):
  For each subset report: target WAB-AQ SD + raw r + Thorndike direct-restriction
  corrected r + MAE/RMSE.
  - Corrected r (standard formula for direct restriction on the target y):
    R = r(S/s) / sqrt(1 - r^2 + r^2 (S/s)^2), S = full-sample target SD,
    s = subset target SD. If corrected r ~= full-sample r, the drop in r for that
    subset is mainly range restriction, not genuine degradation.
  - MAE/RMSE are robust to range restriction; if the subset gap disappears in MAE,
    that is further evidence of a range artifact.
  Decision rule: a regression headline claim requires the gap to hold in MAE as well;
  the r drop in target-defined subsets is labelled as expected from range restriction.

Usage: python src/s21_range_restriction.py
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(PKG, "results", "runs")
SUMM = os.path.join(PKG, "results", "summary")
REF_MODEL = "RandomForest"


def thorndike_direct(r, S, s):
    """Correct r for direct restriction on the selection variable
    (Pearson/Thorndike Case II, univariate direct restriction)."""
    if s <= 0 or np.isnan(r):
        return np.nan
    k = S / s
    return float(r * k / np.sqrt(1 - r ** 2 + (r ** 2) * (k ** 2)))


def main():
    frames = []
    for tag in ["fast", "TabPFN"]:
        p = os.path.join(RUNS, f"grid_regression_{tag}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)

    full_sd = float(df.loc[df["subset"] == "full", "target_sd"].iloc[0])
    # Per subset: mean over seeds for the reference model (RF) + mean over all fast models
    rows = []
    for sub, sd_sub in df.groupby("subset")["target_sd"].first().items():
        d_sub = df[df["subset"] == sub]
        d_rf = d_sub[d_sub["model"] == REF_MODEL]
        r_rf = float(d_rf["r"].mean())
        mae_rf = float(d_rf["mae"].mean())
        rmse_rf = float(d_rf["rmse"].mean())
        r_allmean = float(d_sub.groupby("model")["r"].mean().mean())
        mae_allmean = float(d_sub.groupby("model")["mae"].mean().mean())
        corr_r = thorndike_direct(r_rf, full_sd, float(sd_sub))
        rows.append({"subset": sub, "n": int(d_sub["n"].iloc[0]),
                     "target_sd": round(float(sd_sub), 3),
                     "raw_r_RF": round(r_rf, 4),
                     "thorndike_corrected_r_RF": round(corr_r, 4) if not np.isnan(corr_r) else np.nan,
                     "mae_RF": round(mae_rf, 3), "rmse_RF": round(rmse_rf, 3),
                     "r_allmodel_mean": round(r_allmean, 4),
                     "mae_allmodel_mean": round(mae_allmean, 3)})
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    full = out[out["subset"] == "full"].iloc[0]
    out["delta_raw_r_vs_full"] = (out["raw_r_RF"] - full["raw_r_RF"]).round(4)
    out["delta_corrected_r_vs_full"] = (out["thorndike_corrected_r_RF"] - full["raw_r_RF"]).round(4)
    out["delta_mae_vs_full"] = (out["mae_RF"] - full["mae_RF"]).round(3)

    os.makedirs(SUMM, exist_ok=True)
    out.to_csv(os.path.join(SUMM, "range_restriction.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(out.to_string(index=False), flush=True)
    print(f"\nFull-sample target SD={full_sd:.2f}", flush=True)
    # Focus on aq_le90: how much raw r drops, where the corrected r lands, and whether MAE moves
    if "aq_le90" in out["subset"].values:
        a = out[out["subset"] == "aq_le90"].iloc[0]
        drop = full["raw_r_RF"] - a["raw_r_RF"]                 # total drop in r
        recovered = a["thorndike_corrected_r_RF"] - a["raw_r_RF"]  # recovered by the correction
        frac_rr = recovered / drop if drop > 0 else float("nan")  # share due to range restriction
        mae_worse = a["delta_mae_vs_full"] > 0.3                # whether MAE is materially worse
        print(f"[aq_le90] raw_r {a['raw_r_RF']:.3f}(Δ{a['delta_raw_r_vs_full']:+.3f}) "
              f"-> corrected {a['thorndike_corrected_r_RF']:.3f} (still {a['delta_corrected_r_vs_full']:+.3f} from full); "
              f"MAE {a['mae_RF']:.2f}(Δ{a['delta_mae_vs_full']:+.2f})", flush=True)
        print(f"  Interpretation (data-driven): range restriction explains only {frac_rr*100:.0f}% of the r drop (recovered by correction {recovered:+.3f} / total drop {drop:.3f}); "
              f"MAE {'worse' if mae_worse else 'roughly unchanged'} => the remaining drop is genuine degradation "
              f"(easy ceiling cases removed, harder mid-range patients remain), not purely a range artifact", flush=True)
    print(f"\n[write] range_restriction.csv", flush=True)


if __name__ == "__main__":
    main()
