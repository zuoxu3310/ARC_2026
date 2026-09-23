#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s21_range_restriction.py
========================
任务3 第 5 步: 回归量程截断处理。靶定义子集(aq_le90)在【靶】上截断 -> Pearson r
机械性下降。本脚本隔离"量程截断" vs "真成分变化"。

设计(refine-logs/TASK3_DESIGN_PREREG_2026-06-13.md §4):
  每子集报: target WAB-AQ SD + 原始 r + Thorndike 直接截断校正 r + MAE/RMSE。
  - 校正 r(直接对靶 y 截断的标准公式): R = r(S/s)/sqrt(1-r²+r²(S/s)²), S=全集靶SD, s=子集靶SD。
    若校正 r ≈ 全集 r -> 该子集 r 跌主要是量程截断, 非真退化。
  - MAE/RMSE 对量程截断稳健; 若子集摆幅在 MAE 上消失 -> 进一步证明是量程伪影。
  结论: 回归 headline 要求摆幅在 MAE 上也成立才下; 靶定义子集 r 跌标注为量程预期。

用法: PYENV_VERSION=data-analysis python src/s21_range_restriction.py
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
    """直接对选择变量截断的 r 校正(Pearson/Thorndike Case II 单变量直接限制)。"""
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
    # 每子集: 参考模型 RF 的逐种子均值 + 全快模型均值
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
    print(f"\n全集靶 SD={full_sd:.2f}", flush=True)
    # 重点看 aq_le90: raw_r 跌多少, 校正后回到哪, MAE 动没动
    if "aq_le90" in out["subset"].values:
        a = out[out["subset"] == "aq_le90"].iloc[0]
        drop = full["raw_r_RF"] - a["raw_r_RF"]                 # 总 r 跌幅
        recovered = a["thorndike_corrected_r_RF"] - a["raw_r_RF"]  # 校正追回
        frac_rr = recovered / drop if drop > 0 else float("nan")  # 量程截断占比
        mae_worse = a["delta_mae_vs_full"] > 0.3                # MAE 是否显著变差
        print(f"[aq_le90] raw_r {a['raw_r_RF']:.3f}(Δ{a['delta_raw_r_vs_full']:+.3f}) "
              f"-> 校正 {a['thorndike_corrected_r_RF']:.3f}(离全集仍 {a['delta_corrected_r_vs_full']:+.3f}); "
              f"MAE {a['mae_RF']:.2f}(Δ{a['delta_mae_vs_full']:+.2f})", flush=True)
        print(f"  解读(数据驱动): 量程截断只解释 r 跌的 {frac_rr*100:.0f}%(校正追回 {recovered:+.3f}/总跌 {drop:.3f}); "
              f"MAE {'变差' if mae_worse else '基本不动'} => 剩余跌幅是【真退化】"
              f"(丢天花板易测个案、剩中段难病人), 非纯量程伪影", flush=True)
    print(f"\n[write] range_restriction.csv", flush=True)


if __name__ == "__main__":
    main()
