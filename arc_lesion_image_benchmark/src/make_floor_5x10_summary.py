#!/usr/bin/env python
"""把 5x10 同协议地板线(floor=lesion_volume+age_at_stroke 两列)整理成对外汇总。

读 results/tuning/nested_tuning_floor.csv(由 s5_nested_tuning_regression.py 在
重复分层 5x10 折下跑出, 与主表同协议, 非留一)。对每个模型取 default / tuned 的
Pearson r, 再用 5x10 的 ceiling(JHU_fine RandomForest: default 0.704 / tuned 0.7152)
做分母, 算 floor 占 ceiling 的百分比。

输出 results/summary/floor_5x10.csv:
  每行一个模型, 列出 floor default/tuned r 及其各自相对两个 ceiling 口径的百分比;
  外加一行 BEST_FLOOR(跨所有模型取最高的 floor r)。
"""
from __future__ import annotations
import os
import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PKG, "results", "tuning", "nested_tuning_floor.csv")
OUT = os.path.join(PKG, "results", "summary", "floor_5x10.csv")

# 5x10 同协议 ceiling(分母): JHU_fine RandomForest, 与主表口径一致。
CEIL_DEFAULT = 0.704      # RF default, 5x10, JHU
CEIL_TUNED = 0.7152       # RF tuned,   5x10, JHU
# 备查: arterial_coarse RF 5x10 default 0.6852 / tuned 0.6954


def pct(num, den):
    return round(100.0 * num / den, 1)


def main():
    df = pd.read_csv(SRC)
    rows = []
    for _, r in df.iterrows():
        model = r["model"]
        d_r = r.get("default_r")
        t_r = r.get("tuned_r")
        # floor 该模型的最佳 r(default vs tuned 取大; TabPFN 无 tuned)
        cands = [v for v in (d_r, t_r) if pd.notna(v)]
        best_r = max(cands)
        rows.append({
            "model": model,
            "floor_default_r": d_r,
            "floor_tuned_r": t_r if pd.notna(t_r) else "",
            "floor_best_r": round(best_r, 4),
            # 占 5x10 ceiling 的百分比(同协议)
            "pct_of_ceiling_RFdefault_0.704": pct(best_r, CEIL_DEFAULT),
            "pct_of_ceiling_RFtuned_0.7152": pct(best_r, CEIL_TUNED),
        })

    out = pd.DataFrame(rows)

    # BEST_FLOOR 汇总行: 跨所有模型最高 floor r
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
