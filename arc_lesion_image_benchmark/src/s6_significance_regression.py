#!/usr/bin/env python
"""
s6_significance_regression.py  ——  回归【患者级配对显著性】检验

对比(都是"病灶单模态信息天花板"叙事的关键零假设, 看差的 95%CI 是否跨 0 / p 是否不显著):
  1) 细 vs 粗 (JHU_fine vs arterial_coarse), RandomForest:
       - Steiger's z: 比两个【相依】Pearson r(同一批人、同一 y)。
       - 患者级配对 bootstrap(B=2000, 对"人"重抽样): r 差 与 MAE 差 -> 差值+95%CI+双尾 p。
  2) tuned vs default (RandomForest, JHU_fine):
       - 患者级配对 bootstrap: r 差 与 MAE 差 -> 差值+95%CI+双尾 p。

铁律(逐条对照任务要求)
----------------------
- 用【患者级】逐人预测(每人一个值: reg_*_patient_pred.csv 的 y_pred_mean), 不用逐折,
  避免重复 CV 相关性把样本量灌水。
- 配对、同一批病人: 用 participant_id 做 inner merge 对齐; bootstrap 对【人】重抽样,
  同一次重抽样里两套预测取相同的人, 保证配对。
- 回归两个 r 是【相依相关】(同一批人、同一 y), 用 Steiger's z。
- 只输出每个对比的【原始 p + 差值 + 95%CI】, 不做多重比较校正(主控统一 Holm/FDR)。

输出: results/summary/significance_regression.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm, t as tdist
from sklearn.metrics import mean_absolute_error

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(PKG, "results", "runs")
OUT = os.path.join(PKG, "results", "summary", "significance_regression.csv")
B = 2000
SEED = 42


def load_pred(name, model, config):
    """读一个逐人 dump: participant_id, y_true, y_pred_mean。返回按 id 索引的 DataFrame。"""
    path = os.path.join(RUNS, f"reg_{name}_{model}_{config}_patient_pred.csv")
    df = pd.read_csv(path)
    return df.set_index("participant_id")


def align_two(a, b):
    """按 participant_id 取交集对齐两套预测, 校验 y_true 一致(同一批人、同一 y)。

    返回 (y_true[N], pred_a[N], pred_b[N], ids[N])。
    """
    ids = a.index.intersection(b.index)
    ids = ids.sort_values()                       # 固定顺序, 可复现
    aa = a.loc[ids]
    bb = b.loc[ids]
    # 同一批人、同一 y: 两边 y_true 必须一致(允许浮点误差)
    dy = np.abs(aa["y_true"].to_numpy() - bb["y_true"].to_numpy()).max()
    assert dy < 1e-6, f"y_true 不一致, max abs diff={dy}"
    y = aa["y_true"].to_numpy(dtype=float)
    pa = aa["y_pred_mean"].to_numpy(dtype=float)
    pb = bb["y_pred_mean"].to_numpy(dtype=float)
    return y, pa, pb, list(ids)


def steiger_z(y, pa, pb):
    """Steiger's z (1980): 两个相依相关 r(y,pa) 与 r(y,pb) 之差的检验。

    共享变量 y, 比较 r_ya 与 r_yb; 还需 r_ab(两套预测之间的相关)。
    返回 (r_ya, r_yb, z, p_two_sided)。
    """
    n = len(y)
    r_ya = pearsonr(y, pa)[0]
    r_yb = pearsonr(y, pb)[0]
    r_ab = pearsonr(pa, pb)[0]
    # Steiger 1980, eq for dependent correlations sharing one variable:
    # determinant of the 3x3 correlation matrix
    det = (1 - r_ya**2 - r_yb**2 - r_ab**2 + 2 * r_ya * r_yb * r_ab)
    rbar = (r_ya + r_yb) / 2.0
    # t-statistic form (Steiger 1980, Williams' test variant for dependent r)
    num = (r_ya - r_yb) * np.sqrt((n - 1) * (1 + r_ab))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * det + (rbar**2) * ((1 - r_ab)**3))
    tstat = num / den
    p = 2 * tdist.sf(np.abs(tstat), df=n - 3)
    return float(r_ya), float(r_yb), float(tstat), float(p)


def paired_bootstrap_diff(y, pa, pb, b=B, seed=SEED):
    """患者级配对 bootstrap: 对【人】重抽样, 同一次抽样里两套预测取相同的人。

    返回两个 metric 差(a-b)的 dict:
      r_diff   = r(y,pa) - r(y,pb)
      mae_diff = MAE(y,pa) - MAE(y,pb)
    每个含 point(全样本观测差), ci(2.5/97.5 百分位), p(双尾, 基于 bootstrap 分布跨 0)。
    """
    n = len(y)
    rng = np.random.default_rng(seed)

    def metrics(yy, qa, qb):
        r_diff = pearsonr(yy, qa)[0] - pearsonr(yy, qb)[0]
        mae_diff = mean_absolute_error(yy, qa) - mean_absolute_error(yy, qb)
        return r_diff, mae_diff

    point_r, point_mae = metrics(y, pa, pb)
    boot_r, boot_mae = [], []
    for _ in range(b):
        idx = rng.integers(0, n, size=n)          # 对"人"重抽样, 同一组 idx 用于两套预测 -> 配对
        yy = y[idx]
        if np.ptp(yy) == 0:                       # 退化样本(全相同 y), 跳过
            continue
        qa, qb = pa[idx], pb[idx]
        if np.ptp(qa) == 0 or np.ptp(qb) == 0:
            continue
        rd, md = metrics(yy, qa, qb)
        boot_r.append(rd)
        boot_mae.append(md)
    boot_r = np.asarray(boot_r)
    boot_mae = np.asarray(boot_mae)

    def summarize(point, arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        # 双尾 bootstrap p: 2 * min(P(diff<=0), P(diff>=0)), 经验分布
        p_le = np.mean(arr <= 0)
        p_ge = np.mean(arr >= 0)
        p = min(1.0, 2 * min(p_le, p_ge))
        return float(point), float(lo), float(hi), float(p), len(arr)

    pr = summarize(point_r, boot_r)
    pm = summarize(point_mae, boot_mae)
    return {"r": pr, "mae": pm}


def point_metrics(y, p):
    return pearsonr(y, p)[0], mean_absolute_error(y, p)


def main():
    rows = []

    # ---- 对比 1: 细 vs 粗, RandomForest (default) ----
    # 注: 主表口径为默认超参(DECISIONS: 主表保持默认超参); 故粗细对比用 default 预测。
    fine = load_pred("JHU_fine", "RandomForest", "default")
    coarse = load_pred("arterial_coarse", "RandomForest", "default")
    y, p_fine, p_coarse, ids = align_two(fine, coarse)
    r_fine, mae_fine = point_metrics(y, p_fine)
    r_coarse, mae_coarse = point_metrics(y, p_coarse)
    # Steiger's z(细 vs 粗, 同 y 的相依 r)
    sr_fine, sr_coarse, sz, sp = steiger_z(y, p_fine, p_coarse)
    # 配对 bootstrap(细 - 粗)
    bd = paired_bootstrap_diff(y, p_fine, p_coarse)
    print(f"[fine vs coarse RF] n_paired={len(ids)}  "
          f"r_fine={r_fine:.3f} r_coarse={r_coarse:.3f}  "
          f"Steiger z={sz:.3f} p={sp:.4g}", flush=True)
    rows.append({
        "comparison": "fine_vs_coarse__RandomForest_default",
        "n_paired": len(ids),
        "metric": "pearson_r",
        "value_A": round(r_fine, 4), "value_B": round(r_coarse, 4),
        "label_A": "JHU_fine", "label_B": "arterial_coarse",
        "diff_A_minus_B": round(bd["r"][0], 4),
        "ci_low": round(bd["r"][1], 4), "ci_high": round(bd["r"][2], 4),
        "p_bootstrap": round(bd["r"][3], 4),
        "steiger_z": round(sz, 4), "p_steiger": round(sp, 4),
        "boot_n_valid": bd["r"][4],
    })
    rows.append({
        "comparison": "fine_vs_coarse__RandomForest_default",
        "n_paired": len(ids),
        "metric": "mae",
        "value_A": round(mae_fine, 4), "value_B": round(mae_coarse, 4),
        "label_A": "JHU_fine", "label_B": "arterial_coarse",
        "diff_A_minus_B": round(bd["mae"][0], 4),
        "ci_low": round(bd["mae"][1], 4), "ci_high": round(bd["mae"][2], 4),
        "p_bootstrap": round(bd["mae"][3], 4),
        "steiger_z": "", "p_steiger": "",
        "boot_n_valid": bd["mae"][4],
    })

    # ---- 对比 2: tuned vs default, RandomForest, JHU_fine ----
    tuned = load_pred("JHU_fine", "RandomForest", "tuned")
    default = load_pred("JHU_fine", "RandomForest", "default")
    y2, p_tuned, p_default, ids2 = align_two(tuned, default)
    r_tuned, mae_tuned = point_metrics(y2, p_tuned)
    r_default, mae_default = point_metrics(y2, p_default)
    bd2 = paired_bootstrap_diff(y2, p_tuned, p_default)
    # tuned vs default 也是同 y 的相依 r, 一并给 Steiger 作交叉佐证
    _, _, sz2, sp2 = steiger_z(y2, p_tuned, p_default)
    print(f"[tuned vs default RF JHU] n_paired={len(ids2)}  "
          f"r_tuned={r_tuned:.3f} r_default={r_default:.3f}  "
          f"d_r={bd2['r'][0]:+.3f} CI=[{bd2['r'][1]:.3f},{bd2['r'][2]:.3f}] "
          f"p={bd2['r'][3]:.4g}", flush=True)
    rows.append({
        "comparison": "tuned_vs_default__RandomForest_JHU_fine",
        "n_paired": len(ids2),
        "metric": "pearson_r",
        "value_A": round(r_tuned, 4), "value_B": round(r_default, 4),
        "label_A": "tuned", "label_B": "default",
        "diff_A_minus_B": round(bd2["r"][0], 4),
        "ci_low": round(bd2["r"][1], 4), "ci_high": round(bd2["r"][2], 4),
        "p_bootstrap": round(bd2["r"][3], 4),
        "steiger_z": round(sz2, 4), "p_steiger": round(sp2, 4),
        "boot_n_valid": bd2["r"][4],
    })
    rows.append({
        "comparison": "tuned_vs_default__RandomForest_JHU_fine",
        "n_paired": len(ids2),
        "metric": "mae",
        "value_A": round(mae_tuned, 4), "value_B": round(mae_default, 4),
        "label_A": "tuned", "label_B": "default",
        "diff_A_minus_B": round(bd2["mae"][0], 4),
        "ci_low": round(bd2["mae"][1], 4), "ci_high": round(bd2["mae"][2], 4),
        "p_bootstrap": round(bd2["mae"][3], 4),
        "steiger_z": "", "p_steiger": "",
        "boot_n_valid": bd2["mae"][4],
    })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nsignificance -> {OUT}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
