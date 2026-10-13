#!/usr/bin/env python
"""
s6_significance_regression.py -- patient-level paired significance tests (regression)

Contrasts (key null hypotheses for the "lesion-only information ceiling" argument; check whether
the 95% CI of the difference crosses 0 / whether p is non-significant):
  1) fine vs coarse (JHU_fine vs arterial_coarse), RandomForest:
       - Steiger's z: compares two dependent Pearson r (same patients, same y).
       - Patient-level paired bootstrap (B=2000, resampling patients): r difference and MAE
         difference -> difference + 95% CI + two-sided p.
  2) tuned vs default (RandomForest, JHU_fine):
       - Patient-level paired bootstrap: r difference and MAE difference -> difference + 95% CI
         + two-sided p.

Rules
-----
- Use patient-level predictions (one value per patient: y_pred_mean in reg_*_patient_pred.csv),
  not per-fold values, so correlated repeated-CV predictions do not inflate the sample size.
- Paired, same patients: align by inner merge on participant_id; the bootstrap resamples
  patients, and each resample uses the same patients for both prediction sets, keeping the pairing.
- The two regression r values are dependent correlations (same patients, same y): use Steiger's z.
- Report only the raw p + difference + 95% CI per contrast; no multiple-comparison correction
  here (Holm/FDR is applied jointly downstream).

Output: results/summary/significance_regression.csv
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
    """Read one per-patient dump: participant_id, y_true, y_pred_mean. Returns a DataFrame indexed by id."""
    path = os.path.join(RUNS, f"reg_{name}_{model}_{config}_patient_pred.csv")
    df = pd.read_csv(path)
    return df.set_index("participant_id")


def align_two(a, b):
    """Align two prediction sets on the shared participant_ids; check that y_true agrees (same patients, same y).

    Returns (y_true[N], pred_a[N], pred_b[N], ids[N]).
    """
    ids = a.index.intersection(b.index)
    ids = ids.sort_values()                       # fixed order, reproducible
    aa = a.loc[ids]
    bb = b.loc[ids]
    # same patients, same y: y_true must agree on both sides (up to float error)
    dy = np.abs(aa["y_true"].to_numpy() - bb["y_true"].to_numpy()).max()
    assert dy < 1e-6, f"y_true mismatch, max abs diff={dy}"
    y = aa["y_true"].to_numpy(dtype=float)
    pa = aa["y_pred_mean"].to_numpy(dtype=float)
    pb = bb["y_pred_mean"].to_numpy(dtype=float)
    return y, pa, pb, list(ids)


def steiger_z(y, pa, pb):
    """Steiger's z (1980): test of the difference between two dependent correlations r(y,pa) and r(y,pb).

    Shared variable y; compares r_ya with r_yb; also needs r_ab (correlation between the two prediction sets).
    Returns (r_ya, r_yb, z, p_two_sided).
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
    """Patient-level paired bootstrap: resample patients; each resample uses the same patients for both prediction sets.

    Returns a dict with two metric differences (a - b):
      r_diff   = r(y,pa) - r(y,pb)
      mae_diff = MAE(y,pa) - MAE(y,pb)
    Each holds point (observed full-sample difference), ci (2.5/97.5 percentiles) and p (two-sided,
    from how much of the bootstrap distribution lies on each side of 0).
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
        idx = rng.integers(0, n, size=n)          # resample patients; the same idx is used for both prediction sets -> paired
        yy = y[idx]
        if np.ptp(yy) == 0:                       # degenerate resample (constant y), skip
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
        # two-sided bootstrap p: 2 * min(P(diff<=0), P(diff>=0)), empirical distribution
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

    # ---- Contrast 1: fine vs coarse, RandomForest (default) ----
    # Note: the main table uses default hyperparameters, so the fine-vs-coarse contrast uses the
    # default predictions.
    fine = load_pred("JHU_fine", "RandomForest", "default")
    coarse = load_pred("arterial_coarse", "RandomForest", "default")
    y, p_fine, p_coarse, ids = align_two(fine, coarse)
    r_fine, mae_fine = point_metrics(y, p_fine)
    r_coarse, mae_coarse = point_metrics(y, p_coarse)
    # Steiger's z (fine vs coarse, dependent r on the same y)
    sr_fine, sr_coarse, sz, sp = steiger_z(y, p_fine, p_coarse)
    # paired bootstrap (fine - coarse)
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

    # ---- Contrast 2: tuned vs default, RandomForest, JHU_fine ----
    tuned = load_pred("JHU_fine", "RandomForest", "tuned")
    default = load_pred("JHU_fine", "RandomForest", "default")
    y2, p_tuned, p_default, ids2 = align_two(tuned, default)
    r_tuned, mae_tuned = point_metrics(y2, p_tuned)
    r_default, mae_default = point_metrics(y2, p_default)
    bd2 = paired_bootstrap_diff(y2, p_tuned, p_default)
    # tuned vs default are also dependent r on the same y; Steiger's z is reported as a cross-check
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
