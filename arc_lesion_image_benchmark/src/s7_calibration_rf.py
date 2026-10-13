#!/usr/bin/env python
"""
s7_calibration_rf.py - calibration analysis of RandomForest out-of-fold probabilities
(at the original prevalence).

Background: the common protocol applies StandardScaler -> SMOTE -> clf within each fold.
SMOTE balances the training sub-fold to 50/50, so out-of-fold probabilities from an RF
trained on the balanced distribution are systematically too high (relative to the true
prevalence of ~0.319). This script quantifies the miscalibration on the original label
distribution and reports how far it can be repaired (Platt / isotonic recalibration).

Key no-leakage constraints:
- Raw Brier / ECE / reliability curves are computed directly on the dumped per-patient
  out-of-fold probabilities (the dump is itself a 5x10 out-of-fold aggregate; no leakage).
- Fitting a recalibration (Platt / isotonic) on the full sample and evaluating on the
  full sample would be optimistically biased. The calibration map is therefore fitted
  with stratified 5-fold CV: the calibrator is fit on the calibration training folds
  only and predicts on the held-out fold; predictions are reassembled for all 226
  patients -> out-of-sample recalibrated probabilities. The recalibrated Brier is thus
  also an honest out-of-sample estimate.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

RUNS = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
SUMMARY = os.path.join(os.path.dirname(__file__), "..", "results", "summary")
RUNS = os.path.abspath(RUNS)
SUMMARY = os.path.abspath(SUMMARY)
os.makedirs(SUMMARY, exist_ok=True)

N_BINS = 10
CALIB_FOLDS = 5
SEED = 42

CONFIGS = [
    ("JHU_fine", "default"),
    ("JHU_fine", "tuned"),
    ("arterial_coarse", "default"),
    ("arterial_coarse", "tuned"),
]


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def reliability(y, p, n_bins=N_BINS):
    """Equal-width 10 bins: returns (bin_lo, bin_hi, n, mean_pred, frac_pos) per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    # Bins are [lo, hi); the last bin includes the right edge
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append((edges[b], edges[b + 1], 0, np.nan, np.nan))
        else:
            rows.append(
                (edges[b], edges[b + 1], n, float(p[m].mean()), float(y[m].mean()))
            )
    return rows


def ece(y, p, n_bins=N_BINS):
    """Expected Calibration Error: equal-width bins, |mean_pred - frac_pos| weighted by bin size."""
    rows = reliability(y, p, n_bins)
    N = len(y)
    e = 0.0
    for lo, hi, n, mp, fp in rows:
        if n > 0:
            e += (n / N) * abs(mp - fp)
    return float(e)


def mce(y, p, n_bins=N_BINS):
    """Maximum Calibration Error: largest single-bin gap."""
    rows = reliability(y, p, n_bins)
    gaps = [abs(mp - fp) for _, _, n, mp, fp in rows if n > 0]
    return float(max(gaps)) if gaps else np.nan


def cv_recalibrate(y, p, method, folds=CALIB_FOLDS, seed=SEED):
    """Fit the calibration map with stratified CV; return out-of-sample recalibrated
    probabilities (no leakage).
    method: 'sigmoid' (Platt) or 'isotonic'.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(p.reshape(-1, 1), y):
        if method == "sigmoid":
            # Platt: univariate LR on logit(raw probability); clip to avoid log(0) at 0/1
            ptr = np.clip(p[tr], 1e-6, 1 - 1e-6)
            z = np.log(ptr / (1 - ptr)).reshape(-1, 1)
            lr = LogisticRegression(C=1e10, solver="lbfgs")
            lr.fit(z, y[tr])
            pte = np.clip(p[te], 1e-6, 1 - 1e-6)
            zte = np.log(pte / (1 - pte)).reshape(-1, 1)
            out[te] = lr.predict_proba(zte)[:, 1]
        elif method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(p[tr], y[tr])
            out[te] = ir.predict(p[te])
        else:
            raise ValueError(method)
    assert not np.isnan(out).any()
    return out


def main():
    summary_rows = []
    rel_rows = []

    for granularity, hp in CONFIGS:
        fp = os.path.join(
            RUNS, f"clf_{granularity}_RandomForest_{hp}_patient_probs.csv"
        )
        d = pd.read_csv(fp)
        y = d["y_true"].to_numpy(dtype=float)
        p_raw = d["y_prob_mean"].to_numpy(dtype=float)
        prevalence = float(y.mean())

        # No-information baseline: Brier of always predicting the raw prevalence
        # (used to judge how much raw improves over the baseline)
        brier_const = brier(y, np.full_like(y, prevalence))

        # Recalibration (CV, no leakage)
        p_platt = cv_recalibrate(y, p_raw, "sigmoid")
        p_iso = cv_recalibrate(y, p_raw, "isotonic")

        for tag, p in [("raw", p_raw), ("platt", p_platt), ("isotonic", p_iso)]:
            summary_rows.append(
                dict(
                    granularity=granularity,
                    hyperparam=hp,
                    calibration=tag,
                    n=len(y),
                    prevalence=round(prevalence, 4),
                    mean_pred=round(float(p.mean()), 4),
                    brier=round(brier(y, p), 5),
                    ece=round(ece(y, p), 5),
                    mce=round(mce(y, p), 5),
                    brier_const_baseline=round(brier_const, 5),
                )
            )
            for lo, hi, n, mp, fp_pos in reliability(y, p):
                rel_rows.append(
                    dict(
                        granularity=granularity,
                        hyperparam=hp,
                        calibration=tag,
                        bin_lo=round(lo, 2),
                        bin_hi=round(hi, 2),
                        bin_mid=round((lo + hi) / 2, 2),
                        n=n,
                        mean_pred=None if np.isnan(mp) else round(mp, 4),
                        frac_pos=None if np.isnan(fp_pos) else round(fp_pos, 4),
                    )
                )

    sdf = pd.DataFrame(summary_rows)
    rdf = pd.DataFrame(rel_rows)

    out_sum = os.path.join(SUMMARY, "calibration_RF.csv")
    out_rel = os.path.join(SUMMARY, "calibration_RF_reliability_curve.csv")
    sdf.to_csv(out_sum, index=False)
    rdf.to_csv(out_rel, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("=== calibration_RF.csv ===")
    print(sdf.to_string(index=False))
    print()
    print("wrote:", out_sum)
    print("wrote:", out_rel)


if __name__ == "__main__":
    main()
