#!/usr/bin/env python
"""
Subgroup error analysis for ARC aphasia-severity benchmark (RandomForest default).

Joins per-patient out-of-fold predictions (regression + classification) with
demographics/clinical and lesion_volume, then reports error by 4 slices:
  - lesion_volume quartile (Q1..Q4)
  - age_at_stroke (median split: <=median / >median)
  - sex (M / F)
  - race (w / b)   [b is small N -> descriptive only]

Regression metrics per group: N, MAE, Pearson r (within group).
Classification metrics per group: N, balanced accuracy, error rate, severe recall.

Output: results/summary/subgroup_{reg,clf}_{granularity}.csv
No retraining; pure post-hoc slicing of frozen patient-level dumps. No leakage risk.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RUNS = os.path.join(ROOT, "arc_lesion_image_benchmark/results/runs")
SUMMARY = os.path.join(ROOT, "arc_lesion_image_benchmark/results/summary")
PARTICIPANTS = os.path.join(ROOT, "datasets/arc_ds004884/participants.tsv")
FEATURE_TSV = {
    "arterial_coarse": os.path.join(ROOT, "arc_lesion_image_benchmark/data/ArterialAtlas156_features_merged.tsv"),
    "JHU_fine": os.path.join(ROOT, "arc_lesion_image_benchmark/data/jhu156_features_merged.tsv"),
}

os.makedirs(SUMMARY, exist_ok=True)


def load_demographics():
    """participant_id, sex, age_at_stroke, race + lesion_volume (from coarse table)."""
    part = pd.read_csv(PARTICIPANTS, sep="\t")[
        ["participant_id", "sex", "age_at_stroke", "race"]
    ]
    # lesion_volume is identical patient-level value across granularities; take coarse table.
    vol = pd.read_csv(FEATURE_TSV["arterial_coarse"], sep="\t")[
        ["participant_id", "lesion_volume"]
    ]
    demo = part.merge(vol, on="participant_id", how="inner")
    return demo


def build_slices(df):
    """Add slice-label columns onto df (must contain lesion_volume, age_at_stroke, sex, race)."""
    out = {}

    # lesion volume quartiles (computed on the modeled cohort)
    q = pd.qcut(df["lesion_volume"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    out["lesion_volume_quartile"] = [
        ("Q1", q == "Q1"), ("Q2", q == "Q2"), ("Q3", q == "Q3"), ("Q4", q == "Q4")
    ]

    # age median split
    med = df["age_at_stroke"].median()
    out["age_median_split"] = [
        (f"age<={med:g}", df["age_at_stroke"] <= med),
        (f"age>{med:g}", df["age_at_stroke"] > med),
    ]

    # sex
    out["sex"] = [("M", df["sex"] == "M"), ("F", df["sex"] == "F")]

    # race (drop n/a); b is small N
    out["race"] = [("w", df["race"] == "w"), ("b", df["race"] == "b")]

    return out, med


def reg_metrics(sub):
    n = len(sub)
    mae = float(np.mean(np.abs(sub["y_true"] - sub["y_pred_mean"])))
    if n >= 3 and sub["y_true"].std() > 0 and sub["y_pred_mean"].std() > 0:
        r = float(pearsonr(sub["y_true"], sub["y_pred_mean"])[0])
    else:
        r = np.nan
    return n, mae, r


def clf_metrics(sub):
    n = len(sub)
    yt = sub["y_true"].astype(int)
    yp = sub["y_pred"].astype(int)
    err = float(np.mean(yt != yp))
    # balanced accuracy needs both classes present
    if yt.nunique() == 2:
        bacc = float(balanced_accuracy_score(yt, yp))
    else:
        bacc = np.nan
    # severe recall (positive class = 1 = severe)
    n_severe = int((yt == 1).sum())
    if n_severe > 0:
        srec = float(recall_score(yt, yp, pos_label=1, zero_division=0))
    else:
        srec = np.nan
    return n, bacc, err, srec, n_severe


def run_regression(granularity, demo):
    path = os.path.join(RUNS, f"reg_{granularity}_RandomForest_default_patient_pred.csv")
    pred = pd.read_csv(path)
    df = pred.merge(demo, on="participant_id", how="inner")
    assert len(df) == len(pred), f"merge dropped rows: {len(df)} vs {len(pred)}"
    slices, med = build_slices(df)

    rows = []
    # overall baseline
    n, mae, r = reg_metrics(df)
    rows.append(dict(slice="overall", group="ALL", N=n, MAE=round(mae, 4),
                     pearson_r=round(r, 4), note=""))
    for slice_name, groups in slices.items():
        for gname, mask in groups:
            sub = df[mask]
            n, mae, r = reg_metrics(sub)
            note = "N small: descriptive only, no strong fairness claim" if (slice_name == "race" and gname == "b") else ""
            rows.append(dict(slice=slice_name, group=gname, N=n,
                             MAE=round(mae, 4),
                             pearson_r=(round(r, 4) if not np.isnan(r) else np.nan),
                             note=note))
    res = pd.DataFrame(rows)
    out = os.path.join(SUMMARY, f"subgroup_reg_{granularity}.csv")
    res.to_csv(out, index=False)
    return res, med, len(df)


def run_classification(granularity, demo):
    path = os.path.join(RUNS, f"clf_{granularity}_RandomForest_default_patient_probs.csv")
    pred = pd.read_csv(path)
    df = pred.merge(demo, on="participant_id", how="inner")
    assert len(df) == len(pred), f"merge dropped rows: {len(df)} vs {len(pred)}"
    slices, med = build_slices(df)

    rows = []
    n, bacc, err, srec, nsev = clf_metrics(df)
    rows.append(dict(slice="overall", group="ALL", N=n, n_severe=nsev,
                     balanced_acc=round(bacc, 4), error_rate=round(err, 4),
                     severe_recall=round(srec, 4), note=""))
    for slice_name, groups in slices.items():
        for gname, mask in groups:
            sub = df[mask]
            n, bacc, err, srec, nsev = clf_metrics(sub)
            note = "N small: descriptive only, no strong fairness claim" if (slice_name == "race" and gname == "b") else ""
            rows.append(dict(slice=slice_name, group=gname, N=n, n_severe=nsev,
                             balanced_acc=(round(bacc, 4) if not np.isnan(bacc) else np.nan),
                             error_rate=round(err, 4),
                             severe_recall=(round(srec, 4) if not np.isnan(srec) else np.nan),
                             note=note))
    res = pd.DataFrame(rows)
    out = os.path.join(SUMMARY, f"subgroup_clf_{granularity}.csv")
    res.to_csv(out, index=False)
    return res, med, len(df)


def main():
    demo = load_demographics()
    print(f"demographics rows: {len(demo)}", flush=True)
    for gran in ["arterial_coarse", "JHU_fine"]:
        print(f"\n########## granularity = {gran} ##########", flush=True)
        reg, med, n = run_regression(gran, demo)
        print(f"[REG {gran}] cohort N={n}, age median={med:g}", flush=True)
        print(reg.to_string(index=False), flush=True)
        clf, med, n = run_classification(gran, demo)
        print(f"[CLF {gran}] cohort N={n}", flush=True)
        print(clf.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
