#!/usr/bin/env python
"""
s27_floor_vs_full_paired.py

Paired significance test for binary classification (severe = WAB-AQ <= 50):
two-feature floor (lesion volume + age) vs full features (JHU 189 regions -> 71
modelling features).

Motivation: the main table reports floor best SVM balanced acc 0.778 vs full-feature
best RF 0.754. Those two numbers come from two independent 5x10 runs (different fold
assignments), so they cannot be compared directly. This script uses the **same**
RepeatedStratifiedKFold (5x10, seed=42) splits and pairs floor SVM with full-feature RF
per repeat (unit = repeat: out-of-fold predictions from the 5 folds are pooled into one
balanced acc per repeat, 10 in total), then runs a paired t-test / Wilcoxon / bootstrap CI.

Settings match s3_classification_benchmark.py exactly (in-fold
StandardScaler -> SMOTE -> clf, SVC(rbf, prob, seed 42), RandomForest(500, seed 42)).
Reproducing the main-table 0.778 / 0.754 serves as a self-check.

Note: this is a **repeat-level** stability test (same 226 patients, different CV
shuffles) showing the difference is not fold-assignment luck. It is not patient-level
generalisation (a new patient sample), which requires a patient-level bootstrap and
gives wider intervals.

Run:
  python src/s27_floor_vs_full_paired.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy import stats

PKG = Path(__file__).resolve().parents[1]
SEED, THR, SPARSE, NS, NR = 42, 50, 0.10, 5, 10
FULL = PKG / "data" / "jhu156_features_merged.tsv"
FLOOR = PKG / "data" / "floor_volage_features.tsv"
OUT = PKG / "results" / "summary" / "floor_vs_full_paired_clf.csv"
LOG = PKG / "results" / "summary" / "floor_vs_full_paired_clf.log"


def load(path):
    return pd.read_csv(path, sep="\t", index_col="participant_id").dropna(subset=["wab_aq"]).fillna(0)


def main():
    full, floor = load(FULL), load(FLOOR)
    common = sorted(full.index.intersection(floor.index))          # same patients in same order -> identical splits
    full, floor = full.loc[common], floor.loc[common]
    y = (full["wab_aq"].values <= THR).astype(int)
    ff = full.drop(columns=["wab_aq"]); n = len(ff)
    keep = [c for c in ff.columns if (ff[c] != 0).sum() > SPARSE * n]
    Xfull = ff[keep].values
    Xfloor = floor.drop(columns=["wab_aq"]).values

    def psvm():
        return ImbPipeline([("s", StandardScaler()), ("sm", SMOTE(random_state=SEED)),
                            ("c", SVC(kernel="rbf", probability=True, random_state=SEED))])

    def prf():
        return ImbPipeline([("s", StandardScaler()), ("sm", SMOTE(random_state=SEED)),
                            ("c", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1))])

    rskf = RepeatedStratifiedKFold(n_splits=NS, n_repeats=NR, random_state=SEED)
    fl, fu = [], []
    of = np.full(len(y), -1); ou = np.full(len(y), -1)
    for i, (tr, te) in enumerate(rskf.split(Xfloor, y)):
        of[te] = psvm().fit(Xfloor[tr], y[tr]).predict(Xfloor[te])
        ou[te] = prf().fit(Xfull[tr], y[tr]).predict(Xfull[te])
        if (i + 1) % NS == 0:
            fl.append(balanced_accuracy_score(y, of)); fu.append(balanced_accuracy_score(y, ou))
            of = np.full(len(y), -1); ou = np.full(len(y), -1)
            print(f"  repeat {len(fl)}/{NR}: floor={fl[-1]:.3f} full={fu[-1]:.3f}", flush=True)

    fl, fu = np.array(fl), np.array(fu); d = fl - fu
    t, pt = stats.ttest_rel(fl, fu)
    w, pw = stats.wilcoxon(fl, fu)
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    ci = np.percentile(boots, [2.5, 97.5])

    pd.DataFrame({"repeat": np.arange(1, NR + 1), "floor_svm_bacc": fl,
                  "full_rf_bacc": fu, "diff_floor_minus_full": d}).to_csv(OUT, index=False)
    summary = (
        f"floor SVM  mean={fl.mean():.4f} sd={fl.std():.4f}  (paper 0.778)\n"
        f"full  RF   mean={fu.mean():.4f} sd={fu.std():.4f}  (paper 0.754)\n"
        f"paired diff (floor-full) mean={d.mean():+.4f}\n"
        f"paired t={t:.3f} p={pt:.4f} | Wilcoxon p={pw:.4f} | "
        f"bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]\n"
        f"unit=repeat (n={NR}); within-sample CV-shuffle stability, NOT patient-level generalization.\n"
    )
    LOG.write_text(summary)
    print("\n" + summary + f"per-repeat -> {OUT.relative_to(PKG)}\nsummary  -> {LOG.relative_to(PKG)}")


if __name__ == "__main__":
    raise SystemExit(main())
