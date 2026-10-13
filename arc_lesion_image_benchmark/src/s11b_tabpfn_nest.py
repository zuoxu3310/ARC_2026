#!/usr/bin/env python
"""
s11b_tabpfn_nest.py -- sweep of TabPFN's internal ensemble size n_estimators (find a value
that fits the time budget on JHU without losing accuracy).

Background: TabPFN defaults to n_estimators=8 (8 internal permutations, one forward pass each,
then voting). On JHU (72 dims) x 5x10 folds this runs over 20 minutes on CPU and times out.
Lowering n_estimators saves time linearly (this is not tuning; it only lowers inference
cost). For each n the script runs the full s11 binary protocol (scaler->SMOTE->TabPFN) and
reports wall-clock time + balanced_acc/AUC to show how much accuracy drops.

Reuses load_xy / protocol constants from s11. Runs TabPFN only (single process, no other models).
Run:
  perl -e 'alarm 5400; exec @ARGV' python \
      src/s11b_tabpfn_nest.py --table data/jhu156_features_merged.tsv --name JHU_fine \
      --task fluent --nest 2 3 4 5
"""
from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import s11_type_classification as s11  # reuse load_xy / constants

warnings.filterwarnings("ignore")
PKG = Path(__file__).resolve().parents[1]
SEED = 42
N_SPLITS, N_REPEATS = 5, 10


def run_one(X, y, n_est, n_repeats):
    from tabpfn import TabPFNClassifier
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    per = {k: [] for k in ["balanced_acc", "auc", "f1_pos", "mcc"]}
    oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
    t0 = time.time()
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        pipe = ImbPipeline([("scaler", StandardScaler()),
                            ("smote", SMOTE(random_state=SEED)),
                            ("clf", TabPFNClassifier(device="cpu", n_estimators=n_est))])
        pipe.fit(X[tr], y[tr])
        oof_pred[te] = pipe.predict(X[te])
        try:
            oof_prob[te] = pipe.predict_proba(X[te])[:, 1]
        except Exception:
            oof_prob[te] = oof_pred[te]
        if (i + 1) % 5 == 0:
            print(f"    [n_est={n_est}] fold {i+1}/{N_SPLITS*n_repeats} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % N_SPLITS == 0:
            per["balanced_acc"].append(balanced_accuracy_score(y, oof_pred))
            per["auc"].append(roc_auc_score(y, oof_prob))
            per["f1_pos"].append(f1_score(y, oof_pred, pos_label=1, zero_division=0))
            per["mcc"].append(matthews_corrcoef(y, oof_pred))
            oof_pred = np.full(len(y), -1); oof_prob = np.zeros(len(y))
    dt = time.time() - t0
    summ = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per.items()}
    return summ, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True, choices=["fluent", "broca"])
    ap.add_argument("--nest", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    tp = PKG / args.table if not Path(args.table).is_absolute() else Path(args.table)
    X, y, feat, _classes, _ids = s11.load_xy(tp, args.task)
    print(f"[{args.name}/{args.task}] n={len(y)}  features={X.shape[1]}  "
          f"protocol=5x{args.repeats}  nest={args.nest}", flush=True)

    rows = []
    for n_est in args.nest:
        summ, dt = run_one(X, y, n_est, args.repeats)
        ba, bas = summ["balanced_acc"]; auc, _ = summ["auc"]; f1, _ = summ["f1_pos"]
        rows.append({"task": args.task, "dataset": args.name, "n_estimators": n_est,
                     "balanced_acc": f"{ba:.3f}±{bas:.3f}", "auc": round(auc, 3),
                     "f1_pos": round(f1, 3), "seconds": round(dt, 1)})
        print(f"  n_est={n_est}: bAcc={ba:.3f}±{bas:.3f}  AUC={auc:.3f}  F1={f1:.3f}  "
              f"[{dt:.0f}s = {dt/60:.1f}min]", flush=True)

    out = PKG / "results" / "summary" / f"tabpfn_nest_sweep_{args.task}_{args.name}.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old["n_estimators"].isin(df["n_estimators"])]
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values("n_estimators").to_csv(out, index=False)
    print(f"\nsummary -> {out.relative_to(PKG)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
