#!/usr/bin/env python
"""
s_reg_tabpfn_nest.py — 严重度【WAB-AQ 回归, 留一法】TabPFN 的 n_estimators 扫描。

目的: 验证之前 benchmark_loo 里 TabPFN 回归分(r≈0.707/0.714, 默认 n_estimators=8)对集成遍数敏不敏感。
复用 arc_wab_aq_benchmark 的 data.load_data / evaluate(同清洗 + 同 bootstrap CI)。
留一法 + 折内 StandardScaler(无泄露); 每个 n 用一个 TabPFNRegressor 重复 fit 226 折(避免反复建实例假死)。

运行(动脉, 快):
  perl -e 'alarm shift; exec @ARGV' 7200 env PYENV_VERSION=data-analysis python \
      src/s_reg_tabpfn_nest.py --table data/ArterialAtlas156_features_merged.tsv \
      --name arterial_coarse --nest 2 3 4 5 6 8
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")
PKG = Path(__file__).resolve().parents[1]
HARNESS_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "arc_wab_aq_benchmark", "src"))
sys.path.insert(0, HARNESS_SRC)
import data as data_mod      # noqa: E402
import evaluate as ev        # noqa: E402

SEED = 42
SPARSE_DROP_FRAC = 0.10
BOOT_N = 2000


def load_xy(table_rel):
    abs_path = table_rel if os.path.isabs(table_rel) else os.path.join(PKG, table_rel)
    cfg = {"data": {"path": abs_path, "sep": "\t", "id_col": "participant_id",
                    "target": "wab_aq", "sparse_drop_frac": SPARSE_DROP_FRAC}}
    X, y, feat = data_mod.load_data(cfg)
    return X, y, feat


def run_one(X, y, n_est):
    from tabpfn import TabPFNRegressor
    try:
        reg = TabPFNRegressor(device="cpu", random_state=SEED, n_estimators=n_est)
    except TypeError:
        # 兼容: 某些版本参数名不同 -> 退回默认并标注
        print(f"    [warn] TabPFNRegressor 不接受 n_estimators, 用默认跑 n_est={n_est} 实为默认", flush=True)
        reg = TabPFNRegressor(device="cpu", random_state=SEED)
    n = len(y)
    yt = np.empty(n); yp = np.empty(n)
    t0 = time.time()
    for i, (tr, te) in enumerate(LeaveOneOut().split(X)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X.iloc[tr]); Xte = sc.transform(X.iloc[te])
        reg.fit(Xtr, y.iloc[tr])
        yp[te] = reg.predict(Xte); yt[te] = y.iloc[te].to_numpy()
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"    [n_est={n_est}] fold {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    m = ev.regression_metrics(yt, yp)
    ci = ev.bootstrap_ci(yt, yp, BOOT_N, SEED)
    return m, ci, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--nest", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8])
    args = ap.parse_args()

    X, y, feat = load_xy(args.table)
    print(f"[{args.name}/wab_aq] n={len(y)}  features={X.shape[1]}  LOO  nest={args.nest}", flush=True)

    rows = []
    for n_est in args.nest:
        m, ci, dt = run_one(X, y, n_est)
        rows.append({"dataset": args.name, "target": "wab_aq", "n_estimators": n_est,
                     "pearson_r": round(m["pearson_r"], 4),
                     "r_CI": f"[{ci['pearson_r'][0]:.3f},{ci['pearson_r'][1]:.3f}]",
                     "mae": round(m["mae"], 3), "rmse": round(m["rmse"], 3),
                     "r2": round(m["r2"], 4), "seconds": round(dt, 1)})
        print(f"  n_est={n_est}: r={m['pearson_r']:.3f} "
              f"[{ci['pearson_r'][0]:.3f},{ci['pearson_r'][1]:.3f}]  MAE={m['mae']:.2f}  "
              f"R2={m['r2']:.3f}  [{dt:.0f}s={dt/60:.1f}min]", flush=True)

    out = PKG / "results" / "summary" / f"tabpfn_nest_reg_{args.name}.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out); old = old[~old["n_estimators"].isin(df["n_estimators"])]
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values("n_estimators").to_csv(out, index=False)
    print(f"\nsummary -> {out.relative_to(PKG)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
