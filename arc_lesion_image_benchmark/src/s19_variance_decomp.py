#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s19_variance_decomp.py
======================
Variance decomposition (primary statistic) of the (subset x model x seed) scores
from the s18 fully crossed grid: is more score variance attributable to the data
subset, to the model, or to the CV split seed?

Used instead of a simple max-min range of scores. Methods (design fixed before running):
  - ANOVA eta^2: score ~ C(subset) + C(model) + C(seed); share of total variance per
    factor (headline, easy-to-read number).
  - Mixed-model variance components sigma^2_subset / sigma^2_model / sigma^2_seed
    (crossed random effects, statsmodels vc_formula).
  - Within-cell bootstrap CI (resample seeds within each cell, B times) -> CIs for
    eta^2_subset, eta^2_model and their difference.
  - Leave-one-subset-out / leave-one-model-out sensitivity: does
    eta^2_subset > eta^2_model still hold after dropping any single level?
  - Regression: Pearson r is Fisher-z transformed before analysis (r is bounded and
    its variance depends on its mean).
  - Two scopes: (A) all subsets; (B) primary set = availability + clinical-threshold
    subsets (excluding the target-defined aq_le90 subset and the exact Teghipco cohort).

Usage: python src/s19_variance_decomp.py --task regression
"""
from __future__ import annotations
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(PKG, "results", "runs")
SUMM = os.path.join(PKG, "results", "summary")
SEED = 12345
BOOT = 2000


def load_grid(task):
    frames = []
    for tag in ["fast", "TabPFN"]:
        p = os.path.join(RUNS, f"grid_{task}_{tag}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no grid_{task}_*.csv found; run s18 first")
    df = pd.concat(frames, ignore_index=True)
    # Regression: Fisher-z transform of score (= r)
    if task == "regression":
        r = df["score"].clip(-0.999, 0.999)
        df["score_z"] = np.arctanh(r)
        df["analysis_y"] = df["score_z"]
    else:
        df["analysis_y"] = df["score"]
    return df


def eta_sq(df, factors=("subset", "model", "seed")):
    """ANOVA eta^2: SS of each factor / total SS. Returns a dict."""
    terms = " + ".join(f"C({f})" for f in factors)
    model = smf.ols(f"analysis_y ~ {terms}", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = aov["sum_sq"].sum()
    out = {}
    for f in factors:
        key = f"C({f})"
        if key in aov.index:
            out[f"eta2_{f}"] = float(aov.loc[key, "sum_sq"] / ss_total)
    out["eta2_resid"] = float(aov.loc["Residual", "sum_sq"] / ss_total)
    return out


def var_components(df):
    """Crossed random-effects variance components (statsmodels vc_formula, single dummy group)."""
    d = df.copy()
    d["grp"] = 1
    vc = {"subset": "0 + C(subset)", "model": "0 + C(model)", "seed": "0 + C(seed)"}
    try:
        md = smf.mixedlm("analysis_y ~ 1", d, groups="grp", vc_formula=vc)
        mf = md.fit(reml=True, method="lbfgs")
        vcs = mf.vcomp
        # statsmodels sorts variance-component names internally; insertion order
        # of vc_formula does not define the order of mf.vcomp.
        names = mf.model.exog_vc.names
        assert len(names) == len(vcs)
        comp = {f"sigma2_{n}": float(v) for n, v in zip(names, vcs)}
        comp["sigma2_resid"] = float(mf.scale)
        return comp
    except Exception as e:
        return {"_mixedlm_error": str(e)}


def boot_eta_diff(df, factors=("subset", "model", "seed"), B=BOOT, seed=SEED):
    """Within-cell bootstrap: resample seed rows within each (subset, model) cell, recompute
    eta^2, and return a CI for eta^2_subset - eta^2_model."""
    rng = np.random.default_rng(seed)
    cells = list(df.groupby(["subset", "model"]).groups.items())
    diffs, es_sub, es_mod = [], [], []
    for _ in range(B):
        parts = []
        for _, idx in cells:
            idx = np.asarray(idx)
            parts.append(df.loc[rng.choice(idx, size=len(idx), replace=True)])
        bd = pd.concat(parts, ignore_index=True)
        try:
            e = eta_sq(bd, factors)
            es_sub.append(e["eta2_subset"]); es_mod.append(e["eta2_model"])
            diffs.append(e["eta2_subset"] - e["eta2_model"])
        except Exception:
            continue
    q = lambda a, p: float(np.percentile(a, p))
    return {
        "eta2_subset_ci": [q(es_sub, 2.5), q(es_sub, 97.5)],
        "eta2_model_ci": [q(es_mod, 2.5), q(es_mod, 97.5)],
        "eta2_subset_minus_model_ci": [q(diffs, 2.5), q(diffs, 97.5)],
        "frac_boot_subset_gt_model": float(np.mean(np.array(diffs) > 0)),
    }


def leave_one_out(df, kind):
    """Leave-one-level-out sensitivity: drop one subset or model at a time and check whether
    eta^2_subset > eta^2_model still holds."""
    levels = sorted(df[kind].unique())
    res = []
    for lv in levels:
        sub = df[df[kind] != lv]
        e = eta_sq(sub)
        res.append({f"dropped_{kind}": lv,
                    "eta2_subset": round(e["eta2_subset"], 4),
                    "eta2_model": round(e["eta2_model"], 4),
                    "subset_gt_model": e["eta2_subset"] > e["eta2_model"]})
    return pd.DataFrame(res)


def run_one(df, task, scope):
    print(f"\n===== [{task}] scope={scope}  (n_cells_rows={len(df)}, "
          f"subsets={df['subset'].nunique()}, models={df['model'].nunique()}, "
          f"seeds={df['seed'].nunique()}) =====", flush=True)
    e = eta_sq(df)
    vc = var_components(df)
    print(f"  η²: subset={e['eta2_subset']:.3f}  model={e['eta2_model']:.3f}  "
          f"seed={e['eta2_seed']:.3f}  resid={e['eta2_resid']:.3f}", flush=True)
    print(f"  σ²: {vc}", flush=True)
    bo = boot_eta_diff(df)
    print(f"  η²_subset-model diff 95%CI={[round(x,3) for x in bo['eta2_subset_minus_model_ci']]}  "
          f"P(subset>model)={bo['frac_boot_subset_gt_model']:.3f}", flush=True)
    loo_s = leave_one_out(df, "subset")
    loo_m = leave_one_out(df, "model")
    surv_s = bool(loo_s["subset_gt_model"].all())
    surv_m = bool(loo_m["subset_gt_model"].all())
    print(f"  subset>model holds after every leave-one-subset-out? {surv_s}; after every leave-one-model-out? {surv_m}", flush=True)
    row = {"task": task, "scope": scope, **{k: round(v, 4) for k, v in e.items()},
           **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in vc.items()},
           "eta2_subset_minus_model_ci_lo": round(bo["eta2_subset_minus_model_ci"][0], 4),
           "eta2_subset_minus_model_ci_hi": round(bo["eta2_subset_minus_model_ci"][1], 4),
           "P_subset_gt_model_boot": bo["frac_boot_subset_gt_model"],
           "loo_subset_all_hold": surv_s, "loo_model_all_hold": surv_m,
           "headline_subset_dominates": bool(e["eta2_subset"] > e["eta2_model"]
                                              and bo["eta2_subset_minus_model_ci"][0] > 0)}
    return row, loo_s, loo_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    args = ap.parse_args()
    df = load_grid(args.task)

    # scope A: all subsets
    rowA, _, _ = run_one(df, args.task, "all_subsets")
    # scope B: primary set = availability + clinical threshold (excludes target_defined + exact_persons)
    primary = df[df["category"].isin(["baseline", "availability", "clinical_threshold"])]
    rowB, loo_s, loo_m = run_one(primary, args.task, "primary_avail_clinical")
    # scope C (headline): primary set x only the strong models that studies actually report;
    #   weak baselines (Ridge/MLP/SVR) are excluded. Weak models inflate model-axis variance
    #   (a confound); published studies report their best competitive model, and those
    #   models score close to each other.
    COMP = {"regression": ["RandomForest", "XGBoost", "LightGBM"],
            "classification": ["RandomForest", "XGBoost", "LightGBM", "LogReg"]}[args.task]
    comp_df = primary[primary["model"].isin(COMP)]
    rowC, loo_sC, loo_mC = run_one(comp_df, args.task, "primary_competitive_models")

    out = pd.DataFrame([rowA, rowB, rowC])
    os.makedirs(SUMM, exist_ok=True)
    out.to_csv(os.path.join(SUMM, f"variance_decomposition_{args.task}.csv"), index=False)
    loo_s.to_csv(os.path.join(SUMM, f"variance_loo_subset_{args.task}.csv"), index=False)
    loo_m.to_csv(os.path.join(SUMM, f"variance_loo_model_{args.task}.csv"), index=False)
    loo_sC.to_csv(os.path.join(SUMM, f"variance_loo_subset_competitive_{args.task}.csv"), index=False)
    loo_mC.to_csv(os.path.join(SUMM, f"variance_loo_model_competitive_{args.task}.csv"), index=False)
    print(f"\n[write] variance_decomposition_{args.task}.csv (3 scopes: all / primary / competitive) + LOO tables", flush=True)


if __name__ == "__main__":
    main()
