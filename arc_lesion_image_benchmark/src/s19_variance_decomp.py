#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s19_variance_decomp.py
======================
任务3 第 3 步(主统计量): 把 s18 全交叉网格的 (subset×model×seed) 分数做方差分解,
回答"子集贡献的方差 vs 模型贡献的方差 vs 划分种子的方差谁大"。

替换被评审否掉的 max-min 摆幅。方法(refine-logs/TASK3_DESIGN_PREREG_2026-06-13.md §3):
  - ANOVA η²: score ~ C(subset)+C(model)+C(seed), 各因子占总方差百分比(headline 直观数)。
  - 混合模型方差成分 σ²_subset/σ²_model/σ²_seed(交叉随机效应, statsmodels vc_formula)。
  - 折内自助 CI(每单元内重采样 seed, B 次) -> η²_subset / η²_model / 及其差的 CI。
  - 留一子集/留一模型敏感性: 掉任一 level 后 η²_subset>η²_model 是否仍成立(答"加/减 level 会翻"那刀)。
  - 回归侧 Pearson r 先 Fisher-z 变换再分析(r 有界、方差依赖均值)。
  - 分两套: (A)全子集; (B)主集=可得性+临床阈值子集(去掉 aq_le90 靶定义 + teghipco 精确同人)。

用法: PYENV_VERSION=data-analysis python src/s19_variance_decomp.py --task regression
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
        raise FileNotFoundError(f"没有 grid_{task}_*.csv, 先跑 s18")
    df = pd.concat(frames, ignore_index=True)
    # 回归: Fisher-z 变换 score(=r)
    if task == "regression":
        r = df["score"].clip(-0.999, 0.999)
        df["score_z"] = np.arctanh(r)
        df["analysis_y"] = df["score_z"]
    else:
        df["analysis_y"] = df["score"]
    return df


def eta_sq(df, factors=("subset", "model", "seed")):
    """ANOVA η²: 各因子 SS / 总 SS。返回 dict。"""
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
    """交叉随机效应方差成分(statsmodels vc_formula, 单 dummy group)。"""
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
    """折内自助: 每 (subset,model) 单元内重采样 seed 行, 重算 η², 得 η²_subset-η²_model 的 CI。"""
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
    """留一 level 敏感性: 逐个掉一个 subset 或 model, 看 η²_subset>η²_model 是否仍成立。"""
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
    print(f"  留一子集后 subset>model 全成立? {surv_s}; 留一模型后? {surv_m}", flush=True)
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

    # scope A: 全子集
    rowA, _, _ = run_one(df, args.task, "all_subsets")
    # scope B: 主集 = 可得性 + 临床阈值(去 target_defined + exact_persons)
    primary = df[df["category"].isin(["baseline", "availability", "clinical_threshold"])]
    rowB, loo_s, loo_m = run_one(primary, args.task, "primary_avail_clinical")
    # scope C(headline): 主集 × 仅"各家真会拿来报结果"的强模型 —— 排除弱基线(Ridge/MLP/SVR)
    #   弱模型把模型轴方差撑大是混淆; 各论文报的是其最佳竞争力模型, 它们彼此挤在一起。
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
    print(f"\n[write] variance_decomposition_{args.task}.csv (3 scopes: all / primary / competitive) + loo 表", flush=True)


if __name__ == "__main__":
    main()
