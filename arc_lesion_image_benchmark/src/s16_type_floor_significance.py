#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s16_type_floor_significance.py — 类型【二分类】"两列地板 vs 复杂模型"配对 + 等价检验。

缺口: type_significance.csv 只覆盖 细vs粗 / 调vs默认 / RF vs LogReg;
从没把【floor(病灶量+年龄)≈复杂(JHU 189 细区)】这条主线断言做正式配对/等价检验。
本脚本补上, 把"类型任务上便宜地板追平贵特征"焊死。

特征关系(已核, 见 data/jhu156_features_merged.tsv 与 floor_volage_features.tsv):
  floor   = {lesion_volume, age_at_stroke}
  complex = floor + 189 个 JHU 细区  →  floor 严格【嵌套】于 complex。
  故"floor vs complex"= 标准的【嵌套特征增量检验】: 在量+年龄之上加 189 区有没有增益。
  复杂模型在信息上不可能比 floor 差; 若加 189 区无显著增益 + TOST 判等价
  → 位置信息在病灶负荷之上无贡献, 对上主线"病灶负荷主导"。

对比(每任务 fluent / broca, A=complex, B=floor, Δ=A-B=区域特征的增量, 正=复杂更好):
  1) best_vs_best         : 复杂最佳模型 vs 地板最佳模型(各按 balanced_acc 自动选, 头条)
  2) LogReg_increment     : JHU_fine/LogReg       vs floor_volage/LogReg
  3) RandomForest_increment: JHU_fine/RandomForest vs floor_volage/RandomForest
  4) SVM_increment        : JHU_fine/SVM_rbf       vs floor_volage/SVM_rbf
  (2-4 同算法变特征, 隔离"区域特征增量", 防 best-vs-best 被指 cherry-pick)

指标与检验(与 s13 / s6 / equivalence_power 完全一致):
  - AUC 差: 成对(correlated)DeLong(1988 + Sun&Xu 2014 快速算法)给原始 p。
  - balanced-acc 差: 患者级配对 bootstrap(对【人】重抽样, 阈值 0.5)给双尾 p + 95%CI。
  - TOST 等价: 对 AUC 与 balanced-acc 各做, 可忽略带 ±0.05;
      用 bootstrap SE → 两个单侧 z 检验, p_TOST=max(p_low,p_up); 同时报 90%CI 落带内判定。
      bound 依据: AUC 0.05 = 诊断意义改善的常规阈; bACC 0.05 = "小"效应 0.10 的一半。
  - MDE: alpha=0.05 双尾 + power=0.8 的最小可检出效应 = 2.8016*SE, 比 |Δ|。

读: results/runs/type_{task}_{feat}_{model}_oof.csv  (s11 产, 每人池化折外: prob_pos)
    results/summary/type_{task}_{feat}.csv            (自动选各端最佳模型)
写: results/summary/type_floor_significance.csv  (配对检验: 原始 p, 多重比较校正由主控统一做)
    results/summary/type_floor_equivalence.csv   (TOST 等价 + MDE)

运行: PYENV_VERSION=data-analysis python src/s16_type_floor_significance.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PKG, "results", "runs")
SUM_DIR = os.path.join(PKG, "results", "summary")
OUT_SIG = os.path.join(SUM_DIR, "type_floor_significance.csv")
OUT_EQ = os.path.join(SUM_DIR, "type_floor_equivalence.csv")

COMPLEX_FEAT = "JHU_fine"
FLOOR_FEAT = "floor_volage"
TASKS = ["fluent", "broca"]
MATCHED_MODELS = ["LogReg", "RandomForest", "SVM_rbf"]

B_BOOT = 10000            # 与 equivalence_power.py 一致
SEED = 20260605
THRESHOLD = 0.5           # 与 s11 患者层聚合一致 (y_pred = prob>=0.5)
BOUND = 0.05              # 等价可忽略带 (AUC 与 bACC 同 0.05)
ALPHA = 0.05
POWER = 0.80
Z_ALPHA = stats.norm.ppf(1 - ALPHA / 2)
Z_POWER = stats.norm.ppf(POWER)
MDE_K = Z_ALPHA + Z_POWER                  # ~2.8016
AUC_NOTE = ("equiv bound +/-0.05 AUC: below the conventional 0.05 threshold "
            "for a meaningful diagnostic AUC improvement")
BACC_NOTE = ("equiv bound +/-0.05 balanced-acc: half of a Cohen-small 0.10 "
             "gap, below any practically meaningful accuracy difference")


# ---------------- DeLong 快速算法 (Sun & Xu 2014) ----------------
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def _fast_delong(preds_sorted_t, m):
    n = preds_sorted_t.shape[1] - m
    pos = preds_sorted_t[:, :m]; neg = preds_sorted_t[:, m:]; k = preds_sorted_t.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_t[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    return aucs, sx / m + sy / n


def delong_paired_test(y_true, p1, p2):
    """成对 DeLong: 正类=1。返回 auc1, auc2, diff(=auc1-auc2), z, p(双尾)。"""
    y_true = np.asarray(y_true, dtype=int)
    order = (-y_true).argsort(kind="mergesort")
    m = int(y_true.sum())
    preds = np.vstack((np.asarray(p1, float)[order], np.asarray(p2, float)[order]))
    aucs, cov = _fast_delong(preds, m)
    a1, a2 = float(aucs[0]), float(aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]; diff = a1 - a2
    if var <= 0:
        return dict(auc1=a1, auc2=a2, diff=diff, z=0.0, p=1.0)
    z = diff / np.sqrt(var); p = 2.0 * stats.norm.sf(abs(z))
    return dict(auc1=a1, auc2=a2, diff=diff, z=float(z), p=float(p))


# ---------------- 快速指标 ----------------
def fast_auc(y, s):
    """Mann-Whitney U AUC (与 equivalence_power 一致, ties 用 rankdata)。"""
    y = np.asarray(y, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fast_bacc(y, pred):
    """二分 balanced-acc = 0.5*(TPR+TNR), 与 sklearn 一致(无 adjusted)。"""
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    P = y == 1; N = y == 0
    nP = int(P.sum()); nN = int(N.sum())
    if nP == 0 or nN == 0:
        return np.nan
    tpr = (pred[P] == 1).sum() / nP
    tnr = (pred[N] == 0).sum() / nN
    return 0.5 * (tpr + tnr)


def tost(delta, se, bound):
    if se <= 0:
        return 1.0 if abs(delta) >= bound else 0.0
    z_low = (delta - (-bound)) / se
    z_up = (delta - bound) / se
    p_low = 1 - stats.norm.cdf(z_low)   # H0: delta <= -bound
    p_up = stats.norm.cdf(z_up)         # H0: delta >=  bound
    return max(p_low, p_up)


# ---------------- 载入 + 对齐 ----------------
def load_oof(task, feat, model):
    f = os.path.join(RUNS_DIR, f"type_{task}_{feat}_{model}_oof.csv")
    d = pd.read_csv(f).set_index("participant_id").sort_index()
    return d


def align(task, featA, modelA, featB, modelB):
    a = load_oof(task, featA, modelA).rename(columns={"prob_pos": "pa", "y_true": "ya"})
    b = load_oof(task, featB, modelB).rename(columns={"prob_pos": "pb", "y_true": "yb"})
    m = a.join(b[["pb", "yb"]], how="inner")
    assert (m["ya"] == m["yb"]).all(), f"y_true mismatch {featA}/{modelA} vs {featB}/{modelB}"
    return m["ya"].to_numpy(int), m["pa"].to_numpy(float), m["pb"].to_numpy(float), len(m)


def best_model(task, feat, metric="balanced_acc"):
    """从 type_{task}_{feat}.csv 选 metric 最高且有 OOF 文件的模型(metric ∈ balanced_acc/auc)。"""
    df = pd.read_csv(os.path.join(SUM_DIR, f"type_{task}_{feat}.csv"))
    df["_m"] = df[metric].astype(str).str.split("±").str[0].astype(float)
    df = df.sort_values("_m", ascending=False)
    for _, r in df.iterrows():
        m = r["model"]
        if m == "Dummy":
            continue
        if os.path.exists(os.path.join(RUNS_DIR, f"type_{task}_{feat}_{m}_oof.csv")):
            return m, float(r["_m"])
    raise RuntimeError(f"no OOF-backed model for {task}/{feat}")


# ---------------- 单对比 ----------------
def run_contrast(task, name, featA, modelA, featB, modelB, labA, labB):
    y, pa, pb, n = align(task, featA, modelA, featB, modelB)
    rng = np.random.default_rng(SEED)

    # 点估计
    d = delong_paired_test(y, pa, pb)              # AUC 原始 p (analytic)
    auc_a, auc_b, dauc = d["auc1"], d["auc2"], d["diff"]
    preda = (pa >= THRESHOLD).astype(int); predb = (pb >= THRESHOLD).astype(int)
    bacc_a = fast_bacc(y, preda); bacc_b = fast_bacc(y, predb); dbacc = bacc_a - bacc_b

    # 患者级配对 bootstrap: 同一组重抽样下同时算 AUC 差与 bACC 差
    auc_diffs = np.empty(B_BOOT); bacc_diffs = np.empty(B_BOOT)
    auc_diffs[:] = np.nan; bacc_diffs[:] = np.nan
    for b in range(B_BOOT):
        idx = rng.integers(0, n, n); yt = y[idx]
        s = yt.sum()
        if s == 0 or s == n:
            continue
        auc_diffs[b] = fast_auc(yt, pa[idx]) - fast_auc(yt, pb[idx])
        bacc_diffs[b] = fast_bacc(yt, preda[idx]) - fast_bacc(yt, predb[idx])
    auc_diffs = auc_diffs[~np.isnan(auc_diffs)]
    bacc_diffs = bacc_diffs[~np.isnan(bacc_diffs)]

    def boot_pack(diffs, point):
        se = diffs.std(ddof=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        lo90, hi90 = np.percentile(diffs, [5.0, 95.0])
        ne = len(diffs)
        p_lo = (np.sum(diffs <= 0) + 1) / (ne + 1)
        p_hi = (np.sum(diffs >= 0) + 1) / (ne + 1)
        p_two = min(2.0 * min(p_lo, p_hi), 1.0)
        p_t = tost(point, se, BOUND)
        equiv = bool(lo90 > -BOUND and hi90 < BOUND)
        return dict(se=se, lo=lo, hi=hi, lo90=lo90, hi90=hi90, p_two=p_two,
                    p_tost=p_t, equiv=equiv, mde=MDE_K * se)

    pa_pack = boot_pack(auc_diffs, dauc)
    pb_pack = boot_pack(bacc_diffs, dbacc)

    comp = f"{labA} vs {labB}"
    sig_rows = [
        dict(task=task, contrast=name, comparison=comp, metric="AUC",
             test="paired DeLong (1988; Sun&Xu 2014)", n=n,
             value_complex=round(auc_a, 4), value_floor=round(auc_b, 4),
             diff=round(dauc, 4), ci_lo=round(pa_pack["lo"], 4), ci_hi=round(pa_pack["hi"], 4),
             p_value=d["p"]),
        dict(task=task, contrast=name, comparison=comp, metric="balanced_acc",
             test=f"paired patient bootstrap (B={B_BOOT})", n=n,
             value_complex=round(bacc_a, 4), value_floor=round(bacc_b, 4),
             diff=round(dbacc, 4), ci_lo=round(pb_pack["lo"], 4), ci_hi=round(pb_pack["hi"], 4),
             p_value=round(pb_pack["p_two"], 4)),
    ]
    def eq_row(metric, va, vb, delta, pack, note):
        return {
            "task": task, "contrast": name, "comparison": comp, "metric": metric,
            "n_paired": n, "value_complex": round(va, 4), "value_floor": round(vb, 4),
            "delta_complex_minus_floor": round(delta, 4), "se_boot": round(pack["se"], 4),
            "ci95_lo": round(pack["lo"], 4), "ci95_hi": round(pack["hi"], 4),
            "ci90_lo": round(pack["lo90"], 4), "ci90_hi": round(pack["hi90"], 4),
            "equiv_bound": BOUND, "p_TOST": round(pack["p_tost"], 4),
            "equivalent_at_0.05": pack["equiv"],
            "MDE_alpha0.05_power0.8": round(pack["mde"], 4),
            "abs_delta_lt_MDE": bool(abs(delta) < pack["mde"]), "B_boot": B_BOOT, "note": note,
        }

    eq_rows = [
        eq_row("AUC", auc_a, auc_b, dauc, pa_pack, AUC_NOTE),
        eq_row("balanced_acc", bacc_a, bacc_b, dbacc, pb_pack, BACC_NOTE),
    ]
    return sig_rows, eq_rows


def main():
    os.makedirs(SUM_DIR, exist_ok=True)
    sig_all, eq_all = [], []
    for task in TASKS:
        cm, cba = best_model(task, COMPLEX_FEAT, "balanced_acc")
        fm, fba = best_model(task, FLOOR_FEAT, "balanced_acc")
        cma, cau = best_model(task, COMPLEX_FEAT, "auc")
        fma, fau = best_model(task, FLOOR_FEAT, "auc")
        print(f"[{task}] best-bAcc complex={COMPLEX_FEAT}/{cm}({cba:.3f}) floor={FLOOR_FEAT}/{fm}({fba:.3f}); "
              f"best-AUC complex={COMPLEX_FEAT}/{cma}({cau:.3f}) floor={FLOOR_FEAT}/{fma}({fau:.3f})")
        contrasts = [
            ("best_bAcc", COMPLEX_FEAT, cm, FLOOR_FEAT, fm, f"{COMPLEX_FEAT}/{cm}", f"floor/{fm}"),
            ("best_AUC", COMPLEX_FEAT, cma, FLOOR_FEAT, fma, f"{COMPLEX_FEAT}/{cma}", f"floor/{fma}"),
        ]
        for m in MATCHED_MODELS:
            contrasts.append((f"{m}_increment", COMPLEX_FEAT, m, FLOOR_FEAT, m,
                              f"{COMPLEX_FEAT}/{m}", f"floor/{m}"))
        for nm, fa, ma, fb, mb, la, lb in contrasts:
            try:
                s, e = run_contrast(task, nm, fa, ma, fb, mb, la, lb)
                sig_all.extend(s); eq_all.extend(e)
            except FileNotFoundError as exc:
                print(f"  [skip] {task}/{nm}: 缺 OOF ({exc})")

    sig = pd.DataFrame(sig_all)
    eq = pd.DataFrame(eq_all)

    # 多重比较校正(每个指标族内: Holm + FDR-BH), 与 severity significance_combined.csv 同口径
    from statsmodels.stats.multitest import multipletests
    sig["p_holm"] = np.nan; sig["p_fdr_bh"] = np.nan; sig["sig_holm_0.05"] = False
    for metric, grp in sig.groupby("metric"):
        idx = grp.index
        rej_h, p_h, _, _ = multipletests(grp["p_value"].values, alpha=0.05, method="holm")
        _, p_f, _, _ = multipletests(grp["p_value"].values, alpha=0.05, method="fdr_bh")
        sig.loc[idx, "p_holm"] = np.round(p_h, 4)
        sig.loc[idx, "p_fdr_bh"] = np.round(p_f, 4)
        sig.loc[idx, "sig_holm_0.05"] = rej_h
    sig.to_csv(OUT_SIG, index=False)
    eq.to_csv(OUT_EQ, index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
    print(f"\n[written] {OUT_SIG}\n")
    print(sig.to_string(index=False))
    print(f"\n[written] {OUT_EQ}\n")
    print(eq[["task", "contrast", "metric", "delta_complex_minus_floor", "ci90_lo", "ci90_hi",
              "equiv_bound", "p_TOST", "equivalent_at_0.05", "abs_delta_lt_MDE"]].to_string(index=False))


if __name__ == "__main__":
    main()
