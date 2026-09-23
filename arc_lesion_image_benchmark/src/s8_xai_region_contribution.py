#!/usr/bin/env python
"""
s8_xai_region_contribution.py  ——  RandomForest 脑区贡献的【泄露安全】可解释性分析

口径硬声明
----------
全程是【预测关联 predictive association】, 不是因果定位。任何 "region X 重要" 只表示
"在本模型本数据下, 扰动该列会让折外预测变差", 不写 "X 区导致严重失语"。

做什么(对 RandomForest, 两任务 × 两粒度; 重点 arterial_coarse 因最可解释)
------------------------------------------------------------------------
任务:
  * clf : 分类 severe = wab_aq <= 50。折内口径 = 主表 s3(StandardScaler -> SMOTE -> RF),
          协议 = RepeatedStratifiedKFold(5x10)。
  * reg : 回归 wab_aq。折内口径 = scaler -> RF(默认超参), 协议 = RepeatedStratifiedKFold(5x10),
          连续目标用分位分箱分层(与 s5 一致)。
粒度: arterial_coarse(15 列, 最可解释) + JHU_fine(~71 列)。

五块分析
--------
1) 折外置换重要度: 每折在【测试折】上 permutation_importance(对拟合好的 pipeline,
   置换 raw X 列 -> 重新 predict -> 看指标掉多少)。scaler/SMOTE 是 pipeline 内步骤,
   只在 fit 时见训练折, predict 阶段不触发 SMOTE, 故置换安全。跨折求 均值±std -> 稳定性。
   比 impurity(基尼)重要度可靠(impurity 偏向高基数/连续列、且在训练数据上算)。
2) SHAP(仅 arterial_coarse, 因维度小可解释): 每折 TreeExplainer 在【测试折】上算 SHAP,
   折外聚合 mean(|SHAP|); 报跨折 top-k 区是否稳定(rank 一致性), 换折就变 = 共线性 artifact 警告。
3) 共线性: 特征间 Spearman 相关 -> 层次聚类(1-|rho| 距离, 阈值 0.4 即 |rho|>=0.6 视为一簇)。
   报【组级】置换重要度(把一簇列一起置换), 并标注 top 区里哪些彼此共线
   -> 避免 "region X 驱动" 实为一簇共损伤区的假象。
4) 体积校正: (a) lesion_volume 单独的折外置换重要度多大;
   (b) 含 vs 不含 lesion_volume 两套模型, 看 top 脑区重要度怎么变(掉很多 = 体积代理);
   (c) 把每个脑区列对 lesion_volume 做线性残差(去掉总病灶量的线性成分)后再跑置换重要度
   -> 区分 "脑区特异" vs "只是总病灶量代理"。
5) 解释边界: 见脚本尾打印 + 返回。

输出
----
  results/summary/xai_perm_importance_{task}_{name}.csv     折外置换重要度 + 稳定性
  results/summary/xai_shap_{task}_{name}.csv                SHAP 聚合 + 跨折稳定性(仅 coarse)
  results/summary/xai_collinearity_{name}.csv               相关簇成员
  results/summary/xai_group_importance_{task}_{name}.csv    组级(共线簇)置换重要度
  results/summary/xai_volume_correction_{task}_{name}.csv   含/不含 volume + 残差校正后重要度

用法
----
  PYENV_VERSION=data-analysis python src/s8_xai_region_contribution.py
  PYENV_VERSION=data-analysis python src/s8_xai_region_contribution.py --quick   # 5x2 折快验
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY = os.path.join(PKG, "results", "summary")
SEED = 42
THRESHOLD = 50          # severe = wab_aq <= 50
SPARSE_DROP_FRAC = 0.10
N_SPLITS, N_REPEATS = 5, 10
N_STRAT_BINS = 4        # 回归连续目标分层分箱(与 s5 一致)
PERM_REPEATS = 20       # 每折每列置换次数
COLLIN_THRESH = 0.6     # |Spearman rho| >= 0.6 视为同簇(距离阈值 1-0.6=0.4)
SHAP_TOPK = 8

TABLES = {
    "arterial_coarse": "data/ArterialAtlas156_features_merged.tsv",
    "JHU_fine":        "data/jhu156_features_merged.tsv",
}


# ----------------------------------------------------------------------------- 数据
def load_df(table_rel):
    path = os.path.join(PKG, table_rel)
    df = pd.read_csv(path, sep="\t", index_col="participant_id")
    df = df.dropna(subset=["wab_aq"]).fillna(0)
    n = len(df)
    keep = [c for c in df.columns
            if c == "wab_aq" or (df[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    df = df[keep]
    feat = [c for c in df.columns if c != "wab_aq"]
    X = df[feat].copy()
    y_cont = df["wab_aq"].astype(float).copy()
    return X, y_cont, feat


def strat_labels(task, y_cont):
    """分类直接 severe 标签; 回归用 wab_aq 分位分箱(只用于 split, 不进特征)。"""
    if task == "clf":
        return (y_cont.values <= THRESHOLD).astype(int)
    return pd.qcut(y_cont, q=N_STRAT_BINS, labels=False, duplicates="drop").values


def target_for(task, y_cont):
    if task == "clf":
        return (y_cont.values <= THRESHOLD).astype(int)
    return y_cont.values.astype(float)


# ----------------------------------------------------------------------------- 折内体积残差化 transformer
class VolumeResidualizer(BaseEstimator, TransformerMixin):
    """折内体积校正: 把指定 region 列对 lesion_volume 列做 OLS 线性残差。

    【泄露安全的关键】OLS 系数 (截距+斜率) 只在 fit() 里用【训练折】拟合,
    transform() 把同一组系数应用到任意折(训练折或测试折)。放进 sklearn pipeline
    后, pipeline.fit 只见训练折 -> 系数永不接触测试折。permutation_importance 置换的是
    raw 测试折的列, 经 pipeline.transform 时用训练折系数残差化 -> 无预处理泄露。

    旧版 residualize_on_volume() 对【全 226 人】拟合系数后再切折, 系数见过测试折 = 泄露; 已弃用。

    参数全用列索引(pipeline 内是 numpy array):
      vol_idx    : lesion_volume 在特征矩阵的列位置; None 则本步为恒等(不做残差化)。
      region_idx : 要残差化的列位置列表(排除 volume 与 age)。其余列原样保留。
    """
    def __init__(self, vol_idx=None, region_idx=None):
        self.vol_idx = vol_idx
        self.region_idx = region_idx

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.vol_idx is None or not self.region_idx:
            self.beta_ = None
            return self
        v = X[:, self.vol_idx]
        v1 = np.column_stack([np.ones_like(v), v])           # [1, volume] 设计矩阵, 仅训练折
        self.beta_ = {}
        for j in self.region_idx:
            beta, *_ = np.linalg.lstsq(v1, X[:, j], rcond=None)
            self.beta_[j] = beta                              # (截距, 斜率), 来自训练折
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if not getattr(self, "beta_", None):
            return X
        v = X[:, self.vol_idx]
        v1 = np.column_stack([np.ones_like(v), v])
        for j, beta in self.beta_.items():
            X[:, j] = X[:, j] - v1 @ beta                     # 用训练折系数算残差
        return X


# ----------------------------------------------------------------------------- 模型
def build_pipe(task, residualizer=None):
    """主表口径: clf = scaler->SMOTE->RF(500); reg = scaler->RF(默认)。
    置换重要度作用在 pipeline 上, 故 scaler/SMOTE/residualizer 只在 fit 时见训练折。

    residualizer: 若给一个 VolumeResidualizer 实例, 插在 scaler 之前(用 raw volume 残差化),
                  其系数在 pipeline.fit 时只见训练折 -> 折内体积校正, 无泄露。"""
    steps = []
    if residualizer is not None:
        steps.append(("resid", residualizer))
    steps.append(("scaler", StandardScaler()))
    if task == "clf":
        steps.append(("smote", SMOTE(random_state=SEED)))
        steps.append(("clf", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)))
    else:
        steps.append(("clf", RandomForestRegressor(random_state=SEED, n_jobs=-1)))
    return ImbPipeline(steps)


def make_residualizer(feat, vol_col="lesion_volume"):
    """据特征名构造一个折内 VolumeResidualizer: 对 region 列(排除 volume/age)残差化。
    若 feat 里没有 volume 列, 返回 None(本粒度不做体积校正)。"""
    if vol_col not in feat:
        return None
    vol_idx = feat.index(vol_col)
    region_idx = [feat.index(c) for c in feat if c not in (vol_col, "age_at_stroke")]
    return VolumeResidualizer(vol_idx=vol_idx, region_idx=region_idx)


def perm_scoring(task):
    # 分类用 balanced_accuracy(主指标); 回归用 R2(permutation_importance 默认 estimator.score=R2)
    return "balanced_accuracy" if task == "clf" else "r2"


# ----------------------------------------------------------------------------- 1) 折外置换重要度
def _oof_perm(task, X, y, strat, feat, n_repeats_outer, group_cols=None, residualize=False):
    """每折在【测试折】上对拟合好的 pipeline 做置换重要度, 跨折聚合。

    group_cols: 若给 {组名: [列名...]}, 改成【组级】置换(一组列一起 shuffle, 共线簇用);
                None 则列级置换, 每列单独。
    residualize: True 则在 pipeline 里插一个折内 VolumeResidualizer(每折新建, fit 只见训练折)。
                 体积校正的 C 套用它 -> 系数永不接触测试折, 无预处理泄露。
    返回 DataFrame: unit = 列名(或组名), importance_mean/std/cv, frac_folds_positive, n_folds。
    置换作用在 raw 测试折上, pipeline.predict 不触发 SMOTE, scaler/resid 已 fit 于训练折 -> 无泄露。
    """
    Xv = X.values
    scorer = perm_scoring(task)
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats_outer,
                                    random_state=SEED)
    if group_cols is None:
        units = feat
        col_idx = {c: [feat.index(c)] for c in feat}
    else:
        units = list(group_cols.keys())
        col_idx = {g: [feat.index(c) for c in cols] for g, cols in group_cols.items()}

    per_fold = {u: [] for u in units}
    n_folds = outer.get_n_splits()
    for i, (tr, te) in enumerate(outer.split(Xv, strat)):
        # 每折新建 residualizer, 避免跨折共享已拟合系数; fit 仅在训练折(pipe.fit 内)
        resid = make_residualizer(feat) if residualize else None
        pipe = build_pipe(task, residualizer=resid)
        pipe.fit(Xv[tr], y[tr])
        Xte, yte = Xv[te], y[te]
        rng = np.random.RandomState(SEED + i)
        if task == "clf":
            base = balanced_accuracy_score(yte, pipe.predict(Xte))
        else:
            base = pipe.score(Xte, yte)   # R2
        for u in units:
            idxs = col_idx[u]
            drops = []
            for _ in range(PERM_REPEATS):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                for j in idxs:                       # 同组列用同一 perm 一起打乱
                    Xp[:, j] = Xte[perm, j]
                if task == "clf":
                    s = balanced_accuracy_score(yte, pipe.predict(Xp))
                else:
                    s = pipe.score(Xp, yte)
                drops.append(base - s)               # 掉得越多越重要
            per_fold[u].append(float(np.mean(drops)))
        if (i + 1) % 5 == 0 or (i + 1) == n_folds:
            print(f"    [perm:{task}] fold {i+1}/{n_folds}", flush=True)

    rows = []
    for u in units:
        arr = np.array(per_fold[u])
        rows.append({
            "unit": u,
            "importance_mean": float(arr.mean()),
            "importance_std": float(arr.std()),
            "importance_cv": float(arr.std() / abs(arr.mean())) if arr.mean() != 0 else np.nan,
            "frac_folds_positive": float((arr > 0).mean()),
            "n_folds": len(arr),
        })
    out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ----------------------------------------------------------------------------- 2) SHAP(仅 coarse)
def oof_shap(task, X, y, strat, feat, n_repeats_outer):
    """每折 TreeExplainer 在测试折上算 SHAP, 折外聚合 mean(|SHAP|)。
    并报跨折 top-k 稳定性: 每折各自的 top-k 集合, 计算各区出现频率 + 跨折 rank 标准差。

    注意: TreeExplainer 直接吃 RF, 不吃 SMOTE; 故用纯 scaler->RF 拟合后对 scaled 测试折算 SHAP,
    特征语义不变(scaler 单调线性), top 区相对顺序可比。"""
    import shap
    Xv = X.values
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats_outer,
                                    random_state=SEED)
    n_folds = outer.get_n_splits()
    abs_shap_per_fold = []     # 每折 mean|shap| over test patients, shape [n_feat]
    rank_per_fold = []         # 每折按 mean|shap| 的 rank
    for i, (tr, te) in enumerate(outer.split(Xv, strat)):
        scaler = StandardScaler().fit(Xv[tr])
        Xtr_s, Xte_s = scaler.transform(Xv[tr]), scaler.transform(Xv[te])
        if task == "clf":
            # SMOTE 仅训练折(与主表口径一致), SHAP 在原始测试折上算
            Xtr_r, ytr_r = SMOTE(random_state=SEED).fit_resample(Xtr_s, y[tr])
            rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
            rf.fit(Xtr_r, ytr_r)
            expl = shap.TreeExplainer(rf)
            sv = expl.shap_values(Xte_s, check_additivity=False)
            sv = _pick_positive_class_shap(sv)
        else:
            rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
            rf.fit(Xtr_s, y[tr])
            expl = shap.TreeExplainer(rf)
            sv = expl.shap_values(Xte_s, check_additivity=False)
        mabs = np.abs(sv).mean(axis=0)          # [n_feat]
        abs_shap_per_fold.append(mabs)
        order = np.argsort(-mabs)               # 大到小
        ranks = np.empty(len(feat), dtype=float)
        ranks[order] = np.arange(1, len(feat) + 1)
        rank_per_fold.append(ranks)
        if (i + 1) % 5 == 0 or (i + 1) == n_folds:
            print(f"    [shap:{task}] fold {i+1}/{n_folds}", flush=True)

    abs_shap_per_fold = np.vstack(abs_shap_per_fold)   # [n_folds, n_feat]
    rank_per_fold = np.vstack(rank_per_fold)
    mean_abs = abs_shap_per_fold.mean(axis=0)
    std_abs = abs_shap_per_fold.std(axis=0)
    mean_rank = rank_per_fold.mean(axis=0)
    std_rank = rank_per_fold.std(axis=0)
    topk = min(SHAP_TOPK, len(feat))
    in_topk_frac = (rank_per_fold <= topk).mean(axis=0)   # 该区落进 top-k 的折比例

    df = pd.DataFrame({
        "feature": feat,
        "mean_abs_shap": mean_abs,
        "std_abs_shap": std_abs,
        "mean_rank": mean_rank,
        "std_rank": std_rank,
        f"frac_in_top{topk}": in_topk_frac,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.insert(0, "global_rank", np.arange(1, len(df) + 1))
    return df


def _pick_positive_class_shap(sv):
    """兼容不同 shap 版本对二分类 RF 的返回形态, 取 severe(正类=1)的 SHAP。"""
    if isinstance(sv, list):                    # 旧版: [class0_arr, class1_arr]
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:                            # 新版: [n, n_feat, n_classes]
        return sv[:, :, 1]
    return sv


# ----------------------------------------------------------------------------- 3) 共线性聚类
def collinearity_clusters(X, feat):
    """Spearman |rho| 层次聚类。距离 = 1 - |rho|, 平均链接, 阈值 1-COLLIN_THRESH。
    返回 (cluster_df 长表, group_cols dict 组名->列名列表, corr DataFrame)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(X.values)
    rho = np.atleast_2d(rho)
    if rho.shape[0] != len(feat):               # spearmanr 在 2 列时返回标量, 兜底
        rho = np.corrcoef(X.values.T)
    absrho = np.abs(rho)
    np.fill_diagonal(absrho, 1.0)
    dist = 1.0 - absrho
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - COLLIN_THRESH, criterion="distance")

    group_cols = {}
    rows = []
    for lab in np.unique(labels):
        members = [feat[k] for k in range(len(feat)) if labels[k] == lab]
        gname = f"cluster{lab}" if len(members) > 1 else members[0]
        group_cols[gname] = members
        # 簇内最大 |rho|(单成员置 1)
        if len(members) > 1:
            idxs = [feat.index(m) for m in members]
            sub = absrho[np.ix_(idxs, idxs)].copy()
            np.fill_diagonal(sub, 0.0)
            max_within = float(sub.max())
        else:
            max_within = 1.0
        rows.append({"cluster": gname, "size": len(members),
                     "members": "|".join(members), "max_within_abs_rho": round(max_within, 3)})
    corr_df = pd.DataFrame(absrho, index=feat, columns=feat)
    cluster_df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return cluster_df, group_cols, corr_df


def annotate_collinear(perm_df, feat, corr_df, top_n=8):
    """给 top-n 区标注: 与哪些其它 top 区 |rho|>=COLLIN_THRESH (共线伙伴)。"""
    top_feats = [u for u in perm_df["unit"].tolist()[:top_n] if u in feat]
    notes = {}
    for f in top_feats:
        partners = [g for g in top_feats if g != f and corr_df.loc[f, g] >= COLLIN_THRESH]
        notes[f] = "|".join(f"{p}({corr_df.loc[f,p]:.2f})" for p in partners) if partners else ""
    perm_df = perm_df.copy()
    perm_df["collinear_with_top"] = perm_df["unit"].map(lambda u: notes.get(u, ""))
    return perm_df


# ----------------------------------------------------------------------------- 4) 体积校正(折内)
def volume_correction(task, X, y, strat, feat, n_repeats_outer):
    """三套折外置换重要度对比:
      A_full     : 原特征(含 volume)
      B_no_vol   : 去掉 lesion_volume 列后重新建模(看 top 脑区重要度是否暴涨=之前被 volume 吸走)
      C_resid    : 脑区列对 volume【折内】残差化后(去总病灶量线性成分), 看脑区是否还重要。
                   残差化系数每折只用训练折拟合 -> 无泄露(VolumeResidualizer 进 pipeline)。
    返回合并长表: feature, imp_full, imp_no_vol, imp_resid (均为 mean), 及对应 std。"""
    a = _oof_perm(task, X, y, strat, feat, n_repeats_outer).set_index("unit")

    feat_b = [c for c in feat if c != "lesion_volume"]
    Xb = X[feat_b]
    b = _oof_perm(task, Xb, y, strat, feat_b, n_repeats_outer).set_index("unit")

    # C 套: 同一份 raw X/feat, 由 pipeline 内 VolumeResidualizer 做折内残差化(系数仅训练折)
    c = _oof_perm(task, X, y, strat, feat, n_repeats_outer, residualize=True).set_index("unit")
    feat_c = feat

    allf = sorted(set(feat) | set(feat_b) | set(feat_c))
    rows = []
    for f in allf:
        rows.append({
            "feature": f,
            "imp_full_mean":   round(float(a.loc[f, "importance_mean"]), 5) if f in a.index else np.nan,
            "imp_full_std":    round(float(a.loc[f, "importance_std"]), 5) if f in a.index else np.nan,
            "imp_noVol_mean":  round(float(b.loc[f, "importance_mean"]), 5) if f in b.index else np.nan,
            "imp_noVol_std":   round(float(b.loc[f, "importance_std"]), 5) if f in b.index else np.nan,
            "imp_resid_mean":  round(float(c.loc[f, "importance_mean"]), 5) if f in c.index else np.nan,
            "imp_resid_std":   round(float(c.loc[f, "importance_std"]), 5) if f in c.index else np.nan,
            # 残差化后该列跨折为正的比例: 用作 "显著>0" 的稳健近似(perm 无解析 p 值)
            "resid_frac_folds_positive": round(float(c.loc[f, "frac_folds_positive"]), 3) if f in c.index else np.nan,
            "resid_retained_ratio": round(float(c.loc[f, "importance_mean"]) /
                                          float(a.loc[f, "importance_mean"]), 3)
                                    if (f in c.index and f in a.index and a.loc[f, "importance_mean"] > 1e-9)
                                    else np.nan,
        })
    out = pd.DataFrame(rows)
    # region_specific_flag = 折内体积校正后仍 "显著>0":
    #   残差后均值 > 0  且  >=60% 折为正(跨折一致, 非偶然单折)。
    #   不依赖与 full 的相对比例(retained_ratio 仅作参考列), 避免把 "残差后仍大" 误判为 "脑区特异"。
    out["region_specific_flag"] = (out["imp_resid_mean"] > 0) & \
        (out["resid_frac_folds_positive"] >= 0.60)
    out = out.sort_values("imp_full_mean", ascending=False).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- 主流程
def run(name, table_rel, n_repeats_outer, do_shap):
    print(f"\n========== {name} ({table_rel}) ==========", flush=True)
    X, y_cont, feat = load_df(table_rel)
    print(f"n={len(X)}  features={len(feat)}  feat={feat}", flush=True)

    cluster_df, group_cols, corr_df = collinearity_clusters(X, feat)
    cluster_df.to_csv(os.path.join(SUMMARY, f"xai_collinearity_{name}.csv"), index=False)
    multi = cluster_df[cluster_df["size"] > 1]
    print(f"  collinear clusters (size>1): {len(multi)}", flush=True)

    results = {}
    for task in ("clf", "reg"):
        print(f"  --- task={task} ---", flush=True)
        strat = strat_labels(task, y_cont)
        y = target_for(task, y_cont)

        # 1) 列级折外置换重要度 + 共线标注
        perm = _oof_perm(task, X, y, strat, feat, n_repeats_outer)
        perm = annotate_collinear(perm, feat, corr_df, top_n=8)
        perm.to_csv(os.path.join(SUMMARY, f"xai_perm_importance_{task}_{name}.csv"), index=False)

        # 3) 组级(共线簇)置换重要度
        grp = _oof_perm(task, X, y, strat, feat, n_repeats_outer, group_cols=group_cols)
        grp = grp.rename(columns={"unit": "cluster"})
        grp["members"] = grp["cluster"].map(lambda g: "|".join(group_cols.get(g, [g])))
        grp.to_csv(os.path.join(SUMMARY, f"xai_group_importance_{task}_{name}.csv"), index=False)

        # 4) 体积校正
        volc = volume_correction(task, X, y, strat, feat, n_repeats_outer)
        volc.to_csv(os.path.join(SUMMARY, f"xai_volume_correction_{task}_{name}.csv"), index=False)

        results[task] = {"perm": perm, "group": grp, "volc": volc}

        top5 = perm.head(5)[["unit", "importance_mean", "importance_std",
                             "frac_folds_positive", "collinear_with_top"]]
        print(f"    top5 perm ({task}):", flush=True)
        for _, r in top5.iterrows():
            print(f"      {r['unit']:16s} imp={r['importance_mean']:+.4f}±{r['importance_std']:.4f} "
                  f"pos_folds={r['frac_folds_positive']:.0%} collin=[{r['collinear_with_top']}]",
                  flush=True)

        # 2) SHAP(仅 coarse, 两任务都做)
        if do_shap:
            shp = oof_shap(task, X, y, strat, feat, n_repeats_outer)
            shp.to_csv(os.path.join(SUMMARY, f"xai_shap_{task}_{name}.csv"), index=False)
            results[task]["shap"] = shp
            topk = min(SHAP_TOPK, len(feat))
            print(f"    SHAP top5 ({task}), frac_in_top{topk} = 跨折稳定性:", flush=True)
            for _, r in shp.head(5).iterrows():
                print(f"      {r['feature']:16s} mean|SHAP|={r['mean_abs_shap']:.4f} "
                      f"rank={r['mean_rank']:.1f}±{r['std_rank']:.1f} "
                      f"in_top{topk}={r[f'frac_in_top{topk}']:.0%}", flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5x2 折快验(默认 5x10)")
    ap.add_argument("--coarse-only", action="store_true", help="只跑 arterial_coarse")
    args = ap.parse_args()
    os.makedirs(SUMMARY, exist_ok=True)
    n_rep = 2 if args.quick else N_REPEATS

    print(f"XAI RandomForest 脑区贡献分析  outer={N_SPLITS}x{n_rep}  perm_repeats={PERM_REPEATS}  "
          f"collin_thresh|rho|>={COLLIN_THRESH}", flush=True)
    print("口径: 预测关联 predictive association, 非因果定位。", flush=True)

    t0 = time.time()
    # arterial_coarse 做全套(含 SHAP); JHU_fine 做置换/组级/体积(SHAP 维度大不强制)
    run("arterial_coarse", TABLES["arterial_coarse"], n_rep, do_shap=True)
    if not args.coarse_only:
        run("JHU_fine", TABLES["JHU_fine"], n_rep, do_shap=False)
    print(f"\nDONE in {time.time()-t0:.0f}s. outputs -> results/summary/xai_*.csv", flush=True)

    print("\n[解释边界] 这些是【预测关联】: 扰动该列降低折外预测准确性, "
          "不代表该脑区因果导致严重失语。共线簇内单区重要度不可单独解读; "
          "残差校正只去 lesion_volume 的线性成分, 非线性共变仍可能残留。", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
