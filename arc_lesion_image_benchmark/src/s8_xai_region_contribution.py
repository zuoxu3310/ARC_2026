#!/usr/bin/env python
"""
s8_xai_region_contribution.py  -  leakage-safe explainability analysis of RandomForest region contributions

Scope of claims
---------------
Everything here is predictive association, not causal localisation. "Region X is important"
only means "with this model and these data, perturbing this column worsens out-of-fold
predictions"; it never means "region X causes severe aphasia".

What it does (RandomForest, two tasks x two granularities; focus on arterial_coarse as the most interpretable)
-------------------------------------------------------------------------------------------------------------
Tasks:
  * clf : classification, severe = wab_aq <= 50. In-fold setup = main s3 table
          (StandardScaler -> SMOTE -> RF), protocol = RepeatedStratifiedKFold (5x10).
  * reg : regression on wab_aq. In-fold setup = scaler -> RF (default hyperparameters),
          protocol = RepeatedStratifiedKFold (5x10); the continuous target is stratified by
          quantile bins (same as s5).
Granularities: arterial_coarse (15 columns, most interpretable) + JHU_fine (~71 columns).

Five analyses
-------------
1) Out-of-fold permutation importance: in each fold, permutation_importance on the test fold
   (fitted pipeline; permute raw X columns -> re-predict -> measure the metric drop).
   scaler/SMOTE are pipeline steps that only see the training fold at fit time, and SMOTE is
   not triggered at predict time, so permutation is leakage-safe. Mean +/- std across folds
   -> stability. More reliable than impurity (Gini) importance, which is biased toward
   high-cardinality/continuous columns and is computed on training data.
2) SHAP (arterial_coarse only, low dimension so interpretable): per fold, TreeExplainer computes
   SHAP on the test fold, aggregated out-of-fold as mean(|SHAP|); reports whether the top-k
   regions are stable across folds (rank consistency). Top-k changing across folds is flagged
   as a possible collinearity artifact.
3) Collinearity: Spearman correlation between features -> hierarchical clustering
   (distance 1-|rho|, threshold 0.4, i.e. |rho| >= 0.6 forms one cluster). Reports group-level
   permutation importance (all columns of a cluster permuted together) and marks which top
   regions are collinear with each other
   -> avoids a spurious "region X drives it" when it is really a cluster of co-lesioned regions.
4) Volume correction: (a) out-of-fold permutation importance of lesion_volume itself;
   (b) models with vs without lesion_volume: how top-region importance changes
       (a large drop = volume proxy);
   (c) linearly residualise each region column on lesion_volume (removing the linear component
       of total lesion load) and rerun permutation importance
   -> separates "region-specific" from "just a proxy for total lesion load".
5) Interpretation limits: printed at the end of the script.

Outputs
-------
  results/summary/xai_perm_importance_{task}_{name}.csv     out-of-fold permutation importance + stability
  results/summary/xai_shap_{task}_{name}.csv                SHAP aggregate + cross-fold stability (coarse only)
  results/summary/xai_collinearity_{name}.csv               correlation cluster membership
  results/summary/xai_group_importance_{task}_{name}.csv    group-level (collinear cluster) permutation importance
  results/summary/xai_volume_correction_{task}_{name}.csv   importance with/without volume and after residual correction

Usage
-----
  python src/s8_xai_region_contribution.py
  python src/s8_xai_region_contribution.py --quick   # quick 5x2-fold check
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
N_STRAT_BINS = 4        # stratification bins for the continuous regression target (same as s5)
PERM_REPEATS = 20       # permutations per column per fold
COLLIN_THRESH = 0.6     # |Spearman rho| >= 0.6 counts as the same cluster (distance threshold 1-0.6=0.4)
SHAP_TOPK = 8

TABLES = {
    "arterial_coarse": "data/ArterialAtlas156_features_merged.tsv",
    "JHU_fine":        "data/jhu156_features_merged.tsv",
}


# ----------------------------------------------------------------------------- data
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
    """Classification: the severe label itself; regression: wab_aq quantile bins (used only for splitting, never as features)."""
    if task == "clf":
        return (y_cont.values <= THRESHOLD).astype(int)
    return pd.qcut(y_cont, q=N_STRAT_BINS, labels=False, duplicates="drop").values


def target_for(task, y_cont):
    if task == "clf":
        return (y_cont.values <= THRESHOLD).astype(int)
    return y_cont.values.astype(float)


# ----------------------------------------------------------------------------- in-fold volume-residualising transformer
class VolumeResidualizer(BaseEstimator, TransformerMixin):
    """In-fold volume correction: OLS-residualise the given region columns on the lesion_volume column.

    Leakage guard: the OLS coefficients (intercept + slope) are fit in fit() on the training
    fold only; transform() applies the same coefficients to any fold (training or test). Inside
    an sklearn pipeline, pipeline.fit only sees the training fold -> the coefficients never touch
    the test fold. permutation_importance permutes raw test-fold columns, which pipeline.transform
    residualises with training-fold coefficients -> no preprocessing leakage.

    Fitting the coefficients on all 226 patients before splitting would let them see the test
    fold (leakage), which is why residualisation is done inside the pipeline.

    All parameters are column indices (the pipeline works on numpy arrays):
      vol_idx    : column position of lesion_volume in the feature matrix; None makes this step
                   the identity (no residualisation).
      region_idx : list of column positions to residualise (excluding volume and age). Other
                   columns are kept as is.
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
        v1 = np.column_stack([np.ones_like(v), v])           # [1, volume] design matrix, training fold only
        self.beta_ = {}
        for j in self.region_idx:
            beta, *_ = np.linalg.lstsq(v1, X[:, j], rcond=None)
            self.beta_[j] = beta                              # (intercept, slope), from the training fold
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if not getattr(self, "beta_", None):
            return X
        v = X[:, self.vol_idx]
        v1 = np.column_stack([np.ones_like(v), v])
        for j, beta in self.beta_.items():
            X[:, j] = X[:, j] - v1 @ beta                     # residuals using training-fold coefficients
        return X


# ----------------------------------------------------------------------------- models
def build_pipe(task, residualizer=None):
    """Main-table setup: clf = scaler->SMOTE->RF(500); reg = scaler->RF (default).
    Permutation importance acts on the pipeline, so scaler/SMOTE/residualizer only see the training fold at fit time.

    residualizer: if a VolumeResidualizer instance is given, it is inserted before the scaler (residualising on raw
                  volume); its coefficients only see the training fold in pipeline.fit -> in-fold volume correction,
                  no leakage."""
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
    """Build an in-fold VolumeResidualizer from the feature names: residualise region columns (excluding volume/age).
    Returns None if feat has no volume column (no volume correction for this granularity)."""
    if vol_col not in feat:
        return None
    vol_idx = feat.index(vol_col)
    region_idx = [feat.index(c) for c in feat if c not in (vol_col, "age_at_stroke")]
    return VolumeResidualizer(vol_idx=vol_idx, region_idx=region_idx)


def perm_scoring(task):
    # classification: balanced_accuracy (primary metric); regression: R2 (permutation_importance default estimator.score = R2)
    return "balanced_accuracy" if task == "clf" else "r2"


# ----------------------------------------------------------------------------- 1) out-of-fold permutation importance
def _oof_perm(task, X, y, strat, feat, n_repeats_outer, group_cols=None, residualize=False):
    """Permutation importance of the fitted pipeline on each test fold, aggregated across folds.

    group_cols: if given as {group name: [column names...]}, switch to group-level permutation
                (a group's columns are shuffled together; used for collinear clusters);
                None means per-column permutation.
    residualize: if True, insert an in-fold VolumeResidualizer into the pipeline (new per fold,
                 fit only on the training fold). Used by volume-correction set C -> coefficients
                 never touch the test fold, no preprocessing leakage.
    Returns a DataFrame: unit = column name (or group name), importance_mean/std/cv,
    frac_folds_positive, n_folds.
    Permutation acts on the raw test fold; pipeline.predict does not trigger SMOTE, and
    scaler/resid are already fit on the training fold -> no leakage.
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
        # new residualizer per fold so fitted coefficients are never shared across folds; fit on the training fold only (inside pipe.fit)
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
                for j in idxs:                       # columns of one group are shuffled together with the same perm
                    Xp[:, j] = Xte[perm, j]
                if task == "clf":
                    s = balanced_accuracy_score(yte, pipe.predict(Xp))
                else:
                    s = pipe.score(Xp, yte)
                drops.append(base - s)               # larger drop = more important
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


# ----------------------------------------------------------------------------- 2) SHAP (coarse only)
def oof_shap(task, X, y, strat, feat, n_repeats_outer):
    """Per fold, TreeExplainer computes SHAP on the test fold; aggregated out-of-fold as mean(|SHAP|).
    Also reports cross-fold top-k stability: from each fold's own top-k set, each region's frequency
    in the top-k + cross-fold rank std.

    Note: TreeExplainer takes the RF directly, not the SMOTE pipeline; so scaler and RF are fit
    separately and SHAP is computed on the scaled test fold. Feature meaning is unchanged (the scaler
    is a monotone linear map), so the relative order of top regions is comparable."""
    import shap
    Xv = X.values
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats_outer,
                                    random_state=SEED)
    n_folds = outer.get_n_splits()
    abs_shap_per_fold = []     # per fold: mean|shap| over test patients, shape [n_feat]
    rank_per_fold = []         # per fold: rank by mean|shap|
    for i, (tr, te) in enumerate(outer.split(Xv, strat)):
        scaler = StandardScaler().fit(Xv[tr])
        Xtr_s, Xte_s = scaler.transform(Xv[tr]), scaler.transform(Xv[te])
        if task == "clf":
            # SMOTE on the training fold only (same as the main table); SHAP on the non-oversampled test fold
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
        order = np.argsort(-mabs)               # descending
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
    in_topk_frac = (rank_per_fold <= topk).mean(axis=0)   # fraction of folds where the region is in the top-k

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
    """Handle the return shapes of different shap versions for a binary RF; return SHAP for the severe (positive = 1) class."""
    if isinstance(sv, list):                    # older shap: [class0_arr, class1_arr]
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:                            # newer shap: [n, n_feat, n_classes]
        return sv[:, :, 1]
    return sv


# ----------------------------------------------------------------------------- 3) collinearity clustering
def collinearity_clusters(X, feat):
    """Hierarchical clustering on Spearman |rho|. Distance = 1 - |rho|, average linkage, threshold 1-COLLIN_THRESH.
    Returns (cluster_df long table, group_cols dict group name -> list of column names, corr DataFrame)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(X.values)
    rho = np.atleast_2d(rho)
    if rho.shape[0] != len(feat):               # spearmanr returns a scalar for 2 columns; fallback
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
        # maximum |rho| within the cluster (1 for single-member clusters)
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
    """Annotate the top-n regions with the other top regions they are collinear with (|rho| >= COLLIN_THRESH)."""
    top_feats = [u for u in perm_df["unit"].tolist()[:top_n] if u in feat]
    notes = {}
    for f in top_feats:
        partners = [g for g in top_feats if g != f and corr_df.loc[f, g] >= COLLIN_THRESH]
        notes[f] = "|".join(f"{p}({corr_df.loc[f,p]:.2f})" for p in partners) if partners else ""
    perm_df = perm_df.copy()
    perm_df["collinear_with_top"] = perm_df["unit"].map(lambda u: notes.get(u, ""))
    return perm_df


# ----------------------------------------------------------------------------- 4) volume correction (in-fold)
def volume_correction(task, X, y, strat, feat, n_repeats_outer):
    """Compare three sets of out-of-fold permutation importance:
      A_full     : original features (including volume)
      B_no_vol   : model rebuilt without the lesion_volume column (does top-region importance jump,
                   i.e. was it previously absorbed by volume?)
      C_resid    : region columns residualised on volume within fold (linear component of total
                   lesion load removed); are the regions still important?
                   Residualisation coefficients are fit on the training fold only in each fold
                   -> no leakage (VolumeResidualizer inside the pipeline).
    Returns a merged long table: feature, imp_full, imp_no_vol, imp_resid (all means), plus the matching std."""
    a = _oof_perm(task, X, y, strat, feat, n_repeats_outer).set_index("unit")

    feat_b = [c for c in feat if c != "lesion_volume"]
    Xb = X[feat_b]
    b = _oof_perm(task, Xb, y, strat, feat_b, n_repeats_outer).set_index("unit")

    # set C: same raw X/feat; in-fold residualisation by the VolumeResidualizer inside the pipeline (coefficients from the training fold only)
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
            # fraction of folds with positive importance after residualisation: robust proxy for "significantly > 0" (permutation has no analytic p-value)
            "resid_frac_folds_positive": round(float(c.loc[f, "frac_folds_positive"]), 3) if f in c.index else np.nan,
            "resid_retained_ratio": round(float(c.loc[f, "importance_mean"]) /
                                          float(a.loc[f, "importance_mean"]), 3)
                                    if (f in c.index and f in a.index and a.loc[f, "importance_mean"] > 1e-9)
                                    else np.nan,
        })
    out = pd.DataFrame(rows)
    # region_specific_flag = still "significantly > 0" after in-fold volume correction:
    #   residual mean > 0  and  positive in >= 60% of folds (consistent across folds, not a single-fold fluke).
    #   Does not depend on the ratio to the full model (retained_ratio is a reference column only),
    #   to avoid misreading "still large after residualisation" as "region-specific".
    out["region_specific_flag"] = (out["imp_resid_mean"] > 0) & \
        (out["resid_frac_folds_positive"] >= 0.60)
    out = out.sort_values("imp_full_mean", ascending=False).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- main flow
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

        # 1) column-level out-of-fold permutation importance + collinearity annotation
        perm = _oof_perm(task, X, y, strat, feat, n_repeats_outer)
        perm = annotate_collinear(perm, feat, corr_df, top_n=8)
        perm.to_csv(os.path.join(SUMMARY, f"xai_perm_importance_{task}_{name}.csv"), index=False)

        # 3) group-level (collinear cluster) permutation importance
        grp = _oof_perm(task, X, y, strat, feat, n_repeats_outer, group_cols=group_cols)
        grp = grp.rename(columns={"unit": "cluster"})
        grp["members"] = grp["cluster"].map(lambda g: "|".join(group_cols.get(g, [g])))
        grp.to_csv(os.path.join(SUMMARY, f"xai_group_importance_{task}_{name}.csv"), index=False)

        # 4) volume correction
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

        # 2) SHAP (coarse only, both tasks)
        if do_shap:
            shp = oof_shap(task, X, y, strat, feat, n_repeats_outer)
            shp.to_csv(os.path.join(SUMMARY, f"xai_shap_{task}_{name}.csv"), index=False)
            results[task]["shap"] = shp
            topk = min(SHAP_TOPK, len(feat))
            print(f"    SHAP top5 ({task}), frac_in_top{topk} = cross-fold stability:", flush=True)
            for _, r in shp.head(5).iterrows():
                print(f"      {r['feature']:16s} mean|SHAP|={r['mean_abs_shap']:.4f} "
                      f"rank={r['mean_rank']:.1f}±{r['std_rank']:.1f} "
                      f"in_top{topk}={r[f'frac_in_top{topk}']:.0%}", flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quick 5x2-fold check (default 5x10)")
    ap.add_argument("--coarse-only", action="store_true", help="run arterial_coarse only")
    args = ap.parse_args()
    os.makedirs(SUMMARY, exist_ok=True)
    n_rep = 2 if args.quick else N_REPEATS

    print(f"XAI RandomForest region-contribution analysis  outer={N_SPLITS}x{n_rep}  perm_repeats={PERM_REPEATS}  "
          f"collin_thresh|rho|>={COLLIN_THRESH}", flush=True)
    print("Scope: predictive association, not causal localisation.", flush=True)

    t0 = time.time()
    # arterial_coarse gets the full set (incl. SHAP); JHU_fine gets permutation/group/volume (SHAP optional at high dimension)
    run("arterial_coarse", TABLES["arterial_coarse"], n_rep, do_shap=True)
    if not args.coarse_only:
        run("JHU_fine", TABLES["JHU_fine"], n_rep, do_shap=False)
    print(f"\nDONE in {time.time()-t0:.0f}s. outputs -> results/summary/xai_*.csv", flush=True)

    print("\n[Interpretation limits] These are predictive associations: perturbing a column lowers out-of-fold predictive accuracy; "
          "this does not mean the region causes severe aphasia. Single-region importance within a collinear cluster should not be read on its own; "
          "residual correction removes only the linear component of lesion_volume, so nonlinear covariation may remain.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
