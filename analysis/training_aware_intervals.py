"""Training-variability-aware intervals for the key paired contrasts, from saved predictions only.

No model is fitted here. Every number is recomputed from stored out-of-fold predictions,
stored partitions and stored bootstrap weights; anchors are checked before any interval is formed.

Methods:
  BF_fold          Bouckaert & Frank (2004) corrected repeated k-fold CV t-test on fold-level paired
                   differences: Var = (1/J + n_test/n_train) * S^2, J = r*k, df = J-1
                   (Nadeau & Bengio 2003 variance correction).
  BF_fold_fitonly  Same, but n_train = model-fitting patients only (image case: excludes the
                   validation patients used for threshold/config selection) -> larger correction.
  NB_repeat        Repeat-level analogue for the paper's pooled-per-repeat estimand: under the
                   B&F equicorrelation model (rho = n_test/n per fold pair) the r repeat means have
                   correlation k*rho/(1+(k-1)rho), so Var = (1/r + n/n_train) * S_R^2, df = r-1.
  repeat_plus_boot Conservative: Var = Var_boot (existing conditional patient bootstrap of the
                   repeat-averaged difference) + NB_repeat variance, df = r-1. Centered on the paper's
                   estimate; always at least as wide as the existing interval.
  patient_bootstrap_existing  The paper's existing percentile interval, reproduced (reference row).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import sys
# ROOT is the project root of the original analysis (argument 1, default: repository root).
# Inputs: image-case evidence (supplementary-controls/evidence/image-controls in the
# supplementary results), CNN partitions, fixed-prediction control results and subset lists.
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/training-aware-intervals'
IMG = ROOT / 'outputs/supplementary-controls/evidence/image-controls'
PART = ROOT / 'outputs/cnn-same-input-2026-10-03/partitions.json'
FRZ = ROOT / 'arc_lesion_image_benchmark/results/frozen_prediction_control_2026-09-22'
SUBS = ROOT / 'arc_lesion_image_benchmark/data/subsets'
TOF = ROOT / 'outputs/supplementary-controls/evidence/training-only-features'
SEEDS = list(range(1000, 1020))
MODELS = ['RandomForest', 'XGBoost']
SUBSETS = ['teghipco_idlist', 'has_flair']
SUBSET_LABEL = {'teghipco_idlist': 'published-list intersection (172)', 'has_flair': 'FLAIR-eligible (135)'}
TOL_FILE = 1e-10      # recomputation vs stored CSV value
ANCHOR_DECIMALS = 6   # anchors are the 6-decimal values reported in the paper

USED: dict[str, Path] = {}


def use(path: Path) -> Path:
    USED[str(path)] = path
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ANCHORS: list[dict] = []


def anchor(family, name, reproduced, expected, decimals=None, tol=None):
    if decimals is not None:
        ok = round(float(reproduced), decimals) == round(float(expected), decimals)
        rule = f'equal at {decimals} decimals'
    else:
        ok = abs(float(reproduced) - float(expected)) <= tol
        rule = f'|diff| <= {tol:g}'
    ANCHORS.append(dict(family=family, check=name, reproduced=float(reproduced), expected=float(expected),
                        abs_diff=abs(float(reproduced) - float(expected)), rule=rule, passed=bool(ok)))
    return ok


def t_row(center, se, df):
    tcrit = float(stats.t.ppf(0.975, df))
    p = float(2 * stats.t.sf(abs(center / se), df)) if se > 0 else float('nan')
    return tcrit, center - tcrit * se, center + tcrit * se, p


def rows_for_contrast(base, rep_d, fold_d, boot_draws, anchor_delta, boot_ci, sizes, k, r):
    """Build all interval rows for one contrast.

    rep_d: (r,) repeat-level paired differences of the pooled metric (paper estimand)
    fold_d: (r*k,) fold-level paired differences
    boot_draws: (B,) existing conditional patient-bootstrap draws of the repeat-averaged difference
    sizes: dict n, n_test_fold, n_train_fold, optional n_fit_fold
    """
    out = []
    n, n_te, n_tr = sizes['n'], sizes['n_test_fold'], sizes['n_train_fold']
    s_r = float(np.std(rep_d, ddof=1))
    s_f = float(np.std(fold_d, ddof=1))
    var_boot = float(np.var(boot_draws, ddof=1))
    common = dict(base, paper_anchor_difference=anchor_delta, existing_bootstrap_ci_low=boot_ci[0],
                  existing_bootstrap_ci_high=boot_ci[1])

    # Existing conditional patient bootstrap (reference row)
    out.append(dict(common, method='patient_bootstrap_existing', unit='patient (bootstrap draw)', J=len(boot_draws),
                    n_train=np.nan, n_test=n, correction_factor=np.nan, mean_difference=anchor_delta,
                    naive_SD=np.nan, naive_SE=np.nan, bootstrap_SD=np.sqrt(var_boot), corrected_SE=np.sqrt(var_boot),
                    df=np.nan, t_crit=np.nan, ci_low=boot_ci[0], ci_high=boot_ci[1], p_value=np.nan,
                    note='Existing percentile interval; conditional on stored fits (no training variation)'))

    J = len(fold_d)
    fac = 1 / J + n_te / n_tr
    se = np.sqrt(fac) * s_f
    tc, lo, hi, p = t_row(float(np.mean(fold_d)), se, J - 1)
    out.append(dict(common, method='BF_fold', unit=f'fold (r={r} x k={k})', J=J, n_train=n_tr, n_test=n_te,
                    correction_factor=fac, mean_difference=float(np.mean(fold_d)), naive_SD=s_f,
                    naive_SE=s_f / np.sqrt(J), bootstrap_SD=np.nan, corrected_SE=se, df=J - 1, t_crit=tc,
                    ci_low=lo, ci_high=hi, p_value=p,
                    note='Canonical corrected repeated k-fold CV test; centre is the mean of per-fold metric differences '
                         '(per-fold estimand, differs from pooled anchor)'))
    if 'n_fit_fold' in sizes:
        fac2 = 1 / J + n_te / sizes['n_fit_fold']
        se2 = np.sqrt(fac2) * s_f
        tc, lo, hi, p = t_row(float(np.mean(fold_d)), se2, J - 1)
        out.append(dict(common, method='BF_fold_fitonly', unit=f'fold (r={r} x k={k})', J=J,
                        n_train=sizes['n_fit_fold'], n_test=n_te, correction_factor=fac2,
                        mean_difference=float(np.mean(fold_d)), naive_SD=s_f, naive_SE=s_f / np.sqrt(J),
                        bootstrap_SD=np.nan, corrected_SE=se2, df=J - 1, t_crit=tc, ci_low=lo, ci_high=hi, p_value=p,
                        note='As BF_fold but n_train = fitting patients only (validation patients excluded); '
                             'larger, conservative correction'))

    fac_r = 1 / r + n / n_tr
    se_r = np.sqrt(fac_r) * s_r
    tc, lo, hi, p = t_row(anchor_delta, se_r, r - 1)
    out.append(dict(common, method='NB_repeat', unit=f'repeat of {k}-fold CV', J=r, n_train=n_tr, n_test=n,
                    correction_factor=fac_r, mean_difference=anchor_delta, naive_SD=s_r, naive_SE=s_r / np.sqrt(r),
                    bootstrap_SD=np.nan, corrected_SE=se_r, df=r - 1, t_crit=tc, ci_low=lo, ci_high=hi, p_value=p,
                    note='Repeat-level analogue of B&F (n_test = all n patients scored once per repeat). Across-repeat '
                         'spread contains no test-sample variation (every repeat scores the same patients), so this '
                         'row alone can be too narrow'))

    v_t = fac_r * s_r ** 2
    se_c = np.sqrt(var_boot + v_t)
    # Welch-Satterthwaite: bootstrap variance treated as known, the repeat term has r-1 df
    df_ws = (var_boot + v_t) ** 2 / (v_t ** 2 / (r - 1))
    for method, dfx, extra in [('repeat_plus_boot', df_ws, 'Welch-Satterthwaite df'),
                               ('repeat_plus_boot_df19', r - 1, 'df fixed at r-1 (most conservative)')]:
        tc, lo, hi, p = t_row(anchor_delta, se_c, dfx)
        out.append(dict(common, method=method, unit=f'repeat of {k}-fold CV + patient bootstrap', J=r,
                        n_train=n_tr, n_test=n, correction_factor=fac_r, mean_difference=anchor_delta, naive_SD=s_r,
                        naive_SE=s_r / np.sqrt(r), bootstrap_SD=np.sqrt(var_boot), corrected_SE=se_c, df=dfx,
                        t_crit=tc, ci_low=lo, ci_high=hi, p_value=p,
                        training_share_of_variance=v_t / (var_boot + v_t),
                        note='Conservative: existing test-sample (bootstrap) variance + corrected training/partition '
                             f'variance; centred on the paper estimate; {extra}'))
    for row in out:
        row['includes_zero'] = bool(row['ci_low'] <= 0 <= row['ci_high'])
    return out


# --------------------------------------------------------------------------------------------
# (a) Shared-image case
# --------------------------------------------------------------------------------------------
def image_case():
    fam = 'image_173'
    plan = json.loads(use(IMG / 'image-plan.json').read_text())
    part = json.loads(use(PART).read_text())
    ids = plan['ids']
    assert ids == part['subject_ids'] and len(ids) == 173
    units = part['units']
    assert len(units) == 120
    for u in units:
        te, dev, fit, va = (set(u[k]) for k in ['test', 'development', 'cnn_train', 'cnn_validation'])
        assert len(te) + len(dev) == 173 and not te & dev and dev == fit | va and not fit & va
    n_te = np.mean([len(u['test']) for u in units]); n_dev = np.mean([len(u['development']) for u in units])
    n_fit = np.mean([len(u['cnn_train']) for u in units])
    assert abs(n_te - 173 / 6) < 1e-12

    allpred = pd.read_csv(use(IMG / 'all-predictions.csv'))
    rep_csv = pd.read_csv(use(IMG / 'repeat-metrics.csv'))
    mean_csv = pd.read_csv(use(IMG / 'mean-metrics.csv'))
    pc_csv = pd.read_csv(use(IMG / 'paired-contrasts.csv'))
    W = np.load(use(IMG / 'bootstrap-weights.npz'))['weights']
    assert W.shape == (5000, 173)

    y = (allpred.drop_duplicates('participant_id').set_index('participant_id').loc[ids, 'y_true'].to_numpy())
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    metrics = ['F1', 'balanced_accuracy', 'AUC']

    # Repeat-level (pooled) metrics and fold-level metrics
    rep, fold, boot = {}, {}, {}
    for (m, dec), g in allpred.groupby(['method', 'decision']):
        P = g.pivot(index='repeat', columns='participant_id', values='y_pred').loc[range(1, 21), ids].to_numpy()
        S = g.pivot(index='repeat', columns='participant_id', values='score').loc[range(1, 21), ids].to_numpy()
        rv = {mt: [] for mt in metrics}
        for i in range(20):
            rv['F1'].append(f1_score(y, P[i])); rv['balanced_accuracy'].append(balanced_accuracy_score(y, P[i]))
            rv['AUC'].append(roc_auc_score(y, S[i]))
        rep[(m, dec)] = {mt: np.array(v) for mt, v in rv.items()}
        fv = {mt: [] for mt in metrics}
        for (rr, ff), gf in g.sort_values(['repeat', 'fold']).groupby(['repeat', 'fold']):
            yt, yp, sc = gf.y_true.to_numpy(), gf.y_pred.to_numpy(), gf.score.to_numpy()
            fv['F1'].append(f1_score(yt, yp, zero_division=0)); fv['balanced_accuracy'].append(balanced_accuracy_score(yt, yp))
            fv['AUC'].append(roc_auc_score(yt, sc))
        fold[(m, dec)] = {mt: np.array(v) for mt, v in fv.items()}
        # Same bootstrap as image_controls.analyze(): shared patient weights across repeats and methods
        tp = W @ ((P == 1) * (y == 1)).T; fp = W @ ((P == 1) * (y == 0)).T
        fn = len(pos) - tp; tn = len(neg) - fp
        bs = {'F1': (2 * tp / (2 * tp + fp + fn)).mean(axis=1),
              'balanced_accuracy': (.5 * (tp / len(pos) + tn / len(neg))).mean(axis=1)}
        A = (S[:, pos, None] > S[:, None, neg]).astype(float) + .5 * (S[:, pos, None] == S[:, None, neg])
        bs['AUC'] = np.einsum('bi,ij,bj->b', W[:, pos], A.mean(axis=0), W[:, neg]) / (len(pos) * len(neg))
        boot[(m, dec)] = bs
        # file consistency
        sub = rep_csv[(rep_csv.method == m) & (rep_csv.decision == dec)].sort_values('repeat')
        for mt in metrics:
            anchor(fam, f'{m}/{dec}/{mt} per-repeat vs repeat-metrics.csv (max)',
                   np.max(np.abs(rep[(m, dec)][mt] - sub[mt].to_numpy())), 0.0, tol=TOL_FILE)
            mm = mean_csv[(mean_csv.method == m) & (mean_csv.decision == dec)][mt].iloc[0]
            anchor(fam, f'{m}/{dec}/{mt} mean vs mean-metrics.csv', rep[(m, dec)][mt].mean(), mm, tol=TOL_FILE)

    # Paper anchors (6 decimals)
    ok = True
    ok &= anchor(fam, 'RF (cnn-ratio, validation-F1) F1', rep[('RF_cnnratio', 'validation_F1')]['F1'].mean(), 0.659465, ANCHOR_DECIMALS)
    ok &= anchor(fam, 'CNN (validation-F1) F1', rep[('CNN', 'validation_F1')]['F1'].mean(), 0.668293, ANCHOR_DECIMALS)
    ok &= anchor(fam, 'XGB developed (validation-F1) F1', rep[('XGB_developed', 'validation_F1')]['F1'].mean(), 0.606456, ANCHOR_DECIMALS)
    ok &= anchor(fam, 'RF AUC', rep[('RF_cnnratio', 'validation_F1')]['AUC'].mean(), 0.829361, ANCHOR_DECIMALS)
    ok &= anchor(fam, 'CNN AUC', rep[('CNN', 'validation_F1')]['AUC'].mean(), 0.833590, ANCHOR_DECIMALS)
    ok &= anchor(fam, 'RF F1 at fixed 0.5', rep[('RF_cnnratio', 'fixed_0.5')]['F1'].mean(), 0.447559, ANCHOR_DECIMALS)

    contrasts = [
        ('CNN - RF (cnn-ratio), validation-F1 threshold', ('CNN', 'validation_F1'), ('RF_cnnratio', 'validation_F1'), metrics, True),
        ('CNN - XGB developed, validation-F1 threshold', ('CNN', 'validation_F1'), ('XGB_developed', 'validation_F1'), metrics, True),
        ('RF (cnn-ratio): validation-F1 minus fixed-0.5 threshold', ('RF_cnnratio', 'validation_F1'), ('RF_cnnratio', 'fixed_0.5'), ['F1'], False),
        ('CNN - RF (inverse-frequency), validation-F1 threshold', ('CNN', 'validation_F1'), ('RF_inverse', 'validation_F1'), ['F1'], False),
        ('CNN - RF (cnn-ratio), fixed 0.5 threshold', ('CNN', 'fixed_0.5'), ('RF_cnnratio', 'fixed_0.5'), ['F1'], False),
        # Not reported in the paper (no stored contrast row); both component means are anchor-checked above.
        ('RF (cnn-ratio) - XGB developed, validation-F1 threshold', ('RF_cnnratio', 'validation_F1'), ('XGB_developed', 'validation_F1'), ['F1'], False),
    ]
    expected = {('CNN - RF (cnn-ratio), validation-F1 threshold', 'F1'): (0.008828, -0.015743, 0.035571),
                ('CNN - XGB developed, validation-F1 threshold', 'F1'): (0.061837, 0.026729, 0.103470)}
    rows, units_rep, units_fold = [], [], []
    for label, L, R, mts, key in contrasts:
        for mt in mts:
            rep_d = rep[L][mt] - rep[R][mt]; fold_d = fold[L][mt] - fold[R][mt]
            draws = boot[L][mt] - boot[R][mt]; lo, hi = np.quantile(draws, [.025, .975])
            stored = pc_csv[(pc_csv.left_method == L[0]) & (pc_csv.left_decision == L[1]) & (pc_csv.right_method == R[0])
                            & (pc_csv.right_decision == R[1]) & (pc_csv.metric == mt)]
            c_ok = True
            if len(stored):
                stored = stored.iloc[0]
                c_ok &= anchor(fam, f'{label} {mt} delta vs paired-contrasts.csv', rep_d.mean(), stored.delta, tol=TOL_FILE)
                c_ok &= anchor(fam, f'{label} {mt} CI low vs paired-contrasts.csv', lo, stored.ci_low, tol=TOL_FILE)
                c_ok &= anchor(fam, f'{label} {mt} CI high vs paired-contrasts.csv', hi, stored.ci_high, tol=TOL_FILE)
            else:
                c_ok &= anchor(fam, f'{label} {mt} delta = difference of anchor-checked means', rep_d.mean(),
                               rep[L][mt].mean() - rep[R][mt].mean(), tol=TOL_FILE)
            if (label, mt) in expected:
                e = expected[(label, mt)]
                c_ok &= anchor(fam, f'{label} {mt} delta (paper)', rep_d.mean(), e[0], ANCHOR_DECIMALS)
                c_ok &= anchor(fam, f'{label} {mt} CI low (paper)', lo, e[1], ANCHOR_DECIMALS)
                c_ok &= anchor(fam, f'{label} {mt} CI high (paper)', hi, e[2], ANCHOR_DECIMALS)
            base = dict(family=fam, contrast=label, metric=mt, key_contrast=key)
            if not (ok and c_ok):
                rows.append(dict(base, method='ANCHOR_FAIL', note='Anchor not reproduced; contrast skipped'))
                continue
            rows += rows_for_contrast(base, rep_d, fold_d, draws, float(rep_d.mean()), (lo, hi),
                                      dict(n=173, n_test_fold=n_te, n_train_fold=n_dev, n_fit_fold=n_fit), k=6, r=20)
            units_rep += [dict(family=fam, contrast=label, metric=mt, repeat=i + 1, difference=v) for i, v in enumerate(rep_d)]
            units_fold += [dict(family=fam, contrast=label, metric=mt, repeat=i // 6 + 1, fold=i % 6 + 1, difference=v)
                           for i, v in enumerate(fold_d)]
    return rows, units_rep, units_fold


# --------------------------------------------------------------------------------------------
# (b)/(c) ARC fixed-prediction scoring contrasts
# --------------------------------------------------------------------------------------------
def weighted_pearson(weights, y, P):
    """Identical algebra to s31_frozen_prediction_control.weighted_metrics (regression score)."""
    W = np.atleast_2d(weights).astype(float)
    mass = W.sum(axis=1)[:, None]
    sy = (W @ y)[:, None]; sp = W @ P.T
    cross = W @ (P * y).T - sy * sp / mass
    vy = (W @ (y * y))[:, None] - sy * sy / mass
    vp = W @ (P * P).T - sp * sp / mass
    return cross / np.sqrt(np.maximum(vy * vp, 0))


def arc_family(source):
    fam = 'arc_frozen_dictionary' if source == 'frozen' else 'arc_training_only_features'
    ids = [l.strip() for l in use(SUBS / 'full_ids.txt').read_text().splitlines() if l.strip()]
    masks = {}
    for s in SUBSETS:
        sid = {l.strip() for l in use(SUBS / f'{s}_ids.txt').read_text().splitlines() if l.strip()}
        masks[s] = np.isin(ids, sorted(sid))
    member = pd.read_csv(use(FRZ / 'patient_cohort_membership.csv')).set_index('participant_id').loc[ids]
    for s in SUBSETS:
        assert np.array_equal(member[s].to_numpy(bool), masks[s])
    assert masks['teghipco_idlist'].sum() == 172 and masks['has_flair'].sum() == 135
    boot = np.load(use(FRZ / 'resampling_patient_weights.npz'))['patient_bootstrap']
    assert boot.shape == (2000, 226)
    if source == 'frozen':
        summ = pd.read_csv(use(FRZ / 'frozen_prediction_summary.csv'))
        seeds_csv = pd.read_csv(use(FRZ / 'frozen_scores_by_seed.csv'))
        dist = np.load(use(FRZ / 'resampling_metric_distributions.npz'))
    else:
        fc = pd.read_csv(use(TOF / 'factorial_contrasts.csv'))
        rs = pd.read_csv(use(TOF / 'repeat_scores.csv'))
    rows, units_rep, units_fold = [], [], []
    for model in MODELS:
        frames, n_tr_list, n_te_list = [], [], []
        for seed in SEEDS:
            if source == 'frozen':
                f = pd.read_csv(use(FRZ / 'predictions' / f'regression_{model}_{seed}.csv'))
                folds = json.loads(use(FRZ / 'folds' / f'regression_{model}_{seed}.json').read_text())
            else:
                u = TOF / 'units' / f'full__regression__{model}__{seed}'
                f = pd.read_csv(use(u / 'predictions.csv')); assert (f.train_subset == 'full').all()
                folds = json.loads(use(u / 'folds.json').read_text())
            assert f.participant_id.tolist() == ids
            # fold labels in predictions must match stored fold membership
            for fd in folds:
                te = set(fd['test_ids']); assert te == set(f.participant_id[f.fold == fd['fold']])
                assert not te & set(fd['train_ids']) and len(te) + len(fd['train_ids']) == 226
                n_tr_list.append(len(fd['train_ids'])); n_te_list.append(len(te))
            frames.append(f)
        y = frames[0].y_true.to_numpy(float)
        assert all(np.array_equal(f.y_true.to_numpy(float), y) for f in frames)
        P = np.stack([f.prediction.to_numpy(float) for f in frames])
        fold_lab = np.stack([f.fold.to_numpy() for f in frames])
        full_rep = weighted_pearson(np.ones(226), y, P)[0]
        full_boot = weighted_pearson(boot, y, P)
        for s in SUBSETS:
            mask = masks[s]
            sub_rep = weighted_pearson(mask, y, P)[0]
            rep_d = sub_rep - full_rep
            draws = (weighted_pearson(boot * mask, y, P) - full_boot).mean(axis=1)
            lo, hi = np.quantile(draws, [.025, .975])
            label = f'{model}: {SUBSET_LABEL[s]} minus full pool (226), fixed predictions'
            c_ok = True
            if source == 'frozen':
                sm = summ[(summ.task == 'regression') & (summ.model == model) & (summ.metric == 'score')].set_index('subset')
                c_ok &= anchor(fam, f'{model} full r vs summary', full_rep.mean(), sm.loc['full', 'observed'], tol=TOL_FILE)
                c_ok &= anchor(fam, f'{model} {s} r vs summary', sub_rep.mean(), sm.loc[s, 'observed'], tol=TOL_FILE)
                c_ok &= anchor(fam, f'{model} {s} delta CI low vs summary', lo, sm.loc[s, 'delta_ci_low'], tol=TOL_FILE)
                c_ok &= anchor(fam, f'{model} {s} delta CI high vs summary', hi, sm.loc[s, 'delta_ci_high'], tol=TOL_FILE)
                saved = dist[f'regression__{model}__{s}__score__bootstrap_delta']
                c_ok &= anchor(fam, f'{model} {s} bootstrap draws vs saved npz (max)', np.max(np.abs(saved - draws)), 0.0, tol=TOL_FILE)
                for sub_name, vals in [('full', full_rep), (s, sub_rep)]:
                    sc = seeds_csv[(seeds_csv.task == 'regression') & (seeds_csv.model == model) & (seeds_csv.subset == sub_name)].sort_values('seed')
                    c_ok &= anchor(fam, f'{model} {sub_name} per-seed r vs frozen_scores_by_seed.csv (max)',
                                   np.max(np.abs(sc.score.to_numpy() - vals)), 0.0, tol=TOL_FILE)
                if model == 'RandomForest':
                    c_ok &= anchor(fam, 'RF full-pool mean r (paper)', full_rep.mean(), 0.712333, ANCHOR_DECIMALS)
                    c_ok &= anchor(fam, f'RF {s} mean r (paper)', sub_rep.mean(),
                                   {'teghipco_idlist': 0.755883, 'has_flair': 0.625496}[s], ANCHOR_DECIMALS)
            else:
                st = fc[(fc.source == 'training_only') & (fc.task == 'regression') & (fc.model == model) & (fc.subset == s)
                        & (fc.metric == 'score') & (fc.contrast == 'scoring_at_full_training')].iloc[0]
                c_ok &= anchor(fam, f'{model} {s} S01-S00 delta vs factorial_contrasts.csv', rep_d.mean(), st.delta, tol=TOL_FILE)
                c_ok &= anchor(fam, f'{model} {s} S01-S00 CI low vs factorial_contrasts.csv', lo, st.ci_low, tol=TOL_FILE)
                c_ok &= anchor(fam, f'{model} {s} S01-S00 CI high vs factorial_contrasts.csv', hi, st.ci_high, tol=TOL_FILE)
                for cell, vals in [('S00', full_rep), ('S01', sub_rep)]:
                    sc = rs[(rs.source == 'training_only') & (rs.task == 'regression') & (rs.model == model) & (rs.subset == s)
                            & (rs.cell == cell) & (rs.metric == 'score')].sort_values('seed')
                    c_ok &= anchor(fam, f'{model} {s} {cell} per-seed r vs repeat_scores.csv (max)',
                                   np.max(np.abs(sc.value.to_numpy() - vals)), 0.0, tol=TOL_FILE)
                if model == 'RandomForest':
                    c_ok &= anchor(fam, f'RF training-only {s} scoring contrast (paper, 3 d.p.)', rep_d.mean(),
                                   {'teghipco_idlist': 0.044, 'has_flair': -0.088}[s], 3)
            # fold-level differences: r on fold-test ∩ subset minus r on full fold-test
            fold_d = []
            for i in range(len(SEEDS)):
                for fl in range(1, 6):
                    te = fold_lab[i] == fl
                    r_full = np.corrcoef(y[te], P[i, te])[0, 1]
                    r_sub = np.corrcoef(y[te & mask], P[i, te & mask])[0, 1]
                    fold_d.append(r_sub - r_full)
            fold_d = np.array(fold_d)
            base = dict(family=fam, contrast=label, metric='Pearson r', key_contrast=True)
            if not c_ok:
                rows.append(dict(base, method='ANCHOR_FAIL', note='Anchor not reproduced; contrast skipped'))
                continue
            rows += rows_for_contrast(base, rep_d, fold_d, draws, float(rep_d.mean()), (lo, hi),
                                      dict(n=226, n_test_fold=float(np.mean(n_te_list)), n_train_fold=float(np.mean(n_tr_list))),
                                      k=5, r=20)
            units_rep += [dict(family=fam, contrast=label, metric='Pearson r', repeat=i + 1, difference=v) for i, v in enumerate(rep_d)]
            units_fold += [dict(family=fam, contrast=label, metric='Pearson r', repeat=i // 5 + 1, fold=i % 5 + 1, difference=v)
                           for i, v in enumerate(fold_d)]
    return rows, units_rep, units_fold


def main():
    rows, ur, uf = image_case()
    for src in ['frozen', 'training_only']:
        a, b, c = arc_family(src); rows += a; ur += b; uf += c
    cols = ['family', 'contrast', 'metric', 'key_contrast', 'method', 'unit', 'J', 'n_train', 'n_test', 'correction_factor',
            'mean_difference', 'paper_anchor_difference', 'naive_SD', 'naive_SE', 'bootstrap_SD', 'corrected_SE', 'df',
            't_crit', 'ci_low', 'ci_high', 'p_value', 'includes_zero', 'training_share_of_variance', 'existing_bootstrap_ci_low',
            'existing_bootstrap_ci_high', 'note']
    df = pd.DataFrame(rows).reindex(columns=cols)
    df.to_csv(OUT / 'corrected_intervals.csv', index=False, float_format='%.6f')
    pd.DataFrame(ur).to_csv(OUT / 'unit_differences_repeat.csv', index=False, float_format='%.10f')
    pd.DataFrame(uf).to_csv(OUT / 'unit_differences_fold.csv', index=False, float_format='%.10f')
    an = pd.DataFrame(ANCHORS); an.to_csv(OUT / 'anchor_checks.csv', index=False)
    pd.DataFrame([dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in
                  sorted(USED.values())]).to_csv(OUT / 'input_sha256.csv', index=False)
    print(f'anchors: {int(an.passed.sum())}/{len(an)} passed; rows: {len(df)}; inputs hashed: {len(USED)}')
    if not an.passed.all():
        print(an[~an.passed].to_string())


if __name__ == '__main__':
    main()
