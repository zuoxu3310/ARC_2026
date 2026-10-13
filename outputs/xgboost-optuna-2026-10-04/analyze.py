"""Verify every saved selection/model and compare paired held-out predictions."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

import run

B = 5000
BOOTSTRAP_SEED = 2026100409
METHODS = ['CNN', 'XGBoost', *run.METHODS]


def manual(y, pred, score):
    tp = int(((y==1)&(pred==1)).sum())
    fn = int(((y==1)&(pred==0)).sum())
    fp = int(((y==0)&(pred==1)).sum())
    tn = int(((y==0)&(pred==0)).sum())
    positive, negative = score[y==1][:, None], score[y==0][None, :]
    return dict(F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                BA=.5*(tp/(tp+fn)+tn/(tn+fp)),
                AUC=float(((positive>negative)+.5*(positive==negative)).mean()),
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                TP=tp, FN=fn, FP=fp, TN=tn)


def independently_select_threshold(y, score):
    candidates = [(float(t), float(f1_score(y, score>=t, zero_division=0)))
                  for t in run.THRESHOLDS]
    return min(candidates, key=lambda item: (-item[1], round(abs(item[0]-.5),8), item[0]))[0]


def verify():
    ids, flat, y, units = run.load_frozen()
    rows, trial_checks, warnings = [], 0, []
    old_part = json.loads((run.OLD/'partitions.json').read_text())
    for unit, original in zip(units, old_part['units']):
        for key in ['name','development','test','fit_seed','repeat','fold']:
            assert unit[key] == original[key]
        folder = run.OUT/'units'/unit['name']
        record = json.loads((folder/'complete.json').read_text())
        selection = json.loads((folder/'selection.json').read_text())
        assert record['trials'] == run.N_TRIALS == 100
        assert run.sha(folder/'model.ubj') == record['checkpoint_sha256']
        assert run.sha(folder/'predictions.csv') == record['prediction_sha256']
        assert run.sha(folder/'selection.json') == record['selection_sha256']
        assert selection['selected_before_outer_test_unix'] < record['completed_unix']
        dev, te = np.array(unit['development']), np.array(unit['test'])
        assert not set(dev)&set(te) and set(dev)|set(te) == set(range(173))
        assert selection['development_ids'] == ids[dev].tolist()
        inner_seen = np.zeros(len(dev), dtype=int)
        for tr, va in unit['optuna_inner']:
            assert not set(tr)&set(va)
            assert set(tr)|set(va) == set(range(len(dev)))
            assert not set(dev[tr])&set(te) and not set(dev[va])&set(te)
            inner_seen[va] += 1
        assert np.all(inner_seen == 1)
        state = joblib.load(folder/'search-state.joblib')
        trials = state['study'].trials
        saved = np.load(folder/'inner-predictions.npz')
        scores = saved['score']
        assert scores.shape == (100, len(dev))
        np.testing.assert_array_equal(scores, np.stack(state['oof']))
        np.testing.assert_array_equal(saved['y'], y[dev])
        np.testing.assert_array_equal(saved['participant_ids'], ids[dev])
        assert trials[0].params == run.BASELINE
        values = []
        for index, trial in enumerate(trials):
            assert trial.number == index and trial.state.name == 'COMPLETE'
            p = scores[index] >= .5
            tp = np.sum((y[dev]==1)&p)
            fp = np.sum((y[dev]==0)&p)
            fn = np.sum((y[dev]==1)&~p)
            value = float(2*tp/(2*tp+fp+fn))
            assert np.isclose(value, trial.value, atol=1e-12)
            values.append(value)
            warnings.extend(trial.user_attrs['warnings'])
            trial_checks += 1
        chosen = int(np.argmax(values))
        assert chosen == selection['best_trial'] == record['best_trial']
        assert selection['parameters'] == trials[chosen].params
        assert np.isclose(selection['inner_F1'], values[chosen])
        assert selection['threshold'] == independently_select_threshold(y[dev], scores[chosen])
        assert selection['baseline_threshold'] == independently_select_threshold(y[dev], scores[0])
        model = XGBClassifier(n_jobs=1)
        model.load_model(folder/'model.ubj')
        reloaded_score = model.predict_proba(flat[te])[:,1]
        frame = pd.read_csv(folder/'predictions.csv')
        assert sorted(frame.method.unique()) == sorted(run.METHODS)
        expected_thresholds = dict(XGB_Optuna=.5, XGB_Optuna_threshold=selection['threshold'],
                                   XGB_fixed_threshold=selection['baseline_threshold'])
        old = pd.read_csv(run.OLD/'classical/units'/f"{unit['name']}.csv")
        fixed_score = old.loc[old.method=='XGBoost','score'].to_numpy()
        for method, group in frame.groupby('method', sort=False):
            assert group.participant_id.tolist() == ids[te].tolist()
            np.testing.assert_array_equal(group.y_true, y[te])
            np.testing.assert_array_equal(group.threshold, expected_thresholds[method])
            np.testing.assert_array_equal(group.y_pred, group.score>=group.threshold)
            expected = fixed_score if method=='XGB_fixed_threshold' else reloaded_score
            np.testing.assert_allclose(group.score, expected, rtol=1e-12, atol=1e-12)
        warnings.extend(record['warnings'])
        rows.append(frame)
    old = pd.read_csv(run.OLD/'all-predictions.csv')
    frame = pd.concat([old[old.method.isin(['CNN','XGBoost'])], *rows], ignore_index=True)
    assert len(frame) == 5*20*173
    assert not frame.duplicated(['participant_id','method','repeat']).any()
    records = []
    for (method, repeat), g in frame.groupby(['method','repeat']):
        assert len(g)==173 and g.participant_id.nunique()==173 and g.fold.nunique()==6
        values = manual(g.y_true.to_numpy(),g.y_pred.to_numpy(),g.score.to_numpy())
        assert np.isclose(values['F1'], f1_score(g.y_true,g.y_pred))
        assert np.isclose(values['BA'], balanced_accuracy_score(g.y_true,g.y_pred))
        assert np.isclose(values['AUC'], roc_auc_score(g.y_true,g.score))
        records.append(dict(method=method, repeat=int(repeat), **values))
    metrics = pd.DataFrame(records)
    frame.to_csv(run.OUT/'all-predictions.csv',index=False)
    metrics.to_csv(run.OUT/'repeat-scores.csv',index=False)
    summary = metrics.groupby('method').agg(
        repeats=('repeat','count'), F1_mean=('F1','mean'), F1_sd=('F1','std'),
        BA_mean=('BA','mean'), BA_sd=('BA','std'), AUC_mean=('AUC','mean'), AUC_sd=('AUC','std'),
        precision_mean=('precision','mean'), recall_mean=('recall','mean'),
        TP_mean=('TP','mean'), FN_mean=('FN','mean'), FP_mean=('FP','mean'), TN_mean=('TN','mean'))
    summary.to_csv(run.OUT/'summary.csv')
    evidence = dict(status='passed', checked_unix=time.time(), completed_studies=120,
        completed_trials=trial_checks, checkpoints_reloaded=120, prediction_rows=len(frame),
        repeat_metric_checks=len(records)*3, warnings=warnings,
        original_outer_partitions_preserved=True, selections_recomputed_from_inner_predictions=True,
        all_frozen_sources_unchanged=True, formal_manuscript_unchanged=True)
    run.write_json(run.OUT/'verification.json', evidence)
    return frame, summary


def intervals(frame):
    ids = sorted(frame.participant_id.unique())
    y = frame.drop_duplicates('participant_id').set_index('participant_id').loc[ids,'y_true'].to_numpy()
    pos,neg = np.flatnonzero(y==1),np.flatnonzero(y==0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws_p = rng.choice(pos,(B,len(pos)))
    draws_n = rng.choice(neg,(B,len(neg)))
    wp = np.stack([(draws_p==i).sum(axis=1) for i in pos],axis=1)
    wn = np.stack([(draws_n==i).sum(axis=1) for i in neg],axis=1)
    bootstrap, estimates, matrices = {}, {}, {}
    for method in METHODS:
        g=frame[frame.method==method]
        pred=g.pivot(index='repeat',columns='participant_id',values='y_pred').loc[range(1,21),ids].to_numpy()
        score=g.pivot(index='repeat',columns='participant_id',values='score').loc[range(1,21),ids].to_numpy()
        tp=wp@pred[:,pos].T; fp=wn@pred[:,neg].T
        fn=len(pos)-tp; tn=len(neg)-fp
        f1=(2*tp/(2*tp+fp+fn)).mean(axis=1)
        ba=(.5*(tp/len(pos)+tn/len(neg))).mean(axis=1)
        positive,negative=score[:,pos,None],score[:,None,neg]
        concordance=((positive>negative)+.5*(positive==negative)).mean(axis=0)
        auc=np.einsum('bi,ij,bj->b',wp,concordance,wn,optimize=True)/(len(pos)*len(neg))
        bootstrap[method]=np.column_stack([f1,ba,auc])
        estimates[method]=np.mean([[manual(y,p,s)[k] for k in ['F1','BA','AUC']]
                                    for p,s in zip(pred,score)],axis=0)
        matrices[method]=(pred,score)
        for b in range(3):
            ix=np.concatenate([draws_p[b],draws_n[b]])
            reference=np.mean([[manual(y[ix],p[ix],s[ix])[k] for k in ['F1','BA','AUC']]
                               for p,s in zip(pred,score)],axis=0)
            np.testing.assert_allclose(reference,bootstrap[method][b],atol=1e-12)
    contrasts=[(m,'XGBoost') for m in run.METHODS]+[('CNN',m) for m in run.METHODS]
    rows=[]
    for a,b in contrasts:
        delta=bootstrap[a]-bootstrap[b]
        lo,hi=np.quantile(delta,[.025,.975],axis=0)
        for i,key in enumerate(['F1','BA','AUC']):
            rows.append(dict(contrast=f'{a} minus {b}', metric=key,
                difference=float(estimates[a][i]-estimates[b][i]),
                lower=float(lo[i]),upper=float(hi[i]),bootstrap=B,seed=BOOTSTRAP_SEED,
                scope='Exploratory paired stratified patient-cluster bootstrap; saved models fixed; unadjusted'))
    result=pd.DataFrame(rows)
    result.to_csv(run.OUT/'paired-intervals.csv',index=False)
    return result


def report(summary, paired):
    records=[]
    for path in sorted((run.OUT/'units').glob('*/selection.json')):
        item=json.loads(path.read_text())
        records.append(dict(unit=item['unit'], best_trial=item['best_trial'],
            inner_F1=item['inner_F1'], threshold=item['threshold'],
            baseline_threshold=item['baseline_threshold'], **item['parameters']))
    selections=pd.DataFrame(records)
    selections.to_csv(run.OUT/'selected-parameters.csv',index=False)
    lines=[
        'Optuna XGBoost same-input comparison: completion report',
        '',
        'Data: 173 public patients, 54 severe; same complete 7106-value image input, author labels, 20 repeats x 6 outer folds.',
        'Budget: 120 outer training pools, 100 Optuna TPE candidates per pool, 3 inner folds per candidate: 12000 candidates / 36000 inner fits, plus 120 final fits.',
        'Parameters are selected by inner F1 within each training pool at a fixed 0.5 threshold; the additional threshold is also selected from inner predictions only. CNN and the original fixed XGBoost predictions are reused.',
        'The 100 trials include the original fixed parameters as trial 0. No final version was chosen, no partition dropped and no search extended on the basis of test results.',
        '',
        'Three outputs: XGB_Optuna = tuned parameters only; XGB_Optuna_threshold = the same tuned model with a tuned threshold; XGB_fixed_threshold = fixed model with a tuned threshold only.',
        '',
        summary.to_string(float_format=lambda x:f'{x:.6f}'),
        '',
        'Paired differences and 95% intervals:',
        paired[['contrast','metric','difference','lower','upper']].to_string(index=False,float_format=lambda x:f'{x:.6f}'),
        '',
        'Threshold distribution:', selections[['threshold','baseline_threshold']].describe().to_string(),
        '',
        'Checks: all 120 models reloaded; inner F1 of all 12000 candidates and every selection recomputed; outer patient lists, predictions and metrics agree.',
        'Intervals use 5000 stratified paired patient resamples; each sampled patient is resampled jointly across the 20 repeats. Models are fixed; exploratory intervals are not adjusted for multiplicity.',
        'This is an exploratory extension specified after the fixed-model results were known. The additional optimisation budget went to XGBoost only; it is not equal-budget tuning of all architectures.',
        'There is no prespecified equivalence or non-inferiority margin; an interval that crosses zero does not establish equality. The results neither assume parity nor replace a historical reproduction of the original full-cohort study.',
        '',
    ]
    (run.OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        frame,summary=verify()
        paired=intervals(frame)
        report(summary,paired)
