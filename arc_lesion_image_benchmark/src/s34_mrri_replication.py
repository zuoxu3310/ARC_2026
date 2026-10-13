#!/usr/bin/env python
"""Independent MRRI behavioral extension using the public Walker et al. release.

Requires xlrd >= 2.0 for the original .xls. No clinical data are redistributed.
The manuscript describes the exploratory protocol and nine eligibility rules.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from threadpoolctl import threadpool_limits
import s18_crossed_grid as base
import s31_frozen_prediction_control as metrics

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'datasets/365_participants_alldata.xls'
OUT = ROOT / 'arc_lesion_image_benchmark/results/mrri_replication_2026-10-02'
RAW_HASH = '199b68795158d5ae494af4a5a4bdc6046d1e7a494faabc074d3d364129690bc5'
FEATURES = ['PNT_C','PNT_S','PNT_F','PNT_M','PNT_U','PNT_N','PNT_AN','PNT_NA','Age_when_tested']
TESTS = {
    'has_synonymy': 'Synonymy_triplets_Total',
    'has_ppvt': 'Peabody_Picture_Vocabulary_Test',
    'has_cct': 'Camel_and_Cactus_Test',
    'has_prt': 'PRT_C',
    'has_nwr': 'Nonword_repetition_PNT',
    'has_isr': 'Short_term_memory_ISR',
    'has_ppt': 'Pyramids_and_Palms_Test',
}

def load_data(raw):
    assert metrics.sha(raw) == RAW_HASH, 'Downloaded workbook differs from the inspected release.'
    df = pd.read_excel(raw, engine='xlrd')
    assert len(df) == 365
    ids = df.Anonymous_Subject_ID.astype(str).str.strip()
    assert ids.nunique() == 365
    mrri = df.loc[~ids.str.match(r'^P\d+$')].copy()
    assert len(mrri) == 275 and mrri.index.tolist() == list(range(275))
    # Reconcile institutional assignment against source Table 1, rather than
    # assuming that a different download or archive is an independent cohort.
    assert round(mrri.Age_when_tested.mean(), 1) == 58.8
    assert round(mrri.Months_post_onset.mean(), 1) == 29.9
    assert round(mrri.WAB_Aphasia_Quotient.mean(), 1) == 72.7
    assert round(mrri.WAB_Aphasia_Quotient.std(), 1) == 18.0
    assert mrri.Gender.str.strip().value_counts().to_dict() == {'M':157, 'F':118}
    assert (mrri[FEATURES[:8]].sum(axis=1) == 175).all()
    d = mrri.loc[mrri.WAB_Aphasia_Quotient.notna()].copy()
    d.index = d.Anonymous_Subject_ID.str.strip()
    assert len(d) == 183 and int((d.WAB_Aphasia_Quotient <= 50).sum()) == 25
    assert d[FEATURES + ['Months_post_onset']].notna().all().all()
    masks = {'full': np.ones(len(d), bool),
             'chronic_6m': d.Months_post_onset.ge(6).to_numpy(),
             'chronic_12m': d.Months_post_onset.ge(12).to_numpy()}
    masks.update({name:d[col].notna().to_numpy() for name,col in TESTS.items()})
    return d, masks

def prepare(raw, out):
    d, masks = load_data(raw)
    for subdir in ['predictions','folds']:
        (out/subdir).mkdir(parents=True, exist_ok=True)
    versions = {p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','imbalanced-learn','xgboost','xlrd']}
    p = {'source_url':'https://sites.socsci.uci.edu/~alns/getSubjData.php',
         'doi':'10.1037/pas0000529', 'source_sha256':metrics.sha(raw),
         'code_sha256':metrics.sha(Path(__file__)),
         'base_code_sha256':metrics.sha(Path(base.__file__)),
         'metric_code_sha256':metrics.sha(Path(metrics.__file__)),
         'versions':versions, 'features':FEATURES, 'n_mrri_source':275,
         'n_with_aq':183, 'n_severe':25, 'seeds':metrics.SEEDS,
         'bootstrap_seed':20261002, 'bootstrap_draws':2000}
    dest = out/'provenance.json'
    if dest.exists(): assert json.loads(dest.read_text()) == p
    else: metrics.write_json(dest,p)
    pd.DataFrame([{'subset':k,'n':int(v.sum()),'n_severe':int(((d.WAB_Aphasia_Quotient.to_numpy()<=50)&v).sum()),
                   'aq_sd':float(d.WAB_Aphasia_Quotient.to_numpy()[v].std(ddof=1))} for k,v in masks.items()]).to_csv(out/'subset_manifest.csv',index=False)
    pd.DataFrame({'participant_id':d.index,'y_aq':d.WAB_Aphasia_Quotient,**masks}).to_csv(out/'patient_cohort_membership.csv',index=False)
    return d,masks

def fit_cell(job):
    raw,out,subset,task,model,seed = job
    d,masks = load_data(raw)
    X = d[FEATURES].to_numpy(); aq=d.WAB_Aphasia_Quotient
    y=aq.to_numpy() if task=='regression' else (aq.to_numpy()<=50).astype(int)
    strat = pd.qcut(aq,4,labels=False).to_numpy() if task=='regression' else y
    pred=np.full(len(y),np.nan); prob=np.full(len(y),np.nan); foldid=np.zeros(len(y),int)
    records=[]
    with threadpool_limits(limits=2):
        for fold,(tr0,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,strat),1):
            start=time.monotonic(); tr=tr0[masks[subset][tr0]]
            assert not set(tr)&set(te)
            steps=[('scaler',StandardScaler())]
            minority=None
            if task=='classification':
                minority=int(np.bincount(y[tr],minlength=2).min()); assert minority>1
                steps.append(('smote',SMOTE(random_state=seed,k_neighbors=min(5,minority-1))))
                estimator=base.clf_estimator(model,seed)
            else: estimator=base.reg_estimator(model,seed)
            estimator.set_params(n_jobs=2)
            pipe=Pipeline(steps+[('estimator',estimator)])
            pipe.fit(X[tr],y[tr]); pred[te]=pipe.predict(X[te]); foldid[te]=fold
            if task=='classification':
                prob[te]=pipe.predict_proba(X[te])[:,1]
                assert pipe.named_steps['smote'].nn_k_._fit_X.shape[0]==minority
            sc=pipe.named_steps['scaler']; err=float(np.max(np.abs(sc.mean_-X[tr].mean(axis=0))))
            assert sc.n_samples_seen_==len(tr) and err<1e-10
            records.append({'fold':fold,'original_train_ids':d.index[tr0].tolist(),
                'train_ids':d.index[tr].tolist(),'test_ids':d.index[te].tolist(),
                'scaler_n':int(sc.n_samples_seen_),'scaler_train_mean_max_error':err,
                'smote_original_minority_n':minority})
            metrics.log(f'MRRI {subset} {task} {model} {seed} fold={fold} train={len(tr)} seconds={time.monotonic()-start:.2f}')
    assert np.isfinite(pred).all() and (foldid>0).all()
    stem=f'{subset}__{task}__{model}__{seed}'
    frame=pd.DataFrame({'participant_id':d.index,'task':task,'model':model,'seed':seed,'train_subset':subset,
        'fold':foldid,'y_true':y,'y_aq':aq.to_numpy(),'prediction':pred,'probability_severe':prob})
    tmp=out/'predictions'/f'{stem}.tmp'; tmpdest=tmp.with_suffix('.csv')
    frame.to_csv(tmp,index=False);tmp.replace(tmpdest)
    metrics.write_json(out/'folds'/f'{stem}.json',records)
    return {'subset':subset,'task':task,'model':model,'seed':seed,'sha256':metrics.sha(tmpdest)}

def train(raw,out,workers):
    _,masks=prepare(raw,out)
    jobs=[(raw,out,subset,task,model,seed) for seed in metrics.SEEDS for subset in masks for task in metrics.TASKS for model in metrics.MODELS]
    completed=[];pending=[]
    for j in jobs:
        _,_,sub,task,model,seed=j;stem=f'{sub}__{task}__{model}__{seed}'
        p=out/'predictions'/f'{stem}.csv';f=out/'folds'/f'{stem}.json'
        if p.exists() and f.exists():
            frame=pd.read_csv(p);assert len(frame)==183 and frame.participant_id.nunique()==183
            completed.append({'subset':sub,'task':task,'model':model,'seed':seed,'sha256':metrics.sha(p)})
        else: pending.append(j)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(fit_cell,j) for j in pending]):
            completed.append(f.result())
            metrics.write_json(out/'progress.json',{'completed':len(completed),'total':len(jobs),'updated_utc':metrics.stamp()})
            metrics.log(f'MRRI_CELL_COMPLETE {len(completed)}/{len(jobs)}')
    assert len(completed)==800
    metrics.write_json(out/'TRAINING_COMPLETE.json',{'completed_utc':metrics.stamp(),'fits':4000,'cells':completed})

def analyze(raw,out):
    d,masks=prepare(raw,out);assert (out/'TRAINING_COMPLETE.json').exists()
    rng=np.random.default_rng(20261002)
    boot=rng.multinomial(len(d),np.full(len(d),1/len(d)),size=2000)
    np.savez_compressed(out/'patient_bootstrap.npz',weights=boot)
    cells=[];contrasts=[];repeats=[]
    coeffs={'training_at_full_scoring':{'S10':1,'S00':-1},'training_at_selected_scoring':{'S11':1,'S01':-1},
        'scoring_at_full_training':{'S01':1,'S00':-1},'scoring_at_selected_training':{'S11':1,'S10':-1},
        'joint_change':{'S11':1,'S00':-1},'interaction':{'S11':1,'S10':-1,'S01':-1,'S00':1}}
    for task in metrics.TASKS:
        for model in metrics.MODELS:
            def bank(sub):
                frames=[pd.read_csv(out/'predictions'/f'{sub}__{task}__{model}__{s}.csv') for s in metrics.SEEDS]
                for f in frames: assert f.participant_id.tolist()==d.index.tolist()
                return frames,np.stack([f.prediction.to_numpy() for f in frames])
            full,p0=bank('full');y=full[0].y_true.to_numpy()
            for subset in masks:
                if subset=='full':continue
                frames,p1=bank(subset)
                for old,new in zip(full,frames):np.testing.assert_array_equal(old.fold,new.fold)
                point={};draw={}
                for name,p,mask in [('S00',p0,masks['full']),('S01',p0,masks[subset]),('S10',p1,masks['full']),('S11',p1,masks[subset])]:
                    point[name]=metrics.weighted_metrics(mask,y,p,task)
                    draw[name]=metrics.weighted_metrics(boot*mask,y,p,task)
                    for metric in point[name]:
                        lo,hi=np.quantile(draw[name][metric].mean(axis=1),[.025,.975])
                        cells.append({'task':task,'model':model,'subset':subset,'cell':name,'metric':metric,
                                      'mean':float(point[name][metric].mean()),'ci_low':lo,'ci_high':hi})
                        for i,seed in enumerate(metrics.SEEDS):
                            repeats.append({'task':task,'model':model,'subset':subset,'cell':name,'metric':metric,
                                            'seed':seed,'score':float(point[name][metric][0,i])})
                for contrast,terms in coeffs.items():
                    for metric in point['S00']:
                        delta=sum(k*point[n][metric].mean() for n,k in terms.items())
                        bs=sum(k*draw[n][metric].mean(axis=1) for n,k in terms.items())
                        lo,hi=np.quantile(bs,[.025,.975])
                        contrasts.append({'task':task,'model':model,'subset':subset,'metric':metric,'contrast':contrast,
                                          'delta':delta,'ci_low':lo,'ci_high':hi})
                metrics.log(f'MRRI_ANALYSIS {task} {model} {subset}')
    pd.DataFrame(cells).to_csv(out/'factorial_cells.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'factorial_contrasts.csv',index=False)
    pd.DataFrame(repeats).to_csv(out/'scores_by_seed.csv',index=False)
    metrics.write_json(out/'ANALYSIS_COMPLETE.json',{'completed_utc':metrics.stamp(),'cells':len(cells),
        'contrasts':len(contrasts),'repeated_scores':len(repeats),'bootstrap_draws':2000})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['train','analyze'],required=True)
    ap.add_argument('--data',type=Path,default=RAW);ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    if args.stage=='train':train(args.data,args.output,args.workers)
    else:analyze(args.data,args.output)

if __name__=='__main__': main()
