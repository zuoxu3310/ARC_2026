"""All original population controls, with a training-only feature screen."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parents[1] / 'evidence' / 'training-only-features'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'arc_lesion_image_benchmark/src'))
import s31_frozen_prediction_control as old
import s33_fixed_test_training as restricted
import arc_comparison_run as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def inputs():
    X, aq, features, manifest, masks = old.load_data()
    raw = pd.read_csv(old.base.TABLE, sep='\t', index_col='participant_id').dropna(subset=['wab_aq']).fillna(0)
    raw = raw.loc[X.index].drop(columns='wab_aq')
    assert raw.shape == (226, 191)
    np.testing.assert_array_equal(raw[features], X)
    return raw, aq, features, manifest, masks


def prepare():
    raw, aq, features, manifest, masks = inputs()
    sources = [Path(__file__), Path(runner.__file__), Path(old.__file__), Path(old.base.__file__),
               Path(restricted.__file__), Path(old.base.TABLE), OUT.parents[1] / 'PLAN.md',
               old.OUT / 'resampling_patient_weights.npz']
    sources += sorted(Path(old.base.SUBSET_DIR).glob('*_ids.txt'))
    sources += sorted((old.OUT / 'folds').glob('*.json'))
    sources += sorted((old.OUT / 'predictions').glob('*.csv'))
    sources += sorted((restricted.OUT / 'predictions').glob('*.csv'))
    plan = dict(source_sha256={str(p):sha(p) for p in sources},
                versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','imbalanced-learn','xgboost','joblib']},
                features=raw.columns.tolist(), frozen_features=features, ids=raw.index.tolist(),
                subsets=manifest.subset.tolist(), seeds=old.SEEDS, models=old.MODELS, tasks=old.TASKS,
                bootstrap='Original 2000 shared patient weights; saved fits conditional',
                workers=4, threads_per_estimator=1, new_scientific_fits=4000, anchor_fits=20)
    p = OUT / 'frozen.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan, 'Frozen inputs or implementation changed'
    else:
        write(p, plan)
    return plan


def cell(job):
    subset, task, model, seed, anchor = job
    raw, aq, frozen_features, _, masks = inputs()
    ids=raw.index.to_numpy(); loc={p:i for i,p in enumerate(ids)}
    features=frozen_features if anchor else raw.columns.tolist()
    values=raw[features].to_numpy()
    y=aq.to_numpy() if task=='regression' else (aq.to_numpy()<=50).astype(int)
    original=json.loads((old.OUT/'folds'/f'{task}_{model}_{seed}.json').read_text())
    stem=f'{subset}__{task}__{model}__{seed}'
    folder=OUT/('anchors' if anchor else 'units')/stem
    frozen_sha=sha(OUT/'frozen.json')
    if (folder/'complete.json').exists():
        meta=json.loads((folder/'complete.json').read_text())
        assert meta['frozen_sha256']==frozen_sha
        assert all(sha(folder/name)==digest for name,digest in meta['files'].items())
        return meta
    prediction=np.full(len(ids),np.nan); probability=np.full(len(ids),np.nan); folds=np.zeros(len(ids),int)
    records=[]; warnings_seen=[]; begin=time.monotonic()
    with threadpool_limits(limits=1):
        for rec in original:
            original_tr=np.array([loc[p] for p in rec['train_ids']])
            tr=original_tr[masks[subset][original_tr]]
            te=np.array([loc[p] for p in rec['test_ids']])
            assert len(tr)>1 and not set(tr)&set(te)
            pipe=runner.pipeline(task,{'model':model},{},seed,1,y[tr],
                                 'frozen_dictionary' if anchor else 'training_nonzero')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); pipe.fit(values[tr],y[tr])
                pred,score=runner.predict(pipe,values[te],task,'estimator')
            warnings_seen.extend(dict(fold=rec['fold'],category=w.category.__name__,message=str(w.message)) for w in caught)
            prediction[te]=pred
            if score is not None: probability[te]=score
            folds[te]=rec['fold']
            support=np.ones(len(features),bool) if anchor else pipe.named_steps['features'].support_
            if not anchor:
                np.testing.assert_array_equal(support,(values[tr]!=0).sum(axis=0)>.1*len(tr))
                assert pipe.named_steps['features'].fit_n_==len(tr)
            scaler=pipe.named_steps['scaler']
            error=float(np.abs(scaler.mean_-values[tr][:,support].mean(axis=0)).max())
            assert int(scaler.n_samples_seen_)==len(tr) and error<1e-10
            minority=int(np.bincount(y[tr],minlength=2).min()) if task=='classification' else None
            if task=='classification': assert pipe.named_steps['smote'].nn_k_._fit_X.shape[0]==minority
            records.append(dict(fold=rec['fold'],train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),
                                feature_columns=np.asarray(features)[support].tolist(),scaler_n=int(scaler.n_samples_seen_),
                                scaler_mean_max_error=error,smote_minority_n=minority,
                                selected_n=int(support.sum()),original_train_n=len(original_tr)))
    assert np.isfinite(prediction).all() and (folds>0).all()
    data=pd.DataFrame(dict(participant_id=ids,task=task,model=model,seed=seed,train_subset=subset,
                           fold=folds,y_true=y,prediction=prediction,probability_severe=probability))
    anchor_error=None
    if anchor:
        previous=pd.read_csv(old.OUT/'predictions'/f'{task}_{model}_{seed}.csv')
        assert previous.participant_id.tolist()==ids.tolist()
        np.testing.assert_array_equal(previous.fold,folds)
        if model=='XGBoost':
            np.testing.assert_array_equal(prediction.astype(np.float32),previous.prediction.to_numpy().astype(np.float32))
        else: np.testing.assert_allclose(prediction,previous.prediction,atol=1e-10,rtol=0)
        if task=='classification': np.testing.assert_allclose(probability,previous.probability_severe,atol=1e-7,rtol=0)
        anchor_error=float(np.max(np.abs(prediction-previous.prediction)))
    folder.mkdir(parents=True,exist_ok=True)
    data.to_csv(folder/'predictions.csv',index=False)
    write(folder/'folds.json',records)
    meta=dict(subset=subset,task=task,model=model,seed=seed,anchor=anchor,
              seconds=time.monotonic()-begin,fits=len(records),anchor_max_error=anchor_error,
              frozen_sha256=frozen_sha,warnings=warnings_seen,
              files={n:sha(folder/n) for n in ['predictions.csv','folds.json']})
    write(folder/'complete.json',meta)
    print(f"COMPLETE {stem} anchor={anchor} seconds={meta['seconds']:.1f}",flush=True)
    return meta


def train(anchors=False):
    p=prepare(); begin=time.monotonic()
    if anchors:
        jobs=[('full',t,m,1000,True) for t in p['tasks'] for m in p['models']]
    else:
        assert (OUT/'ANCHORS_PASS.json').exists()
        jobs=[(s,t,m,seed,False) for seed in p['seeds'] for s in p['subsets'] for t in p['tasks'] for m in p['models']]
    results=[]
    with parallel_config(backend='loky',n_jobs=4,inner_max_num_threads=1):
        for result in Parallel(return_as='generator_unordered')(delayed(cell)(j) for j in jobs):
            results.append(result)
            write(OUT/'progress.json',dict(phase='anchor' if anchors else 'train',completed=len(results),total=len(jobs),
                                          elapsed_seconds=time.monotonic()-begin,last=result))
    write(OUT/('ANCHORS_PASS.json' if anchors else 'TRAINING_COMPLETE.json'),dict(units=results,fits=sum(r['fits'] for r in results)))


def predictions(subset,task,model,source):
    frames=[]
    for seed in old.SEEDS:
        if source=='training_only':
            path=OUT/'units'/f'{subset}__{task}__{model}__{seed}'/'predictions.csv'
        elif subset=='full': path=old.OUT/'predictions'/f'{task}_{model}_{seed}.csv'
        else: path=restricted.OUT/'predictions'/f'{subset}__{task}__{model}__{seed}.csv'
        frames.append(pd.read_csv(path))
    assert all(f.participant_id.tolist()==frames[0].participant_id.tolist() for f in frames)
    return np.stack([f.prediction.to_numpy() for f in frames]),frames[0].y_true.to_numpy()


def analyze():
    prepare(); assert (OUT/'TRAINING_COMPLETE.json').exists()
    raw,aq,_,manifest,masks=inputs()
    boot=np.load(old.OUT/'resampling_patient_weights.npz')['patient_bootstrap']
    contrasts={'scoring_at_full_training':{'S01':1,'S00':-1},
               'scoring_at_selected_training':{'S11':1,'S10':-1},
               'training_at_full_scoring':{'S10':1,'S00':-1},
               'training_at_selected_scoring':{'S11':1,'S01':-1},
               'joint_change':{'S11':1,'S00':-1},'interaction':{'S11':1,'S10':-1,'S01':-1,'S00':1}}
    rows=[]; cellrows=[]; repeats=[]; cache={}
    with threadpool_limits(limits=2):
        for source in ['frozen_dictionary','training_only']:
            for task in old.TASKS:
                for model in old.MODELS:
                    p0,y=predictions('full',task,model,source)
                    for sub in manifest.itertuples():
                        if sub.subset=='full': continue
                        p1,_=predictions(sub.subset,task,model,source)
                        point={};draws={}
                        for name,pred,mask in [('S00',p0,masks['full']),('S01',p0,masks[sub.subset]),
                                               ('S10',p1,masks['full']),('S11',p1,masks[sub.subset])]:
                            point[name]=old.weighted_metrics(mask,y,pred,task)
                            draws[name]=old.weighted_metrics(boot*mask,y,pred,task)
                            for metric in point[name]:
                                v=point[name][metric].ravel(); bs=draws[name][metric].mean(axis=1)
                                low,high=np.quantile(bs,[.025,.975])
                                cellrows.append(dict(source=source,task=task,model=model,subset=sub.subset,cell=name,
                                                     metric=metric,mean=float(v.mean()),ci_low=low,ci_high=high))
                                repeats.extend(dict(source=source,task=task,model=model,subset=sub.subset,cell=name,
                                                    metric=metric,seed=seed,value=value) for seed,value in zip(old.SEEDS,v))
                                cache[(source,task,model,sub.subset,name,metric)]=(v,bs)
                        for contrast,coefs in contrasts.items():
                            for metric in point['S00']:
                                v=sum(w*point[c][metric].ravel() for c,w in coefs.items())
                                bs=sum(w*draws[c][metric].mean(axis=1) for c,w in coefs.items())
                                lo,hi=np.quantile(bs,[.025,.975])
                                rows.append(dict(source=source,task=task,model=model,subset=sub.subset,
                                                 metric=metric,contrast=contrast,delta=float(v.mean()),ci_low=lo,ci_high=hi,
                                                 repeat_min=float(v.min()),repeat_max=float(v.max())))
                        print(f'ANALYZED {source} {task} {model} {sub.subset}',flush=True)
        methods=[]; pipeline_changes=[]
        for source in ['frozen_dictionary','training_only']:
            for task in old.TASKS:
                for subset in manifest.subset:
                    if subset=='full':continue
                    for cellname in ['S00','S01','S10','S11']:
                        for metric in (['score','mae'] if task=='regression' else ['score']):
                            a,b=[cache[(source,task,m,subset,cellname,metric)] for m in old.MODELS]
                            v=b[0]-a[0];bs=b[1]-a[1];lo,hi=np.quantile(bs,[.025,.975])
                            methods.append(dict(source=source,task=task,subset=subset,cell=cellname,metric=metric,
                                                contrast='XGBoost_minus_RandomForest',delta=float(v.mean()),ci_low=lo,ci_high=hi,
                                                repeat_min=float(v.min()),repeat_max=float(v.max())))
        for key,(v,b) in cache.items():
            source,task,model,subset,cellname,metric=key
            if source!='training_only':continue
            v0,b0=cache[('frozen_dictionary',task,model,subset,cellname,metric)]
            lo,hi=np.quantile(b-b0,[.025,.975])
            pipeline_changes.append(dict(task=task,model=model,subset=subset,cell=cellname,metric=metric,
                                         delta=float((v-v0).mean()),ci_low=lo,ci_high=hi))
    for name,records in [('factorial_contrasts',rows),('factorial_cells',cellrows),('repeat_scores',repeats),
                         ('method_contrasts',methods),('pipeline_changes',pipeline_changes)]:
        pd.DataFrame(records).to_csv(OUT/(name+'.csv'),index=False)
    write(OUT/'ANALYSIS_COMPLETE.json',dict(rows=len(rows),cells=len(cellrows),method_contrasts=len(methods),
                                         bootstrap_draws=len(boot),conditional_on_saved_fits=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','anchor','train','analyze'])
    stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='anchor':train(True)
    elif stage=='train':train(False)
    else:analyze()
