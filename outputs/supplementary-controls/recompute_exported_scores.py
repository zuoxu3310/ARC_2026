"""Recompute new scientific scores from exported predictions, without raw inputs.

Usage:
  python recompute_exported_scores.py --root /path/to/evidence

The directory must contain training-only-features/ and image-controls/.
Keep feature-patient-bootstrap-weights.npz beside this script, or supply
--feature-weights. This script writes no files; its JSON report goes to stdout.
Requires Python, numpy, pandas, scipy, scikit-learn, threadpoolctl.
Historical frozen-dictionary predictions and pipeline-change intervals are
outside this portable check. They were separately audited with local originals.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def discrepancy(a,b):
    value=float(np.max(abs(np.asarray(a)-np.asarray(b))))
    assert value < 2e-11, (value,a,b)
    return value

def regression_and_ba(y,pred,w,task):
    w=np.atleast_2d(w).astype(float)
    if task=='classification':
        pos,neg=y==1,y==0
        return {'score':.5*((w[:,pos]@(pred[:,pos]==1).T)/w[:,pos].sum(1)[:,None]+(w[:,neg]@(pred[:,neg]==0).T)/w[:,neg].sum(1)[:,None])}
    yy=y-y.mean(); pp=pred-pred.mean(1)[:,None]; mass=w.sum(1)[:,None]
    my=(w@yy)[:,None]/mass;mp=(w@pp.T)/mass
    covariance=w@(pp*yy).T/mass-my*mp
    variance_y=(w@(yy**2))[:,None]/mass-my**2
    variance_p=w@(pp**2).T/mass-mp**2
    return {'score':covariance/np.sqrt(variance_y*variance_p),'mae':w@abs(pred-y).T/mass}

def audit_features(root,weight_file):
    d=root/'training-only-features';plan=read(d/'frozen.json');ids=plan['ids'];rows=[];bank={};masks={}
    weights=np.load(weight_file)['patient_bootstrap'];assert weights.shape==(2000,226)
    assert np.all(weights.sum(1)==226)
    for sub in plan['subsets']:
        f=d/'units'/f'{sub}__regression__RandomForest__1000'/'folds.json'
        selected=set(pid for fold in read(f) for pid in fold['train_ids'])
        masks[sub]=np.isin(ids,list(selected))
    assert masks['full'].all()
    checked_records=0
    for task in plan['tasks']:
        for model in plan['models']:
            for sub in plan['subsets']:
                values=[];truth=None
                for seed in plan['seeds']:
                    folder=d/'units'/f'{sub}__{task}__{model}__{seed}'
                    complete=read(folder/'complete.json')
                    assert complete['frozen_sha256']==digest(d/'frozen.json')
                    assert all(digest(folder/name)==h for name,h in complete['files'].items())
                    f=pd.read_csv(folder/'predictions.csv');assert f.participant_id.tolist()==ids
                    if truth is None:truth=f.y_true.to_numpy()
                    else:assert np.array_equal(truth,f.y_true)
                    assert np.isfinite(f.prediction).all()
                    full=read(d/'units'/f'full__{task}__{model}__{seed}'/'folds.json')
                    current=read(folder/'folds.json');allowed={pid for pid,flag in zip(ids,masks[sub]) if flag}
                    assert len(full)==len(current)==5
                    expected_fold={}
                    for actual,original in zip(current,full):
                        assert actual['train_ids']==[pid for pid in original['train_ids'] if pid in allowed]
                        assert actual['test_ids']==original['test_ids']
                        assert not set(actual['train_ids'])&set(actual['test_ids'])
                        expected_fold.update({pid:actual['fold'] for pid in actual['test_ids']});checked_records+=1
                    assert f.fold.tolist()==[expected_fold[i] for i in ids]
                    values.append(f.prediction.to_numpy())
                bank[(task,model,sub)]=(truth,np.stack(values))
    cells=pd.read_csv(d/'factorial_cells.csv').query("source=='training_only'")
    contrasts=pd.read_csv(d/'factorial_contrasts.csv').query("source=='training_only'")
    methods=pd.read_csv(d/'method_contrasts.csv').query("source=='training_only'")
    cache={};errors=[]
    for task in plan['tasks']:
        for model in plan['models']:
            for sub in plan['subsets']:
                if sub=='full':continue
                for cell,training,scoring in [('S00','full','full'),('S01','full',sub),('S10',sub,'full'),('S11',sub,sub)]:
                    y,pred=bank[(task,model,training)]
                    point=regression_and_ba(y,pred,masks[scoring],task)
                    bootstrap=regression_and_ba(y,pred,weights*masks[scoring],task)
                    for metric in point:
                        v=point[metric].ravel();b=bootstrap[metric].mean(1);cache[(task,model,sub,cell,metric)]=(v,b)
                        row=cells.query('task==@task and model==@model and subset==@sub and cell==@cell and metric==@metric').iloc[0]
                        errors.append(discrepancy([v.mean(),*np.quantile(b,[.025,.975])],[row['mean'],row.ci_low,row.ci_high]))
    coefs={'scoring_at_full_training':{'S01':1,'S00':-1},'scoring_at_selected_training':{'S11':1,'S10':-1},
           'training_at_full_scoring':{'S10':1,'S00':-1},'training_at_selected_scoring':{'S11':1,'S01':-1},
           'joint_change':{'S11':1,'S00':-1},'interaction':{'S11':1,'S10':-1,'S01':-1,'S00':1}}
    for row in contrasts.itertuples():
        v=sum(w*cache[(row.task,row.model,row.subset,c,row.metric)][0] for c,w in coefs[row.contrast].items())
        b=sum(w*cache[(row.task,row.model,row.subset,c,row.metric)][1] for c,w in coefs[row.contrast].items())
        errors.append(discrepancy([v.mean(),*np.quantile(b,[.025,.975])],[row.delta,row.ci_low,row.ci_high]))
    for row in methods.itertuples():
        a=cache[(row.task,'XGBoost',row.subset,row.cell,row.metric)];b=cache[(row.task,'RandomForest',row.subset,row.cell,row.metric)]
        errors.append(discrepancy([(a[0]-b[0]).mean(),*np.quantile(a[1]-b[1],[.025,.975])],[row.delta,row.ci_low,row.ci_high]))
    return {'scope':'New training-only prediction-bank metrics and record consistency; raw feature filtering is a separate local audit.',
            'training_fold_records':checked_records,'recomputed_summary_rows':len(errors),'maximum_difference':max(errors),
            'patient_bootstrap_sha256':digest(weight_file)}

def auc_by_ranks(y,probabilities,W):
    out=[]
    for score in probabilities:
        order=np.argsort(score,kind='stable');s=score[order];target=y[order];w=W[:,order]
        cumulative=np.cumsum(w*(target==0),axis=1)
        left=np.searchsorted(s,s,side='left')-1;right=np.searchsorted(s,s,side='right')-1
        before=np.where(left[None,:]>=0,cumulative[:,np.maximum(left,0)],0)
        through=cumulative[:,right]
        numerator=(w*(target==1)*(before+through)/2).sum(1)
        out.append(numerator/(W[:,y==1].sum(1)*W[:,y==0].sum(1)))
    return np.stack(out,axis=1)

def audit_images(root):
    d=root/'image-controls';plan=read(d/'image-plan.json');ids=plan['ids']
    pred=pd.read_csv(d/'all-predictions.csv');truth=pred.groupby('participant_id').y_true
    assert (truth.nunique()==1).all();y=truth.first().loc[ids].to_numpy()
    W=np.load(d/'bootstrap-weights.npz')['weights'];assert W.shape==(5000,173)
    pos,neg=y==1,y==0;assert pos.sum()==54 and neg.sum()==119
    assert (W[:,pos].sum(1)==54).all() and (W[:,neg].sum(1)==119).all()
    rng=np.random.default_rng(plan['bootstrap_seed'])
    ix=np.concatenate([rng.choice(np.flatnonzero(pos),(5000,54)),rng.choice(np.flatnonzero(neg),(5000,119))],axis=1)
    assert np.array_equal(W,np.stack([np.bincount(i,minlength=173) for i in ix]))
    summary=pd.read_csv(d/'summary-intervals.csv');contrasts=pd.read_csv(d/'paired-contrasts.csv');saved=pd.read_csv(d/'repeat-metrics.csv')
    cache={};errors=[]
    for (method,decision),f in pred.groupby(['method','decision']):
        assert not f.duplicated(['participant_id','repeat']).any()
        score=f.pivot(index='repeat',columns='participant_id',values='score').loc[range(1,21),ids].to_numpy()
        binary=f.pivot(index='repeat',columns='participant_id',values='y_pred').loc[range(1,21),ids].to_numpy()
        assert score.shape==(20,173) and np.isfinite(score).all()
        assert np.array_equal(f.y_pred.to_numpy(),f.score.to_numpy()>=f.threshold.to_numpy())
        point={'F1':np.array([f1_score(y,b) for b in binary]),'balanced_accuracy':np.array([balanced_accuracy_score(y,b) for b in binary]),'AUC':np.array([roc_auc_score(y,s) for s in score])}
        tp=W[:,pos]@binary[:,pos].T;fp=W[:,neg]@binary[:,neg].T
        bs={'F1':(2*tp/(54+tp+fp)).mean(1),'balanced_accuracy':(.5*(tp/54+(119-fp)/119)).mean(1),'AUC':auc_by_ranks(y,score,W).mean(1)}
        for metric in point:
            v,b=point[metric],bs[metric];cache[(method,decision,metric)]=(v,b)
            row=summary.query('method==@method and decision==@decision and metric==@metric').iloc[0]
            errors.append(discrepancy([v.mean(),*np.quantile(b,[.025,.975])],[row['mean'],row.ci_low,row.ci_high]))
            errors.append(discrepancy(saved.query('method==@method and decision==@decision').sort_values('repeat')[metric].to_numpy(),v))
    for row in contrasts.itertuples():
        a=cache[(row.left_method,row.left_decision,row.metric)];b=cache[(row.right_method,row.right_decision,row.metric)]
        errors.append(discrepancy([(a[0]-b[0]).mean(),*np.quantile(a[1]-b[1],[.025,.975])],[row.delta,row.ci_low,row.ci_high]))
    return {'scope':'All new image F1, balanced accuracy, AUC and paired conditional intervals.',
            'method_decision_pairs':len(cache)//3,'score_checks':len(errors),'maximum_difference':max(errors),
            'patient_bootstrap_sha256':digest(d/'bootstrap-weights.npz')}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--feature-weights',type=Path,default=Path(__file__).with_name('feature-patient-bootstrap-weights.npz'))
    p.add_argument('--stage',choices=['features','images','both'],default='both');args=p.parse_args();report={}
    with threadpool_limits(limits=1):
        if args.stage in ['features','both']:report['features']=audit_features(args.root,args.feature_weights)
        if args.stage in ['images','both']:report['images']=audit_images(args.root)
    print(json.dumps(report,indent=2,ensure_ascii=False))
