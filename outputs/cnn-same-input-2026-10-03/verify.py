"""Independent saved-model prediction and metric checks; supports partial audits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import study


def manual(y,p,s):
    tp=int(np.sum((y==1)&(p==1)));tn=int(np.sum((y==0)&(p==0)))
    fp=int(np.sum((y==0)&(p==1)));fn=int(np.sum((y==1)&(p==0)))
    # Pairwise rank definition handles ties independently of sklearn's ROC code.
    a=s[y==1][:,None];b=s[y==0][None,:]
    return dict(F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),
                AUC=float(np.mean((a>b)+.5*(a==b))),
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
                accuracy=(tp+tn)/len(y),n=len(y))


def verify(reload_all=True):
    import torch
    torch.set_num_threads(4)
    ids,x,flat,y,strata,units=study.load_frozen()
    subject_index={v:i for i,v in enumerate(ids)}
    all_frames=[];reloaded=0;max_cnn_delta=0.;warnings=[]
    for worker in ['cnn','classical']:
        for unit in units:
            path=study.OUT/worker/'units'/f"{unit['name']}.json"
            if not path.exists(): continue
            record=json.loads(path.read_text());pred=path.with_suffix('.csv')
            assert study.sha(pred)==record['prediction_sha256']
            frame=pd.read_csv(pred);all_frames.append(frame)
            tr,va,te,dev=(np.array(unit[k]) for k in ['cnn_train','cnn_validation','test','development'])
            assert not (set(tr)&set(va) or set(dev)&set(te))
            assert set(tr)|set(va)==set(dev)
            for _,g in frame.groupby('method'):
                assert g.participant_id.tolist()==ids[te].tolist()
                np.testing.assert_array_equal(g.y_true,y[te])
                np.testing.assert_array_equal(g.y_pred,(g.score>=g.threshold).astype(int))
            if worker=='cnn':
                assert record['train_ids']==ids[tr].tolist() and record['validation_ids']==ids[va].tolist()
                curve=pd.read_csv(study.OUT/'cnn/curves'/f"{unit['name']}.csv")
                best=curve.loc[curve.validation_loss.idxmin()]
                assert int(best.epoch)==record['best_epoch']
                assert np.isclose(best.validation_loss,record['best_validation_loss'])
                checkpoint=study.OUT/'cnn'/record['checkpoint']
                assert study.sha(checkpoint)==record['checkpoint_sha256']
                if reload_all:
                    factory=runpy.run_path(str(study.NETWORK))['generate_model']
                    model=factory(drop=.8,model_depth=3,num_classes=1,in_channels=1)
                    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
                    with torch.inference_mode(): score=model(torch.from_numpy(x[te])).ravel().sigmoid().numpy()
                    np.testing.assert_allclose(score,frame.score,atol=2e-6,rtol=2e-5)
                    np.testing.assert_array_equal(score>=.5,frame.y_pred)
                    max_cnn_delta=max(max_cnn_delta,float(np.max(np.abs(score-frame.score.to_numpy()))))
                    reloaded+=1
            else:
                for fit in record['fits']:
                    method=fit['method'];g=frame[frame.method==method]
                    fit_indices=np.array([subject_index[v] for v in fit['train_ids']])
                    expected=tr if method.endswith('_CNNfit') else dev
                    np.testing.assert_array_equal(fit_indices,expected)
                    assert not set(fit_indices)&set(te)
                    checkpoint=study.OUT/'classical/checkpoints'/f"{unit['name']}_{method}.joblib"
                    assert study.sha(checkpoint)==fit['checkpoint_sha256']
                    warnings.extend([dict(unit=unit['name'],method=method,message=v) for v in fit.get('warnings',[])])
                    if method.startswith('SVM_'):
                        kind=method.split('_')[1]
                        log=json.loads((study.OUT/'classical/search'/f"{unit['name']}_{kind}.json").read_text())
                        assert len(log['trials'])==300
                        for t in log['trials']:
                            assert np.isclose(t['mean_loss'],np.mean(t['inner_losses']))
                        assert log['selected']['candidate']==int(np.argmin([t['mean_loss'] for t in log['trials']]))
                    if reload_all:
                        item=joblib.load(checkpoint)
                        if isinstance(item,dict):
                            z=study.kernel_matrix(flat[te].astype(float),flat[item['train_indices']].astype(float),item['kernel'],item['scale'])
                            score=item['model'].decision_function(z)
                        else: score=item.predict_proba(flat[te].astype(float))[:,1]
                        np.testing.assert_allclose(score,g.score,atol=1e-10,rtol=1e-10)
                        reloaded+=1
    if not all_frames: return dict(status='no_completed_units')
    frame=pd.concat(all_frames,ignore_index=True)
    assert not frame.duplicated(['participant_id','repeat','method']).any()
    metric_checks=0;counts={}
    for (method,repeat),g in frame.groupby(['method','repeat']):
        if len(g)!=173: continue
        assert g.participant_id.nunique()==173 and g.fold.nunique()==6
        a=manual(g.y_true.to_numpy(),g.y_pred.to_numpy(),g.score.to_numpy());b=study.scores(g)
        for key in a: assert np.isclose(a[key],b[key],rtol=1e-12,atol=1e-12),(method,repeat,key)
        metric_checks+=6;counts[method]=counts.get(method,0)+1
    done=all(counts.get(m,0)==20 for m in study.EXPECTED_METHODS)
    out=dict(status='complete' if done else 'partial',completed_repeats=counts,
             prediction_rows=len(frame),checkpoints_reloaded=reloaded,metric_checks=metric_checks,
             maximum_cnn_cpu_mps_probability_difference=max_cnn_delta,warnings=warnings,
             formal_manuscript_unchanged=study.sha(study.ROOT/'paper/main.pdf')==json.loads((study.OUT/'frozen.json').read_text())['paper_sha256'])
    study.write_json(study.OUT/'verification.json',out)
    return out


def paired_intervals(bootstrap=5000):
    frame=pd.read_csv(study.OUT/'all-predictions.csv')
    cnn=frame[frame.method=='CNN'].sort_values(['repeat','participant_id'])
    if cnn.repeat.nunique()!=20 or len(cnn)!=20*173:
        return
    ids=sorted(cnn.participant_id.unique());truth=cnn[cnn.repeat==1].set_index('participant_id').loc[ids].y_true.to_numpy()
    def matrix(method):
        g=frame[frame.method==method]
        if len(g)!=20*173:return None
        return g.pivot(index='repeat',columns='participant_id',values='y_pred').loc[range(1,21),ids].to_numpy()
    cp=matrix('CNN');positive=np.flatnonzero(truth==1);negative=np.flatnonzero(truth==0)
    rng=np.random.default_rng(2026100399)
    selections=np.concatenate([rng.choice(positive,(bootstrap,len(positive))),rng.choice(negative,(bootstrap,len(negative)))],axis=1)
    def estimates(pred,indices):
        v=pred[:,indices];yy=truth[indices]
        tp=np.sum((v==1)&(yy==1),axis=-1);fp=np.sum((v==1)&(yy==0),axis=-1)
        fn=np.sum((v==0)&(yy==1),axis=-1);tn=np.sum((v==0)&(yy==0),axis=-1)
        return np.stack([2*tp/(2*tp+fp+fn),.5*(tp/(tp+fn)+tn/(tn+fp))],axis=-1).mean(axis=0)
    output=[]
    for method in study.EXPECTED_METHODS:
        if method in ['CNN','Majority']:continue
        p=matrix(method)
        if p is None:continue
        base=estimates(cp,np.arange(173)[None,:])-estimates(p,np.arange(173)[None,:])
        diffs=[]
        for offset in range(0,bootstrap,100):
            idx=selections[offset:offset+100]
            diffs.append(estimates(cp,idx)-estimates(p,idx))
        diffs=np.concatenate(diffs)
        lo,hi=np.quantile(diffs,[.025,.975],axis=0)
        for i,metric in enumerate(['F1','balanced_accuracy']):
            output.append(dict(contrast='CNN minus '+method,metric=metric,difference=float(base[0,i]),
                              lower=float(lo[i]),upper=float(hi[i]),bootstrap=bootstrap,
                              scope='paired stratified patient-cluster bootstrap; saved models fixed; unadjusted intervals'))
    pd.DataFrame(output).to_csv(study.OUT/'paired-intervals.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-reload',action='store_true');p.add_argument('--intervals',action='store_true')
    a=p.parse_args();r=verify(not a.no_reload)
    print(json.dumps({k:v for k,v in r.items() if k!='warnings'},indent=2))
    if a.intervals:paired_intervals()
