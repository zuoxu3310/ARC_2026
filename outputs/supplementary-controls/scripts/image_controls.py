"""Matched fitting/validation patients and decision-policy controls for the CNN case."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, runpy, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]/'evidence'/'image-controls'
sys.path.insert(0,str(ROOT/'outputs/cnn-same-input-2026-10-03'))
import study


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def prepare():
    ids,x,flat,y,strata,units=study.load_frozen()
    fixed=dict(n_estimators=400,max_depth=3,learning_rate=.05)
    configs=[fixed]+[dict(n_estimators=n,max_depth=d,learning_rate=lr) for n in [200,400,800] for d in [1,2,3,5] for lr in [.03,.1]]
    sources=[Path(__file__),Path(study.__file__),study.OUT/'partitions.json',study.OUT/'frozen.json',study.DATA,study.NETWORK]
    sources+=sorted((study.OUT/'cnn/units').glob('*.json'))+sorted((study.OUT/'cnn/units').glob('*.csv'))
    sources+=sorted((study.OUT/'cnn/checkpoints').glob('*.pt'))
    sources += [study.OUT/'classical/units/r01_f1.csv',study.OUT/'classical/units/r01_f1.json']
    plan=dict(question='Residual image-method gain after fitting/validation membership, class-weight ratio and threshold controls',
              source_sha256={str(p):sha(p) for p in sources},
              versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','torch']},
              ids=ids.tolist(),units=[u['name'] for u in units],configs=configs,
              thresholds=np.round(np.arange(.10,.901,.01),2).tolist(),
              threshold_tie_rule='Highest validation F1, nearest 0.5, then lower threshold',
              config_tie_rule='Highest validation F1 at 0.5, first config (original config first)',
              fit='Original CNN fitting patients only; no validation refit; same image normalisation',
              class_weights={'inverse_frequency':'N/N_class; sample mean 2',
                             'cnn_ratio':'positive/negative ratio N/N_positive, normalized to sample mean 2'},
              selection='25 XGB configurations under cnn_ratio, on original CNN validation patients only',
              variants=['CNN','RF_inverse','RF_cnnratio','XGB_inverse','XGB_cnnratio','XGB_developed'],
              decision_rules=['fixed_0.5','validation_F1'],
              new_fits=3360,cnn_retraining=False,bootstrap_draws=5000,bootstrap_seed=2026100399,
              bootstrap='Stratified patient clusters; same patient weight across 20 repeats/methods; saved fits fixed',
              primary_contrasts=['CNN minus each XGB under corresponding threshold rule',
                                 'XGB_developed minus XGB_cnnratio under corresponding threshold rule'],
              report_all_metrics=['F1','balanced_accuracy','AUC','precision','recall','accuracy'],
              prospective_supplement_after_original_results=True)
    if (OUT/'image-plan.json').exists():assert json.loads((OUT/'image-plan.json').read_text())==plan,'Frozen plan changed'
    else:write(OUT/'image-plan.json',plan)
    return plan


def weights(y,policy):
    if policy=='inverse':return study.inverse_weights(y)
    ratio=len(y)/np.sum(y==1)
    w=np.array([1.,ratio]);w*=2/np.mean(w[y])
    assert np.isclose(np.mean(w[y]),2)
    return w


def estimator(name,seed,y,policy,params=None):
    model,_=study.estimator(name,seed,y);model.set_params(n_jobs=1)
    w=weights(y,policy)
    if name=='RandomForest':
        model.set_params(class_weight={0:float(w[0]),1:float(w[1])});sample_weight=None
    else:sample_weight=w[y]
    if params:model.set_params(**params)
    return model,sample_weight


def select_threshold(y,score,grid):
    rows=[(float(f1_score(y,score>=t,zero_division=0)),float(t)) for t in grid]
    return min(rows,key=lambda z:(-z[0],abs(z[1]-.5),z[1]))[1]


def fit_model(model,X,y,w):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if w is None:model.fit(X,y)
        else:model.fit(X,y,sample_weight=w)
    return [dict(category=a.category.__name__,message=str(a.message)) for a in caught]


def anchor():
    plan=prepare();ids,x,flat,y,strata,units=study.load_frozen();unit=units[0];flat=flat.astype(float)
    prior=pd.read_csv(study.OUT/'classical/units/r01_f1.csv');checks=[]
    for name,training,old_name in [('XGBoost','development','XGBoost'),('RandomForest','cnn_train','RandomForest_CNNfit')]:
        tr=np.array(unit[training]);te=np.array(unit['test']);model,w=estimator(name,unit['fit_seed'],y[tr],'inverse')
        with threadpool_limits(limits=1):
            fit_model(model,flat[tr],y[tr],w);score=model.predict_proba(flat[te])[:,1]
        target=prior[prior.method==old_name]
        assert target.participant_id.tolist()==ids[te].tolist()
        np.testing.assert_allclose(score,target.score,atol=1e-10,rtol=1e-10)
        checks.append(dict(method=old_name,max_probability_error=float(np.abs(score-target.score.to_numpy()).max())))
    write(OUT/'ANCHORS_PASS.json',dict(checks=checks,new_anchor_fits=2))
    print('Image historical anchors passed',flush=True)


def cnn_outputs():
    import torch
    torch.set_num_threads(2);plan=prepare();ids,x,flat,y,strata,units=study.load_frozen()
    factory=runpy.run_path(str(study.NETWORK))['generate_model'];checks=[]
    for unit in units:
        folder=OUT/'cnn'/unit['name']
        if (folder/'complete.json').exists():
            meta=json.loads((folder/'complete.json').read_text());assert meta['plan_sha256']==sha(OUT/'image-plan.json')
            assert sha(folder/'probabilities.csv')==meta['probabilities_sha256'];checks.append(meta);continue
        tr,va,te=[np.array(unit[k]) for k in ['cnn_train','cnn_validation','test']]
        record=json.loads((study.OUT/'cnn/units'/f"{unit['name']}.json").read_text())
        assert record['train_ids']==ids[tr].tolist() and record['validation_ids']==ids[va].tolist()
        checkpoint=study.OUT/'cnn'/record['checkpoint'];assert sha(checkpoint)==record['checkpoint_sha256']
        model=factory(drop=.8,model_depth=3,num_classes=1,in_channels=1)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        with torch.inference_mode():
            val=model(torch.from_numpy(x[va])).ravel().sigmoid().numpy()
            score=model(torch.from_numpy(x[te])).ravel().sigmoid().numpy()
        old=pd.read_csv(study.OUT/'cnn/units'/f"{unit['name']}.csv")
        assert old.participant_id.tolist()==ids[te].tolist()
        np.testing.assert_allclose(score,old.score,atol=2e-6,rtol=2e-5)
        flips=int(np.sum((score>=.5)!=old.y_pred.to_numpy()))
        assert flips==0,'CPU/MPS .5 classification changed; stop for explicit investigation'
        threshold=select_threshold(y[va],val,plan['thresholds'])
        frame=pd.DataFrame(dict(participant_id=ids[np.r_[va,te]],split=['validation']*len(va)+['test']*len(te),
                                y_true=y[np.r_[va,te]],score=np.r_[val,score]))
        folder.mkdir(parents=True,exist_ok=True);frame.to_csv(folder/'probabilities.csv',index=False)
        meta=dict(unit=unit['name'],plan_sha256=sha(OUT/'image-plan.json'),checkpoint_sha256=sha(checkpoint),
                  probabilities_sha256=sha(folder/'probabilities.csv'),threshold=threshold,
                  max_cpu_probability_error=float(np.abs(score-old.score.to_numpy()).max()),fixed_threshold_flips=flips,
                  train_ids=ids[tr].tolist(),validation_ids=ids[va].tolist(),test_ids=ids[te].tolist())
        write(folder/'complete.json',meta);checks.append(meta)
        print(f"CNN_RELOADED {unit['name']} {len(checks)}/120",flush=True)
    write(OUT/'CNN_RELOAD_COMPLETE.json',dict(units=checks,cnn_retraining=False))


def one_unit(name):
    plan=json.loads((OUT/'image-plan.json').read_text());ids,x,flat,y,strata,units=study.load_frozen()
    unit=next(u for u in units if u['name']==name);folder=OUT/'units'/name;plan_sha=sha(OUT/'image-plan.json')
    if (folder/'complete.json').exists():
        meta=json.loads((folder/'complete.json').read_text());assert meta['plan_sha256']==plan_sha
        assert all(sha(folder/n)==h for n,h in meta['files'].items());return meta
    flat=flat.astype(float);tr,va,te=[np.array(unit[k]) for k in ['cnn_train','cnn_validation','test']]
    assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    folder.mkdir(parents=True,exist_ok=True);begin=time.monotonic();models={};validation={};trials=[];warn=[]
    with threadpool_limits(limits=1):
        for family in ['RandomForest','XGBoost']:
            for policy in ['inverse','cnnratio']:
                if family=='XGBoost' and policy=='cnnratio':continue
                key=('RF' if family=='RandomForest' else 'XGB')+'_'+policy
                model,w=estimator(family,unit['fit_seed'],y[tr],policy)
                warn.extend(fit_model(model,flat[tr],y[tr],w));models[key]=model
                validation[key]=model.predict_proba(flat[va])[:,1]
        best_score=-np.inf;best_config=None
        for ci,params in enumerate(plan['configs']):
            model,w=estimator('XGBoost',unit['fit_seed'],y[tr],'cnnratio',params)
            warn.extend(fit_model(model,flat[tr],y[tr],w));v=model.predict_proba(flat[va])[:,1]
            value=float(f1_score(y[va],v>=.5,zero_division=0))
            trials.append(dict(config=ci,parameters=params,validation_f1_at_half=value,validation_scores=v.tolist()))
            if ci==0:models['XGB_cnnratio']=model;validation['XGB_cnnratio']=v
            if value>best_score:
                best_score=value;best_config=ci;models['XGB_developed']=model;validation['XGB_developed']=v
        thresholds={k:select_threshold(y[va],v,plan['thresholds']) for k,v in validation.items()}
        # Save the complete decision record before querying any classical test probabilities.
        selection=dict(unit=name,train_ids=ids[tr].tolist(),validation_ids=ids[va].tolist(),test_ids=ids[te].tolist(),
                       weight_vectors={p:weights(y[tr],p).tolist() for p in ['inverse','cnnratio']},
                       selected_config=best_config,thresholds=thresholds,trials=trials,
                       validation_scores={k:v.tolist() for k,v in validation.items()},chosen_before_test_unix=time.time())
        write(folder/'selection.json',selection)
        rows=[];checkpoints={}
        for key,model in models.items():
            path=folder/(key+'.joblib');joblib.dump(model,path,compress=3);checkpoints[path.name]=sha(path)
            score=model.predict_proba(flat[te])[:,1]
            for decision,threshold in [('fixed_0.5',.5),('validation_F1',thresholds[key])]:
                rows.extend(dict(participant_id=p,y_true=int(yy),repeat=unit['repeat'],fold=unit['fold'],
                                 method=key,decision=decision,threshold=threshold,score=float(s),y_pred=int(s>=threshold))
                            for p,yy,s in zip(ids[te],y[te],score))
    pd.DataFrame(rows).to_csv(folder/'predictions.csv',index=False)
    meta=dict(unit=name,plan_sha256=plan_sha,fit_count=28,seconds=time.monotonic()-begin,warnings=warn,
              files={n:sha(folder/n) for n in ['selection.json','predictions.csv']},checkpoints=checkpoints)
    write(folder/'complete.json',meta)
    print(f"IMAGE_UNIT {name} 28 fits {meta['seconds']:.1f}s",flush=True)
    return meta


def train():
    p=prepare();assert (OUT/'ANCHORS_PASS.json').exists();results=[];begin=time.monotonic()
    with parallel_config(backend='loky',n_jobs=3,inner_max_num_threads=1):
        for r in Parallel(return_as='generator_unordered')(delayed(one_unit)(n) for n in p['units']):
            results.append(r);write(OUT/'progress.json',dict(completed=len(results),total=len(p['units']),elapsed_seconds=time.monotonic()-begin,last=r))
    write(OUT/'TRAINING_COMPLETE.json',dict(units=results,new_fits=sum(r['fit_count'] for r in results)))


def analyze():
    p=prepare();assert (OUT/'TRAINING_COMPLETE.json').exists() and (OUT/'CNN_RELOAD_COMPLETE.json').exists()
    ids,x,flat,y,strata,units=study.load_frozen();frames=[]
    for unit in units:
        name=unit['name'];frames.append(pd.read_csv(OUT/'units'/name/'predictions.csv'))
        meta=json.loads((OUT/'cnn'/name/'complete.json').read_text());f=pd.read_csv(OUT/'cnn'/name/'probabilities.csv')
        f=f[f.split=='test'].drop(columns='split')
        for decision,threshold in [('fixed_0.5',.5),('validation_F1',meta['threshold'])]:
            frames.append(f.assign(method='CNN',decision=decision,threshold=threshold,y_pred=(f.score>=threshold).astype(int),repeat=unit['repeat'],fold=unit['fold']))
    allpred=pd.concat(frames,ignore_index=True);assert not allpred.duplicated(['participant_id','repeat','method','decision']).any()
    allpred.to_csv(OUT/'all-predictions.csv',index=False)
    byrepeat=[]
    for (method,decision,repeat),g in allpred.groupby(['method','decision','repeat']):
        g=g.set_index('participant_id').loc[ids];assert len(g)==173 and g.fold.nunique()==6
        yy=g.y_true.to_numpy();pr=g.y_pred.to_numpy();sc=g.score.to_numpy()
        assert np.array_equal(yy,y)
        tp=int(np.sum((yy==1)&(pr==1)));fp=int(np.sum((yy==0)&(pr==1)));fn=int(np.sum((yy==1)&(pr==0)));tn=int(np.sum((yy==0)&(pr==0)))
        byrepeat.append(dict(method=method,decision=decision,repeat=repeat,F1=f1_score(yy,pr),
                             balanced_accuracy=balanced_accuracy_score(yy,pr),AUC=roc_auc_score(yy,sc),
                             precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),accuracy=(tp+tn)/len(yy),TP=tp,FP=fp,FN=fn,TN=tn))
    metrics=pd.DataFrame(byrepeat);metrics.to_csv(OUT/'repeat-metrics.csv',index=False)
    metrics.groupby(['method','decision']).mean(numeric_only=True).drop(columns='repeat').to_csv(OUT/'mean-metrics.csv')
    rng=np.random.default_rng(p['bootstrap_seed']);positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0);B=p['bootstrap_draws']
    indices=np.concatenate([rng.choice(positive,(B,len(positive))),rng.choice(negative,(B,len(negative)))],axis=1)
    W=np.stack([np.bincount(r,minlength=len(ids)) for r in indices]);np.savez_compressed(OUT/'bootstrap-weights.npz',weights=W)
    cache={};summaries=[]
    with threadpool_limits(limits=2):
        for (method,decision),g in allpred.groupby(['method','decision']):
            P=g.pivot(index='repeat',columns='participant_id',values='y_pred').loc[range(1,21),ids].to_numpy()
            S=g.pivot(index='repeat',columns='participant_id',values='score').loc[range(1,21),ids].to_numpy()
            tp=W@((P==1)*(y==1)).T;fp=W@((P==1)*(y==0)).T;fn=len(positive)-tp;tn=len(negative)-fp
            bs={'F1':(2*tp/(2*tp+fp+fn)).mean(axis=1),
                'balanced_accuracy':(.5*(tp/len(positive)+tn/len(negative))).mean(axis=1)}
            # Weighted Mann–Whitney form, retaining ties and the same cluster draw across repeats.
            A=(S[:,positive,None]>S[:,None,negative]).astype(float)+.5*(S[:,positive,None]==S[:,None,negative])
            bs['AUC']=np.einsum('bi,ij,bj->b',W[:,positive],A.mean(axis=0),W[:,negative])/(len(positive)*len(negative))
            cache[(method,decision)]=bs
            sub=metrics[(metrics.method==method)&(metrics.decision==decision)]
            for metric,draws in bs.items():
                lo,hi=np.quantile(draws,[.025,.975]);summaries.append(dict(method=method,decision=decision,metric=metric,
                     mean=float(sub[metric].mean()),ci_low=lo,ci_high=hi,repeat_min=float(sub[metric].min()),repeat_max=float(sub[metric].max())))
    pairs=[]
    for decision in p['decision_rules']:
        pairs += [(('CNN',decision),(m,decision)) for m in p['variants'] if m!='CNN']
        pairs += [(('XGB_developed',decision),('XGB_cnnratio',decision)),(('XGB_cnnratio',decision),('XGB_inverse',decision))]
    pairs += [((m,'validation_F1'),(m,'fixed_0.5')) for m in p['variants']]
    contrasts=[]
    for left,right in pairs:
        for metric in ['F1','balanced_accuracy','AUC']:
            a=metrics[(metrics.method==left[0])&(metrics.decision==left[1])][metric].to_numpy()
            b=metrics[(metrics.method==right[0])&(metrics.decision==right[1])][metric].to_numpy()
            bs=cache[left][metric]-cache[right][metric];lo,hi=np.quantile(bs,[.025,.975])
            contrasts.append(dict(left_method=left[0],left_decision=left[1],right_method=right[0],right_decision=right[1],
                                  metric=metric,delta=float(np.mean(a-b)),ci_low=lo,ci_high=hi,
                                  repeat_min=float(np.min(a-b)),repeat_max=float(np.max(a-b))))
    pd.DataFrame(summaries).to_csv(OUT/'summary-intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired-contrasts.csv',index=False)
    write(OUT/'ANALYSIS_COMPLETE.json',dict(methods=len(cache),repeat_scores=len(metrics),contrasts=len(contrasts),bootstrap_draws=B,conditional_on_saved_fits=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','anchor','cnn','train','analyze']);stage=parser.parse_args().stage
    {'prepare':prepare,'anchor':anchor,'cnn':cnn_outputs,'train':train,'analyze':analyze}[stage]()
