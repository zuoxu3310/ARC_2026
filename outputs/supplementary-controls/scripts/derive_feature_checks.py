"""Descriptive ordering and effect-stability summaries, after the frozen training run."""
import itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
import feature_sensitivity as a


def main():
    _,aq,_,manifest,masks=a.inputs()
    weights=np.load(a.old.OUT/'resampling_patient_weights.npz')['patient_bootstrap']
    primary=manifest.loc[~manifest.category.isin(['target_defined','exact_persons']),'subset'].tolist()
    rows=[];counts=[]
    with threadpool_limits(limits=2):
        for source in ['frozen_dictionary','training_only']:
            for task in a.old.TASKS:
                values={};boots={}
                for model in a.old.MODELS:
                    P,y=a.predictions('full',task,model,source)
                    for subset in manifest.subset:
                        v=a.old.weighted_metrics(masks[subset],y,P,task)
                        b=a.old.weighted_metrics(weights*masks[subset],y,P,task)
                        for metric in v:
                            values[model,subset,metric]=float(v[metric].mean())
                            boots[model,subset,metric]=b[metric].mean(axis=1)
                for metric in (['score','mae'] if task=='regression' else ['score']):
                    current=[]
                    for left,right in itertools.permutations(primary,2):
                        contrast={};bs={}
                        for name,sr,sx in [('cross',left,right),('same_rf_set',left,left),('same_xgb_set',right,right)]:
                            contrast[name]=values['XGBoost',sx,metric]-values['RandomForest',sr,metric]
                            bs[name]=boots['XGBoost',sx,metric]-boots['RandomForest',sr,metric]
                        signs={k:np.sign(v) for k,v in contrast.items()}
                        reversal=bool(signs['cross']!=0 and signs['same_rf_set']==signs['same_xgb_set']==-signs['cross'])
                        row=dict(source=source,task=task,metric=metric,rf_set=left,xgb_set=right,reversal=reversal)
                        for k,v in contrast.items():
                            lo,hi=np.quantile(bs[k],[.025,.975]);row.update({k:v,k+'_low':lo,k+'_high':hi})
                        rows.append(row);current.append(row)
                    assert len(current)==56
                    counts.append(dict(source=source,task=task,metric=metric,total=56,reversals=sum(r['reversal'] for r in current)))
    pd.DataFrame(rows).to_csv(a.OUT/'ordering-contrasts.csv',index=False)
    pd.DataFrame(counts).to_csv(a.OUT/'ordering-counts.csv',index=False)
    historical=pd.read_csv(a.old.ROOT/'results/comparison_population_2026-10-02/primary_ordering_counts.csv')
    for r in counts:
        if r['source']=='frozen_dictionary':
            expected=historical[(historical.task==r['task'])&(historical.metric==r['metric'])]
            assert r['reversals']==int(expected.strict_point_ordering_reversals.iloc[0])
    contrasts=pd.read_csv(a.OUT/'factorial_contrasts.csv')
    index=['task','model','subset','metric','contrast']
    c0=contrasts[contrasts.source=='frozen_dictionary'].set_index(index)
    c1=contrasts[contrasts.source=='training_only'].set_index(index)
    changed=(np.sign(c0.delta)!=np.sign(c1.delta))
    feature_counts=[]
    for p in (a.OUT/'units').glob('*/folds.json'):
        records=json.loads(p.read_text())
        feature_counts.extend(r['selected_n'] for r in records)
    checks=dict(ordering_counts=counts,selected_feature_min=min(feature_counts),selected_feature_max=max(feature_counts),
                selected_feature_median=float(np.median(feature_counts)),fits=len(feature_counts),
                all_contrasts=len(changed),contrast_sign_changes=int(changed.sum()),
                max_scoring_effect_change=float((c1.delta-c0.delta).reset_index(name='delta').query('contrast=="scoring_at_full_training"').delta.abs().max()))
    a.write(a.OUT/'scientific-summary.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
