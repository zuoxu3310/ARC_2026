"""Frozen same-image re-evaluation. See PLAN.txt for estimands and source gaps."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / 'datasets/arc_ds004884/teghipco_figshare/DLAphasiaSeverityARCSubset.mat'
PILOT = ROOT / 'outputs/cnn-pilot-2026-10-03/run_pilot.py'
NETWORK = ROOT / 'outputs/targeted-replication-feasibility-2026-10-03/raw/deep-vgg2.py.txt'
BASE_SEED = 2026100307
REPEATS = 20
CONFIG = dict(name='author_recommended', lr=1e-4, model_depth=3, drop=0.8, l2=0.001)
FIXED_MODELS = ['RandomForest', 'XGBoost', 'LightGBM', 'LogReg', 'MLP']
EXPECTED_METHODS = ['CNN', 'SVM_linear', 'SVM_rbf', 'SVM_selected',
                    'SVM_linear_CNNfit', 'SVM_rbf_CNNfit',
                    *FIXED_MODELS, 'RandomForest_CNNfit', 'Majority']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float,np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def event(worker, stage, **kw):
    folder = OUT / worker
    folder.mkdir(exist_ok=True)
    obj = dict(unix_time=time.time(), stage=stage, worker=worker, **kw)
    with (folder / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')
    write_json(folder / 'heartbeat.json', obj)
    print(json.dumps(obj, allow_nan=False), flush=True)


def inputs():
    m = loadmat(DATA, squeeze_me=True)
    raw_ids = np.array([str(s) if str(s).startswith('sub-') else 'sub-' + str(s)
                        for s in m['subsOnly']])
    order = np.argsort(raw_ids)
    ids = raw_ids[order]
    raw = np.moveaxis(m['data'][..., order], -1, 0).astype(np.float32)
    assert raw.shape == (173, 17, 22, 19) and len(set(ids)) == 173
    assert np.isfinite(raw).all()
    y = np.asarray(m['wabClassi'][order], dtype=int)
    strata = np.asarray(m['wabClass2i'][order], dtype=int)
    assert np.array_equal(y, (strata >= 3).astype(int))
    lo, hi = raw.min((1, 2, 3), keepdims=True), raw.max((1, 2, 3), keepdims=True)
    assert np.all(hi > lo)
    x = np.ascontiguousarray((2 * (raw - lo) / (hi - lo) - 1)[:, None])
    flat = x.reshape(len(ids), -1)
    assert flat.shape == (173, 7106) and np.array_equal(flat.reshape(x.shape), x)
    assert np.bincount(y).tolist() == [119, 54]
    return ids, x, flat, y, strata


def prepare():
    if (OUT / 'frozen.json').exists():
        raise FileExistsError('Protocol already frozen; do not replace it')
    ids, x, flat, y, strata = inputs()
    rng = np.random.default_rng(BASE_SEED)
    units = []
    cspace = np.geomspace(1e-3, 2e4, 300)
    scales = np.geomspace(1e-3, 1e3, 300)
    for r in range(REPEATS):
        outer_seed = int(rng.integers(1, 2**31 - 1))
        outer = StratifiedKFold(6, shuffle=True, random_state=outer_seed)
        for f, (dev, test) in enumerate(outer.split(ids, strata)):
            seed = int(rng.integers(1, 2**31 - 1))
            tr, val = train_test_split(dev, test_size=.3, random_state=seed,
                                      stratify=strata[dev])
            inner = [(dev[a].tolist(), dev[b].tolist()) for a, b in
                     StratifiedKFold(8, shuffle=True, random_state=seed+1).split(dev, strata[dev])]
            pairs = rng.choice(90000, 300, replace=False)
            candidates = [dict(C=float(cspace[i // 300]), scale=float(scales[i % 300]))
                          for i in pairs]
            units.append(dict(name=f'r{r+1:02d}_f{f+1}', repeat=r+1, fold=f+1,
                              outer_seed=outer_seed, fit_seed=seed,
                              development=dev.tolist(), test=test.tolist(),
                              cnn_train=np.sort(tr).tolist(), cnn_validation=np.sort(val).tolist(),
                              inner=inner, svm_candidates=candidates))
    write_json(OUT / 'partitions.json', dict(subject_ids=ids.tolist(), units=units))
    sources = [OUT / 'PLAN.txt', OUT / 'source-alignment.tsv', OUT / 'study.py', OUT / 'watchdog.py',
               OUT / 'verify.py', OUT / 'test_study.py', OUT / 'partitions.json',
               DATA, NETWORK, PILOT]
    write_json(OUT / 'frozen.json', dict(created_unix=time.time(), base_seed=BASE_SEED,
               n=173, shape=list(x.shape), classes=np.bincount(y).tolist(),
               cnn_config=CONFIG, n_outer_units=len(units), expected_methods=EXPECTED_METHODS,
               sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
               paper_sha256=sha(ROOT / 'paper/main.pdf')))
    print(json.dumps(dict(frozen=True, n=173, cnn_fits=len(units), methods=EXPECTED_METHODS)))


def load_frozen():
    frozen = json.loads((OUT / 'frozen.json').read_text())
    for p, expected in frozen['sources'].items():
        assert sha(ROOT / p) == expected, f'Frozen source changed: {p}'
    ids, x, flat, y, strata = inputs()
    spec = json.loads((OUT / 'partitions.json').read_text())
    assert spec['subject_ids'] == ids.tolist()
    return ids, x, flat, y, strata, spec['units']


def import_trainer():
    spec = importlib.util.spec_from_file_location('frozen_pilot_trainer', PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT / 'cnn'
    module.OUT.mkdir(exist_ok=True)
    module.event = lambda stage, **kwargs: event('cnn', stage, **kwargs)
    return module


def prediction_frame(ids, y, unit, method, score, threshold, fitted_n, score_kind):
    te = np.array(unit['test'])
    score = np.asarray(score, dtype=float)
    assert score.shape == (len(te),) and np.isfinite(score).all()
    return pd.DataFrame(dict(participant_id=ids[te], repeat=unit['repeat'], fold=unit['fold'],
                             method=method, y_true=y[te], score=score,
                             y_pred=(score >= threshold).astype(int), threshold=threshold,
                             score_kind=score_kind, fitted_n=fitted_n,
                             development_n=len(unit['development'])))


def complete_unit(worker, unit, frame, record):
    folder = OUT / worker / 'units'
    folder.mkdir(exist_ok=True)
    path = folder / f"{unit['name']}.csv"
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    record['prediction_sha256'] = sha(path)
    write_json(folder / f"{unit['name']}.json", record)
    event(worker, 'unit_complete', unit=unit['name'], repeat=unit['repeat'], fold=unit['fold'],
          duration_seconds=record['duration_seconds'], methods=sorted(frame.method.unique().tolist()))


def already_done(worker, unit):
    p = OUT / worker / 'units' / f"{unit['name']}.json"
    if not p.exists():
        return False
    record = json.loads(p.read_text())
    assert sha(p.with_suffix('.csv')) == record['prediction_sha256']
    return True


def run_cnn(max_units=None):
    import torch
    assert torch.backends.mps.is_available(), 'MPS required; do not silently use CPU'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '1'
    ids, x, flat, y, strata, units = load_frozen()
    module = import_trainer()
    torch.set_num_threads(4)
    processed = 0
    for unit in units:
        if already_done('cnn', unit):
            continue
        if max_units is not None and processed >= max_units:
            break
        started = time.monotonic()
        tr, va, te = (np.array(unit[k]) for k in ['cnn_train','cnn_validation','test'])
        # fit_cnn has no argument for outer test data or labels.
        model, record = module.fit_cnn(CONFIG, x[tr], y[tr], x[va], y[va],
                                       ids[tr], ids[va], unit['name'], unit['fit_seed'])
        record['chosen_before_test_unix'] = time.time()
        with torch.inference_mode():
            score = model(torch.from_numpy(x[te]).to('mps')).ravel().sigmoid().cpu().numpy()
        frame = prediction_frame(ids, y, unit, 'CNN', score, .5, len(tr), 'probability')
        record['duration_seconds'] = time.monotonic()-started
        complete_unit('cnn', unit, frame, record)
        del model
        gc.collect()
        torch.mps.empty_cache()
        processed += 1
    event('cnn', 'worker_finished', completed=sum(already_done('cnn',u) for u in units), total=len(units))


def inverse_weights(y):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    assert np.all(counts > 0)
    return len(y) / counts


def kernel_matrix(a, b, kind, scale):
    dot = a @ b.T
    if kind == 'linear':
        return np.ascontiguousarray(dot / scale**2)
    dist = np.maximum(np.sum(a*a,axis=1)[:,None] + np.sum(b*b,axis=1)[None,:] - 2*dot, 0)
    return np.ascontiguousarray(np.exp(-dist/scale**2))


def tune_svm(dev_x, dev_y, inner_local, candidates, kind, unit_name):
    """Only development images/labels enter selection; outer-test data are absent."""
    dot = dev_x @ dev_x.T
    dist = np.maximum(np.diag(dot)[:,None] + np.diag(dot)[None,:] - 2*dot, 0)
    records = []
    best = None
    for ci, config in enumerate(candidates):
        gram = dot/config['scale']**2 if kind=='linear' else np.exp(-dist/config['scale']**2)
        losses = []
        for train, val in inner_local:
            w = inverse_weights(dev_y[train])
            model = SVC(C=config['C'], kernel='precomputed', class_weight={0:w[0],1:w[1]},
                        tol=1e-3, shrinking=True, probability=False)
            model.fit(np.ascontiguousarray(gram[np.ix_(train,train)]), dev_y[train])
            assert model.fit_status_ == 0
            margins = model.decision_function(np.ascontiguousarray(gram[np.ix_(val,train)]))
            hinge = np.maximum(0, 1-(2*dev_y[val]-1)*margins)
            losses.append(float(np.average(hinge, weights=w[dev_y[val]])))
        row = dict(candidate=ci, **config, inner_losses=losses, mean_loss=float(np.mean(losses)))
        records.append(row)
        if best is None or row['mean_loss'] < best['mean_loss']:
            best = row
        if ci == 0 or (ci+1)%25 == 0:
            event('classical','svm_search',unit=unit_name,kernel=kind,candidate=ci+1,total=len(candidates))
    return best, records


def estimator(name, seed, y):
    # The common inverse-frequency principle follows the original classical methods.
    w = inverse_weights(y)
    cw = {0:float(w[0]),1:float(w[1])}
    if name=='RandomForest':
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=4,class_weight=cw), None
    if name=='LogReg':
        return LogisticRegression(max_iter=5000,random_state=seed,class_weight=cw), None
    if name=='MLP':
        return MLPClassifier(hidden_layer_sizes=(64,32),max_iter=2000,random_state=seed), w[y]
    if name=='XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400,learning_rate=.05,max_depth=3,subsample=.8,
                  colsample_bytree=.8,random_state=seed,n_jobs=4,verbosity=0,eval_metric='logloss'), w[y]
    if name=='LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400,learning_rate=.05,num_leaves=15,subsample=.8,
                  colsample_bytree=.8,random_state=seed,n_jobs=4,verbose=-1,class_weight=cw), None
    raise KeyError(name)


def run_classical(max_units=None):
    ids, x, flat, y, strata, units = load_frozen()
    flat = flat.astype(np.float64)
    checkpoint_dir = OUT/'classical/checkpoints'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    processed = 0
    with threadpool_limits(limits=4):
        for unit in units:
            if already_done('classical',unit):
                continue
            if max_units is not None and processed>=max_units:
                break
            start = time.monotonic()
            event('classical','unit_start',unit=unit['name'])
            dev, te, ct = (np.array(unit[k]) for k in ['development','test','cnn_train'])
            local = {i:j for j,i in enumerate(dev)}
            inner = [(np.array([local[i] for i in a]),np.array([local[i] for i in b])) for a,b in unit['inner']]
            frames, fit_records, selected = [], [], {}
            for kind in ['linear','rbf']:
                best, trials = tune_svm(flat[dev],y[dev],inner,unit['svm_candidates'],kind,unit['name'])
                # Save all choices before accessing outer-test feature rows.
                write_json(OUT/'classical/search'/f"{unit['name']}_{kind}.json",
                           dict(selected=best,trials=trials,chosen_before_test_unix=time.time()))
                for tr,suffix in [(dev,''),(ct,'_CNNfit')]:
                    method = 'SVM_'+kind+suffix
                    w = inverse_weights(y[tr])
                    model=SVC(C=best['C'],kernel='precomputed',class_weight={0:w[0],1:w[1]},
                              tol=1e-3,shrinking=True,probability=False)
                    tick=time.monotonic()
                    model.fit(kernel_matrix(flat[tr],flat[tr],kind,best['scale']),y[tr])
                    assert model.fit_status_==0
                    path=checkpoint_dir/f"{unit['name']}_{method}.joblib"
                    joblib.dump(dict(model=model,train_indices=tr,kernel=kind,scale=best['scale']),path)
                    score=model.decision_function(kernel_matrix(flat[te],flat[tr],kind,best['scale']))
                    frames.append(prediction_frame(ids,y,unit,method,score,0,len(tr),'margin'))
                    fit_records.append(dict(method=method,train_ids=ids[tr].tolist(),
                         checkpoint_sha256=sha(path),duration_seconds=time.monotonic()-tick,selected=best))
                    if not suffix:
                        selected[kind]=(best['mean_loss'],frames[-1].copy())
            selected_kind=min(selected,key=lambda k:selected[k][0])
            chosen=selected[selected_kind][1]
            chosen['method']='SVM_selected'
            frames.append(chosen)
            for name in FIXED_MODELS:
                for tr,suffix in ([(dev,''),(ct,'_CNNfit')] if name=='RandomForest' else [(dev,'')]):
                    method=name+suffix
                    event('classical','fit_start',unit=unit['name'],method=method)
                    model, weights=estimator(name,unit['fit_seed'],y[tr])
                    tick=time.monotonic()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        if weights is None:
                            model.fit(flat[tr],y[tr])
                        else:
                            model.fit(flat[tr],y[tr],sample_weight=weights)
                        path=checkpoint_dir/f"{unit['name']}_{method}.joblib"
                        joblib.dump(model,path)
                        score=model.predict_proba(flat[te])[:,1]
                    frames.append(prediction_frame(ids,y,unit,method,score,.5,len(tr),'probability'))
                    fit_records.append(dict(method=method,train_ids=ids[tr].tolist(),checkpoint_sha256=sha(path),
                         duration_seconds=time.monotonic()-tick,parameters=json_safe(model.get_params()),
                         warnings=[str(v.message) for v in caught]))
                    event('classical','fit_complete',unit=unit['name'],method=method,
                          duration_seconds=time.monotonic()-tick,warnings=len(caught))
            majority=int(np.argmax(np.bincount(y[dev])))
            frames.append(prediction_frame(ids,y,unit,'Majority',np.full(len(te),majority),.5,len(dev),'probability'))
            record=dict(unit=unit['name'],duration_seconds=time.monotonic()-start,
                        selected_svm_kernel=selected_kind,fits=fit_records)
            complete_unit('classical',unit,pd.concat(frames,ignore_index=True),record)
            processed+=1
    event('classical','worker_finished',completed=sum(already_done('classical',u) for u in units),total=len(units))


def scores(group):
    y,p,s=group.y_true.to_numpy(),group.y_pred.to_numpy(),group.score.to_numpy()
    return dict(F1=float(f1_score(y,p,zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y,p)),AUC=float(roc_auc_score(y,s)),
                precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p)),
                accuracy=float(accuracy_score(y,p)),n=len(y))


def aggregate():
    paths=sorted((OUT/'cnn/units').glob('*.csv'))+sorted((OUT/'classical/units').glob('*.csv'))
    if not paths:
        return
    frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    assert not frame.duplicated(['participant_id','repeat','method']).any()
    complete=[]
    for (method,repeat),g in frame.groupby(['method','repeat']):
        if len(g)==173 and g.fold.nunique()==6:
            complete.append(dict(method=method,repeat=int(repeat),**scores(g)))
    if complete:
        r=pd.DataFrame(complete)
        r.to_csv(OUT/'repeat-scores.csv',index=False)
        summary=r.groupby('method').agg(repeats=('repeat','count'),F1_mean=('F1','mean'),
            F1_sd=('F1','std'),BA_mean=('balanced_accuracy','mean'),BA_sd=('balanced_accuracy','std'),
            AUC_mean=('AUC','mean'),AUC_sd=('AUC','std'))
        summary.to_csv(OUT/'summary.csv')
        print(summary.to_string())
    frame.to_csv(OUT/'all-predictions.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','cnn','classical','aggregate'])
    p.add_argument('--max-units',type=int)
    args=p.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='aggregate': aggregate()
    else:
        try:
            {'cnn':run_cnn,'classical':run_classical}[args.action](args.max_units)
        except BaseException:
            event(args.action,'failed',traceback=traceback.format_exc())
            raise


if __name__=='__main__':
    main()
