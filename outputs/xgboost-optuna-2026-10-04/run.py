"""Exploratory nested Optuna XGBoost extension; see frozen PLAN.txt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = OUT.parent / 'cnn-same-input-2026-10-03'
SEED = 2026100407
N_TRIALS = 100
WORKERS = 8
METHODS = ['XGB_Optuna', 'XGB_Optuna_threshold', 'XGB_fixed_threshold']
THRESHOLDS = np.round(np.arange(.10, .901, .01), 2)
BASELINE = dict(n_estimators=400, learning_rate=.05, max_depth=3,
                min_child_weight=1., subsample=.8, colsample_bytree=.8,
                reg_alpha=0., reg_lambda=1., gamma=0., positive_multiplier=1.)


def previous():
    spec = importlib.util.spec_from_file_location('prior_same_input', OLD/'study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def save_pickle(path, value):
    tmp = path.with_suffix('.tmp')
    joblib.dump(value, tmp)
    tmp.replace(path)


def event(worker, stage, **kwargs):
    value = dict(time=time.time(), worker=worker, stage=stage, **kwargs)
    write_json(OUT/'workers'/f'{worker}.json', value)
    print(json.dumps(value), flush=True)


def inner_partitions(dev, strata, seed):
    # Only development labels enter stratification.
    return [(a, b) for a,b in StratifiedKFold(3, shuffle=True, random_state=seed)
            .split(np.zeros(len(dev)), strata[dev])]


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int('n_estimators', 100, 800, step=50),
        learning_rate=trial.suggest_float('learning_rate', .01, .2, log=True),
        max_depth=trial.suggest_int('max_depth', 1, 6),
        min_child_weight=trial.suggest_float('min_child_weight', .1, 20, log=True),
        subsample=trial.suggest_float('subsample', .6, 1.),
        colsample_bytree=trial.suggest_float('colsample_bytree', .4, 1.),
        reg_alpha=trial.suggest_categorical('reg_alpha', [0., .001, .01, .1, 1., 10.]),
        reg_lambda=trial.suggest_float('reg_lambda', .1, 100., log=True),
        gamma=trial.suggest_float('gamma', 0., 5.),
        positive_multiplier=trial.suggest_float('positive_multiplier', .5, 4., log=True),
    )


def fit(config, x, y, seed):
    config = dict(config)
    multiplier = config.pop('positive_multiplier')
    counts = np.bincount(y, minlength=2)
    assert np.all(counts > 0)
    w = len(y)/counts
    weights = w[y] * np.where(y == 1, multiplier, 1.)
    model = XGBClassifier(**config, objective='binary:logistic', tree_method='hist',
                          eval_metric='logloss', n_jobs=1, random_state=seed, verbosity=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x, y, sample_weight=weights)
    return model, [str(v.message) for v in caught]


def evaluate_config(config, dev_x, dev_y, inner, seed):
    # Deliberately has no outer-test data, labels, IDs, or previous scores argument.
    oof = np.full(len(dev_y), np.nan)
    seen = np.zeros(len(dev_y), dtype=int)
    notes = []
    for k,(train, val) in enumerate(inner):
        assert not (set(train) & set(val))
        model, caught = fit(config, dev_x[train], dev_y[train], seed+k)
        oof[val] = model.predict_proba(dev_x[val])[:, 1]
        seen[val] += 1
        notes.extend(caught)
    assert np.all(seen == 1) and np.isfinite(oof).all()
    value = float(f1_score(dev_y, oof >= .5, zero_division=0))
    return value, oof, notes


def select_threshold(y, score):
    p = score[:, None] >= THRESHOLDS[None, :]
    tp = ((y[:, None] == 1) & p).sum(0)
    fp = ((y[:, None] == 0) & p).sum(0)
    fn = ((y[:, None] == 1) & ~p).sum(0)
    values = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(THRESHOLDS)), where=(2*tp+fp+fn)>0)
    chosen = min(np.flatnonzero(values == values.max()),
                 key=lambda i: (round(abs(float(THRESHOLDS[i])-.5), 8), float(THRESHOLDS[i])))
    return float(THRESHOLDS[chosen]), values.tolist()


def prepare():
    assert not (OUT/'frozen.json').exists(), 'Do not overwrite the frozen plan'
    old = previous()
    ids,x,flat,y,strata,units = old.load_frozen()
    for index,unit in enumerate(units):
        dev = np.array(unit['development'])
        unit['optuna_seed'] = SEED + index*101
        unit['inner_seed'] = SEED + index*101 + 1
        unit['optuna_inner'] = [(a.tolist(), b.tolist()) for a,b in
                                inner_partitions(dev, strata, unit['inner_seed'])]
        # Keep the old outer and fitting seeds unchanged.
        unit.pop('svm_candidates')
        unit.pop('inner')
    write_json(OUT/'partitions.json', dict(subject_ids=ids.tolist(), units=units))
    sources = [OUT/'PLAN.txt', OUT/'run.py', OUT/'analyze.py', OUT/'test_run.py',
               OUT/'partitions.json', OLD/'frozen.json', OLD/'partitions.json',
               OLD/'all-predictions.csv', OLD/'summary.csv', OLD/'study.py',
               old.DATA, ROOT/'paper/main.pdf']
    import sklearn, xgboost
    write_json(OUT/'frozen.json', dict(created_unix=time.time(), seed=SEED, trials=N_TRIALS,
        workers=WORKERS, outer_units=len(units), baseline=BASELINE, methods=METHODS,
        packages=dict(optuna=optuna.__version__, xgboost=xgboost.__version__,
                      sklearn=sklearn.__version__, numpy=np.__version__),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    write_json(OUT/'status.json', dict(stage='prepared', time=time.time(), units=120, trials_per_unit=100))
    print('Frozen: 120 outer units x 100 Optuna trials x 3 inner fits.', flush=True)


def load_frozen():
    frozen = json.loads((OUT/'frozen.json').read_text())
    for relative, expected in frozen['sources'].items():
        assert sha(ROOT/relative) == expected, f'Frozen file changed: {relative}'
    old = previous()
    ids,x,flat,y,strata,_ = old.load_frozen()
    part = json.loads((OUT/'partitions.json').read_text())
    assert part['subject_ids'] == ids.tolist()
    return ids, flat, y, part['units']


def run_unit(unit, ids, flat, y, worker):
    start = time.monotonic()
    name = unit['name']
    folder = OUT/'units'/name
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder/'complete.json'
    if completed.exists():
        record = json.loads(completed.read_text())
        assert sha(folder/'predictions.csv') == record['prediction_sha256']
        return
    dev, te = np.array(unit['development']), np.array(unit['test'])
    assert not set(dev) & set(te)
    inner = [(np.array(a), np.array(b)) for a,b in unit['optuna_inner']]
    dev_x, dev_y = flat[dev], y[dev]
    state_path = folder/'search-state.joblib'
    if state_path.exists():
        state = joblib.load(state_path)
        search, predictions, elapsed_before = state['study'], state['oof'], state['elapsed_seconds']
        assert len(search.trials) == len(predictions)
    else:
        search = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
                                    seed=unit['optuna_seed'], n_startup_trials=20),
                                    pruner=optuna.pruners.NopPruner())
        search.enqueue_trial(BASELINE)
        predictions, elapsed_before = [], 0.
    event(worker, 'unit_start', unit=name, completed_trials=len(predictions))
    while len(predictions) < N_TRIALS:
        tick = time.monotonic()
        trial = search.ask()
        config = suggest(trial)
        value, oof, notes = evaluate_config(config, dev_x, dev_y, inner, unit['fit_seed'])
        trial.set_user_attr('warnings', notes)
        trial.set_user_attr('duration_seconds', time.monotonic()-tick)
        search.tell(trial, value)
        predictions.append(oof)
        assert len(search.trials) == len(predictions)
        save_pickle(state_path, dict(study=search, oof=predictions,
                    elapsed_seconds=elapsed_before + time.monotonic()-start))
        event(worker, 'trial_complete', unit=name, trials=len(predictions), total=N_TRIALS)
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in search.trials)
    best = min(search.trials, key=lambda t: (-t.value, t.number))
    threshold, threshold_f1 = select_threshold(dev_y, predictions[best.number])
    baseline_threshold, baseline_threshold_f1 = select_threshold(dev_y, predictions[0])
    selected = dict(unit=name, best_trial=best.number, parameters=best.params,
        inner_F1=best.value, threshold=threshold, threshold_grid=THRESHOLDS.tolist(),
        threshold_grid_F1=threshold_f1, baseline_threshold=baseline_threshold,
        baseline_threshold_grid_F1=baseline_threshold_f1,
        development_ids=ids[dev].tolist(), selected_before_outer_test_unix=time.time())
    write_json(folder/'selection.json', selected)
    np.savez_compressed(folder/'inner-predictions.npz', score=np.stack(predictions),
                        y=dev_y, participant_ids=ids[dev])
    search.trials_dataframe().to_csv(folder/'trials.csv', index=False)
    event(worker, 'final_fit', unit=name, trials=N_TRIALS)
    model, final_warnings = fit(best.params, dev_x, dev_y, unit['fit_seed'])
    checkpoint = folder/'model.ubj'
    model.save_model(checkpoint)
    # First new-model access to outer-test rows happens after selection was saved.
    score = model.predict_proba(flat[te])[:, 1]
    previous_frame = pd.read_csv(OLD/'classical/units'/f'{name}.csv')
    fixed = previous_frame[previous_frame.method == 'XGBoost']
    assert fixed.participant_id.tolist() == ids[te].tolist()
    old = previous()
    frames = [old.prediction_frame(ids,y,unit,'XGB_Optuna',score,.5,len(dev),'probability'),
              old.prediction_frame(ids,y,unit,'XGB_Optuna_threshold',score,threshold,len(dev),'probability'),
              old.prediction_frame(ids,y,unit,'XGB_fixed_threshold',fixed.score.to_numpy(),
                                   baseline_threshold,len(dev),'probability')]
    pd.concat(frames, ignore_index=True).to_csv(folder/'predictions.csv', index=False)
    write_json(completed, dict(unit=name, trials=N_TRIALS, best_trial=best.number,
        duration_seconds=elapsed_before + time.monotonic()-start,
        prediction_sha256=sha(folder/'predictions.csv'), checkpoint_sha256=sha(checkpoint),
        selection_sha256=sha(folder/'selection.json'), warnings=final_warnings,
        completed_unix=time.time()))
    event(worker, 'unit_complete', unit=name, trials=N_TRIALS,
          duration_seconds=elapsed_before + time.monotonic()-start)


def worker(index):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ids,flat,y,units = load_frozen()
    with threadpool_limits(limits=1):
        for unit in units[index::WORKERS]:
            run_unit(unit, ids, flat, y, index)
    event(index, 'worker_finished')


def run():
    load_frozen()
    start = time.time()
    (OUT/'workers').mkdir(exist_ok=True)
    processes, logs = [], []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONHASHSEED=str(SEED))
    write_json(OUT/'status.json', dict(stage='running', started_unix=start,
                                     pid=os.getpid(), workers=WORKERS, trials_per_unit=N_TRIALS))
    try:
        for index in range(WORKERS):
            log = (OUT/'workers'/f'{index}.log').open('a', buffering=1)
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-u', str(OUT/'run.py'),
                                                'worker', '--index', str(index)],
                                               env=env, stdout=log, stderr=subprocess.STDOUT))
        write_json(OUT/'pids.json', dict(supervisor=os.getpid(), children=[p.pid for p in processes]))
        last_print = 0.
        while any(p.poll() is None for p in processes):
            now = time.time()
            assert now-start < 6*3600, 'Six-hour supervisor runtime limit reached'
            for index,p in enumerate(processes):
                if p.poll() is not None:
                    assert p.returncode == 0, f'Worker {index} failed with code {p.returncode}'
                elif now-start > 600:
                    h = json.loads((OUT/'workers'/f'{index}.json').read_text())
                    assert now-h['time'] < 600, f'Worker {index} has not produced a heartbeat in 10 minutes'
            if now-last_print >= 30:
                complete = len(list((OUT/'units').glob('*/complete.json')))
                heartbeats = [json.loads(p.read_text()) for p in (OUT/'workers').glob('[0-7].json')]
                state = dict(stage='running', started_unix=start, updated_unix=now,
                             elapsed_seconds=now-start, completed_units=complete, total_units=120,
                             workers=heartbeats, pid=os.getpid())
                write_json(OUT/'status.json', state)
                print(json.dumps({k:v for k,v in state.items() if k!='workers'}), flush=True)
                last_print = now
            time.sleep(1)
        assert all(p.returncode == 0 for p in processes)
        assert len(list((OUT/'units').glob('*/complete.json'))) == 120
        write_json(OUT/'status.json', dict(stage='verifying', started_unix=start,
                                          fitting_seconds=time.time()-start))
        subprocess.run([sys.executable, '-u', str(OUT/'analyze.py')], env=env, check=True)
        write_json(OUT/'status.json', dict(stage='complete', started_unix=start,
            completed_unix=time.time(), total_seconds=time.time()-start,
            completed_units=120, completed_trials=12000))
        print('COMPLETED: 120 studies, 12000 trials, saved-model and metric verification passed.', flush=True)
    except BaseException:
        write_json(OUT/'failure.json', dict(time=time.time(), traceback=traceback.format_exc()))
        write_json(OUT/'status.json', dict(stage='failed', time=time.time(),
                                          detail='See failure.json; all completed trials preserved'))
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill()
        raise
    finally:
        for log in logs: log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'run', 'worker'])
    p.add_argument('--index', type=int)
    args = p.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run()
    else:
        try: worker(args.index)
        except BaseException:
            event(args.index, 'failed', traceback=traceback.format_exc())
            raise
