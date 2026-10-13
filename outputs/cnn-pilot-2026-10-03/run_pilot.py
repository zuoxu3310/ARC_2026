"""A predeclared one-outer-fold local pilot; see PROTOCOL.txt before use."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import time
import traceback

import numpy as np
import pandas as pd
import psutil
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = ROOT / 'outputs/targeted-replication-feasibility-2026-10-03/raw'
DATA = ROOT / 'datasets/arc_ds004884/teghipco_figshare/DLAphasiaSeverityARCSubset.mat'
TABLE = ROOT / 'arc_lesion_image_benchmark/data/jhu156_features_merged.tsv'
NETWORK = RAW / 'deep-vgg2.py.txt'
SEED = 20261003
MAX_EPOCHS = 800
CONFIGS = [dict(name='lr8e-5', lr=8e-5, model_depth=3, drop=0.8, l2=0.001),
           dict(name='lr1e-4', lr=1e-4, model_depth=3, drop=0.8, l2=0.001)]
NETWORK_HASH = '6088e711ae2486f327d5fcbe7d6e703c5dd46792c0932847eefada29d4a59623'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def event(stage, **kwargs):
    item = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                unix_time=time.time(), stage=stage, **kwargs)
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(item, allow_nan=False) + '\n')
    save_json(OUT / 'heartbeat.json', item)
    print(json.dumps(item, ensure_ascii=False, allow_nan=False), flush=True)


def load_inputs():
    assert sha(NETWORK) == NETWORK_HASH, 'Author network snapshot changed'
    mat = loadmat(DATA, squeeze_me=True)
    all_ids = np.array([str(x) if str(x).startswith('sub-') else 'sub-' + str(x)
                        for x in mat['subsOnly']])
    assert len(all_ids) == len(set(all_ids)) == 173
    table = pd.read_csv(TABLE, sep='\t').set_index('participant_id', verify_integrity=True)
    ids = np.array(sorted(set(all_ids) & set(table.index)))
    assert len(ids) == 172
    ix = np.array([np.flatnonzero(all_ids == p)[0] for p in ids])
    maps = np.moveaxis(mat['data'][..., ix], -1, 0).astype(np.float32)
    assert maps.shape == (172, 17, 22, 19) and np.isfinite(maps).all()
    y = mat['wabClassi'][ix].astype(np.int64)
    subtype = mat['wabClass2i'][ix].astype(np.int64)
    assert np.array_equal(y, (subtype >= 3).astype(int))
    lo = maps.min(axis=(1, 2, 3), keepdims=True)
    hi = maps.max(axis=(1, 2, 3), keepdims=True)
    assert np.all(hi > lo)
    normalized = 2 * (maps - lo) / (hi - lo) - 1
    normalized = np.ascontiguousarray(normalized[:, None])
    assert np.allclose(normalized.min(axis=(1, 2, 3, 4)), -1)
    assert np.allclose(normalized.max(axis=(1, 2, 3, 4)), 1)
    frame = table.loc[ids]
    feature_names = [c for c in frame.columns if c != 'wab_aq']
    assert len(feature_names) == 191 and 'wab_aq' not in feature_names
    features = frame[feature_names].fillna(0).to_numpy(dtype=np.float64)
    assert np.isfinite(features).all()
    mismatch = (frame['wab_aq'].to_numpy() <= 50).astype(int) != y
    audit = dict(n_public=173, n_common=len(ids), excluded=sorted(set(all_ids) - set(ids)),
                 class_counts=np.bincount(y).tolist(), subtype_counts=np.bincount(subtype)[1:].tolist(),
                 shape=list(normalized.shape), n_maps_without_lesion_code=int(np.sum(hi.ravel() < 4)),
                 label_source='author wabClassi', feature_names=feature_names,
                 outcome_mismatches=[dict(participant_id=p, author_label=int(a), current_aq=float(q))
                                     for p, a, q in zip(ids[mismatch], y[mismatch],
                                                      frame['wab_aq'].to_numpy()[mismatch])])
    assert audit['excluded'] == ['sub-M2019']
    assert [r['participant_id'] for r in audit['outcome_mismatches']] == ['sub-M2087']
    return ids, normalized, y, subtype, features, audit


def partition(ids, subtype):
    outer = list(StratifiedKFold(6, shuffle=True, random_state=SEED).split(ids, subtype))
    dev, test = outer[0]
    inner = [(dev[a], dev[b]) for a, b in
             StratifiedKFold(8, shuffle=True, random_state=SEED + 1).split(dev, subtype[dev])]
    tr, val = train_test_split(dev, test_size=0.30, random_state=SEED + 2, stratify=subtype[dev])
    tr, val, test = np.sort(tr), np.sort(val), np.sort(test)
    assert not (set(tr) & set(val) or set(tr) & set(test) or set(val) & set(test))
    assert set(tr) | set(val) == set(dev)
    assert sorted(np.concatenate([b for a, b in inner]).tolist()) == sorted(dev.tolist())
    for a, b in inner:
        assert not (set(a) & set(b) or set(a) & set(test) or set(b) & set(test))
        assert set(a) | set(b) == set(dev)
    spec = dict(seed=SEED, chosen_outer_fold=1,
                all_outer_test_ids=[ids[b].tolist() for a, b in outer],
                development_ids=ids[dev].tolist(), test_ids=ids[test].tolist(),
                final_train_ids=ids[tr].tolist(), final_validation_ids=ids[val].tolist(),
                inner=[dict(fold=i + 1, train_ids=ids[a].tolist(), validation_ids=ids[b].tolist())
                       for i, (a, b) in enumerate(inner)])
    return tr, val, test, inner, spec


def prepare():
    ids, x, y, subtype, feats, audit = load_inputs()
    tr, val, test, inner, spec = partition(ids, subtype)
    files = [Path(__file__), OUT / 'PROTOCOL.txt', OUT / 'watchdog.py', DATA, TABLE, NETWORK]
    frozen = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  inputs_and_code={str(p.relative_to(ROOT)): sha(p) for p in files},
                  paper_pdf_sha256=sha(ROOT / 'paper/main.pdf'),
                  configs=CONFIGS, max_epochs=MAX_EPOCHS, n_cnn_fits=17,
                  n_train=len(tr), n_validation=len(val), n_test=len(test))
    for name, value in [('data-audit.json', audit), ('partitions.json', spec), ('frozen.json', frozen)]:
        if (OUT / name).exists():
            raise FileExistsError(f'Refusing to overwrite prepared file: {name}')
        save_json(OUT / name, value)
    print(json.dumps(frozen, indent=2), flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


def classify(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(n=len(y), f1_severe=float(f1_score(y, pred, zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                roc_auc=float(roc_auc_score(y, p)),
                precision_severe=float(precision_score(y, pred, zero_division=0)),
                recall_severe=float(recall_score(y, pred)),
                accuracy=float(accuracy_score(y, pred)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def quick_metrics(y, p):
    pred = p >= 0.5
    tp = np.sum((y == 1) & pred)
    fp = np.sum((y == 0) & pred)
    fn = np.sum((y == 1) & ~pred)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0, float(np.mean(pred == y))


def fit_cnn(config, train_x, train_y, val_x, val_y, train_ids, val_ids, name, seed):
    """No test data or test labels are accepted by the fitting function."""
    seed_everything(seed)
    gc.collect()
    torch.mps.empty_cache()
    generator = runpy.run_path(str(NETWORK))['generate_model']
    model = generator(drop=config['drop'], model_depth=config['model_depth'],
                      num_classes=1, in_channels=1).to('mps')
    assert sum(p.numel() for p in model.parameters()) == 522145
    tx, vx = torch.from_numpy(train_x).to('mps'), torch.from_numpy(val_x).to('mps')
    ty = torch.tensor(train_y, dtype=torch.float32, device='mps')
    vy = torch.tensor(val_y, dtype=torch.float32, device='mps')
    pos_weight = len(train_y) / float(np.sum(train_y == 1))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device='mps'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999),
                                  weight_decay=config['l2'], amsgrad=True, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-10)
    checkpoint = OUT / 'checkpoints' / f'{name}.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    trace_path = OUT / 'curves' / f'{name}.csv'
    trace_path.parent.mkdir(exist_ok=True)
    best_loss, best_epoch, best_f1, stagnant, anti_run = float('inf'), 0, 0.0, 0, 0
    train_history, val_history, durations = [], [], []
    sampled = dict(mps_driver_bytes=0, mps_current_bytes=0, process_rss_bytes=0)
    first_gradient_norm = None
    best_state = None
    started = time.monotonic()
    event('fit_start', name=name, config=config, n_train=len(train_y), n_validation=len(val_y))

    def sample_memory():
        readings = dict(mps_driver_bytes=int(torch.mps.driver_allocated_memory()),
                        mps_current_bytes=int(torch.mps.current_allocated_memory()),
                        process_rss_bytes=int(psutil.Process().memory_info().rss))
        for k, v in readings.items():
            sampled[k] = max(sampled[k], v)

    fields = ['epoch', 'lr', 'train_loss', 'validation_loss', 'train_f1', 'validation_f1',
              'train_accuracy', 'validation_accuracy', 'seconds', 'is_best']
    with trace_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, MAX_EPOCHS + 1):
            tick = time.monotonic()
            model.train()
            order = torch.randperm(len(train_y))
            lr = optimizer.param_groups[0]['lr']
            for batch in order.split(128):
                batch = batch.to('mps')
                optimizer.zero_grad(set_to_none=True)
                logits = model(tx[batch]).ravel()
                loss = criterion(logits, ty[batch])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite loss: {name}, epoch {epoch}')
                loss.backward()
                sample_memory()
                if epoch == 1:
                    first_gradient_norm = float(next(model.parameters()).grad.norm().item())
                    assert np.isfinite(first_gradient_norm) and first_gradient_norm > 0
                optimizer.step()
            scheduler.step(epoch)
            model.eval()
            with torch.inference_mode():
                train_logits = model(tx).ravel()
                val_logits = model(vx).ravel()
                train_loss = float(criterion(train_logits, ty).item())
                val_loss = float(criterion(val_logits, vy).item())
                train_p = train_logits.sigmoid().cpu().numpy()
                val_p = val_logits.sigmoid().cpu().numpy()
            assert np.isfinite([train_loss, val_loss]).all()
            assert np.isfinite(train_p).all() and np.isfinite(val_p).all()
            train_f1, train_acc = quick_metrics(train_y, train_p)
            val_f1, val_acc = quick_metrics(val_y, val_p)
            train_history.append(train_f1)
            val_history.append(val_f1)
            best_f1 = max(best_f1, val_f1)
            is_best = val_loss < best_loss
            if is_best:
                best_loss, best_epoch, stagnant = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stagnant += 1
            if epoch >= 30:
                a, b = np.array(train_history[-30:]), np.array(val_history[-30:])
                corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0
                anti_run = anti_run + 1 if corr < -0.3 else 0
            torch.mps.synchronize()
            duration = time.monotonic() - tick
            durations.append(duration)
            sample_memory()
            writer.writerow(dict(epoch=epoch, lr=lr, train_loss=train_loss, validation_loss=val_loss,
                                 train_f1=train_f1, validation_f1=val_f1, train_accuracy=train_acc,
                                 validation_accuracy=val_acc, seconds=duration, is_best=int(is_best)))
            handle.flush()
            reason = 'epoch_limit'
            triggers = []
            if best_f1 >= 0.7:
                if train_acc > 0.94:
                    triggers.append('train_accuracy_gt_0.94')
                if val_f1 > 0.9:
                    triggers.append('validation_f1_gt_0.9')
                if stagnant >= 50:
                    triggers.append('validation_loss_patience_50')
                if anti_run >= 50:
                    triggers.append('inverse_f1_correlation_50')
            if epoch == 1 or epoch % 20 == 0 or triggers or epoch == MAX_EPOCHS:
                event('epoch', name=name, epoch=epoch, best_epoch=best_epoch,
                      best_validation_loss=best_loss, best_validation_f1=best_f1,
                      train_accuracy=train_acc, elapsed_seconds=time.monotonic() - started,
                      sampled_memory=sampled)
            if triggers:
                reason = '+'.join(triggers)
                break
    assert best_state is not None
    torch.save(best_state, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    with torch.inference_mode():
        restored = float(criterion(model(vx).ravel(), vy).item())
    assert np.isclose(restored, best_loss, rtol=1e-5, atol=1e-6)
    record = dict(name=name, config=config, seed=seed, train_ids=list(train_ids), validation_ids=list(val_ids),
                  epochs=epoch, best_epoch=best_epoch, best_validation_loss=best_loss,
                  best_validation_f1_seen=best_f1, stop_reason=reason, pos_weight=pos_weight,
                  duration_seconds=time.monotonic() - started,
                  median_epoch_seconds=float(np.median(durations)), sampled_memory_max=sampled,
                  first_gradient_norm=first_gradient_norm, checkpoint_sha256=sha(checkpoint),
                  checkpoint=str(checkpoint.relative_to(OUT)), checkpoint_reload_loss=restored)
    save_json(OUT / 'checkpoints' / f'{name}.json', record)
    event('fit_complete', name=name, epochs=epoch, best_epoch=best_epoch,
          duration_seconds=record['duration_seconds'], stop_reason=reason,
          best_validation_loss=best_loss)
    # Only the final model is retained by the caller; tensors/optimizer leave scope.
    return model, record


def run():
    if (OUT / 'started.json').exists():
        raise FileExistsError('This attempt already started; preserve it, do not overwrite')
    frozen = json.loads((OUT / 'frozen.json').read_text())
    for path, expected in frozen['inputs_and_code'].items():
        assert sha(ROOT / path) == expected, f'Frozen file changed: {path}'
    assert sha(ROOT / 'paper/main.pdf') == frozen['paper_pdf_sha256']
    ids, x, y, subtype, feats, audit = load_inputs()
    tr, val, test, inner, spec = partition(ids, subtype)
    assert spec == json.loads((OUT / 'partitions.json').read_text())
    assert torch.backends.mps.is_available(), 'MPS unavailable; run authorized host process, do not fall back'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '1'
    torch.set_num_threads(4)
    started = time.monotonic()
    versions = dict(python=os.sys.version, torch=torch.__version__, numpy=np.__version__,
                    pandas=pd.__version__, device='mps', fallback=False,
                    recommended_max_mps_bytes=int(torch.mps.recommended_max_memory()))
    save_json(OUT / 'started.json', dict(unix_time=time.time(), versions=versions,
                                      frozen_sha256=sha(OUT / 'frozen.json')))
    event('pilot_start', n_cnn_fits=17, device='mps', max_epochs=MAX_EPOCHS)
    records = []
    tuning = []
    for config in CONFIGS:
        losses = []
        for fold, (itr, iva) in enumerate(inner, start=1):
            model, record = fit_cnn(config, x[itr], y[itr], x[iva], y[iva], ids[itr], ids[iva],
                                    f"{config['name']}_inner{fold}", SEED + 100 + fold)
            records.append(record)
            losses.append(record['best_validation_loss'])
            del model
            gc.collect()
            torch.mps.empty_cache()
        tuning.append(dict(config=config, inner_best_losses=losses,
                           mean_inner_best_loss=float(np.mean(losses))))
        save_json(OUT / 'tuning-progress.json', tuning)
    winner = min(tuning, key=lambda r: r['mean_inner_best_loss'])
    save_json(OUT / 'selected-before-test.json', dict(selected=winner, tuning=tuning, unix_time=time.time()))
    event('hyperparameters_selected', config=winner['config'], mean_inner_loss=winner['mean_inner_best_loss'])
    final, record = fit_cnn(winner['config'], x[tr], y[tr], x[val], y[val], ids[tr], ids[val],
                            'final_cnn', SEED + 1000)
    records.append(record)

    event('reference_fit_start')
    ref_started = time.monotonic()
    keep = np.sum(feats[tr] != 0, axis=0) > 0.10 * len(tr)
    assert keep.any()
    k = min(5, int(min(np.bincount(y[tr]))) - 1)
    jhu = Pipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=SEED, k_neighbors=k)),
                    ('rf', RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=4))])
    jhu.fit(feats[tr][:, keep], y[tr])
    jhu_time = time.monotonic() - ref_started
    ref_started = time.monotonic()
    flat = x.reshape(len(x), -1)
    same_map = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=4,
                                      class_weight='balanced')
    same_map.fit(flat[tr], y[tr])
    map_time = time.monotonic() - ref_started
    import joblib
    joblib.dump(dict(model=jhu, feature_mask=keep), OUT / 'checkpoints/rf_jhu.joblib')
    joblib.dump(same_map, OUT / 'checkpoints/rf_same_map.joblib')
    references = dict(train_ids=ids[tr].tolist(), jhu_feature_names=np.array(audit['feature_names'])[keep].tolist(),
                      rf_jhu_seconds=jhu_time, rf_same_map_seconds=map_time)
    save_json(OUT / 'references-before-test.json', references)
    event('outer_test_scoring_start', n_test=len(test))
    with torch.inference_mode():
        cnn_p = final(torch.from_numpy(x[test]).to('mps')).ravel().sigmoid().cpu().numpy()
    probabilities = dict(CNN=cnn_p, RF_JHU_SMOTE=jhu.predict_proba(feats[test][:, keep])[:, 1],
                         RF_same_map=same_map.predict_proba(flat[test])[:, 1])
    rows, metrics = [], {}
    for method, p in probabilities.items():
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        metrics[method] = classify(y[test], p)
        rows.extend(dict(participant_id=pid, repeat=0, fold=1, method=method,
                         y_true=int(label), probability=float(prob), y_pred=int(prob >= 0.5),
                         final_training_n=len(tr), development_n=len(tr) + len(val))
                    for pid, label, prob in zip(ids[test], y[test], p))
    pd.DataFrame(rows).to_csv(OUT / 'test-predictions.csv', index=False)
    assert sha(ROOT / 'paper/main.pdf') == frozen['paper_pdf_sha256']
    memory_max = {key: max(r['sampled_memory_max'][key] for r in records)
                  for key in records[0]['sampled_memory_max']}
    result = dict(status='complete', elapsed_seconds=time.monotonic() - started,
                  n_cnn_fits=len(records), total_cnn_epochs=sum(r['epochs'] for r in records),
                  cnn_training_seconds=sum(r['duration_seconds'] for r in records),
                  sampled_memory_max=memory_max, selected_config=winner['config'],
                  n_development=len(tr) + len(val), n_final_train=len(tr), n_final_validation=len(val),
                  n_test=len(test), test_class_counts=np.bincount(y[test]).tolist(),
                  metrics=metrics, references=references, fits=records,
                  formal_manuscript_unchanged=True, versions=versions,
                  scope='one selected outer fold; reduced two-setting CNN tuning grid; descriptive only')
    save_json(OUT / 'results.json', result)
    event('pilot_complete', elapsed_seconds=result['elapsed_seconds'], metrics=metrics,
          total_cnn_epochs=result['total_cnn_epochs'], sampled_memory_max=memory_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        try:
            run()
        except Exception:
            save_json(OUT / 'failure.json', dict(unix_time=time.time(), traceback=traceback.format_exc()))
            raise
