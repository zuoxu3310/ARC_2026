#!/usr/bin/env python
"""Independently recompute scores, validate folds, and check archived checksums."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'arc_lesion_image_benchmark'
BASE = PKG / 'results/frozen_prediction_control_2026-09-22'
sys.path.insert(0, str(Path(__file__).parent))
from reproduce_frozen import verify_original_inputs


def main():
    verify_original_inputs()
    manifest_path = ROOT / 'checksums.sha256.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    membership = pd.read_csv(BASE / 'patient_cohort_membership.csv').set_index('participant_id')
    summary = pd.read_csv(BASE / 'frozen_prediction_summary.csv')
    subsets = membership.columns.drop('y_aq')
    assert len(membership) == 226 and membership.index.is_unique
    grid = pd.concat([pd.read_csv(p) for p in sorted((PKG / 'results/runs').glob('grid_*.csv'))])
    assert len(grid) == 3000
    assert not grid.duplicated(['task', 'subset', 'model', 'seed']).any()
    estimates, anchors, count, fold_count = {}, [], 0, 0
    for path in sorted((BASE / 'predictions').glob('*.csv')):
        frame = pd.read_csv(path).set_index('participant_id')
        assert frame.index.equals(membership.index)
        assert np.array_equal(frame.y_aq, membership.y_aq)
        task, model, seed = frame.iloc[0][['task', 'model', 'seed']]
        count += len(frame)
        expected_target = membership.y_aq if task == 'regression' else (membership.y_aq <= 50).astype(int)
        assert np.array_equal(frame.y_true, expected_target)
        folds = json.loads((BASE / 'folds' / f'{path.stem}.json').read_text())
        assert len(folds) == 5
        all_test = []
        for fold in folds:
            train_ids, test_ids = set(fold['train_ids']), set(fold['test_ids'])
            assert not train_ids & test_ids
            assert train_ids | test_ids == set(membership.index)
            assert fold['scaler_n'] == len(train_ids)
            assert fold['scaler_train_mean_max_error'] < 1e-10
            assert (frame.loc[list(test_ids), 'fold'] == fold['fold']).all()
            if task == 'classification':
                values = frame.loc[list(train_ids), 'y_true'].astype(int)
                assert fold['smote_original_minority_n'] == np.bincount(values).min()
            all_test.extend(test_ids)
            fold_count += 1
        assert len(all_test) == len(set(all_test)) == 226
        for subset in subsets:
            sub = frame.loc[membership[subset]]
            score = (float(pearsonr(sub.y_true, sub.prediction).statistic)
                     if task == 'regression' else
                     float(balanced_accuracy_score(sub.y_true, sub.prediction)))
            estimates.setdefault((task, model, subset, 'score'), []).append(score)
            if task == 'regression':
                estimates.setdefault((task, model, subset, 'mae'), []).append(
                    float(mean_absolute_error(sub.y_true, sub.prediction)))
            if subset == 'full':
                anchor = grid.query('task==@task and model==@model and seed==@seed and subset=="full"')
                assert len(anchor) == 1
                anchors.append(abs(score - float(anchor.score.iloc[0])))
    errors = []
    for row in summary.itertuples():
        vals = estimates[(row.task, row.model, row.subset, row.metric)]
        assert len(vals) == 20
        errors.append(abs(np.mean(vals) - row.observed))
    assert len(errors) == 60 and len(anchors) == 80
    assert count == 18080 and fold_count == 400
    assert max(errors) < 1e-12 and max(anchors) < 1e-12
    result = {'status': 'passed', 'grid_cells': len(grid),
              'held_out_predictions': count, 'folds_checked': fold_count,
              'summary_estimates_recomputed': len(errors),
              'max_summary_error': max(errors), 'grid_anchors_checked': len(anchors),
              'max_grid_anchor_error': max(anchors), 'model_training_repeated': False}
    out = ROOT / 'outputs/verification.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
