"""Refit the first repeat of each core task/model and reconcile stored anchors."""
from pathlib import Path
import json
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'arc_lesion_image_benchmark'
sys.path.insert(0, str(PKG / 'src'))
import s31_frozen_prediction_control as analysis

if __name__ == '__main__':
    X, aq, _, _, _ = analysis.load_data()
    records = []
    for task in analysis.TASKS:
        grid = pd.read_csv(PKG / f'results/runs/grid_{task}_fast.csv')
        for model in analysis.MODELS:
            _, folds, score = analysis.one_cell(X, aq, task, model, 1000)
            anchor = grid.query('subset=="full" and model==@model and seed==1000').iloc[0].score
            difference = float(score - anchor)
            assert abs(difference) < analysis.TOL, (task, model, difference)
            records.append({'task': task, 'model': model, 'seed': 1000,
                            'fold_fits': len(folds), 'score': score,
                            'archived_score': float(anchor), 'difference': difference})
    out = ROOT / 'outputs/training_smoke.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'status': 'passed', 'cells': records}, indent=2)+'\n')
    print(out.read_text())
