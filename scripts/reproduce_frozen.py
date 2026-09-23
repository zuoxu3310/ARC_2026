#!/usr/bin/env python
"""Portable entry point for the manuscript's fixed-prediction experiment.

The original s18/s31 training modules remain byte-for-byte unchanged. New runs
have their own provenance and never overwrite the archived prediction bank.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'arc_lesion_image_benchmark'
ARCHIVE = PKG / 'results/frozen_prediction_control_2026-09-22'
sys.path.insert(0, str(PKG / 'src'))
import s31_frozen_prediction_control as analysis


def verify_original_inputs():
    provenance = json.loads((ARCHIVE / 'provenance.json').read_text())
    for relative, expected in provenance['sha256'].items():
        source = PKG / relative
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Archived analysis input changed: {relative}')
    features = analysis.load_data()[2]
    if features != provenance['features']:
        raise ValueError('Feature dictionary differs from the archived analysis')
    return provenance


def correct_ties(output):
    # Execute the archived downstream correction with only its output location
    # changed. All statistical code remains the original recorded correction.
    source = PKG / 'src/s31_resampling_tie_fix.py'
    text = source.read_text()
    original = 'OUT = ROOT / "results/frozen_prediction_control_2026-09-22"'
    assert text.count(original) == 1
    text = text.replace(original, 'OUT = Path(REPRODUCTION_OUTPUT)')
    # The correction's manifest records paths relative to the package.
    # Reproduction output is therefore constrained to a package subdirectory.
    exec(compile(text, str(source), 'exec'), {
        '__name__': '__main__', '__file__': str(source),
        'REPRODUCTION_OUTPUT': str(output),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['analyze', 'train'])
    parser.add_argument('--output', type=Path,
                        default=PKG / 'reproduced/frozen_prediction_control')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(PKG / 'reproduced'):
        parser.error('--output must be inside arc_lesion_image_benchmark/reproduced/')
    source_provenance = verify_original_inputs()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARCHIVE / 'ANALYSIS_PLAN.md', out / 'ANALYSIS_PLAN.md')
    analysis.OUT = out
    if args.stage == 'train':
        # New local provenance is written by the original training function.
        # Every one of the 80 trained cells is reconciled against the grid.
        analysis.train()
    else:
        if not (out / 'provenance.json').exists():
            for folder in ['predictions', 'folds']:
                shutil.copytree(ARCHIVE / folder, out / folder, dirs_exist_ok=True)
            for name in ['TRAINING_COMPLETE.json', 'subset_manifest.csv',
                         'patient_cohort_membership.csv', 'anchor_reconciliation.csv']:
                shutil.copy2(ARCHIVE / name, out / name)
            (out / 'source_provenance.json').write_text(
                json.dumps(source_provenance, indent=2) + '\n')
            analysis.write_json(out / 'provenance.json',
                                analysis.make_provenance(analysis.load_data()[2]))
        analysis.analyze()
        correct_ties(out)
        import numpy as np
        import pandas as pd
        current = pd.read_csv(out / 'frozen_prediction_summary.csv')
        archived = pd.read_csv(ARCHIVE / 'frozen_prediction_summary.csv')
        assert current.shape == archived.shape
        for col in archived.select_dtypes(include='number').columns:
            assert np.allclose(current[col], archived[col], rtol=0, atol=1e-10,
                               equal_nan=True), col
        old_draws = np.load(ARCHIVE / 'resampling_metric_distributions.npz')
        new_draws = np.load(out / 'resampling_metric_distributions.npz')
        assert set(old_draws.files) == set(new_draws.files)
        error = max(float(np.max(np.abs(old_draws[k] - new_draws[k])))
                    for k in old_draws.files)
        assert error < 1e-10, error
        old_weights = np.load(ARCHIVE / 'resampling_patient_weights.npz')
        new_weights = np.load(out / 'resampling_patient_weights.npz')
        assert set(old_weights.files) == set(new_weights.files)
        assert all(np.array_equal(old_weights[k], new_weights[k]) for k in old_weights.files)
        result = {'summary_rows': len(current), 'distributions': len(old_draws.files),
                  'maximum_distribution_error': error,
                  'sampling_weights_identical': True, 'training_repeated': False}
        analysis.write_json(out / 'reproduction_comparison.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
