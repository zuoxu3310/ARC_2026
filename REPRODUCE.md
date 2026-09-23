# Reproduction guide

Run commands from the repository root with Python 3.11. The core package list is
`requirements.txt`; the original frozen-run environment is separately preserved
in `arc_lesion_image_benchmark/results/frozen_prediction_control_2026-09-22/requirements-lock.txt`.
These pin the listed packages, not every transitive dependency. Numerical
verification reports the tested environment rather than assuming bitwise
identity across operating systems.

## 1. Saved-prediction reproduction

```bash
python scripts/verify_release.py
python scripts/reproduce_frozen.py analyze
```

The second command regenerates sampling weights and metric distributions, not
just a copy of the summary CSV. It compares the results against the archive.
The original `s18_crossed_grid.py` and `s31_frozen_prediction_control.py` remain
unchanged, preserving their recorded hashes. The public wrapper creates a new
output directory and records the current executable and platform. The original
strict resume check is retained for each new run.

## 2. Training and grid

For the main fixed-prediction control, use the two commands in the README under
"Retrain the core experiment". For the complete crossed grid:

```bash
python scripts/run_grid.py --task regression --seeds 20
python scripts/run_grid.py --task classification --seeds 20
```

The grid wrapper calls the original model factories and scoring functions, and
saves to `arc_lesion_image_benchmark/reproduced/grid/results/runs/`. Each command
writes its requested grid in full; partial model/subset runs do not append to a
previous output. The shipped 3,000-cell grid remains unchanged.

TabPFN is optional and runs in a separate process:

```bash
python -m pip install -r requirements-optional.txt
python scripts/run_grid.py --task regression --models TabPFN --seeds 20
python scripts/run_grid.py --task classification --models TabPFN --seeds 20
```

The archived TabPFN grid used package version 6.3.2. Model weights are obtained
through that package; access and model-license terms belong to its upstream
provider. The saved grid permits all manuscript score analyses without fetching
those weights. The TabPFN classifier's original factory uses its package
default random state; the CV splits and SMOTE use the stated repeat seeds.

## 3. Cohort membership and image features

The supplied feature and metadata tables support all main experiments. To
reconstruct the ID lists, run:

```bash
python arc_lesion_image_benchmark/src/s17_freeze_pool_subsets.py
```

To reconstruct JHU and arterial features from normalized lesion masks:

```bash
python arc_lesion_image_benchmark/src/s0_download_lesion_masks.py
python arc_lesion_image_benchmark/src/s1_verify_masks.py
python arc_lesion_image_benchmark/src/s2_extract_atlas_features.py --atlas jhu156
python arc_lesion_image_benchmark/src/s2_extract_atlas_features.py --atlas ArterialAtlas156
```

The downloader pins the ARC demo commit. The valid age/WAB label source is
`datasets/arc_ds004884/derived_features_arc_demo/merged_artery_participants.tsv`.
The earlier upstream merged-JHU label file is not used. The modeling intersection
has 226 patients; 228 normalized lesion masks are available.

For the exploratory AAL/Brodmann/AICHA sweep, run
`python scripts/download_optional_inputs.py atlases`, then `s23_resample_atlases.py`
and `s2_extract_atlas_features.py --atlas NAME` for `aal116`, `brodmann41`, and
`aicha384`. The download manifest checks the exact source-image hashes; the
resampler writes int16 labels so the 384-region atlas is preserved. Precomputed
feature tables are already included.

The optional `python scripts/download_optional_inputs.py participant-list`
retrieves the CC0 Figshare input used by `teghipco_inspect_labels.py`. The
derived ID list and label cross-check are already included. This reconstructs
the secondary membership control.

## 4. Grid summaries and figures

```bash
python scripts/model_set_sensitivity.py
python scripts/reproduce_variance.py --task regression
python scripts/reproduce_variance.py --task classification
python scripts/reproduce_variance.py --task regression --include-tabpfn
python scripts/reproduce_variance.py --task classification --include-tabpfn
python arc_lesion_image_benchmark/src/fig_manuscript_rewrite_20260922.py
python scripts/build_manuscript_tables.py
```

The regression headline competitive set includes RF/XGBoost/LightGBM/TabPFN;
the classification headline set includes RF/XGBoost/LightGBM/logistic regression.
The classification set with TabPFN is a sensitivity. `s19` alone uses the older
three-model regression competitive set; `--include-tabpfn` calls `s19b` and is
required for the manuscript's regression headline. Variance outputs are written
to `arc_lesion_image_benchmark/reproduced/variance/`.

## 5. Supporting analyses

These sources and their saved outputs are supplied because the corresponding
analyses appear in the Appendix. Use a separate working copy when rerunning the
older scripts: their original output paths point to `results/`. Primary frozen
reproduction uses the portable wrapper above.

| Analysis | Source scripts |
|---|---|
| Size and severity matched refitting (300 draws) | `s20_matchedN_permutation.py` |
| Range restriction | `s21_range_restriction.py` |
| Atlas representations | `s22_granularity_spectrum.py`, `s23_resample_atlases.py`, `s2_extract_atlas_features.py` |
| Low-dimensional severe-aphasia baseline | `s3_classification_benchmark.py`, `s27_floor_vs_full_paired.py` |
| Nested tuning and paired atlas comparisons | `s5_nested_tuning_regression.py`, `s3b_classification_nested_dump.py`, `s6_significance_regression.py`, `s6_significance_classification.py` |
| Margin sensitivity | `fig_equivalence.py` |
| Calibration, subgroup errors, regional importance | `s7_calibration_rf.py`, `s7_subgroup_error.py`, `s8_xai_region_contribution.py` |
| Syndrome classification | `s11_type_classification.py` through `s16_type_floor_significance.py` |
| Exploratory composition explanation | `s28_cohort_explainers.py`, `s29_difficulty_composition.py`, `s30_mechanism_partition.py` |
| Pipeline diagnostics | `s26_leakage_probe.py` |

The older paired comparisons average patient predictions across ten repeats;
the main grid and frozen control average scores across twenty repeats. The
individual model–repeat arrays for the historical atlas margin intervals were
not retained; only their summaries are archived, as stated in the Appendix.
Rebuilding that interval calculation requires rerunning its original analysis.
