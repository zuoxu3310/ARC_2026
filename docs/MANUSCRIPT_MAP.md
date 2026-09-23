# Manuscript-to-code map

Manuscript: **Higher Scores, Unchanged Predictions: Cohort Selection in
Lesion-Based Aphasia Severity Prediction** (submission draft, 2026-09-23).

Paths below are relative to `arc_lesion_image_benchmark/`, except where marked.

| Manuscript component | Method source | Numerical source |
|---|---|---|
| Analysis pool and eligibility table | `src/s17_freeze_pool_subsets.py` | `data/subsets/subset_manifest.csv`, `*_ids.txt`, `data/jhu156_features_merged.tsv` |
| Cohort–model grid | `src/s18_crossed_grid.py` | `results/runs/grid_{regression,classification}_{fast,TabPFN}.csv` |
| Explained score variation | `src/s19_variance_decomp.py`, `src/s19b_variance_decomp_with_tabpfn.py` | `results/summary/variance_decomposition*.csv` |
| Fixed-prediction scores, paired intervals | `src/s31_frozen_prediction_control.py` | `results/frozen_prediction_control_2026-09-22/` predictions, folds, summary and bootstrap arrays |
| Size-only / severity-bin matched sampling | Same `s31`, plus `s31_resampling_tie_fix.py` | `resampling_patient_weights.npz`, `resampling_metric_distributions.npz` in the frozen directory |
| Figure 1: story and control design | `src/fig_first_story_20260923.py` | Frozen `patient_cohort_membership.csv` and `frozen_prediction_summary.csv` |
| Figures 2–4: fixed shifts, composition, refitted comparison | `src/fig_manuscript_rewrite_20260922.py` | Frozen summary, distributions, `original_vs_frozen.csv` |
| Complete fixed-prediction / matching tables | Root `scripts/build_manuscript_tables.py` | Frozen summary |
| Model-threshold sensitivity | Root `scripts/model_set_sensitivity.py` | Four complete grid files; root `analysis/model_sensitivity/` |
| Additional composition analysis | `src/s20_*`, `s21_*`, `s28_*`, `s29_*`, `s30_*` | `results/summary/matchedN_perm_*.csv`, `range_restriction.csv`, `cohort_explainers_*.csv`, `subset_difficulty_composition.csv` |
| Atlas representations and simple baselines | `src/s22_*`, `s27_*` | `results/summary/granularity_spectrum.csv`, `granularity_tost.csv`, `floor_5x10.csv`, `floor_vs_full_paired_clf.csv` |
| Paired atlas / tuning comparisons | `src/s5_*`, `s3b_*`, `s6_*` | Patient predictions in `results/runs/`, `results/summary/significance_*.csv` |
| Calibration / subgroup / prediction errors | `src/s7_*`, `s8_*` | `results/summary/calibration_RF*.csv`, `subgroup_*.csv`, `xai_*.csv` |
| Syndrome prediction | `src/s11_*` through `s16_*` | `results/summary/type_*.csv` and patient probabilities in `results/runs/` |
| Verification | Root `scripts/verify_release.py` | 80 frozen prediction files and 80 files containing five fold records each |

The central regression anchors are means of 20 repeat-level correlations:

| Model | Full (226) | FLAIR (135) | Published-list intersection (172, secondary) |
|---|---:|---:|---:|
| Random forest | 0.712333 | 0.625496 | 0.755883 |
| XGBoost | 0.693973 | 0.613572 | 0.741900 |

Exact values and conditional paired intervals are in the archived summary.
The interpretation of the primary composition results concerns FLAIR and
task-fMRI regression decreases. The secondary positive list effect is retained
as a separate finding.
