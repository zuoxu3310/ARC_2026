# ARC_2026

Code for *Higher Scores, Unchanged Predictions: Testing Added Value in Aphasia
Severity Prediction*. It reproduces the analyses on the Aphasia Recovery Cohort
(ARC) and provides a reusable reference benchmark and a controlled protocol for
comparing a new prediction method with a reference.

The tasks are WAB-AQ regression and severe-aphasia classification
(WAB-AQ <= 50) from lesion-atlas features, with Random Forest, XGBoost,
LightGBM, linear, kernel, MLP and TabPFN models.

## Contents

| Path | Purpose |
|---|---|
| `reproduce.py` | Stage-by-stage reproduction of the paper's ARC and MRRI analyses |
| `benchmark.py` | Reference benchmark: query, prepare, train, verify and compare against recorded baselines |
| `scripts/` | Public-data download and feature construction, manuscript tables, model-set sensitivity |
| `arc_lesion_image_benchmark/src/` | Analysis scripts (`s1`-`s37`), comparison engines, figure scripts and unit tests |
| `arc_lesion_image_benchmark/*.md` | Documentation of the reference benchmark, comparison protocol and controlled runs |
| `arc_lesion_image_benchmark/examples/`, `examples/` | Example selection policies and an example candidate method |
| `arc_wab_aq_benchmark/` | Leave-one-out regression harness reused by the earlier atlas analyses |
| `outputs/cnn-same-input-2026-10-03/`, `outputs/cnn-pilot-2026-10-03/` | Same-input image comparison (CNN and classical models) |
| `outputs/xgboost-optuna-2026-10-04/` | Exploratory Optuna-tuned XGBoost extension of the image comparison |
| `outputs/supplementary-controls/` | Training-only feature-selection control, matched image development controls, and a data-free recomputation script |
| `IMAGE_CASE_README.txt` | Inputs and commands for the image-input case |
| `analysis/training_aware_intervals.py` | Paired intervals that add retraining variation across repeated partitions (appendix table; no refitting) |
| `figures/figure1/` | Editable draw.io source of Figure 1, its generator, icon stencils and fonts |
| `bibliography.bib` | References |

The directory names under `outputs/` are used as fixed paths by the scripts; keep
the directory structure unchanged.

## Requirements

Python 3.11.8 was used for all recorded results.

```bash
pip install -r requirements.txt            # core analyses, figures and tables
pip install -r requirements-optional.txt   # adds TabPFN, Optuna and SHAP
```

TabPFN model weights are obtained from their provider under its access
conditions. The image-input case additionally needs PyTorch with Apple Metal (MPS)
support for CNN training; see `IMAGE_CASE_README.txt`.

## Data

No medical images, clinical tables, feature matrices or model weights are
included. `python reproduce.py prepare` downloads the public inputs at fixed
versions and checks each file against a recorded checksum:

- ARC lesion masks (228 MNI-normalised masks), the JHU and arterial atlases and
  the demo participant table from the ARC demo repository
  (`neurolabusc/AphasiaRecoveryCohortDemo`, commit `1d24ad4`);
- `participants.tsv` and the modality listing of OpenNeuro `ds004884` v1.0.2;
- the public image subset `DLAphasiaSeverityARCSubset.mat`
  (figshare, doi:10.6084/m9.figshare.23579943.v1);
- with `--extra-atlases`: AAL, Brodmann and AICHA atlases from NeuroParc.

`python reproduce.py mrri` downloads the Walker et al. (2018) behavioural
workbook used for the MRRI extension (`scripts/prepare_mrri.py`). The image-input
case also needs the authors' `vgg2.py` from `alexteghipco/volDNN`
(see `IMAGE_CASE_README.txt`). Use these data under their providers' terms.

Two result archives accompany the paper as supplementary files: the reference
results (150 reference entries with per-patient predictions and split records)
and the supplementary results (aggregate tables and the predictions of the
supplementary controls). They are distributed separately from this code.

## Reproducing the paper

```bash
python reproduce.py prepare        # public inputs, features, 10 cohort definitions
python reproduce.py grid           # 20-repeat crossed cohort x model grid (add --tabpfn for the full grid)
python reproduce.py fixed          # full-pool fixed-prediction control and matched sampling
python reproduce.py comparisons    # comparison populations and fixed-test training controls
python reproduce.py mrri           # independent behavioural-cohort extension
python reproduce.py statistics     # variance decomposition and model-set sensitivity
python reproduce.py figures        # manuscript figures and tables
python reproduce.py protocol-demo  # worked ARC/MRRI comparisons from completed prediction banks
```

TabPFN runs in separate processes. Long stages print per-fold progress.

| Paper section | Main code |
|---|---|
| Patient pool, tasks and eligibility rules; fixed feature dictionary | `scripts/prepare_data.py`, `s1_verify_masks.py`, `s2_extract_atlas_features.py`, `s17_freeze_pool_subsets.py` |
| Crossed cohort-model grid | `s18_crossed_grid.py`, `s19_variance_decomp.py`, `s19b_*`, `s19c_*` |
| Fixed-prediction control; size-only and severity-matched sampling | `s31_frozen_prediction_control.py`, `s31_resampling_tie_fix.py`, `s20_matchedN_permutation.py`, `s21_range_restriction.py` |
| Comparison populations and fixed-test training controls | `s32_comparison_population.py`, `s33_fixed_test_training.py` |
| Independent behavioural-cohort extension (MRRI) | `scripts/prepare_mrri.py`, `s34_mrri_replication.py` |
| Controlled comparison protocol | `s35_compare_predictions.py`, `s36_protocol_examples.py`, `arc_comparison_run.py` |
| Same-input image-based comparison | `outputs/cnn-same-input-2026-10-03/`, `outputs/xgboost-optuna-2026-10-04/` |
| Versioned reference predictions and reuse | `benchmark.py`, `arc_reference.py`, `s37_reference_benchmark.py` |
| Appendix: training-only feature selection | `outputs/supplementary-controls/scripts/feature_sensitivity.py`, `derive_feature_checks.py` |
| Appendix: aligned fitting, validation and decision policies | `outputs/supplementary-controls/scripts/image_controls.py`, `paper_assets.py` |
| Appendix: complete model benchmarks; model-set sensitivity | `scripts/build_benchmark_tables.py`, `scripts/model_set_sensitivity.py` |
| Appendix: additional composition analyses | `s28_cohort_explainers.py`, `s29_difficulty_composition.py`, `s30_mechanism_partition.py` |
| Appendix: atlas representations, paired atlas and tuning comparisons | `s22_granularity_spectrum.py`, `s23_resample_atlases.py`, `make_floor_5x10_summary.py`, `s3*`, `s5_*`, `s6_*`, `s10_*`, `s27_*`, `teghipco_same_subset.py`, `arc_wab_aq_benchmark/` |
| Appendix: calibration, subgroups and prediction errors | `s7_calibration_rf.py`, `s7_subgroup_error.py`, `s8_xai_region_contribution.py`, `s9_chronicity_sensitivity.py` |
| Appendix: syndrome classification | `s11_*` to `s16_*` |
| Appendix: verification and reproducibility | `s26_leakage_probe.py`, unit tests |
| Figures and tables | `fig_*.py`, `nature_style.py`, `scripts/build_manuscript_tables.py` |

Scripts without a path prefix are in `arc_lesion_image_benchmark/src/`.

## Using the reference benchmark

Unpack the reference-results archive next to this directory, then:

```bash
python benchmark.py query --catalog reference-results/catalog --cohort has_flair --task regression
python benchmark.py verify --bank reference-results/banks/has_flair/regression
```

To build baselines for your own patient list from the public 226-patient pool:

```bash
python reproduce.py prepare
python benchmark.py prepare --ids my-patients.txt --task regression --models RandomForest XGBoost --output my-reference
python benchmark.py train --bank my-reference --workers 2
```

See [REFERENCE_BENCHMARK.md](arc_lesion_image_benchmark/REFERENCE_BENCHMARK.md)
for reuse of saved references, custom folds, prediction formats and paired
comparison, [COMPARISON_PROTOCOL.md](arc_lesion_image_benchmark/COMPARISON_PROTOCOL.md)
for separating scoring-population, training-population and method changes, and
[CONTROLLED_RUN.md](arc_lesion_image_benchmark/CONTROLLED_RUN.md) for the
integrated fitting-and-comparison workflow.

## Supplementary controls

`outputs/supplementary-controls/ANALYSIS_PLAN.md` describes the training-only
feature-selection control and the matched image development controls. After
unpacking the supplementary-results archive, the reported scores and intervals
can be recomputed from the exported predictions, without images or fitted models:

```bash
cd supplementary-controls
python recompute_exported_scores.py --root evidence
```

## Tests

```bash
python -m unittest discover -s arc_lesion_image_benchmark/src -p 'test_*.py'
```

The 46 tests use synthetic inputs and need no downloaded data.

## Notes on recorded file hashes

Result manifests record SHA-256 hashes of the source files used to produce them.

- `arc_lesion_image_benchmark/src/s18_crossed_grid.py` defines the frozen
  estimators of the reference benchmark. `benchmark.py` checks its hash against
  every reference bank, so the file is kept byte-for-byte as recorded, including
  its original non-English comments. Its purpose: run one five-fold stratified
  cross-validation per (cohort, model, seed) cell with the fixed JHU feature
  pipeline (training-fold scaling, SMOTE for classification, default
  hyperparameters), using a sparse-column filter frozen once on the full
  226-patient pool.
- Comments and console messages of some scripts were edited after the runs;
  code statements are unchanged, so their hashes differ from those recorded in
  earlier run manifests.

## Package information

- Platform: results were produced on macOS 15 (Apple Silicon, CPU; MPS for the CNN case)
  with Python 3.11.8. The analysis code is platform independent apart from the optional
  MPS-based CNN training.
- Contact: Dmitry Ignatov, HSE University (dignatov@hse.ru).

## License

Code: MIT (see [LICENSE](LICENSE)). Data and third-party code remain under
their providers' terms.
