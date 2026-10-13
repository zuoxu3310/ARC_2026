# ARC reference benchmark

The reference consists of a score catalogue, explicit evaluation assignments,
recorded baseline predictions, and commands to train or compare a new method.
The input track uses the original 71 lesion-burden, lesion-volume and age features.
MRI availability defines the reference populations; those modality images are not
additional inputs to this feature track.

The manuscript catalogue has 10 population conditions, eight regression learners
and seven classification learners: 150 entries, each evaluated with 20 repeats of
five-fold cross-validation. Two secondary populations (AQ <= 90 and the published
participant-list intersection) are labelled separately. Explicit availability
rules identify reproducible ARC reference populations, not unpublished historical
patient lists from every study.

## Find a baseline

Unpack the separate reference-results archive and query its `catalog` directory:

```bash
python benchmark.py query --catalog reference-results/catalog --cohort has_flair --task regression
```

Entries identify the population, task, fixed model configuration, primary mean,
repeat-to-repeat standard deviation and secondary metrics. The standard deviation
describes split variation; it is not a confidence interval. The catalogue preserves
the paper's historical scores. Newly recorded predictions have their own scores
and an explicit historical-score reconciliation; paired comparisons use the actual
recorded predictions, never an unrelated historical average.

## Reuse a reference population

Install `requirements.txt` (recorded runtime: Python 3.11.8). Add `requirements-optional.txt` for TabPFN.
Run `python reproduce.py prepare` to obtain the versioned public inputs and features.
If using the supplied completed reference-results bank, restore only its input
features and train your candidate; the recorded baselines need no retraining:

```bash
python benchmark.py hydrate --bank reference-results/banks/has_flair/regression
python examples/reference_candidate.py --bank reference-results/banks/has_flair/regression --output my-method
python benchmark.py compare --bank reference-results/banks/has_flair/regression --model RandomForest --predictions my-method/predictions.csv --folds my-method/folds.json --output my-comparison
```

The result archive omits the feature matrix, raw images and clinical source tables.
Querying, verifying or comparing already supplied predictions does not require
restoring features. `hydrate` checks that regenerated features match the exact
input checksum. To construct and train a new bank instead:

```bash
python benchmark.py prepare --cohort has_flair --task regression --models RandomForest XGBoost --output my-reference
python benchmark.py train --bank my-reference --workers 2
python benchmark.py verify --bank my-reference
```

Preparation defaults to 20 repeats and seeds 1000--1019. Omitting `--models` includes
the complete model panel; `train` automatically runs the conventional models first
and launches TabPFN in a separate process. TabPFN can also be requested alone:

```bash
python benchmark.py train --bank my-reference --models TabPFN --threads 4
```

A completed cell resumes only after checking its identity, prediction
hashes, patients, outcomes, assignments and recomputed metrics. A failed cell does
not overwrite a historical result. Each completed fold logs progress.
Training inside an existing bank requires its recorded dependency versions;
preparing a new bank records the environment used for that run.

The standard cohort names are `full`, `has_dwi`, `has_rsfmri`, `has_taskfmri`,
`has_flair`, `multimodal_complete`, `chronic_365`, `chronic_180`, `aq_le90`, and
`teghipco_idlist`. Exact identities are generated from the public inputs.

## Use your own ARC patients or outer folds

Supply a text file with one exact ARC patient ID per line, or a CSV with a
`patient_id` column. Select from this release's public 226-patient feature pool.
Unknown and duplicate IDs are errors.

```bash
python benchmark.py prepare --ids my-patients.txt --task classification --models RandomForest XGBoost --output my-reference
python benchmark.py train --bank my-reference --workers 2
```

To supply your own splits, add `--folds my-folds.csv`. The file has
`patient_id,repeat,fold`, with exactly one test-fold assignment for every patient
in every repeat. Training membership is the complement of each test fold within
the declared population. Repeat names map to recorded model seeds. The same
feature dictionary, preprocessing and outcome definition are retained.

Custom folds need at least two test folds, nonempty training sets, and both classes
in every classification training set. Automatically generated folds require at
least five patients per stratification stratum. Missing rows are never dropped
silently. A changed patient list or split design creates a different benchmark ID.

## Compare your method

`examples/reference_candidate.py` trains a real, fixed example model using exactly
the bank's training and test patients. Replace `make_model()` with your method:

```bash
python examples/reference_candidate.py --bank my-reference --output my-method
python benchmark.py compare --bank my-reference --model RandomForest --predictions my-method/predictions.csv --folds my-method/folds.json --output my-comparison
```

Candidate predictions need `patient_id,repeat,fold,prediction`. For classification,
`prediction` contains binary decisions and optional `score` holds probabilities or
decision scores for AUC. Optional `y_true` is checked against the reference.
`folds.json` records each repeat/fold's exact `train_ids` and `test_ids` arrays.
Reordered prediction rows are aligned by ID; changed coverage, repeated patients,
different outcomes, mismatched folds or different training memberships are rejected.

`comparison.csv` reports reference and candidate means, candidate-minus-reference
differences and paired patient-bootstrap intervals. Patient weights are shared
across both methods and all repeats. Metrics are calculated from each complete
repeat's held-out predictions, then averaged. Intervals are conditional on saved
fits and folds, pointwise and unadjusted; they do not establish equivalence.
`scores_by_repeat.csv` and `report.json` provide the underlying scores and checks.

Keep candidate feature construction, preprocessing and any tuning inside the
training folds. Recorded membership checks cannot inspect an unseen implementation.
Use identical inputs to isolate an algorithm change; explicitly declare additional
inputs when evaluating an entire method. The baseline's input track remains clear
in either comparison.

## Artifacts and versions

A prepared bank contains `benchmark.json`, `patients.csv`, `features.csv` and
`assignments.csv`. Training adds model/repeat directories with `predictions.csv`,
`folds.json` and `result.json`. Verification creates `summary.csv`,
`scores_by_repeat.csv` and `verification.json`. Incomplete cells remain visible.
`export` writes one model in the candidate-compatible prediction/fold format.

The source release contains code and documentation only. Public data are downloaded
from their providers; the catalogue and generated reference predictions are separate
results. Reference results contain patient identifiers, targets, prediction and split
records needed for exact paired comparison, but not raw images or input feature tables.
TabPFN weights are obtained from its provider, not redistributed with the
code. TabPFN also requires the provider's model-access conditions to be satisfied.

For the broader three-step scoring/training/method analysis, see
[COMPARISON_PROTOCOL.md](COMPARISON_PROTOCOL.md). This benchmark's direct comparison
uses common patients and folds; it does not substitute population changes for a
measured method difference.

The integrated [controlled-run workflow](CONTROLLED_RUN.md) constructs the full
analysis from one full-pool bank: it retains the original outer folds, derives
eligible training intersections, fits both references, and produces predictions
for every original test patient. It also supports declared inner-fold parameter
and threshold selection and saves the complete selection record. This provides
an executable route from the fixed reference bank to a newly developed reference
under the candidate's actual comparison conditions.
