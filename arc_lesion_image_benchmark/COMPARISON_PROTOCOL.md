# Controlled comparison of predictions

Use `src/s35_compare_predictions.py` to separate scoring-population changes,
training-population changes, and a candidate method's difference from a reference.
It reads recorded out-of-fold predictions; it does not train models. Any model
family can supply predictions in this format.

To generate these inputs through one recorded full-pool fold design, use the
[controlled-run workflow](CONTROLLED_RUN.md). It fits the full and eligible
references, predicts all original test patients, and supplies the candidate's
eligible predictions. Fixed and nested-selection policies are recorded before
fitting. External candidate files are checked before fitting the references.

```bash
python arc_lesion_image_benchmark/src/s35_compare_predictions.py --spec comparison.json --output my-comparison
```

Choose the outcome, eligibility rule and outer folds before comparing methods.
For an added imaging input, compare both methods on patients who have that input,
and report their fraction of the declared analysis pool. Keep preprocessing and tuning
inside the training folds. Hold features fixed when testing an algorithm change;
allow the added inputs when testing the entire proposed method.

## Input contract (schema version 1)

Paths are relative to the JSON file. IDs are exact strings, including leading
zeros. All three tables describe the same outcome and repeated outer folds.

```json
{
  "schema_version": 1,
  "task": "regression",
  "outcome": "WAB-AQ at the recorded assessment",
  "protocol": "Shared outer folds; specify preprocessing, tuning and feature conditions here.",
  "patients": "patients.csv",
  "predictions": "predictions.csv",
  "folds": "folds.json",
  "baseline": "reference_full",
  "restricted_baseline": "reference_eligible",
  "candidate": "new_method",
  "candidate_reference": "restricted_baseline",
  "bootstrap": {"draws": 2000, "seed": 20261003}
}
```

- `patients.csv`: `patient_id,y_true,eligible`. One row per analysis patient;
  `eligible` is 0 or 1. This table defines the coverage denominator.
- `predictions.csv`: `patient_id,repeat,fold,model,y_true,prediction`. Exactly one
  prediction per patient/model/repeat. `baseline` and `restricted_baseline`
  cover the entire analysis pool. `candidate` covers exactly the eligible
  patients; filter any additional predictions explicitly before exporting.
- `folds.json`: an array of records with `model`, `repeat`, `fold`, `train_ids`
  and `test_ids`. The latter two are arrays of string IDs from the recorded fit.
  Include each original outer fold even if its candidate test subset is empty.

The baseline's test folds must partition the complete pool once per repeat, with
at least two folds. Its training set is the complement of each test fold.
The restricted baseline trains on that complement intersected with eligibility,
while retaining the full original test fold. The candidate must use the same
training IDs as its declared reference and score the eligible part of each
original test fold. Empty training sets are rejected. Repeats, identities,
outcomes, duplicate rows, finite predictions and these membership relations
are checked exactly; no patient is silently dropped or matched by row position.

`restricted_baseline` and `candidate` are optional. Without a candidate, the
output is explicitly a population-control report, with no method comparison.
If comparing methods trained on the full pool, use `candidate_reference` equal
to `baseline`; the candidate must then have the same full training memberships.

For classification use `task: "classification"`; both truth and prediction must
be 0/1 labels with the positive class defined in `outcome`. Supply labels from
your prespecified decision rule, not probabilities. Both outcome classes must
occur in the full and eligible pools. The regression metrics are Pearson r and
mean absolute error (MAE); classification uses balanced accuracy.

## Outputs and interpretation

`scores.csv` contains the following cells and their paired patient-bootstrap
intervals. Scores are computed separately per repeat and then averaged.

| Cell | Training patients | Scoring patients |
|---|---|---|
| S00 | Full original training fold | Full pool |
| S01 | Full original training fold | Eligible subset |
| S10 | Eligible original training patients | Full pool |
| S11 | Eligible original training patients | Eligible subset |
| candidate | Same as the declared reference | Eligible subset |

`contrasts.csv` reports S01−S00 (scoring), S10−S00 and S11−S01
(training under each scoring population), S11−S10 (scoring after restriction),
S11−S00 (joint), and S11−S10−S01+S00 (interaction). The method difference is
candidate−S11, or candidate−S01 when the reference uses full training.
Positive r or balanced-accuracy differences favor the candidate; negative MAE
differences favor it. Balanced accuracy is a proportion: multiply differences
and interval limits by 100 for percentage points.

`scores_by_repeat.csv` retains the repeated scores. `report.json` records input
hashes, coverage and the checks performed; `comparison.txt` is a brief readable
report. Outputs contain aggregate statistics, not patient IDs or predictions.
Use a separate output directory for each analysis; a failed validation does not
replace an earlier successful report.

Bootstrap draws resample patients in the full pool and reuse the same counts
across models, scoring sets and repeats. To reuse existing draws, replace the
bootstrap object with `{"file": "bootstrap.npz"}`. This NumPy archive must have
only `weights` (draws × patients, nonnegative integer counts summing to full N)
and `patient_ids` (the corresponding string IDs). Column alignment uses IDs.
At least 100 draws are required. The default is 2,000.

Intervals are pointwise percentile 95% intervals conditional on the supplied
fits and splits. No bootstrap refitting or multiplicity adjustment is performed.
Draws missing an outcome class, or having undefined correlation, are counted as
invalid for the affected result; their count is reported. A constant prediction
has undefined Pearson r, not zero correlation. An interval crossing zero does
not establish equivalence. Eligibility coverage describes the supplied analysis
pool, not an independently established clinical target population.

Recorded membership checks cannot prove that preprocessing/tuning excluded test
data or that supplied logs are authentic. Review the original training pipeline
for those properties. These contrasts apply to the specified experiment; they
cannot correct another paper's aggregate score without comparable predictions.

## Reproduce the manuscript example

After the ARC and MRRI analysis stages have produced their prediction banks:

```bash
python arc_lesion_image_benchmark/src/s36_protocol_examples.py
python -m unittest discover -s arc_lesion_image_benchmark/src -p test_comparison_protocol.py
```

The examples use ARC FLAIR availability and MRRI nonword-repetition availability,
with RF as reference and XGBoost as comparator, for both prediction tasks.
The exporter preserves training records and selects the declared candidate
scoring patients. It produces local inputs plus aggregate results without
fitting a new model. Do not publish the generated patient-level input directory
as part of the code release. MRRI is a behavioral evaluation replication, not
external validation of the ARC lesion model.
