# Run a complete controlled comparison

`src/arc_comparison_run.py` connects a full-pool reference bank to the population
comparison protocol. A single outer-fold assignment defines all three learning
procedures. The run produces the predictions, membership records, model-selection
audit and comparison report together.

## Start from one outer design

Use the `full` bank for the required task, or prepare a new full-pool bank with
your own folds. Preparing records patients and folds; the new workflow handles
training. Custom folds contain `patient_id,repeat,fold` for every full-pool
patient in every repeat.

```bash
python arc_lesion_image_benchmark/src/s37_reference_benchmark.py prepare --cohort full --task classification --models RandomForest XGBoost --folds my-full-pool-folds.csv --output my-pool
python arc_lesion_image_benchmark/src/arc_comparison_run.py prepare --pool-bank my-pool --eligible-ids my-eligible-patients.txt --policy arc_lesion_image_benchmark/examples/control_nested_f1.json --output my-controlled-run
python arc_lesion_image_benchmark/src/arc_comparison_run.py run --run my-controlled-run --threads 2
```

The eligible IDs must be drawn from the declared 226-patient ARC pool. The full
reference uses each original training fold. The restricted reference and the
built-in candidate use its intersection with the eligible IDs. Both references
predict every patient in the original test fold; the candidate predicts eligible
test patients. This supplies all four population-control cells and the paired
method comparison automatically.

For existing results, substitute the task's `reference-results/banks/full/...`
directory for `my-pool`. Its recorded features can be restored from the verified
public feature table; the original result bank is read only. `--repeats 1000 1001`
selects existing repeat names for a small implementation check. A within-subset
bank is rejected as the source of the full-pool controls.

The stored 150-entry bank uses partitions generated separately within each
cohort. Its `benchmark.py compare` entry remains useful for a direct paired
comparison on that exact cohort and design. The controlled-run entry instead
starts from the full-pool design and derives every intersection from it. Run IDs
record which design, features, policies and software produced the result.

## Declare how each method is selected

The policy is saved before fitting. It declares one target metric, inner-fold
count, feature policy and the reference/candidate learning procedures. Built-in
procedures support RandomForest and XGBoost, with either a single fixed setting
or a declared list of parameter candidates. Each candidate dictionary overrides
the existing recorded estimator defaults.

The supplied classification example selects severe-class F1; the regression
example selects Pearson correlation. Other supported objectives are balanced
accuracy, AUC, MCC, MAE, RMSE and R-squared, according to task. Each parameter
candidate receives the same inner folds. Parameters and decision thresholds
are chosen from complete inner held-out predictions; the selected procedure is
then fitted on the outer training patients. Ties follow declaration order.
The outer test outcomes never select a setting.

For a fixed policy set `mode` to `fixed`, `parameter_grid` to a list containing
one dictionary, and `thresholds` to a one-element list. `"estimator"` preserves
the estimator's own decision rule. Numerical classification thresholds use
`score >= threshold`. The nested examples declare three parameter settings and
three inner folds for both built-in models, and four threshold options for
classification. The report records this budget explicitly.

`frozen_dictionary` uses the published 71-input reference track.
`training_nonzero` starts from the original feature columns and fits the
greater-than-10-percent nonzero filter inside each fitting set, including every
inner fold. Scaling and classification resampling are also fitted there. The
feature policy is part of the run identity; outputs from different policies
remain separate.

## Supply predictions from another method

Replace the candidate policy with a declared external procedure:

```json
{
  "mode": "external",
  "description": "Describe the actual inputs, preprocessing, search budget, objective and selection rule used to create these predictions."
}
```

Provide `--candidate-predictions predictions.csv --candidate-folds folds.json`
to `prepare`. The prediction format is `patient_id,repeat,fold,prediction`, plus
an optional `score` column and an optional checked `y_true` column. Predictions
cover exactly the eligible patients in each source-bank repeat. Every supplied
fold record contains the original training-fold intersection with eligibility
and the eligible original test patients, including records for empty eligible
test intersections. All planned outer-fold records are supplied.

External predictions and memberships are validated before reference training.
A provided score column must be numeric and finite in every row. AUC requires
scores; missing scores are reported explicitly. Duplicate or missing patients,
changed outcomes, different outer folds and incorrect training memberships are
rejected. Candidate preprocessing and selection are documented by its producer.

## Inspect the result

The run retains:

- `run.json`, `policy.json`, `assignments.csv`, `planned-folds.json`: frozen inputs,
  source/environment hashes and intended memberships.
- `units/*/selection.json` and `inner-predictions.csv`: all evaluated settings,
  inner memberships, scores and the selected configuration/threshold.
- `units/*/result.json`: actual feature columns, scaler fitting counts,
  diagnostics, elapsed time and prediction hashes.
- `comparison-inputs/`: ready-to-use inputs for `s35_compare_predictions.py`.
- `comparison/all-metric-scores.csv` and `all-metric-contrasts.csv`: all available
  metrics for the four cells, candidate and comparison paths. A constant
  regression predictor retains its error measures and reports correlation as
  undefined. Undefined bootstrap draws are counted.
- `comparison/path-verification.json`: checks the telescoping score path on
  shared patient draws. Each interval is computed from its own contrast on those
  shared draws; endpoints are not added.
- `COMPLETE.json`: completed fit count and output hashes.

Repeat `run` to resume verified completed units. Input, implementation or
environment changes require a new run directory. Saved unit changes are rejected.
Long runs emit per-fold progress; retain those logs and set an execution time
limit in the process scheduler. `progress.json` counts only fitted procedures,
so imported candidates do not leave the training progress incomplete.

The comparison reports percentile patient-bootstrap intervals conditional on
the recorded fitted procedures and outer folds. It also preserves training/scoring
interaction and both orders of the population contrasts. The new runs add a
training and audit route to the fixed reference bank; historical scores remain
available under their original version identifiers.
