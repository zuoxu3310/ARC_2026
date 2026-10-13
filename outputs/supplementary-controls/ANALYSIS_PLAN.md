# Supplementary analysis plan

This document describes two supplementary analyses: (A) a training-only feature
dictionary control and (B) matched development controls for the image comparison.

**Timing.** Both analyses were specified after the original results were known.
They are pre-specified only with respect to their own new fits: each design was
frozen in a JSON plan before any new model was fitted. They are not a
preregistration of the original study, and the original comparisons remain part
of the evidence.

| Analysis | Frozen plan | SHA-256 |
|---|---|---|
| A. Training-only feature dictionary | `evidence/training-only-features/frozen.json` | `aa6c05e88777bc3b5d7e9c03f64275cb57f2130a992ee2ff7dd0b8d30c25b272` |
| B. Image development controls | `evidence/image-controls/image-plan.json` | `844d4acab65f6cc604d1719294ad8bdf748f320e08c784a323323e9db9cb21a1` |

The frozen plans also record SHA-256 hashes of the source files, inputs and
original partitions that were used. Those records keep the absolute file paths of
the machine on which the analyses were run.

## A. Training-only feature dictionary

**Question.** Do the population-control and method-comparison results depend on
the feature dictionary having been screened once on the full 226-patient pool?

**Design.**

- Re-run the original Random Forest / XGBoost population-control experiment with
  the nonzero-prevalence feature screen fitted only on the training patients of
  each fit.
- Start from all 191 raw feature columns and keep the columns that are nonzero in
  more than 10% of the current fitting patients. Scaling and, for classification,
  SMOTE are fitted on the same patients.
- Keep the original model settings, the original five-fold memberships, seeds
  1000-1019, the full cohort and all nine recorded eligibility restrictions.
  Every model trained on a restricted cohort predicts the entire original test
  fold.
- Scale: 10 cohorts x 2 tasks x 2 models x 20 seeds = 800 five-fold units, i.e.
  4000 new fits.
- Before the full run, four first-repeat units refitted with the original frozen
  dictionary (20 fits) must reproduce the historical predictions.

**Reporting.**

- All scoring/training four-cell contrasts for both pipelines, paired
  RF-minus-XGBoost differences, and differences between the frozen-dictionary and
  training-only pipelines.
- Intervals reuse the original 2000 shared patient-bootstrap weight vectors; each
  draw keeps a patient's weight across all repeats and methods.
- All 20 repeat scores are retained to show dependence on the recorded partitions.
- The primary checks are the fixed-prediction scoring effects and the method
  contrasts under matched conditions. No single favourable cohort is selected as
  the robustness result.

## B. Image comparison: matched development controls

**Question.** On the public 173-patient image release, how large is the remaining
CNN advantage once the classical models use the same fitting and validation
patients, the same class-cost ratio and a validation-selected decision threshold?

**Design.**

- Use the 120 original CNN fitting/validation/test partitions (20 repeats x 6
  outer folds), the 173 patients and all 7106 image values per patient.
- Reload all 120 saved CNN checkpoints and recover validation and test
  probabilities. The CNN is not retrained.
- Fit classical models only on the CNN's fitting patients with the same image
  normalisation, use only the CNN's validation patients for any decision, and keep
  the original test patients.
- Variants: CNN; Random Forest and XGBoost with inverse-frequency class weights;
  Random Forest and XGBoost with the CNN's positive-to-negative cost ratio (both
  weightings normalised to a mean sample weight of 2); and a developed XGBoost that
  selects one of 25 configurations under the CNN cost ratio by validation F1 at
  threshold 0.5 (ties keep the original configuration).
- Decision rules: a fixed 0.5 threshold, and a threshold chosen by validation F1
  on a 0.10-0.90 grid (ties: nearest to 0.5, then the lower threshold).
- The original study already reported Random Forest and SVM fitted on the smaller
  CNN fitting set; those remain historical results. The new parts are the XGBoost
  variants and the validation and decision-policy controls.
- Scale: 3360 new classical-model fits.

**Reporting.**

- Every planned variant under both decision rules, with F1, balanced accuracy,
  AUC, precision, recall and accuracy.
- Primary contrasts: CNN minus each XGBoost variant under the corresponding
  threshold rule, and developed XGBoost minus XGBoost with the CNN cost ratio.
- Paired intervals use 5000 stratified patient-cluster bootstrap draws (seed
  2026100399). A sampled patient keeps the same weight across the 20 repeats and
  all methods; fitted models are held fixed.

## Interpretation rules (both analyses)

- Intervals are conditional on the saved fitted models and the recorded
  partitions.
- Overlapping intervals, or an interval that crosses zero, are not treated as
  evidence of equivalence.

## Execution records

- New outputs are stored separately from the historical runs.
- Source files, inputs and memberships were hashed before each experiment.
- Predictions and fitted-preprocessing records are saved for every unit, and the
  anchor checks passed before the full runs (`ANCHORS_PASS.json`).
- `recompute_exported_scores.py` recomputes the reported metrics and intervals
  from the exported predictions without raw images or fitted models.
- `frozen.json` lists the plan text under its original file name `PLAN.md`
  (SHA-256 `6bb4419ee778d5fcb7a11da4d08e45c7e2bac364c40976aeb569f3d59701a047`).
  This document restates that plan for readers; it is not the hashed file.
