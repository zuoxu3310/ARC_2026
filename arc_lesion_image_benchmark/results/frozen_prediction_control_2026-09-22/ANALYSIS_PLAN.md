# Frozen-prediction cohort control — analysis plan

Date: 2026-09-22. Status: specified before this new analysis was run.
This is a prospective plan for a secondary analysis of an extensively explored
dataset, **not** a registered confirmatory study or a new untouched test set.

## Scientific question

Can changing the evaluated patients change performance materially when the
learning procedure, fitted fold models, and individual predictions are fixed?
This supplies a direct control for the paper's central challenge: a higher score
across differently selected ARC cohorts does not by itself establish a better
method. It does not estimate what fraction of any particular published gain is
caused by cohort selection.

## Existing evidence to reorganize

Reanalyze the complete s18 grid (all models, 20 seeds). For each model, report
every cohort's mean score and difference from the full-pool reference. Report
the range over all primary cohorts, with the two secondary cohorts separately.
The original grid changes both training and evaluation patients. Same seed
numbers across cohorts do not mean the same folds or paired patient samples.
Do not use seed repetitions as independent patients for significance tests.

## New control

- Input: the aligned `data/jhu156_features_merged.tsv` and existing s18 subset
  ID files. Load the full pool in `full_ids.txt` order, exactly as s18 does.
- Features: reuse the s18 frozen dictionary (nonzero in >10% of 226 patients).
  This is a label-free, full-pool feature dictionary, not a new fold-local
  feature-selection claim. No new feature selection or tuning.
- Models: RandomForest and XGBoost, using the exact s18 factory configurations.
  They are two available model families for a control, not proxies for all
  published architectures. Both regression and classification are evaluated.
- Splits: s18 stratified five-fold CV, seeds 1000–1019. Scaling is fitted on each
  training fold. Classification uses training-fold SMOTE as in s18; severe is
  WAB-AQ <=50. No outcome-driven selection rule is learned.
- Save all held-out predictions and fold assignments. Every patient is predicted
  once per seed. Then freeze these arrays and apply every evaluation-cohort mask
  without fitting, tuning, resampling training data, or selecting predictions.
  This is a bank of fixed cross-fitted predictions, not one model fitted on all
  patients and evaluated in sample. Different test folds share training data;
  this is an internal evaluation rather than external validation.
- Reproduce full-pool s18 anchors before proceeding. First seed for each of the
  four task/model cells must agree in headline score within 1e-6. All subsequent
  full-pool scores are also reconciled to stored anchors within that tolerance.
  Stop and diagnose any mismatch; do not silently redefine the protocol.

## Cohorts and endpoints

Primary: full plus seven availability/chronicity criteria (`has_dwi`,
`has_rsfmri`, `has_taskfmri`, `has_flair`, `multimodal_complete`, `chronic_365`,
`chronic_180`). Secondary: `aq_le90` (outcome-defined restriction) and
`teghipco_idlist` (published patient list). The latter is not a replication of
that paper's pipeline; neither secondary cohort enters the primary range.

Primary endpoints: pooled held-out Pearson r per seed for regression and pooled
balanced accuracy per seed for classification, then the arithmetic mean across
20 seeds. Report full reference, selected-cohort score, and selected-minus-full
difference. Regression MAE is a secondary endpoint to check whether correlation
changes are accompanied by changes in absolute error. Do not average predictions
across seeds before calculating the primary metric.

## Fixed-prediction uncertainty and reference sampling

1. Paired patient bootstrap: 2,000 draws of 226 patient IDs with replacement;
   apply the same multiplicities to the full pool and each cohort, to both
   models, and to all 20 prediction repetitions. Percentile 95% intervals for
   mean score and selected-minus-full difference. Random seed 31092026.
   These intervals are **conditional on the saved predictions**, not confidence
   intervals for an entirely refitted learning pipeline. No independent-seed
   t-test; no claim that overlap between CV training folds disappears.
2. Size-only reference: for each non-full cohort, sample 2,000 random sets of
   the same N without replacement from the 226 patients. Evaluate the identical
   saved predictions. Report the central 95% reference interval and observed
   minus reference mean. This is a finite-pool sampling comparison, not evidence
   that inclusion was randomly assigned. Tail rank: twice the smaller one-sided
   tail, plus-one corrected and capped at one. Holm adjustment across the seven
   primary criteria separately for each task/model. Secondary ranks are separate.
3. Exploratory composition-matched reference: use the same N and exact counts
   in AQ bins (-0.1,25], (25,50], (50,70], (70,90], (90,100.1]. It preserves the
   severe fraction and coarse severity distribution, not exact AQ variance.
   Use 2,000 draws, report the same descriptive intervals and tail ranks. Do not
   interpret an interval containing the observed score as proof of equivalence.
4. Sampling random seeds are 31093026+i for size-only and 31094026+i for AQ-bin
   matching, where i is the cohort's zero-based row in the saved manifest.

## Interpretation fixed before results

- A substantial fixed-prediction shift demonstrates that evaluation composition
  alone is sufficient to move the score in these data; no algorithm update is
  involved. Show all seven rules, not only the largest shift.
- Small shifts do not erase the existing train-plus-test result. They suggest
  that training composition, sample size, or their interaction deserves further
  study; subtracting the two designs is descriptive, not a causal decomposition.
- Size-only reference sampling asks whether a rule selects unusually scored
  patients relative to random sets of the same size. A broad interval or large
  tail rank is not proof that a rule has no effect.
- Published scores remain scale/context references; unavailable deep models,
  simulation pipelines, and their incremental value are not directly tested.
- No new architecture sweep, no fabricated external holdout, no publication.

## Deliverables and verification

Preserve old artifacts. Save code/input/protocol hashes, environment versions,
patient predictions, exact train/test IDs, fold preprocessing checks, progress
and watchdog records, resampling distributions, complete summaries, and editable
SVG/PDF/PNG figures in this new result directory. Independently recompute headline
metrics from patient predictions and inspect train/test separation, matching,
aggregation and primary/secondary cohort scopes. Acceptance is by an agent other
than the executor, followed by source verification of any finding. This run does
not retroactively claim formal research-workflow P1 registration or gate passage.
