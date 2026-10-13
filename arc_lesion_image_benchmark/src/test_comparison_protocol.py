"""Numerical and malformed-input tests for the reusable comparison protocol."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

import s35_compare_predictions as tool


def fixture(root, classification=False):
    ids = [f"{i:03d}" for i in range(12)]
    y = np.array([2, 14, 23, 35, 46, 52, 60, 66, 75, 84, 91, 98], float)
    eligible = np.array([i not in {0, 3, 8, 11} for i in range(12)])
    if classification:
        y = (y <= 50).astype(float)
    pd.DataFrame({"patient_id": ids, "y_true": y, "eligible": eligible.astype(int)}).to_csv(root / "patients.csv", index=False)
    predictions, records = [], []
    for model in ["A_full", "A_restricted", "B"]:
        for repeat in ["0", "1"]:
            assignment = (np.arange(len(ids)) + int(repeat)) % 3
            for fold in range(3):
                test = [p for i, p in enumerate(ids) if assignment[i] == fold]
                train = [p for i, p in enumerate(ids) if assignment[i] != fold and (model == "A_full" or eligible[i])]
                if model == "B":
                    test = [p for p in test if eligible[ids.index(p)]]
                records.append({"model": model, "repeat": repeat, "fold": str(fold), "train_ids": train, "test_ids": test})
            for i, patient in enumerate(ids):
                if model == "B" and not eligible[i]:
                    continue
                if classification:
                    flip = (i + int(repeat) + ["A_full", "A_restricted", "B"].index(model)) % 5 == 0
                    prediction = 1 - y[i] if flip else y[i]
                else:
                    prediction = y[i] * .91 + np.sin(i + int(repeat)) * (7 if model == "A_full" else 11 if model == "A_restricted" else 5)
                predictions.append({"patient_id": patient, "repeat": repeat, "fold": str(assignment[i]),
                                    "model": model, "y_true": y[i], "prediction": prediction})
    pd.DataFrame(predictions).to_csv(root / "predictions.csv", index=False)
    (root / "folds.json").write_text(json.dumps(records))
    # Shared patient counts in intentionally reversed column order.
    w = np.random.default_rng(44).multinomial(12, [1 / 12] * 12, size=120)
    np.savez_compressed(root / "boot.npz", weights=w[:, ::-1], patient_ids=np.asarray(ids[::-1]))
    spec = {"schema_version": 1, "task": "classification" if classification else "regression",
            "outcome": "Synthetic binary outcome" if classification else "Synthetic continuous outcome",
            "protocol": "Analytical fixture for score arithmetic; no model trained.",
            "patients": "patients.csv", "predictions": "predictions.csv", "folds": "folds.json",
            "baseline": "A_full", "restricted_baseline": "A_restricted", "candidate": "B",
            "candidate_reference": "restricted_baseline", "bootstrap": {"file": "boot.npz"}}
    (root / "spec.json").write_text(json.dumps(spec))
    return ids, y, eligible, predictions, w


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def compare(self, folder="result"):
        return tool.compare(self.root / "spec.json", self.root / folder)

    def predictions(self, transform):
        p = self.root / "predictions.csv"
        frame = pd.read_csv(p, dtype={"patient_id": str, "repeat": str, "fold": str})
        transform(frame).to_csv(p, index=False)

    def folds(self, transform):
        p = self.root / "folds.json"
        value = json.loads(p.read_text())
        transform(value)
        p.write_text(json.dumps(value))

    def reject(self, pattern):
        with self.assertRaisesRegex(tool.InputError, pattern):
            self.compare()

    def check_reference(self, classification):
        ids, y, eligible, records, weights = fixture(self.root, classification)
        self.compare()
        df = pd.DataFrame(records)
        roles = {"S00": ("A_full", np.ones(12, bool)), "S01": ("A_full", eligible),
                 "S10": ("A_restricted", np.ones(12, bool)), "S11": ("A_restricted", eligible), "candidate": ("B", eligible)}
        metrics = {"balanced_accuracy": balanced_accuracy_score} if classification else {
            "pearson_r": lambda truth, pred: pearsonr(truth, pred).statistic, "mae": mean_absolute_error}
        expected = {}
        for cell, (model, mask) in roles.items():
            for metric, function in metrics.items():
                values = []
                for weight in [np.ones(12, int), *weights]:
                    chosen = np.repeat(np.arange(12), weight)
                    chosen = chosen[mask[chosen]]
                    if not len(chosen) or (classification and len(set(y[chosen])) != 2):
                        values.append(np.nan)
                        continue
                    by_repeat = []
                    for repeat in ["0", "1"]:
                        one = df.loc[(df.model == model) & (df.repeat == repeat)].set_index("patient_id")
                        pred = one.loc[[ids[i] for i in chosen]].prediction.to_numpy()
                        by_repeat.append(function(y[chosen], pred))
                    values.append(np.mean(by_repeat))
                expected[cell, metric] = np.array(values)
        scores = pd.read_csv(self.root / "result/scores.csv")
        for row in scores.itertuples():
            truth = expected[row.cell, row.metric]
            self.assertAlmostEqual(row.estimate, truth[0], places=11)
            np.testing.assert_allclose([row.ci_low, row.ci_high], np.nanquantile(truth[1:], [.025, .975]), atol=1e-11)
        contrast = pd.read_csv(self.root / "result/contrasts.csv")
        coefficients = {"scoring_at_full_training": {"S01": 1, "S00": -1},
                        "training_at_full_scoring": {"S10": 1, "S00": -1},
                        "training_at_eligible_scoring": {"S11": 1, "S01": -1},
                        "scoring_at_restricted_training": {"S11": 1, "S10": -1},
                        "joint_change": {"S11": 1, "S00": -1},
                        "interaction": {"S11": 1, "S10": -1, "S01": -1, "S00": 1},
                        "method_difference_on_eligible": {"candidate": 1, "S11": -1}}
        for row in contrast.itertuples():
            truth = sum(coef * expected[cell, row.metric] for cell, coef in coefficients[row.contrast].items())
            self.assertAlmostEqual(row.estimate, truth[0], places=11)
            np.testing.assert_allclose([row.ci_low, row.ci_high], np.nanquantile(truth[1:], [.025, .975]), atol=1e-11)
            self.assertEqual(row.valid_bootstrap_draws, int(np.isfinite(truth[1:]).sum()))

    def test_regression_against_direct_scipy_bootstrap(self):
        self.check_reference(False)

    def test_classification_against_direct_sklearn_bootstrap(self):
        self.check_reference(True)

    def test_nearly_constant_predictions_against_direct_scipy(self):
        y = np.array([10., 20., 50., 80., 90.])
        weights = np.array([[1, 1, 1, 1, 1], [0, 2, 1, 1, 1], [3, 0, 0, 1, 1]])
        for amplitude in [1e-2, 1e-4, 1e-6, 1e-7]:
            pred = 50. + amplitude * np.arange(1., 6.)
            predictions = np.array([pred, 100. - pred])
            actual = tool.weighted_scores(weights, y, predictions, 'regression')['pearson_r']
            expected = []
            for counts in weights:
                chosen = np.repeat(np.arange(len(y)), counts)
                expected.append([pearsonr(y[chosen], p[chosen]).statistic for p in predictions])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_correlation_stable_under_small_and_large_prediction_scales(self):
        y = np.array([10., 20., 50., 80., 90.])
        pred = np.array([1., 2., 3., 4., 5.])
        predictions = np.array([pred * 1e-200, pred, pred * 1e200])
        result = tool.weighted_scores(np.ones(5), y, predictions, 'regression')['pearson_r']
        np.testing.assert_allclose(result, pearsonr(y, pred).statistic, atol=1e-14, rtol=0)

    def test_row_order_and_bootstrap_id_alignment(self):
        self.compare("before")
        for name in ["patients.csv", "predictions.csv"]:
            p = self.root / name
            pd.read_csv(p, dtype=str, keep_default_na=False).sample(frac=1, random_state=25).to_csv(p, index=False)
        self.folds(lambda items: items.reverse())
        self.compare("after")
        for name in ["scores.csv", "contrasts.csv", "scores_by_repeat.csv"]:
            pd.testing.assert_frame_equal(pd.read_csv(self.root / "before" / name), pd.read_csv(self.root / "after" / name), check_exact=False, atol=1e-11, rtol=0)

    def test_no_patient_identifiers_in_outputs(self):
        report = self.compare()
        self.assertEqual(report["full_n"], 12)
        self.assertEqual(report["eligible_n"], 8)
        for p in (self.root / "result").glob("*.csv"):
            self.assertNotIn("patient_id", pd.read_csv(p).columns)

    def test_output_does_not_overwrite_input(self):
        p = self.root / "spec.json"; spec = json.loads(p.read_text())
        original = (self.root / "patients.csv").read_bytes()
        (self.root / "scores.csv").write_bytes(original)
        spec["patients"] = "scores.csv"; p.write_text(json.dumps(spec))
        with self.assertRaisesRegex(tool.InputError, "overwrite an input"):
            tool.compare(p, self.root)
        self.assertEqual((self.root / "scores.csv").read_bytes(), original)

    def test_duplicate_patient_prediction(self):
        self.predictions(lambda f: pd.concat([f, f.iloc[:1]]))
        self.reject("duplicate patient prediction")

    def test_missing_candidate_prediction(self):
        self.predictions(lambda f: f.drop(f.index[f.model == "B"][0]))
        self.reject("exactly cover")

    def test_unknown_patient(self):
        self.predictions(lambda f: f.assign(patient_id=f.patient_id.where(f.index != 0, "foreign")))
        self.reject("unknown patient")

    def test_outcome_disagreement(self):
        self.predictions(lambda f: f.assign(y_true=f.y_true + (f.index == 0)))
        self.reject("outcome differs")

    def test_prediction_nonfinite(self):
        self.predictions(lambda f: f.assign(prediction=f.prediction.where(f.index != 0, np.nan)))
        self.reject("nonfinite|numeric")

    def test_duplicate_fold_record(self):
        self.folds(lambda items: items.append(items[0]))
        self.reject("duplicate fold record")

    def test_train_test_overlap(self):
        self.folds(lambda items: items[0]["train_ids"].append(items[0]["test_ids"][0]))
        self.reject("train/test overlap")

    def test_wrong_restricted_training(self):
        def change(items):
            record = next(x for x in items if x["model"] == "A_restricted")
            record["train_ids"].pop()
        self.folds(change)
        self.reject("training membership")

    def test_candidate_different_training_population(self):
        def change(items):
            record = next(x for x in items if x["model"] == "B")
            record["train_ids"].pop()
        self.folds(change)
        self.reject("training membership")

    def test_changed_candidate_outer_fold(self):
        self.predictions(lambda f: f.assign(fold=f.fold.where(~((f.model == "B") & (f.patient_id == "001")), "99")))
        self.reject("folds differ")

    def test_unexpected_model(self):
        self.predictions(lambda f: f.assign(model=f.model.where(f.index != 0, "extra")))
        self.reject("model names")

    def test_bootstrap_wrong_population(self):
        np.savez(self.root / "boot.npz", weights=np.ones((120, 12)), patient_ids=np.array([str(i) for i in range(12)]))
        self.reject("bootstrap patient IDs differ")

    def test_bootstrap_fractional_weights(self):
        np.savez(self.root / "boot.npz", weights=np.full((120, 12), .5), patient_ids=np.asarray(self.data[0]))
        self.reject("integer counts")

    def test_constant_regression_prediction_is_not_zero_correlation(self):
        self.predictions(lambda f: f.assign(prediction=f.prediction.where(f.model != "B", 50)))
        self.compare()
        results = pd.read_csv(self.root / "result/scores.csv")
        row = results.query("cell == 'candidate' and metric == 'pearson_r'").iloc[0]
        self.assertTrue(np.isnan(row.estimate))
        self.assertEqual(row.valid_bootstrap_draws, 0)
        self.assertTrue(np.isfinite(results.query("cell == 'candidate' and metric == 'mae'").estimate.iloc[0]))

    def test_classification_probability_rejected(self):
        fixture(self.root, True)
        self.predictions(lambda f: f.assign(prediction=.3))
        self.reject("0/1 label")

    def test_diagnostic_mode_without_candidate(self):
        p = self.root / "spec.json"; spec = json.loads(p.read_text())
        del spec["candidate"]; del spec["candidate_reference"]
        p.write_text(json.dumps(spec))
        self.predictions(lambda f: f.loc[f.model != "B"])
        self.folds(lambda items: items.__setitem__(slice(None), [x for x in items if x["model"] != "B"]))
        self.assertEqual(self.compare()["status"], "population_controls_only")
        self.assertNotIn("method_difference_on_eligible", pd.read_csv(self.root / "result/contrasts.csv").contrast.tolist())

    def test_method_comparison_with_full_training_reference(self):
        p = self.root / "spec.json"; spec = json.loads(p.read_text())
        del spec["restricted_baseline"]
        spec["candidate_reference"] = "baseline"; p.write_text(json.dumps(spec))
        self.predictions(lambda f: f.loc[f.model != "A_restricted"])
        def change(items):
            base = {(x["repeat"], x["fold"]): x["train_ids"] for x in items if x["model"] == "A_full"}
            for row in items:
                if row["model"] == "B":
                    row["train_ids"] = list(base[row["repeat"], row["fold"]])
            items[:] = [x for x in items if x["model"] != "A_restricted"]
        self.folds(change)
        self.assertEqual(self.compare()["status"], "controlled_method_comparison")
        results = pd.read_csv(self.root / "result/contrasts.csv")
        self.assertEqual(set(results.contrast), {"scoring_at_full_training", "method_difference_on_eligible"})

    def test_duplicate_population_id(self):
        p = self.root / "patients.csv"; frame = pd.read_csv(p, dtype=str)
        pd.concat([frame, frame.iloc[:1]]).to_csv(p, index=False)
        self.reject("duplicate IDs")

    def test_no_eligible_patients(self):
        p = self.root / "patients.csv"; frame = pd.read_csv(p, dtype=str)
        frame["eligible"] = 0; frame.to_csv(p, index=False)
        self.reject("eligible patients")


if __name__ == "__main__":
    unittest.main()
