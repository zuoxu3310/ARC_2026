"""Behavior checks for reference banks and paired candidate evaluation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score, mean_absolute_error, mean_squared_error,
                             r2_score)
from scipy.stats import pearsonr

import arc_reference as ref


class ReferenceContractTests(unittest.TestCase):
    def fixture(self):
        patients = pd.DataFrame({"patient_id": ["001", "002", "003", "004"],
                                 "y_true": [0, 1, 0, 1]})
        assignments = pd.DataFrame({"patient_id": patients.patient_id, "repeat": "runA",
                                    "fold": ["a", "a", "b", "b"]})
        info = {"task": "classification", "repeat_seeds": {"runA": 1000}}
        predictions = assignments.assign(prediction=[0, 1, 1, 0], y_true=patients.y_true)
        folds = ref.expected_folds(info, patients, assignments, "runA")
        return info, patients, assignments, predictions, folds

    def test_matching_candidate_in_any_row_order(self):
        info, pts, assignment, preds, folds = self.fixture()
        actual = ref.aligned_candidate(info, pts, assignment, preds.iloc[::-1], folds)
        self.assertEqual(actual[0].patient_id.tolist(), ["001", "002", "003", "004"])

    def test_duplicate_missing_extra_or_wrong_fold_rejected(self):
        info, pts, assignment, preds, folds = self.fixture()
        variants = [pd.concat([preds, preds.iloc[:1]]), preds.iloc[:-1],
                    preds.assign(patient_id=["other", "002", "003", "004"]),
                    preds.assign(fold=["b", "a", "b", "a"])]
        for invalid in variants:
            with self.subTest(rows=invalid.to_dict("list")), self.assertRaises(ValueError):
                ref.aligned_candidate(info, pts, assignment, invalid, folds)

    def test_training_membership_and_outcomes_checked(self):
        info, pts, assignment, preds, folds = self.fixture()
        invalid = copy.deepcopy(folds)
        invalid[0]["train_ids"][0] = "001"
        with self.assertRaisesRegex(ValueError, "train_ids"):
            ref.aligned_candidate(info, pts, assignment, preds, invalid)
        with self.assertRaisesRegex(ValueError, "outcomes"):
            ref.aligned_candidate(info, pts, assignment, preds.assign(y_true=[1, 1, 0, 1]), folds)

    def test_one_class_training_folds_rejected(self):
        _, pts, assignment, _, _ = self.fixture()
        bad = assignment.assign(fold=["a", "b", "a", "b"])
        with self.assertRaisesRegex(ValueError, "One-class"):
            ref.validate_assignments(bad, pts.patient_id.tolist(), pts.set_index("patient_id").y_true, "classification")

    def test_ids_preserve_zeros_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ids.csv"
            path.write_text("patient_id\n001\n002\n")
            self.assertEqual(ref.read_ids(path), ["001", "002"])
            path.write_text("patient_id\n001\n001\n")
            with self.assertRaises(ValueError):
                ref.read_ids(path)

    def test_model_metadata_is_strict_json(self):
        params = ref.model_parameters("regression", "XGBoost", 1000)
        json.dumps(params, allow_nan=False)
        self.assertEqual(params["missing"], {"nonfinite_parameter": "NaN"})


class PairedMetricTests(unittest.TestCase):
    def test_classification_weighted_metrics_match_explicit_replication(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        p = np.array([[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0]])
        q = np.array([[.2, .2, .7, .7, .4, .9], [.1, .8, .2, .8, .2, .4]])
        W = np.array([[1, 1, 1, 1, 1, 1], [3, 2, 0, 1, 1, 2]])
        actual = ref.weighted_metrics("classification", y, p, W, q)
        for b, weights in enumerate(W):
            idx = np.repeat(np.arange(len(y)), weights)
            for k in range(len(p)):
                expected = {"balanced_acc": balanced_accuracy_score(y[idx], p[k, idx]),
                            "f1_severe": f1_score(y[idx], p[k, idx]),
                            "mcc": matthews_corrcoef(y[idx], p[k, idx]),
                            "auc": roc_auc_score(y[idx], q[k, idx])}
                for name, value in expected.items():
                    self.assertAlmostEqual(actual[name][b, k], value, places=12)

    def test_regression_weighted_metrics_match_explicit_replication(self):
        y = np.array([10., 35., 70., 81., 95.])
        p = np.array([[12., 33., 68., 82., 90.], [1e7+.01, 1e7+.05, 1e7+.03, 1e7+.07, 1e7+.10]])
        W = np.array([[1, 1, 1, 1, 1], [3, 2, 0, 1, 2]])
        actual = ref.weighted_metrics("regression", y, p, W)
        for b, weights in enumerate(W):
            idx = np.repeat(np.arange(len(y)), weights)
            for k in range(len(p)):
                expected = {"r": pearsonr(y[idx], p[k, idx]).statistic,
                            "mae": mean_absolute_error(y[idx], p[k, idx]),
                            "rmse": np.sqrt(mean_squared_error(y[idx], p[k, idx])),
                            "r2": r2_score(y[idx], p[k, idx])}
                for name, value in expected.items():
                    np.testing.assert_allclose(actual[name][b, k], value, rtol=1e-10, atol=1e-10)

    def test_undefined_metrics_are_not_invented_zeroes(self):
        y = np.array([1., 2., 3.])
        values = ref.weighted_metrics("regression", y, np.ones((1, 3)), np.ones((1, 3)))
        self.assertTrue(np.isnan(values["r"][0, 0]))
        self.assertIsNone(ref.scores("regression", y, np.ones(3))["r"])

    def test_reference_correlation_preserves_small_and_large_scales(self):
        y = np.array([1., 2., 3., 4., 7.])
        pattern = np.array([2., 1., 4., 3., 8.])
        W = np.array([[1, 1, 1, 1, 1], [2, 1, 0, 3, 2]])
        for scale in [1e-100, 1e100, -1e-100]:
            p = pattern*scale
            np.testing.assert_allclose(ref.scores("regression", y, p)["r"], pearsonr(y, p).statistic, atol=1e-14)
            actual = ref.weighted_metrics("regression", y, p[None, :], W)["r"][:, 0]
            expected = []
            for weights in W:
                idx = np.repeat(np.arange(len(y)), weights)
                expected.append(pearsonr(y[idx], p[idx]).statistic)
            np.testing.assert_allclose(actual, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
