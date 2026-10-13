"""Tests of the integrated fitting, selection and comparison boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import arc_comparison_run as run
from test_comparison_protocol import fixture


def policy():
    return {"schema_version": 1, "objective": "f1_severe", "inner_folds": 3,
            "feature_policy": "frozen_dictionary",
            "reference": {"model": "RandomForest", "mode": "nested",
                          "parameter_grid": [{"n_estimators": 12, "max_depth": 2}, {"n_estimators": 12, "max_depth": 4}],
                          "thresholds": [0.3, 0.5, 0.7]},
            "candidate": {"model": "XGBoost", "mode": "fixed", "parameter_grid": [{}]}}


class ControlledFittingTests(unittest.TestCase):
    def test_two_references_cover_full_test_fold(self):
        pts = pd.DataFrame({"patient_id": ["001", "002", "003", "004", "005", "006"]})
        assignment = pd.DataFrame({"patient_id": pts.patient_id, "repeat": "1", "fold": ["a", "a", "b", "b", "c", "c"]})
        info = {"repeat_seeds": {"1": 1}}
        eligible = ["001", "003", "005"]
        records = run.planned_folds(info, pts, assignment, eligible)
        for fold in ["a", "b", "c"]:
            by_role = {r["model"]: r for r in records if r["fold"] == fold}
            self.assertEqual(by_role["baseline"]["test_ids"], by_role["restricted_baseline"]["test_ids"])
            self.assertEqual(set(by_role["candidate"]["train_ids"]), set(by_role["restricted_baseline"]["train_ids"]))
            self.assertEqual(set(by_role["candidate"]["test_ids"]), set(by_role["baseline"]["test_ids"]) & set(eligible))
            self.assertFalse(set(by_role["restricted_baseline"]["train_ids"]) & set(by_role["baseline"]["test_ids"]))

    def test_target_and_selection_are_explicit(self):
        run.validate_policy(policy(), "classification")
        for change in [{"objective": "unknown"}, {"feature_policy": "global_unknown"}, {"inner_folds": 1}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                run.validate_policy(policy() | change, "classification")
        invalid = policy()
        invalid["reference"]["mode"] = "fixed"
        with self.assertRaisesRegex(ValueError, "Fixed policy"):
            run.validate_policy(invalid, "classification")

    def test_unknown_parameters_and_hidden_seed_override_rejected(self):
        for params in [{"made_up_parameter": 1}, {"random_state": 44}, {"n_jobs": 4}]:
            p = policy()
            p["reference"]["parameter_grid"] = [params]
            with self.subTest(params=params), self.assertRaises(ValueError):
                run.validate_policy(p, "classification")

    def test_nested_scores_and_selection_match_independent_metric(self):
        rng = np.random.default_rng(25)
        X = rng.normal(size=(36, 6)); y = (X[:, 0] + .3 * X[:, 1] > 0).astype(int)
        ids = [f"{i:03d}" for i in range(len(y))]
        item = policy()["reference"]
        selected, trials, rows, folds, fits = run.select_policy(X, y, ids, "classification", item,
                                                               "f1_severe", 3, "frozen_dictionary", 91, 1)
        self.assertEqual(fits, 6)
        pred = pd.DataFrame(rows)
        for trial in trials:
            sub = pred[pred.configuration == trial["configuration"]]
            value = f1_score(sub.y_true, sub.score >= trial["threshold"], zero_division=0)
            self.assertAlmostEqual(value, trial["objective_value"], places=14)
        self.assertEqual(selected, max(trials, key=lambda t: t["objective_value"]))
        for fold in folds:
            self.assertFalse(set(fold["train_ids"]) & set(fold["validation_ids"]))
            self.assertEqual(set(fold["train_ids"]) | set(fold["validation_ids"]), set(ids))
        self.assertEqual(sum(len(f["validation_ids"]) for f in folds), len(ids))

    def test_test_patients_cannot_change_training_feature_dictionary(self):
        X = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 0], [4, 1, 0], [5, 0, 0], [6, 0, 0],
                      [7, 0, 0], [8, 0, 0], [9, 0, 0], [10, 0, 0]], dtype=float)
        selector = run.TrainingNonzeroSelector().fit(X)
        self.assertEqual(selector.support_.tolist(), [True, False, False])
        transformed = selector.transform(np.array([[9, 99999, 99999]]))
        self.assertEqual(transformed.tolist(), [[9]])
        self.assertEqual(selector.fit_n_, 10)

    def test_fitting_is_independent_of_unused_outer_test_labels(self):
        rng = np.random.default_rng(71)
        X = rng.normal(size=(42, 5)); y = np.tile([0, 1], 21)
        train = np.arange(30); test = np.arange(30, 42)
        item = policy()["reference"]
        first = run.select_policy(X[train], y[train], list(map(str, train)), "classification", item,
                                  "f1_severe", 3, "training_nonzero", 4, 1)
        y[test] = 1 - y[test]; X[test] *= 1000000
        second = run.select_policy(X[train], y[train], list(map(str, train)), "classification", item,
                                   "f1_severe", 3, "training_nonzero", 4, 1)
        self.assertEqual(first, second)

    def test_fixed_policy_keeps_estimator_decision_rule(self):
        p = policy()["candidate"]
        selected, trials, preds, folds, fits = run.select_policy(np.ones((4, 2)), np.array([0, 1, 0, 1]),
                ["1", "2", "3", "4"], "classification", p, "f1_severe", 3, "frozen_dictionary", 1, 1)
        self.assertEqual(selected["threshold"], "estimator")
        self.assertEqual((trials, preds, folds, fits), ([], [], [], 0))

    def test_provided_bad_scores_fail_instead_of_disappearing(self):
        for value in ["broken", np.inf, -np.inf, np.nan, ""]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "numeric and finite"):
                run.checked_scores(pd.DataFrame({"score": [.1, value]}))
        self.assertIsNone(run.checked_scores(pd.DataFrame({"prediction": [0, 1]})))

    def test_constant_candidate_keeps_defined_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            f = pd.read_csv(root / "predictions.csv", dtype={"patient_id": str})
            f.loc[f.model == "B", "prediction"] = 50.
            f.to_csv(root / "predictions.csv", index=False)
            run.all_metrics(root / "spec.json", root / "result")
            results = pd.read_csv(root / "result/all-metric-contrasts.csv")
            r = results[(results.metric == "r") & (results.contrast == "method")].iloc[0]
            mae = results[(results.metric == "mae") & (results.contrast == "method")].iloc[0]
            self.assertTrue(np.isnan(r.estimate))
            self.assertEqual(r.valid_bootstrap_draws, 0)
            self.assertTrue(np.isfinite(mae.estimate))
            info = json.loads((root / "result/path-verification.json").read_text())
            entry = next(m for m in info["metrics"] if m["metric"] == "r")
            self.assertEqual(entry["status"], "undefined_metric")
            self.assertIsNone(entry["point_residual"])

    def test_all_metrics_rejects_bad_or_infinite_auc_inputs(self):
        for value in ["broken", np.inf, ""]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture(root, classification=True)
                f = pd.read_csv(root / "predictions.csv", dtype={"patient_id": str})
                f["score"] = f.prediction.astype(object)
                f.loc[f.index[f.model == "B"][0], "score"] = value
                f.to_csv(root / "predictions.csv", index=False)
                with self.assertRaisesRegex(ValueError, "numeric and finite"):
                    run.all_metrics(root / "spec.json", root / "result")


if __name__ == "__main__":
    unittest.main()
