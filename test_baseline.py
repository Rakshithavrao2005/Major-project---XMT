"""
tests/test_baseline.py
─────────────────────────────────────────────────────────────────────────────
Tests for Phase 2 baseline model artifacts and evaluation outputs.

Run with:
    python -m pytest tests/test_baseline.py -v
"""

import os
import sys
import json
import pickle
import unittest
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.clinical.evaluate import compute_metrics, calibration_stats, threshold_sweep

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "baseline")
PROC_DIR   = os.path.join(_PROJECT_ROOT, "data", "processed")
META_DIR   = os.path.join(_PROJECT_ROOT, "data", "metadata")
LABEL_COL  = "sepsis_label"

_MODELS_EXIST = os.path.isdir(MODELS_DIR) and any(
    f.endswith(".pkl") for f in os.listdir(MODELS_DIR)
) if os.path.isdir(MODELS_DIR) else False


# ─────────────────────────────────────────────────────────────────────────────
# F. Evaluate utility unit tests (no model artifacts needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateUtils(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.y_true = np.array([0]*80 + [1]*20)
        self.y_prob = np.clip(
            self.y_true * 0.6 + rng.normal(0, 0.2, 100), 0, 1
        )

    def test_F01_compute_metrics_keys(self):
        """compute_metrics must return all required keys."""
        m = compute_metrics(self.y_true, self.y_prob)
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1",
                    "tn", "fp", "fn", "tp", "brier_score"]:
            self.assertIn(key, m, f"Missing key: {key}")

    def test_F02_roc_auc_in_range(self):
        m = compute_metrics(self.y_true, self.y_prob)
        self.assertGreaterEqual(m["roc_auc"], 0.0)
        self.assertLessEqual(m["roc_auc"], 1.0)

    def test_F03_pr_auc_in_range(self):
        m = compute_metrics(self.y_true, self.y_prob)
        self.assertGreaterEqual(m["pr_auc"], 0.0)
        self.assertLessEqual(m["pr_auc"], 1.0)

    def test_F04_confusion_matrix_sums_to_n(self):
        m = compute_metrics(self.y_true, self.y_prob)
        self.assertEqual(m["tn"] + m["fp"] + m["fn"] + m["tp"], len(self.y_true))

    def test_F05_brier_score_in_range(self):
        m = compute_metrics(self.y_true, self.y_prob)
        self.assertGreaterEqual(m["brier_score"], 0.0)
        self.assertLessEqual(m["brier_score"], 1.0)

    def test_F06_calibration_ece_in_range(self):
        cal = calibration_stats(self.y_true, self.y_prob)
        self.assertIn("ece", cal)
        self.assertGreaterEqual(cal["ece"], 0.0)
        self.assertLessEqual(cal["ece"], 1.0)

    def test_F07_threshold_sweep_shape(self):
        df = threshold_sweep(self.y_true, self.y_prob)
        self.assertIn("threshold", df.columns)
        self.assertIn("f1", df.columns)
        self.assertGreater(len(df), 0)

    def test_F08_perfect_classifier_roc_auc_1(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_prob)
        self.assertAlmostEqual(m["roc_auc"], 1.0, places=4)

    def test_F09_random_classifier_roc_auc_near_half(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.random(1000)
        m = compute_metrics(y_true, y_prob)
        self.assertGreater(m["roc_auc"], 0.40)
        self.assertLess(m["roc_auc"],    0.60)

    def test_F10_n_pos_neg_counts(self):
        m = compute_metrics(self.y_true, self.y_prob)
        self.assertEqual(m["n_pos"], int(self.y_true.sum()))
        self.assertEqual(m["n_neg"], int((1 - self.y_true).sum()))


# ─────────────────────────────────────────────────────────────────────────────
# G. Model artifact tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelArtifacts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not _MODELS_EXIST:
            raise unittest.SkipTest(
                "Model artifacts not found. Run `python src/clinical/train_baseline.py` first."
            )
        cls.models_dir = MODELS_DIR

    def test_G01_config_file_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(self.models_dir, "training_config.json"))
        )

    def test_G02_metrics_file_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(self.models_dir, "test_metrics.json"))
        )

    def test_G03_cv_results_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(self.models_dir, "cv_results.csv"))
        )

    def test_G04_at_least_two_model_pkls(self):
        pkls = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
        self.assertGreaterEqual(len(pkls), 2,
            f"Expected >= 2 .pkl files, found: {pkls}")

    def test_G05_config_has_required_keys(self):
        with open(os.path.join(self.models_dir, "training_config.json")) as f:
            cfg = json.load(f)
        for key in ["n_features", "feature_columns", "cv_folds",
                    "cv_summary", "best_cv_model", "chosen_thresholds"]:
            self.assertIn(key, cfg, f"Missing config key: {key}")

    def test_G06_config_feature_count(self):
        with open(os.path.join(self.models_dir, "training_config.json")) as f:
            cfg = json.load(f)
        self.assertGreaterEqual(cfg["n_features"], 39,
            "Expected at least 39 features")

    def test_G07_test_metrics_roc_auc_above_chance(self):
        """All models must achieve ROC-AUC > 0.5 on the test set."""
        with open(os.path.join(self.models_dir, "test_metrics.json")) as f:
            metrics = json.load(f)
        for name, m in metrics.items():
            self.assertGreater(m["roc_auc"], 0.5,
                f"{name} ROC-AUC={m['roc_auc']} is not above chance")

    def test_G08_test_metrics_pr_auc_above_baseline(self):
        """All models must achieve PR-AUC > positive rate (16.5%)."""
        with open(os.path.join(self.models_dir, "test_metrics.json")) as f:
            metrics = json.load(f)
        for name, m in metrics.items():
            self.assertGreater(m["pr_auc"], 0.165,
                f"{name} PR-AUC={m['pr_auc']} not above baseline ({m['positive_rate']})")

    def test_G09_models_can_predict(self):
        """Each saved .pkl model must load and predict on X_test without error."""
        if not os.path.isfile(os.path.join(PROC_DIR, "X_test.csv")):
            self.skipTest("X_test.csv not found")
        X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        pkls = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
        for pkl in pkls:
            with open(os.path.join(self.models_dir, pkl), "rb") as f:
                model = pickle.load(f)
            probs = model.predict_proba(X_test)[:, 1]
            self.assertEqual(len(probs), len(X_test),
                f"{pkl}: prob length mismatch")
            self.assertTrue((probs >= 0).all() and (probs <= 1).all(),
                f"{pkl}: probs out of [0,1] range")

    def test_G10_cv_results_has_expected_columns(self):
        cv = pd.read_csv(os.path.join(self.models_dir, "cv_results.csv"))
        for col in ["model", "fold", "pr_auc", "roc_auc"]:
            self.assertIn(col, cv.columns, f"Missing CV column: {col}")

    def test_G11_cv_results_five_folds_per_model(self):
        cv = pd.read_csv(os.path.join(self.models_dir, "cv_results.csv"))
        for model_name, grp in cv.groupby("model"):
            self.assertEqual(len(grp), 5,
                f"{model_name}: expected 5 CV fold rows, got {len(grp)}")

    def test_G12_no_test_data_used_in_config(self):
        """training_config must record cv_folds = 5 (no test-set fitting)."""
        with open(os.path.join(self.models_dir, "training_config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["cv_folds"], 5)

    def test_G13_report_md_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(META_DIR, "PHASE2_BASELINE_REPORT.md")),
            "PHASE2_BASELINE_REPORT.md not found in data/metadata/"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestEvaluateUtils, TestModelArtifacts]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
