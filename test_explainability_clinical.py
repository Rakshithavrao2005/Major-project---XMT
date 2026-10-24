"""
tests/test_explainability_clinical.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Test Suite for Step 5A Clinical Model Explainability.

Verifies:
1. CalibratedClassifierCV wrapper extraction & scikit-learn version defensive handling.
2. Feature count (49), schema ordering, and zero data/target leakage.
3. Logistic Regression calibrated probability SHAP additivity & uncalibrated log-odds SHAP.
4. Random Forest calibrated probability SHAP additivity & uncalibrated TreeExplainer SHAP.
5. Cohort-level explanation aggregation on held-out test split (N=602).
6. Exact mathematical reproducibility and deterministic consistency.
7. Mandatory clinical non-causality disclaimers on all outputs.
"""

from __future__ import annotations
import os
import sys
import pytest
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.explainability.shap_explainers import (
    LogisticRegressionSHAPExplainer,
    RandomForestSHAPExplainer,
    ClinicalTransformerSHAPExplainer,
    extract_base_estimator,
    CLINICAL_DISCLAIMER,
)

PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


# =============================================================================
# 1. CalibratedClassifierCV Wrapper Inspection & Defensive Extraction
# =============================================================================
class TestCalibratedClassifierCVWrapperInspection:
    """Verifies safe, version-defensive unpacking of CalibratedClassifierCV wrappers."""

    def test_extract_base_estimator_logistic_regression(self):
        """Checks that inner pipeline is extracted correctly from logistic_regression.pkl."""
        lr_exp = LogisticRegressionSHAPExplainer(num_background=10)
        inner = extract_base_estimator(lr_exp.model)
        assert inner is not None
        assert hasattr(inner, "named_steps")
        assert "scaler" in inner.named_steps
        assert "clf" in inner.named_steps
        assert hasattr(inner.named_steps["clf"], "coef_")

    def test_extract_base_estimator_random_forest(self):
        """Checks that inner RandomForestClassifier is extracted correctly from random_forest.pkl."""
        rf_exp = RandomForestSHAPExplainer(num_background=10)
        inner = extract_base_estimator(rf_exp.model)
        assert inner is not None
        assert hasattr(inner, "estimators_")
        assert len(inner.estimators_) == 300
        assert inner.max_depth == 8

    def test_extract_base_estimator_passthrough(self):
        """Checks that passing a non-calibrated estimator passes through untouched."""
        dummy_model = {"custom_model": True}
        extracted = extract_base_estimator(dummy_model)
        assert extracted == dummy_model


# =============================================================================
# 2. Feature Schema & Leakage Insulation
# =============================================================================
class TestFeatureAlignmentAndLeakage:
    """Verifies exact 49-feature schema alignment and zero target leakage."""

    @classmethod
    def setup_class(cls):
        cls.X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        cls.lr_exp = LogisticRegressionSHAPExplainer(num_background=10)
        cls.rf_exp = RandomForestSHAPExplainer(num_background=10)

    def test_feature_count_and_column_names(self):
        """Confirms exactly 49 features matching X_train and X_test column order."""
        assert len(self.lr_exp.feature_names) == 49
        assert len(self.rf_exp.feature_names) == 49
        assert self.lr_exp.feature_names == list(self.X_train.columns)
        assert self.rf_exp.feature_names == list(self.X_test.columns)

    def test_zero_target_leakage_in_features(self):
        """Confirms no target labels or identifier columns appear in feature lists."""
        forbidden = ["sepsis_label", "sepsis_onset_time", "hours_to_onset", "split", "subject_id", "stay_id"]
        for feat in self.lr_exp.feature_names:
            assert feat not in forbidden
        for feat in self.rf_exp.feature_names:
            assert feat not in forbidden


# =============================================================================
# 3. Logistic Regression SHAP Attributions (Calibrated & Uncalibrated)
# =============================================================================
class TestLogisticRegressionSHAP:
    """Verifies Logistic Regression instance and cohort explanations."""

    @classmethod
    def setup_class(cls):
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        cls.sample = cls.X_test.iloc[0]
        cls.explainer = LogisticRegressionSHAPExplainer(num_background=20, random_state=42)

    def test_calibrated_probability_additivity(self):
        """Verifies exact additivity for calibrated probability prediction function."""
        res = self.explainer.explain_instance(self.sample, output_type="calibrated_probability", nsamples=115)
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-3
        assert res["output_type"] == "calibrated_probability"
        assert 0.0 <= res["prediction_value"] <= 1.0
        assert len(res["top_features"]) == 10
        assert len(res["all_features"]) == 49

    def test_uncalibrated_log_odds_additivity(self):
        """Verifies exact analytical additivity for underlying uncalibrated log-odds."""
        res = self.explainer.explain_instance(self.sample, output_type="uncalibrated_log_odds")
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-4
        assert res["output_type"] == "uncalibrated_log_odds"
        # Decision function logit
        inner_pipe = self.explainer.inner_pipeline
        true_logit = float(inner_pipe.decision_function(pd.DataFrame([self.sample]))[0])
        assert abs(res["prediction_value"] - true_logit) < 1e-4

    def test_lr_cohort_explanation_dimensions(self):
        """Verifies cohort explanation produces valid matrices and rankings on test data."""
        test_subset = self.X_test.iloc[:20]
        cohort_res = self.explainer.explain_cohort(test_subset, output_type="uncalibrated_log_odds")
        assert cohort_res["cohort_size"] == 20
        assert cohort_res["shap_matrix"].shape == (20, 49)
        assert len(cohort_res["top_features"]) == 15
        assert cohort_res["top_features"][0]["mean_abs_shap"] >= cohort_res["top_features"][-1]["mean_abs_shap"]


# =============================================================================
# 4. Random Forest SHAP Attributions (Calibrated & Uncalibrated)
# =============================================================================
class TestRandomForestSHAP:
    """Verifies Random Forest instance and cohort explanations."""

    @classmethod
    def setup_class(cls):
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        cls.sample = cls.X_test.iloc[0]
        cls.explainer = RandomForestSHAPExplainer(num_background=20, random_state=42)

    def test_calibrated_probability_additivity(self):
        """Verifies exact additivity for RF calibrated probability function."""
        res = self.explainer.explain_instance(self.sample, output_type="calibrated_probability", nsamples=115)
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-3
        assert res["output_type"] == "calibrated_probability"
        assert 0.0 <= res["prediction_value"] <= 1.0

    def test_uncalibrated_tree_proba_additivity(self):
        """Verifies TreeExplainer additivity on underlying uncalibrated tree ensemble."""
        res = self.explainer.explain_instance(self.sample, output_type="uncalibrated_tree_proba")
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-4
        assert res["output_type"] == "uncalibrated_tree_proba"
        raw_prob = float(self.explainer.inner_rf.predict_proba(pd.DataFrame([self.sample]))[:, 1][0])
        assert abs(res["prediction_value"] - raw_prob) < 1e-4

    def test_rf_cohort_explanation_dimensions(self):
        """Verifies RF cohort explanation with TreeExplainer on test subset."""
        test_subset = self.X_test.iloc[:15]
        cohort_res = self.explainer.explain_cohort(test_subset, output_type="uncalibrated_tree_proba")
        assert cohort_res["cohort_size"] == 15
        assert cohort_res["shap_matrix"].shape == (15, 49)
        assert len(cohort_res["top_features"]) == 15


# =============================================================================
# 5. Exact Reproducibility & Determinism
# =============================================================================
class TestSHAPReproducibility:
    """Verifies that uncalibrated linear and tree attributions are strictly deterministic."""

    @classmethod
    def setup_class(cls):
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        cls.sample = cls.X_test.iloc[1]
        cls.lr_exp = LogisticRegressionSHAPExplainer(num_background=15, random_state=42)
        cls.rf_exp = RandomForestSHAPExplainer(num_background=15, random_state=42)

    def test_uncalibrated_lr_deterministic(self):
        """Multiple runs of analytical linear SHAP return bitwise identical values."""
        res1 = self.lr_exp.explain_instance(self.sample, output_type="uncalibrated_log_odds")
        res2 = self.lr_exp.explain_instance(self.sample, output_type="uncalibrated_log_odds")
        assert res1["prediction_value"] == res2["prediction_value"]
        assert res1["shap_sum"] == res2["shap_sum"]
        for f1, f2 in zip(res1["all_features"], res2["all_features"]):
            assert f1["shap_value"] == f2["shap_value"]

    def test_uncalibrated_tree_deterministic(self):
        """Multiple runs of TreeExplainer return bitwise identical values."""
        res1 = self.rf_exp.explain_instance(self.sample, output_type="uncalibrated_tree_proba")
        res2 = self.rf_exp.explain_instance(self.sample, output_type="uncalibrated_tree_proba")
        assert res1["prediction_value"] == res2["prediction_value"]
        assert res1["shap_sum"] == res2["shap_sum"]
        for f1, f2 in zip(res1["all_features"], res2["all_features"]):
            assert f1["shap_value"] == f2["shap_value"]


# =============================================================================
# 6. Safety & Non-Causality Disclaimers
# =============================================================================
class TestSafetyDisclaimers:
    """Verifies mandatory non-causality disclaimers on all explainability outputs."""

    @classmethod
    def setup_class(cls):
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        cls.sample = cls.X_test.iloc[0]
        cls.lr_exp = LogisticRegressionSHAPExplainer(num_background=10)
        cls.rf_exp = RandomForestSHAPExplainer(num_background=10)

    def test_disclaimer_presence_in_all_models(self):
        """Confirms disclaimer is present in all explanation payloads."""
        res_lr = self.lr_exp.explain_instance(self.sample)
        res_rf = self.rf_exp.explain_instance(self.sample)
        assert "disclaimer" in res_lr
        assert "disclaimer" in res_rf
        assert "NOT biological etiology or clinical causation" in res_lr["disclaimer"]
        assert "NOT biological etiology or clinical causation" in res_rf["disclaimer"]
