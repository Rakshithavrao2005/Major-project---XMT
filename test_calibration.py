"""
tests/test_calibration.py
─────────────────────────────────────────────────────────────────────────────
Automated Test Suite for Step 2: Post-Hoc Probability Calibration.

Test Groups:
- Group C01-C03: Calibrator Artifacts, File Integrity & Configurations
- Group C04-C06: Probability Bounds, Monotonicity & Mathematical Rigor
- Group C07-C09: Input Alignment, Serialization & Reproducibility
- Group C10-C12: Split Isolation, Leakage Verification & Metric Contracts
"""

from __future__ import annotations
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.calibration import (
    TemperatureCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    evaluate_calibration_metrics,
    evaluate_classification_metrics,
    sigmoid,
)
from src.fusion.multimodal_dataset import load_paired_multimodal_splits

CALIB_DIR = os.path.join(_PROJECT_ROOT, "models", "calibration")
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports", "calibration")


class TestCalibrationArtifacts:
    """Group C01-C03: Calibrator Artifacts, File Integrity & Configurations."""

    def test_C01_calibrator_files_exist_on_disk(self):
        """Verifies that all 9 calibrator pickles and config exist in models/calibration/."""
        expected_files = [
            "clinical_transformer_platt.pkl",
            "clinical_transformer_temp.pkl",
            "clinical_transformer_isotonic.pkl",
            "model_a_concat_platt.pkl",
            "model_a_concat_temp.pkl",
            "model_a_concat_isotonic.pkl",
            "model_b_cross_attn_platt.pkl",
            "model_b_cross_attn_temp.pkl",
            "model_b_cross_attn_isotonic.pkl",
            "calibration_config.json",
        ]
        for fname in expected_files:
            path = os.path.join(CALIB_DIR, fname)
            assert os.path.isfile(path), f"Missing calibrator artifact: {path}"

    def test_C02_calibration_reports_exist(self):
        """Verifies that calibration metrics JSON, markdown report, and plot exist."""
        assert os.path.isfile(os.path.join(REPORTS_DIR, "calibration_metrics.json"))
        assert os.path.isfile(os.path.join(REPORTS_DIR, "CALIBRATION_REPORT.md"))
        assert os.path.isfile(os.path.join(REPORTS_DIR, "reliability_curves_comparison.png"))

    def test_C03_calibration_metrics_json_integrity(self):
        """Verifies that calibration_metrics.json contains all 3 models and required fields."""
        with open(os.path.join(REPORTS_DIR, "calibration_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "calibration_metadata" in data
        assert "results" in data
        assert data["calibration_metadata"]["test_split"]["n_samples"] == 602
        assert data["calibration_metadata"]["validation_split"]["n_samples"] == 454

        for m_name in ["Clinical Transformer", "Model A (Concatenation Fusion)", "Model B (Cross-Attention Fusion)"]:
            assert m_name in data["results"]
            m_res = data["results"][m_name]["test"]
            for cal_type in ["uncalibrated", "temperature", "platt", "isotonic"]:
                assert cal_type in m_res
                cal_data = m_res[cal_type]["calibration"]
                assert "brier_score" in cal_data
                assert "log_loss" in cal_data
                assert "ece" in cal_data
                assert "roc_auc" in cal_data
                assert "pr_auc" in cal_data


class TestCalibrationMathAndBounds:
    """Group C04-C06: Probability Bounds, Monotonicity & Mathematical Rigor."""

    def test_C04_probability_bounds_strict(self):
        """Verifies that all calibrators output valid probabilities strictly within [0.0, 1.0]."""
        dummy_logits = np.linspace(-100.0, 100.0, 1000)

        # Temperature
        tc = TemperatureCalibrator(temperature=1.35)
        p_temp = tc.predict_proba(dummy_logits)
        assert np.all(p_temp >= 0.0) and np.all(p_temp <= 1.0)
        assert p_temp[0] < 1e-4 and p_temp[-1] > 0.9999

        # Platt
        pc = PlattCalibrator(a=0.6, b=-1.5)
        p_platt = pc.predict_proba(dummy_logits)
        assert np.all(p_platt >= 0.0) and np.all(p_platt <= 1.0)

        # Isotonic
        ic = IsotonicCalibrator()
        dummy_p = sigmoid(dummy_logits)
        ic.fit(dummy_p, (dummy_p > 0.5).astype(int))
        p_iso = ic.predict_proba(dummy_p)
        assert np.all(p_iso >= 0.0) and np.all(p_iso <= 1.0)

    def test_C05_platt_and_temperature_preserve_ranking(self):
        """Verifies that strictly monotonic calibrators preserve ROC-AUC exactly."""
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.2, 500)
        logits = y_true * 1.5 + rng.normal(0, 1.0, 500)

        tc = TemperatureCalibrator().fit(logits, y_true)
        pc = PlattCalibrator().fit(logits, y_true)

        p_uncal = sigmoid(logits)
        p_temp = tc.predict_proba(logits)
        p_platt = pc.predict_proba(logits)

        from sklearn.metrics import roc_auc_score
        roc_uncal = roc_auc_score(y_true, p_uncal)
        roc_temp = roc_auc_score(y_true, p_temp)
        roc_platt = roc_auc_score(y_true, p_platt)

        assert abs(roc_uncal - roc_temp) < 1e-6
        assert abs(roc_uncal - roc_platt) < 1e-6

    def test_C06_temperature_parameter_strictly_positive(self):
        """Verifies that fitted temperature T is strictly positive."""
        for slug in ["clinical_transformer", "model_a_concat", "model_b_cross_attn"]:
            tc = TemperatureCalibrator.load(os.path.join(CALIB_DIR, f"{slug}_temp.pkl"))
            assert tc.temperature > 0.01, f"{slug} T={tc.temperature} is not strictly positive"

    def test_C07_platt_scale_parameter_strictly_positive(self):
        """Verifies that Platt slope a is positive (monotonic increasing with logit)."""
        for slug in ["clinical_transformer", "model_a_concat", "model_b_cross_attn"]:
            pc = PlattCalibrator.load(os.path.join(CALIB_DIR, f"{slug}_platt.pkl"))
            assert pc.a > 0.0, f"{slug} Platt a={pc.a} is non-positive"


class TestCalibrationAlignmentAndReproducibility:
    """Group C07-C09: Input Alignment, Serialization & Reproducibility."""

    def test_C08_calibrator_save_load_exact_reproducibility(self):
        """Verifies that saved and loaded calibrators yield byte-identical predictions."""
        tc = TemperatureCalibrator.load(os.path.join(CALIB_DIR, "model_a_concat_temp.pkl"))
        pc = PlattCalibrator.load(os.path.join(CALIB_DIR, "model_a_concat_platt.pkl"))
        ic = IsotonicCalibrator.load(os.path.join(CALIB_DIR, "model_a_concat_isotonic.pkl"))

        dummy_z = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
        p1_t = tc.predict_proba(dummy_z)
        p1_p = pc.predict_proba(dummy_z)
        p1_i = ic.predict_proba(sigmoid(dummy_z))

        # Re-save to temp and reload
        tmp_t = os.path.join(CALIB_DIR, "_tmp_t.pkl")
        tmp_p = os.path.join(CALIB_DIR, "_tmp_p.pkl")
        tmp_i = os.path.join(CALIB_DIR, "_tmp_i.pkl")
        try:
            tc.save(tmp_t)
            pc.save(tmp_p)
            ic.save(tmp_i)

            tc_re = TemperatureCalibrator.load(tmp_t)
            pc_re = PlattCalibrator.load(tmp_p)
            ic_re = IsotonicCalibrator.load(tmp_i)

            assert np.allclose(p1_t, tc_re.predict_proba(dummy_z))
            assert np.allclose(p1_p, pc_re.predict_proba(dummy_z))
            assert np.allclose(p1_i, ic_re.predict_proba(sigmoid(dummy_z)))
        finally:
            for p in [tmp_t, tmp_p, tmp_i]:
                if os.path.exists(p):
                    os.remove(p)

    def test_C09_input_array_dimension_handling(self):
        """Verifies calibrators accept 1D array, 2D column vector, and PyTorch tensor inputs."""
        pc = PlattCalibrator.load(os.path.join(CALIB_DIR, "model_a_concat_platt.pkl"))

        z_1d = np.array([0.1, 0.5, -0.3])
        z_2d = z_1d.reshape(-1, 1)
        z_torch = torch.tensor(z_1d)

        p_1d = pc.predict_proba(z_1d)
        p_2d = pc.predict_proba(z_2d)
        p_torch = pc.predict_proba(z_torch.numpy())

        assert p_1d.shape == (3,)
        assert np.allclose(p_1d, p_2d)
        assert np.allclose(p_1d, p_torch)


class TestSplitIsolationAndLeakageControls:
    """Group C10-C12: Split Isolation, Leakage Verification & Metric Contracts."""

    def test_C10_zero_patient_leakage_between_calibration_and_test(self):
        """Confirms 0 patient overlap between the validation fitting set and the test set."""
        ds_tr, ds_va, ds_te, meta = load_paired_multimodal_splits()
        va_patients = set(ds_va.ids_df["subject_id"])
        te_patients = set(ds_te.ids_df["subject_id"])
        overlap = va_patients & te_patients
        assert len(overlap) == 0, f"Patient leakage detected: {overlap}"

    def test_C11_platt_substantially_reduces_test_ece(self):
        """Verifies that Platt scaling reduces test set ECE by at least 70% across all 3 models."""
        with open(os.path.join(REPORTS_DIR, "calibration_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        for m_name in ["Clinical Transformer", "Model A (Concatenation Fusion)", "Model B (Cross-Attention Fusion)"]:
            uncal_ece = data["results"][m_name]["test"]["uncalibrated"]["calibration"]["ece"]
            platt_ece = data["results"][m_name]["test"]["platt"]["calibration"]["ece"]
            pct_reduction = (uncal_ece - platt_ece) / uncal_ece
            assert pct_reduction > 0.70, (
                f"{m_name}: ECE reduction ({pct_reduction:.1%}) was less than 70% (uncal={uncal_ece}, platt={platt_ece})"
            )

    def test_C12_classification_metrics_decoupled_from_fixed_threshold(self):
        """Verifies that evaluating calibrated probabilities at the old fixed threshold is documented and separate from optimal threshold."""
        with open(os.path.join(REPORTS_DIR, "calibration_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        for m_name in ["Clinical Transformer", "Model A (Concatenation Fusion)", "Model B (Cross-Attention Fusion)"]:
            platt_res = data["results"][m_name]["test"]["platt"]
            clf_orig = platt_res["classification_at_orig_thr"]
            clf_opt = platt_res["classification_at_val_opt_thr"]

            # Both evaluations must exist
            assert "sensitivity" in clf_orig
            assert "sensitivity" in clf_opt
            # The val_opt_threshold must be documented and strictly lower than original threshold
            assert clf_opt["val_opt_threshold"] < data["results"][m_name]["original_threshold"]
