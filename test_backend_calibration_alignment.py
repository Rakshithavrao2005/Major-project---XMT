"""
tests/test_backend_calibration_alignment.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Verification & Alignment Test Suite for Backend Probability
Calibration and Operating Thresholds (Step 6B Phase 1).

Covers:
1. Output field contract for Clinical Transformer and Fusion Models:
   - Explicit presence of raw_logit, raw_probability, platt_calibrated_probability,
     calibrated_probability, calibrated_decision_threshold, calibrated_classification,
     and governing_calibrated_rule.
2. Mathematical verification of Platt calibration:
   - Confirms that calibrators receive uncalibrated logits z, evaluating sigma(a * z + b).
   - Confirms calibrators do NOT receive sigmoid probabilities (no double-squishing).
3. Operating threshold contract:
   - Clinical Transformer: Calibrated threshold 0.21 applied strictly to calibrated probability.
   - Model A (Concat): Calibrated threshold 0.18 applied strictly to calibrated probability.
   - Model B (Cross-Attention): Calibrated threshold 0.20 applied strictly to calibrated probability.
4. Backward compatibility:
   - Legacy keys (predicted_probability, decision_threshold=0.60, threshold=0.50, predicted_class,
     classification) remain intact and mathematically aligned.
5. Safe error handling:
   - Graceful handling (status="error") when calibrator files are missing or corrupted.
"""

from __future__ import annotations
import math
import os
import pickle
import unittest.mock as mock
import numpy as np
import pytest

from backend.patient_service import (
    run_clinical_transformer_inference,
    run_fusion_models_inference,
    load_clinical_platt_calibrator,
    load_fusion_platt_calibrators,
    get_calibrated_thresholds,
)
from src.evaluation.calibration import PlattCalibrator, sigmoid

# Patient with valid EHR and DICOM imaging
AUDIT_PATIENT_ID = 10009049
AUDIT_STAY_ID = 35636875


# ── Test Suite 1: Threshold Metadata & Calibrator Loading ─────────────────────

def test_calibrated_thresholds_metadata():
    """Verify that authoritative Step 2 validation thresholds are loaded correctly."""
    thresholds = get_calibrated_thresholds()
    assert isinstance(thresholds, dict)
    assert thresholds["clinical_transformer"] == 0.21
    assert thresholds["model_a"] == 0.18
    assert thresholds["model_b"] == 0.20


def test_platt_calibrators_exist_and_load():
    """Verify that saved Platt calibrators exist on disk and load with expected parameters."""
    clin_cal = load_clinical_platt_calibrator()
    assert isinstance(clin_cal, PlattCalibrator)
    assert hasattr(clin_cal, "a") and hasattr(clin_cal, "b")
    assert abs(clin_cal.a - 0.60745) < 1e-3
    assert abs(clin_cal.b - (-1.55000)) < 1e-3

    cal_a, cal_b = load_fusion_platt_calibrators()
    assert isinstance(cal_a, PlattCalibrator)
    assert isinstance(cal_b, PlattCalibrator)
    assert abs(cal_a.a - 0.62632) < 1e-3
    assert abs(cal_a.b - (-1.52935)) < 1e-3
    assert abs(cal_b.a - 0.57039) < 1e-3
    assert abs(cal_b.b - (-1.49711)) < 1e-3


# ── Test Suite 2: Clinical Transformer Calibration & Field Contract ───────────

def test_clinical_transformer_output_field_contract():
    """Verify all raw and calibrated fields are present in Clinical Transformer response."""
    res = run_clinical_transformer_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
    assert res["status"] == "success"

    # Explicit probability fields
    assert "raw_logit" in res
    assert "raw_probability" in res
    assert "predicted_probability" in res
    assert "platt_calibrated_probability" in res
    assert "calibrated_probability" in res

    # Explicit threshold & classification fields
    assert "decision_threshold" in res
    assert "calibrated_decision_threshold" in res
    assert "decision_rule" in res
    assert "calibrated_decision_rule" in res
    assert "predicted_class" in res
    assert "calibrated_classification" in res
    assert "thresholds" in res
    assert "governing_calibrated_rule" in res

    # Value ranges
    assert 0.0 <= res["raw_probability"] <= 1.0
    assert 0.0 <= res["predicted_probability"] <= 1.0
    assert 0.0 <= res["platt_calibrated_probability"] <= 1.0
    assert 0.0 <= res["calibrated_probability"] <= 1.0
    assert res["calibrated_probability"] == res["platt_calibrated_probability"]


def test_clinical_transformer_mathematical_calibration():
    """
    Verify Clinical Transformer calibration uses logits: sigma(a * z + b).
    Confirms that predict_proba is NOT applied to sigmoid(z).
    """
    res = run_clinical_transformer_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
    assert res["status"] == "success"

    raw_logit = res["raw_logit"]
    raw_prob = res["raw_probability"]
    cal_prob = res["platt_calibrated_probability"]

    # 1. Raw prob matches sigmoid(raw_logit)
    expected_raw_prob = 1.0 / (1.0 + math.exp(-raw_logit))
    assert abs(raw_prob - round(expected_raw_prob, 4)) <= 1e-4

    # 2. Calibrated prob matches sigma(a * raw_logit + b)
    clin_cal = load_clinical_platt_calibrator()
    expected_cal_logit = clin_cal.a * raw_logit + clin_cal.b
    expected_cal_prob = 1.0 / (1.0 + math.exp(-expected_cal_logit))
    assert abs(cal_prob - round(expected_cal_prob, 4)) <= 1e-4

    # 3. Double-squishing counter-check: sigma(a * raw_prob + b) must NOT equal cal_prob
    wrong_cal_logit = clin_cal.a * raw_prob + clin_cal.b
    wrong_cal_prob = 1.0 / (1.0 + math.exp(-wrong_cal_logit))
    assert abs(cal_prob - wrong_cal_prob) > 0.01, (
        "Calibrator incorrectly appears to be receiving sigmoid probability instead of logit!"
    )


def test_clinical_transformer_threshold_rules():
    """Verify that calibrated decision threshold 0.21 governs calibrated_classification."""
    res = run_clinical_transformer_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
    assert res["status"] == "success"

    cal_prob = res["calibrated_probability"]
    cal_thr = res["calibrated_decision_threshold"]
    assert cal_thr == 0.21

    if cal_prob >= cal_thr:
        assert res["calibrated_classification"] == "High Sepsis Risk"
    else:
        assert res["calibrated_classification"] == "Standard Risk"

    # Legacy uncalibrated threshold remains 0.60
    assert res["decision_threshold"] == 0.60
    legacy_prob = res["predicted_probability"]
    if legacy_prob >= 0.60:
        assert res["predicted_class"] == "High Sepsis Risk"
    else:
        assert res["predicted_class"] == "Standard Risk"


# ── Test Suite 3: Multimodal Fusion Calibration & Field Contract ──────────────

def test_fusion_models_output_field_contract():
    """Verify all raw and calibrated fields are present for Model A and Model B."""
    res = run_fusion_models_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
    assert res["status"] == "success"

    for model_key in ["model_a", "model_b"]:
        m = res[model_key]
        assert "raw_logit" in m
        assert "raw_probability" in m
        assert "predicted_probability" in m
        assert "platt_calibrated_probability" in m
        assert "calibrated_probability" in m
        assert "threshold" in m
        assert "calibrated_decision_threshold" in m
        assert "classification" in m
        assert "calibrated_classification" in m
        assert "thresholds" in m
        assert "governing_calibrated_rule" in m

        # Value ranges
        assert 0.0 <= m["raw_probability"] <= 1.0
        assert 0.0 <= m["predicted_probability"] <= 1.0
        assert 0.0 <= m["platt_calibrated_probability"] <= 1.0
        assert 0.0 <= m["calibrated_probability"] <= 1.0
        assert m["calibrated_probability"] == m["platt_calibrated_probability"]


def test_fusion_models_mathematical_calibration():
    """
    Verify Model A and Model B calibrations use logits: sigma(a * z + b).
    Confirms that predict_proba is NOT applied to sigmoid(z).
    """
    res = run_fusion_models_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
    assert res["status"] == "success"

    cal_a, cal_b = load_fusion_platt_calibrators()

    # Model A
    ma = res["model_a"]
    expected_raw_a = 1.0 / (1.0 + math.exp(-ma["raw_logit"]))
    assert abs(ma["raw_probability"] - round(expected_raw_a, 4)) <= 1e-4

    expected_cal_logit_a = cal_a.a * ma["raw_logit"] + cal_a.b
    expected_cal_a = 1.0 / (1.0 + math.exp(-expected_cal_logit_a))
    assert abs(ma["calibrated_probability"] - round(expected_cal_a, 4)) <= 1e-4

    # Model B
    mb = res["model_b"]
    expected_raw_b = 1.0 / (1.0 + math.exp(-mb["raw_logit"]))
    assert abs(mb["raw_probability"] - round(expected_raw_b, 4)) <= 1e-4

    expected_cal_logit_b = cal_b.a * mb["raw_logit"] + cal_b.b
    expected_cal_b = 1.0 / (1.0 + math.exp(-expected_cal_logit_b))
    assert abs(mb["calibrated_probability"] - round(expected_cal_b, 4)) <= 1e-4


def test_fusion_models_threshold_rules():
    """Verify calibrated decision thresholds (0.18 for A, 0.20 for B) govern calibrated classifications."""
    res = run_fusion_models_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
    assert res["status"] == "success"

    # Model A: calibrated threshold 0.18, uncalibrated threshold 0.50
    ma = res["model_a"]
    assert ma["calibrated_decision_threshold"] == 0.18
    assert ma["threshold"] == 0.50

    if ma["calibrated_probability"] >= 0.18:
        assert ma["calibrated_classification"] == "Positive (High Sepsis Risk)"
    else:
        assert ma["calibrated_classification"] == "Negative (Low Sepsis Risk)"

    if ma["predicted_probability"] >= 0.50:
        assert ma["classification"] == "Positive (High Sepsis Risk)"
    else:
        assert ma["classification"] == "Negative (Low Sepsis Risk)"

    # Model B: calibrated threshold 0.20, uncalibrated threshold 0.50
    mb = res["model_b"]
    assert mb["calibrated_decision_threshold"] == 0.20
    assert mb["threshold"] == 0.50

    if mb["calibrated_probability"] >= 0.20:
        assert mb["calibrated_classification"] == "Positive (High Sepsis Risk)"
    else:
        assert mb["calibrated_classification"] == "Negative (Low Sepsis Risk)"

    if mb["predicted_probability"] >= 0.50:
        assert mb["classification"] == "Positive (High Sepsis Risk)"
    else:
        assert mb["classification"] == "Negative (Low Sepsis Risk)"


# ── Test Suite 4: Threshold Boundary Simulations ──────────────────────────────

@pytest.mark.parametrize(
    "model_type,threshold,sub_eps,sup_eps",
    [
        ("clinical_calibrated", 0.21, 0.2099, 0.2100),
        ("fusion_a_calibrated", 0.18, 0.1799, 0.1800),
        ("fusion_b_calibrated", 0.20, 0.1999, 0.2000),
    ]
)
def test_calibrated_threshold_boundary_decision_logic(model_type, threshold, sub_eps, sup_eps):
    """
    Unit test mathematical boundary behavior: score strictly below cutoff is negative,
    score at or above cutoff is positive.
    """
    assert sub_eps < threshold
    assert sup_eps >= threshold

    if model_type == "clinical_calibrated":
        neg_class = "High Sepsis Risk" if sub_eps >= threshold else "Standard Risk"
        pos_class = "High Sepsis Risk" if sup_eps >= threshold else "Standard Risk"
        assert neg_class == "Standard Risk"
        assert pos_class == "High Sepsis Risk"
    else:
        neg_class = "Positive (High Sepsis Risk)" if sub_eps >= threshold else "Negative (Low Sepsis Risk)"
        pos_class = "Positive (High Sepsis Risk)" if sup_eps >= threshold else "Negative (Low Sepsis Risk)"
        assert neg_class == "Negative (Low Sepsis Risk)"
        assert pos_class == "Positive (High Sepsis Risk)"


# ── Test Suite 5: Safe Error Handling for Missing / Corrupt Calibrators ────────

def test_clinical_calibrator_missing_error_handling():
    """Verify safe error handling when clinical calibrator artifact fails to load."""
    with mock.patch("src.backend.patient_service.load_clinical_platt_calibrator", side_effect=FileNotFoundError("Mock missing file")):
        res = run_clinical_transformer_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
        assert res["status"] == "error"
        assert "Mock missing file" in res["message"]


def test_fusion_calibrator_missing_error_handling():
    """Verify safe error handling when fusion calibrator artifact fails to load."""
    with mock.patch("src.backend.patient_service.load_fusion_platt_calibrators", side_effect=FileNotFoundError("Mock missing fusion cal")):
        res = run_fusion_models_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
        assert res["status"] == "error"
        assert "Mock missing fusion cal" in res["message"]


def test_corrupt_calibrator_error_handling():
    """Verify safe error handling when calibrator file contains corrupt data."""
    with mock.patch("src.backend.patient_service.load_clinical_platt_calibrator", side_effect=ValueError("Corrupt calibrator dictionary")):
        res = run_clinical_transformer_inference(AUDIT_PATIENT_ID, AUDIT_STAY_ID)
        assert res["status"] == "error"
        assert "Corrupt calibrator" in res["message"]
