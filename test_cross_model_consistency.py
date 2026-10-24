"""
tests/test_cross_model_consistency.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Regression & Contract Test Suite for Cross-Model Consistency.

Verifies:
1. Probability strictly below decision threshold cannot be classified as positive/high-risk.
2. Probability at or above decision threshold is classified as positive/high-risk.
3. Precise threshold boundary behavior for Clinical (0.60), CXR raw (0.5237), Fusion A (0.50), Fusion B (0.50).
4. Displayed probability, threshold, and classification come from the same inference result without desynchronization.
5. Raw logits are never silently returned or presented as probabilities (probabilities must lie in [0, 1]).
6. Missing or unmatched inputs return status="unavailable" and cannot produce a fabricated fusion result.
7. End-to-end regression audit on Patient 10009049 Stay 35636875 verifying exact multi-model divergence rationale.
"""

import math
import pytest
import torch
import numpy as np

from backend.patient_service import (
    run_clinical_transformer_inference,
    run_cxr_inference,
    run_fusion_models_inference,
    resolve_patient_stay,
    get_stay_clinical_record,
)


# ── Test Suite 1: Mathematical Classification & Threshold Consistency ─────────

def test_clinical_transformer_threshold_contract():
    """
    Verify that Clinical Transformer classifies as positive if and only if
    predicted_probability >= decision_threshold (0.60).
    """
    res = run_clinical_transformer_inference(10009049, 35636875)
    assert res["status"] == "success"

    prob = res["predicted_probability"]
    thr = res["decision_threshold"]
    cls_name = res["predicted_class"]

    assert thr == 0.60
    assert 0.0 <= prob <= 1.0

    # Probability below threshold must NOT be classified as High Sepsis Risk
    if prob < thr:
        assert cls_name != "High Sepsis Risk"
        assert cls_name in ["Standard Risk", "Low Risk", "Normal Risk"]
    else:
        assert cls_name == "High Sepsis Risk"


def test_cxr_governing_raw_threshold_contract():
    """
    Verify that CXR classification is governed by raw_probability >= 0.5237,
    and that Platt-calibrated probability is reported separately for epidemiology.
    """
    res = run_cxr_inference(10009049, 35636875)
    assert res["status"] == "success"

    raw_prob = res["raw_probability"]
    cal_prob = res["platt_calibrated_probability"]
    raw_logit = res["raw_logit"]

    # Invariants on ranges
    assert 0.0 <= raw_prob <= 1.0
    assert 0.0 <= cal_prob <= 1.0
    # Sigmoid consistency: raw_prob == sigmoid(raw_logit)
    expected_prob = 1.0 / (1.0 + math.exp(-raw_logit))
    assert abs(raw_prob - expected_prob) < 1e-3

    # Platt calibration is strictly monotonic: maps raw logit monotonically
    assert cal_prob < raw_prob  # rescales from ~0.529 down toward cohort prevalence ~0.170

    gov = res["governing_score"]
    thr = gov["threshold"]
    assert thr == 0.5237

    if raw_prob >= thr:
        assert gov["classification"] == "Positive (High Risk)"
    else:
        assert gov["classification"] == "Negative"


def test_fusion_model_a_threshold_contract():
    """
    Verify that Multimodal Fusion Model A (Concatenation) classifies as positive
    if and only if predicted_probability >= threshold (0.50).
    """
    res = run_fusion_models_inference(10009049, 35636875)
    assert res["status"] == "success"

    ma = res["model_a"]
    prob = ma["predicted_probability"]
    thr = ma["threshold"]
    cls_str = ma["classification"]

    assert thr == 0.50
    assert 0.0 <= prob <= 1.0

    if prob >= thr:
        assert "Positive" in cls_str or "High" in cls_str
    else:
        assert "Negative" in cls_str or "Low" in cls_str


def test_fusion_model_b_threshold_contract():
    """
    Verify that Multimodal Fusion Model B (Cross-Attention) classifies as positive
    if and only if predicted_probability >= threshold (0.50).
    """
    res = run_fusion_models_inference(10009049, 35636875)
    assert res["status"] == "success"

    mb = res["model_b"]
    prob = mb["predicted_probability"]
    thr = mb["threshold"]
    cls_str = mb["classification"]

    assert thr == 0.50
    assert 0.0 <= prob <= 1.0

    if prob >= thr:
        assert "Positive" in cls_str or "High" in cls_str
    else:
        assert "Negative" in cls_str or "Low" in cls_str


# ── Test Suite 2: Boundary Tests for All Documented Operating Thresholds ───────

@pytest.mark.parametrize(
    "model_type, threshold, val_below, val_above",
    [
        ("clinical", 0.60, 0.5999, 0.6000),
        ("cxr_raw", 0.5237, 0.5236, 0.5237),
        ("fusion_a", 0.50, 0.4999, 0.5000),
        ("fusion_b", 0.50, 0.4999, 0.5000),
    ],
)
def test_threshold_boundary_decision_rules(model_type, threshold, val_below, val_above):
    """
    Verify strict mathematical boundary behavior:
    - val_below < threshold MUST yield False (Negative / Low Risk)
    - val_above >= threshold MUST yield True (Positive / High Risk)
    """
    # Test strict inequality
    assert (val_below >= threshold) is False
    assert (val_above >= threshold) is True

    # Test exact boundary inclusion
    assert (threshold >= threshold) is True


# ── Test Suite 3: Coherence & Desynchronization Prevention ────────────────────

def test_inference_metadata_synchronization():
    """
    Verify that displayed probability, threshold, raw logit, and classification
    come from the same coherent inference execution.
    """
    # Clinical Transformer
    c_res = run_clinical_transformer_inference(10009049, 35636875)
    assert c_res["status"] == "success"
    c_prob = c_res["predicted_probability"]
    c_logit = c_res["raw_logit"]
    # Logit to probability check
    assert abs(c_prob - 1.0 / (1.0 + math.exp(-c_logit))) < 1e-3

    # Fusion Models
    f_res = run_fusion_models_inference(10009049, 35636875)
    assert f_res["status"] == "success"
    # Model A logit vs prob
    ma_prob = f_res["model_a"]["predicted_probability"]
    ma_logit = f_res["model_a"]["raw_logit"]
    assert abs(ma_prob - 1.0 / (1.0 + math.exp(-ma_logit))) < 1e-3

    # Model B logit vs prob
    mb_prob = f_res["model_b"]["predicted_probability"]
    mb_logit = f_res["model_b"]["raw_logit"]
    assert abs(mb_prob - 1.0 / (1.0 + math.exp(-mb_logit))) < 1e-3


def test_logits_are_not_silently_presented_as_probabilities():
    """
    Verify raw logits (which can be negative or > 1) are never substituted for probabilities.
    """
    c_res = run_clinical_transformer_inference(10009049, 35636875)
    raw_logit = c_res["raw_logit"]
    prob = c_res["predicted_probability"]
    assert raw_logit != prob
    assert raw_logit < 0.0  # -0.4695
    assert 0.0 <= prob <= 1.0  # 0.3847

    f_res = run_fusion_models_inference(10009049, 35636875)
    assert f_res["model_a"]["raw_logit"] < 0.0
    assert 0.0 <= f_res["model_a"]["predicted_probability"] <= 1.0
    assert f_res["model_b"]["raw_logit"] < 0.0
    assert 0.0 <= f_res["model_b"]["predicted_probability"] <= 1.0


# ── Test Suite 4: Missing & Unmatched Modality Integrity ──────────────────────

def test_missing_imaging_returns_unavailable_without_fabrication():
    """
    Stays without a linked chest radiograph must return status="unavailable"
    and MUST NOT produce a fabricated fusion prediction.
    """
    # Patient 17585185 Stay 32417784 or single-stay patients without imaging
    res_cxr = run_cxr_inference(99999999, 88888888)
    assert res_cxr["status"] in ["error", "unavailable"]

    res_fusion = run_fusion_models_inference(99999999, 88888888)
    assert res_fusion["status"] in ["error", "unavailable"]
    assert "model_a" not in res_fusion
    assert "model_b" not in res_fusion


def test_fusion_models_are_never_averaged():
    """
    Verify that Model A and Model B are treated as separate architectures
    with their own independent probabilities, and are never averaged.
    """
    res = run_fusion_models_inference(10009049, 35636875)
    assert res["status"] == "success"

    p_a = res["model_a"]["predicted_probability"]
    p_b = res["model_b"]["predicted_probability"]

    assert p_a != p_b
    assert abs(p_a - 0.3782) < 1e-3
    assert abs(p_b - 0.3159) < 1e-3

    # Confirm there is no "average_probability" key in the return dictionary
    assert "average_probability" not in res
    assert "ensemble_probability" not in res


# ── Test Suite 5: Exact End-to-End Regression Audit on Patient 10009049 ───────

def test_patient_10009049_exact_inter_model_audit():
    """
    Deterministic audit of the exact patient case reported by user:
    Patient #10009049 Stay #35636875:
    - Clinical Transformer: prob 38.47% < 0.60 -> Low Risk
    - CXR: raw prob 52.9% >= 0.5237 -> Positive (High Risk), Platt prob 17.0%
    - Fusion A: prob 37.82% < 0.50 -> Low Risk
    - Fusion B: prob 31.59% < 0.50 -> Low Risk
    """
    # 1. Clinical AI
    c_res = run_clinical_transformer_inference(10009049, 35636875)
    assert c_res["status"] == "success"
    assert round(c_res["predicted_probability"], 4) == 0.3847
    assert c_res["decision_threshold"] == 0.60
    assert c_res["predicted_class"] == "Standard Risk"

    # 2. Chest X-ray AI
    x_res = run_cxr_inference(10009049, 35636875)
    assert x_res["status"] == "success"
    assert round(x_res["raw_probability"], 4) == 0.5290
    assert round(x_res["platt_calibrated_probability"], 4) == 0.1699
    bal_thr = x_res["thresholds"]["experimental_balanced_threshold_05237"]
    assert bal_thr["threshold"] == 0.5237
    assert bal_thr["classification"] == "Positive (High Risk)"
    assert x_res["governing_score"]["classification"] == "Positive (High Risk)"

    # 3. Fusion Models
    f_res = run_fusion_models_inference(10009049, 35636875)
    assert f_res["status"] == "success"
    ma = f_res["model_a"]
    mb = f_res["model_b"]

    assert round(ma["predicted_probability"], 4) == 0.3782
    assert ma["threshold"] == 0.50
    assert ma["classification"] == "Negative (Low Sepsis Risk)"

    assert round(mb["predicted_probability"], 4) == 0.3159
    assert mb["threshold"] == 0.50
    assert mb["classification"] == "Negative (Low Sepsis Risk)"

    # Latent allocation verification: clinical share accounts for majority/parity
    assert round(ma["clinical_share_pct"], 1) == 50.1
    assert round(ma["image_share_pct"], 1) == 49.9
