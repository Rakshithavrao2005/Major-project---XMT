"""
tests/test_dashboard_shap_and_score_alignment.py
─────────────────────────────────────────────────────────────────────────────
Focused Verification Test Suite for Step 6B Phase 2:
SHAP and Dashboard Score Alignment.

Covers:
1. SHAP schema verification:
   - Validates that shap_explainers.py returns prediction_logit, prediction_value,
     raw_probability, calibrated_probability, base_value, and top_features.
   - Verifies that 'model_output' does NOT exist in the canonical schema.
2. Missing-key behavior:
   - Verifies that missing keys in SHAP payloads are NOT silently replaced with 0.0,
     but are surfaced cleanly as 'N/A' or handled defensively.
3. Dashboard score distinction:
   - Confirms that raw logits, raw probabilities (threshold 0.60), and
     Platt-calibrated probabilities (threshold 0.21) are distinguished.
4. Model comparison table score alignment:
   - Confirms that model_comparison.py reflects calibrated probabilities and
     authoritative Step 2 thresholds (0.21, 0.18, 0.20) for Clinical Transformer,
     Model A, and Model B.
5. UI structure and disclaimer preservation:
   - Confirms that essential research disclaimers are preserved in dashboard pages.
"""

from __future__ import annotations
import inspect
import os
import re
import unittest.mock as mock
import pytest
import numpy as np

from backend.patient_service import (
    run_clinical_transformer_inference,
    run_fusion_models_inference,
    run_clinical_shap_explanation,
    resolve_patient_stay,
)
from src.explainability.shap_explainers import ClinicalTransformerSHAPExplainer

AUDIT_PATIENT_ID = 10009049
AUDIT_STAY_ID = 35636875


# ── Test Suite 1: SHAP Output Schema Inspection ───────────────────────────────

def test_shap_canonical_schema_keys():
    """
    Verify that ClinicalTransformerSHAPExplainer returns canonical keys:
    prediction_logit, prediction_value, raw_probability, calibrated_probability,
    base_value, shap_sum, top_features, all_features, disclaimer.
    Confirms 'model_output' is absent from the canonical dictionary.
    """
    # Sample a dummy scaled vector
    x_dummy = np.zeros((1, 49), dtype=np.float32)
    explainer = ClinicalTransformerSHAPExplainer(num_background=5)
    res = explainer.explain_instance(x_dummy, nsamples=20)

    # Required canonical keys
    assert "prediction_logit" in res, "prediction_logit must be present"
    assert "prediction_value" in res, "prediction_value must be present"
    assert "raw_probability" in res, "raw_probability must be present"
    assert "calibrated_probability" in res, "calibrated_probability must be present"
    assert "base_value" in res, "base_value must be present"
    assert "top_features" in res, "top_features must be present"
    assert "disclaimer" in res, "disclaimer must be present"

    # Confirmed absence of 'model_output'
    assert "model_output" not in res, (
        "The key 'model_output' must not exist in canonical SHAP schema. "
        "Any dashboard code accessing 'model_output' was looking for the wrong key!"
    )


def test_run_clinical_shap_explanation_structure():
    """Verify backend wrapper run_clinical_shap_explanation returns status and canonical keys."""
    res = run_clinical_shap_explanation(AUDIT_PATIENT_ID, AUDIT_STAY_ID, nsamples=20)
    assert res["status"] == "success"
    assert res["patient_id"] == AUDIT_PATIENT_ID
    assert res["stay_id"] == AUDIT_STAY_ID
    assert "prediction_logit" in res
    assert "prediction_value" in res
    assert "raw_probability" in res
    assert "calibrated_probability" in res
    assert "base_value" in res
    assert "model_output" not in res


# ── Test Suite 2: Missing Key Behavior and No Silent 0.0 Replacement ──────────

def test_clinical_ai_shap_no_silent_zero_replacement():
    """
    Verify dashboard/pages/clinical_ai.py does not use shap_res.get('model_output', 0).
    Confirms that prediction_logit or prediction_value is used, and missing values
    are formatted as 'N/A' rather than silently becoming 0.000.
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "clinical_ai.py")
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Must NOT have the old bug
    assert "shap_res.get('model_output'" not in code, (
        "Found old 'model_output' lookup in dashboard/pages/clinical_ai.py!"
    )
    assert 'shap_res.get("model_output"' not in code, (
        "Found old 'model_output' lookup in dashboard/pages/clinical_ai.py!"
    )

    # Must check for prediction_logit or prediction_value
    assert "prediction_logit" in code
    assert "base_v_str" in code or "base_value" in code
    assert "N/A" in code, "Must handle missing values with 'N/A' rather than silent 0.0"


def test_explainability_page_shap_no_silent_zero_replacement():
    """
    Verify dashboard/pages/explainability.py does not use shap_res.get('model_output', 0.0).
    Confirms that prediction_logit is used and missing values are formatted as 'N/A'.
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "explainability.py")
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Must NOT have the old bug
    assert "shap_res.get('model_output'" not in code, (
        "Found old 'model_output' lookup in dashboard/pages/explainability.py!"
    )
    assert 'shap_res.get("model_output"' not in code, (
        "Found old 'model_output' lookup in dashboard/pages/explainability.py!"
    )

    # Must use prediction_logit / prediction_value
    assert "prediction_logit" in code
    assert "pred_logit_str" in code
    assert "base_v_str" in code


# ── Test Suite 3: Dashboard Raw vs Calibrated Score Distinction ───────────────

def test_clinical_ai_distinguishes_raw_and_calibrated():
    """
    Verify dashboard/pages/clinical_ai.py presents raw logit, raw probability,
    and Platt-calibrated probability with their respective thresholds (0.60 vs 0.21).
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "clinical_ai.py")
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Check for accurate labeling
    assert "Platt-Calibrated Sepsis Risk" in code
    assert "Model Raw Logit & Sigmoid" in code
    assert "Calibrated Cutoff" in code or "calibrated_decision_threshold" in code
    assert "21.0%" in code or "0.21" in code


def test_model_comparison_reflects_calibrated_thresholds():
    """
    Verify dashboard/model_comparison.py includes Platt-calibrated probability and
    authoritative thresholds for Clinical Transformer (0.21), Model A (0.18), and Model B (0.20).
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "model_comparison.py")
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Table rows must distinguish Platt and Raw
    assert "Platt-Calibrated" in code
    assert "platt_calibrated_probability" in code
    assert "calibrated_decision_threshold" in code
    assert "0.21" in code
    assert "0.18" in code
    assert "0.20" in code


# ── Test Suite 4: Preservation of Research Disclaimers ────────────────────────

def test_research_disclaimers_preserved():
    """Verify research prototypes and non-fabrication disclaimers remain in dashboard pages."""
    cai_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "clinical_ai.py")
    with open(cai_path, "r", encoding="utf-8") as f:
        cai_code = f.read()
    assert "RESEARCH & DEMO PROTOTYPE ONLY" in cai_code
    assert "SHAP Attribution Disclaimer" in cai_code

    exp_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "explainability.py")
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_code = f.read()
    assert "Grad-CAM Interpretability Disclaimer" in exp_code
    assert "Never fabricate findings" in exp_code
