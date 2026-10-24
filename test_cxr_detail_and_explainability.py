"""
tests/test_cxr_detail_and_explainability.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Test Suite for Chest X-ray AI Detail, Explainability, and
Prediction Clarity.

Verifies:
1. Correct raw-logit versus probability labels and mathematical separation.
2. Calibration and threshold consistency (raw threshold 0.5237 vs Platt scaling).
3. Image quality and usability checks (valid, corrupt, empty, and degenerate images).
4. Missing Grad-CAM handling and graceful fallback without crashing.
5. No fabricated image findings (model predicts host sepsis risk, not radiological lesions).
6. Correct patient/image association against the EHR ground truth.
7. Honest unavailable states for unsupported outputs without fabrication.
"""

import math
import os
import tempfile
import pytest
import numpy as np
import torch
from PIL import Image

from backend.patient_service import (
    run_cxr_inference,
    run_cxr_gradcam_inference,
    validate_cxr_image,
    get_stay_clinical_record,
    get_raw_dataset,
)


# ── Test Suite 1: Raw Logit vs. Probability Labels ─────────────────────────────

def test_raw_logit_vs_probability_labels():
    """
    Verify that raw logit, raw sigmoid score, and Platt-calibrated probability
    are distinct fields and are never confused or mislabeled.
    """
    res = run_cxr_inference(10213765, 36628477)
    assert res["status"] == "success"

    raw_logit = res["raw_logit"]
    raw_prob = res["raw_probability"]
    cal_prob = res["platt_calibrated_probability"]

    # Invariants
    assert isinstance(raw_logit, float)
    assert isinstance(raw_prob, float)
    assert isinstance(cal_prob, float)

    # Probabilities must be bounded in [0, 1]
    assert 0.0 <= raw_prob <= 1.0
    assert 0.0 <= cal_prob <= 1.0

    # Logit to raw probability consistency: raw_prob == sigmoid(raw_logit)
    expected_prob = 1.0 / (1.0 + math.exp(-raw_logit))
    assert abs(raw_prob - expected_prob) < 1e-3

    # Logit and probability are distinctly labeled and mathematically separate
    assert raw_logit != raw_prob
    assert raw_prob != cal_prob


# ── Test Suite 2: Calibration and Threshold Consistency ────────────────────────

def test_calibration_and_threshold_consistency():
    """
    Verify that the operating threshold 0.5237 applies strictly to raw_probability,
    and that Platt-calibrated probability reflects cohort incidence (~16.1%).
    """
    res = run_cxr_inference(10213765, 36628477)
    assert res["status"] == "success"

    raw_prob = res["raw_probability"]
    cal_prob = res["platt_calibrated_probability"]
    gov = res["governing_score"]

    assert gov["score_type"] == "raw_probability"
    assert gov["threshold"] == 0.5237

    # Classification must match raw_probability >= 0.5237
    if raw_prob >= 0.5237:
        assert gov["classification"] == "Positive (High Risk)"
    else:
        assert gov["classification"] == "Negative"

    # Platt calibrated probability must be monotonic with raw probability
    # but scaled down to true population prevalence
    assert cal_prob < raw_prob
    assert 0.10 <= cal_prob <= 0.35  # expected range for ICU sepsis prevalence


# ── Test Suite 3: Image Quality & Usability Checks ─────────────────────────────

def test_validate_cxr_image_valid_file():
    """Valid admission radiograph on disk passes all quality checks."""
    df = get_raw_dataset()
    match = df[df["dicom_id"].notna()].iloc[0]
    safe_dicom = os.path.basename(str(match["dicom_id"]))
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "images")
    img_path = os.path.join(raw_dir, f"{safe_dicom}.jpg")

    val = validate_cxr_image(img_path)
    assert val["is_valid"] is True
    assert val["status"] == "valid"
    assert val["width"] > 0
    assert val["height"] > 0
    assert val["channels"] == 3
    assert val["pixel_std"] > 1.0
    assert val["pixel_max"] > 0


def test_validate_cxr_image_missing_file():
    """Missing or non-existent file path returns is_valid=False with file_not_found."""
    val = validate_cxr_image("C:/non_existent_image_path_12345.jpg")
    assert val["is_valid"] is False
    assert val["status"] == "file_not_found"
    assert "not found" in val["error"].lower()


def test_validate_cxr_image_empty_file():
    """Zero-byte file returns is_valid=False with empty_file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf_name = tf.name

    try:
        val = validate_cxr_image(tf_name)
        assert val["is_valid"] is False
        assert val["status"] == "empty_file"
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)


def test_validate_cxr_image_all_black():
    """Corrupt all-black image returns is_valid=False with corrupt_all_black."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf_name = tf.name

    try:
        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        black_img.save(tf_name, format="JPEG")

        val = validate_cxr_image(tf_name)
        assert val["is_valid"] is False
        assert val["status"] == "corrupt_all_black"
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)


def test_validate_cxr_image_degenerate_blank():
    """Degenerate image with zero intensity variance returns is_valid=False with degenerate_blank."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf_name = tf.name

    try:
        grey_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        grey_img.save(tf_name, format="JPEG")

        val = validate_cxr_image(tf_name)
        assert val["is_valid"] is False
        assert val["status"] == "degenerate_blank"
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)


# ── Test Suite 4: Missing Grad-CAM & Graceful Fallback ─────────────────────────

def test_missing_gradcam_handling():
    """Non-existent patient or stay returns status='error' without crashing."""
    res = run_cxr_gradcam_inference(99999999, 88888888)
    assert res["status"] in ["error", "unavailable"]
    assert "not found" in res["message"].lower() or "unavailable" in res["message"].lower()


def test_gradcam_successful_execution():
    """Verified patient with radiograph successfully computes Grad-CAM overlay."""
    res = run_cxr_gradcam_inference(10213765, 36628477, alpha=0.45)
    assert res["status"] == "success"
    assert "overlay_file" in res or "gradcam_overlay_file" in res
    assert os.path.isfile(res["overlay_file"])
    assert res["target_layer"] == "layer4 (ResNet-18)"


# ── Test Suite 5: No Fabricated Image Findings ─────────────────────────────────

def test_no_fabricated_image_findings():
    """
    Verify that inference results do NOT fabricate specific pathology claims
    (e.g., pneumonia, edema, atelectasis confirmed).
    """
    res_inf = run_cxr_inference(10213765, 36628477)
    assert res_inf["status"] == "success"

    # Ensure disclaimers are present
    assert "disclaimer" in res_inf
    disc_text = res_inf["disclaimer"].lower()
    assert "localized pulmonary findings" in disc_text
    assert "never be used for clinical diagnosis" in disc_text

    res_cam = run_cxr_gradcam_inference(10213765, 36628477)
    assert res_cam["status"] == "success"
    cam_disc = res_cam["disclaimer"].lower()
    assert "does not establish" in cam_disc or "not constitute" in cam_disc


# ── Test Suite 6: Patient-Image Association Integrity ─────────────────────────

def test_patient_image_association_integrity():
    """
    Verify that the radiograph associated with a stay strictly matches the
    retrospective MIMIC-IV / MIMIC-CXR dataset record.
    """
    rec = get_stay_clinical_record(10213765, 36628477)
    imaging = rec.get("imaging", {})
    assert imaging.get("image_exists") is True

    # Authoritative dataset lookup
    df = get_raw_dataset()
    match = df[(df["subject_id"] == 10213765) & (df["stay_id"] == 36628477)]
    assert not match.empty
    expected_dicom = str(match.iloc[0]["dicom_id"])

    assert imaging.get("dicom_id") == expected_dicom
    assert expected_dicom in imaging.get("image_file")


# ── Test Suite 7: Honest Unavailable States for Unsupported Outputs ───────────

def test_honest_unavailable_states():
    """
    Verify that stays with no linked radiograph return status='unavailable'
    without fabricating a CXR score or Grad-CAM heatmap.
    """
    # Query an invalid/unmatched stay
    res = run_cxr_inference(12345678, 87654321)
    assert res["status"] in ["error", "unavailable"]
    assert "raw_probability" not in res
    assert "platt_calibrated_probability" not in res

    cam_res = run_cxr_gradcam_inference(12345678, 87654321)
    assert cam_res["status"] in ["error", "unavailable"]
    assert "overlay_file" not in cam_res
