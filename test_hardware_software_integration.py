"""
tests/test_hardware_software_integration.py
─────────────────────────────────────────────────────────────────────────────
Automated Integration Tests for XMT Telemedicine Dashboard:
- Separation of SOFTWARE (Historical EHR Cohort) and HARDWARE (Live Sensor Bench).
- Real-time ESP32/MAX30102 telemetry retrieval, staleness detection, and quality flags.
- Temporary monitored session lifecycle and isolation from hospital patient records.
- Session-scoped Chest X-ray file upload validation (formats, size limit, corrupt payloads).
- Unimodal ResNet-18 CXR inference on uploaded image with dual operating points & disclaimer.
- Strict clinical model unavailability safeguard when 49 clinical features are missing.
- Verification that no API keys or IoT credentials leak into telemetry or session payloads.
"""

from __future__ import annotations
import os
import sys
import io
import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.live_session import (
    start_monitoring_session,
    stop_monitoring_session,
    get_monitoring_session,
    list_monitoring_sessions,
    get_latest_sensor_telemetry,
    validate_uploaded_image,
    run_session_cxr_inference,
    load_cached_cxr_model,
    generate_session_id,
    MAX_UPLOAD_BYTES,
    ALLOWED_EXTENSIONS,
    _ACTIVE_SESSIONS,
)
from src.sensor.validate_clean import CLEAN_CSV
from backend.patient_service import (
    search_patient,
    get_stay_clinical_record,
    run_clinical_transformer_inference,
    get_sensor_telemetry_status,
)


class TestSoftwareHardwareSeparation:
    """Group L01-L05: Strict Separation between Historical EHR and Live Hardware."""

    def test_L01_hardware_telemetry_contains_no_hospital_patient_id(self):
        """Live sensor telemetry dict must never expose subject_id or stay_id."""
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)
        assert telemetry["status"] == "success"
        # Must not contain EHR identifiers
        assert "subject_id" not in telemetry
        assert "stay_id" not in telemetry
        assert "hadm_id" not in telemetry
        assert "patient_id" not in telemetry

    def test_L02_session_identifiers_are_distinct_from_hospital_patient_ids(self):
        """Generated session IDs must follow SESSION- format and cannot match numeric subject_id."""
        sid = generate_session_id()
        assert sid.startswith("SESSION-")
        # Must not be a pure integer like MIMIC subject_ids (e.g. 17585185)
        assert not sid.isdigit()

    def test_L03_session_lifecycle_and_isolation(self):
        """Starting and stopping sessions maintains isolated records without mixing state."""
        sid1 = f"SESSION-TEST-A-{os.urandom(4).hex()}"
        sid2 = f"SESSION-TEST-B-{os.urandom(4).hex()}"

        s1 = start_monitoring_session(sid1)
        s2 = start_monitoring_session(sid2)

        assert s1["session_id"] == sid1
        assert s2["session_id"] == sid2
        assert s1["status"] == "active"
        assert s2["status"] == "active"

        # End s1
        stopped_s1 = stop_monitoring_session(sid1)
        assert stopped_s1["status"] == "stopped"
        assert stopped_s1["end_time"] is not None

        # s2 must remain active
        fetched_s2 = get_monitoring_session(sid2)
        assert fetched_s2["status"] == "active"
        assert fetched_s2["end_time"] is None

        # Clean up
        _ACTIVE_SESSIONS.pop(sid1, None)
        _ACTIVE_SESSIONS.pop(sid2, None)

    def test_L04_telemetry_isolation_status_in_patient_service(self):
        """Authoritative patient service confirms hardware telemetry is decoupled from EHR."""
        status = get_sensor_telemetry_status()
        assert status["clean_records_count"] > 0
        assert "NOT associated with any MIMIC-IV hospital patient records" in status["isolation_guarantee"]

    def test_L05_historical_patient_search_does_not_contain_sensor_telemetry(self):
        """Historical patient search returns EHR stays and never injects live hardware readings."""
        res = search_patient(17585185)
        assert res["found"] is True
        for stay in res["stays"]:
            # Stays contain MIMIC-IV clinical timestamps, not ThingSpeak feeds
            assert "intime" in stay
            assert "outtime" in stay
            assert "entry_id" not in stay


class TestSensorTelemetryMonitoring:
    """Group M01-M06: Live Telemetry Ingestion, Quality Indicators, and Staleness."""

    def test_M01_telemetry_data_structure_and_types(self):
        """Telemetry output contains all required physiological and quality fields."""
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)
        assert telemetry["status"] == "success"
        assert telemetry["connected"] is True
        assert telemetry["has_readings"] is True
        assert isinstance(telemetry["timestamp"], str)
        assert isinstance(telemetry["is_stale"], bool)
        assert isinstance(telemetry["staleness_description"], str)
        assert isinstance(telemetry["entry_id"], int)
        assert isinstance(telemetry["heart_rate"], float)
        assert isinstance(telemetry["spo2"], float)
        assert telemetry["ir_ppg"] is None or isinstance(telemetry["ir_ppg"], float)
        assert isinstance(telemetry["risk_status_code"], int)
        assert isinstance(telemetry["firmware_status"], str)
        assert isinstance(telemetry["valid_hr"], bool)
        assert isinstance(telemetry["valid_spo2"], bool)
        assert isinstance(telemetry["finger_detected"], bool)
        assert isinstance(telemetry["physiologically_valid"], bool)
        assert isinstance(telemetry["recent_history"], pd.DataFrame)
        assert "HARDWARE BENCH TELEMETRY" in telemetry["disclaimer"]

    def test_M02_telemetry_recent_history_bounded(self):
        """Telemetry recent history is a dataframe bounded to at most 50 rows."""
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)
        history = telemetry["recent_history"]
        assert len(history) <= 50
        assert "heart_rate" in history.columns
        assert "spo2" in history.columns
        assert "ir_ppg" in history.columns
        assert "timestamp" in history.columns

    def test_M03_firmware_status_rule_verification(self):
        """Firmware status code matches the hardcoded ESP32 threshold rule (HR>=130 & SpO2<92%)."""
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)
        code = telemetry["risk_status_code"]
        if code == 2:
            assert "HIGH RISK" in telemetry["firmware_status"]
            assert "HR >= 130 and SpO2 < 92%" in telemetry["firmware_status"]
        elif code == 0:
            assert "NORMAL" in telemetry["firmware_status"]

    def test_M04_telemetry_quality_flags_logical_consistency(self):
        """Optical quality flags: physiologically_valid implies valid_hr and valid_spo2."""
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)
        if telemetry["physiologically_valid"]:
            assert telemetry["valid_hr"] is True
            assert telemetry["valid_spo2"] is True
            assert telemetry["finger_detected"] is True

    def test_M05_no_credentials_leaked_in_telemetry_response(self):
        """Telemetry response must never leak API keys, tokens, or environment passwords."""
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)
        keys = ["api_key", "thingspeak_api_key", "password", "token", "secret"]
        for k in keys:
            assert k not in telemetry
            assert k.upper() not in telemetry


class TestImageUploadAndCXRInference:
    """Group N01-N08: Upload Validation, Format Protection, and CXREncoder Inference."""

    def test_N01_validate_valid_jpeg_image(self):
        """Valid JPEG byte stream is accepted and converted to RGB."""
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        is_valid, pil_img, err = validate_uploaded_image(raw_bytes, "chest_xray_test.jpg")
        assert is_valid is True
        assert pil_img is not None
        assert err == ""
        assert pil_img.size == (300, 300)
        assert pil_img.mode == "RGB"

    def test_N02_validate_valid_png_image(self):
        """Valid PNG byte stream is accepted and converted to RGB."""
        img = Image.new("RGBA", (256, 256), color=(64, 64, 64, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        is_valid, pil_img, err = validate_uploaded_image(raw_bytes, "admission_scan.png")
        assert is_valid is True
        assert pil_img is not None
        assert err == ""
        assert pil_img.size == (256, 256)
        assert pil_img.mode == "RGB"

    def test_N03_reject_empty_payload(self):
        """Empty byte payload is rejected immediately."""
        is_valid, pil_img, err = validate_uploaded_image(b"", "empty.jpg")
        assert is_valid is False
        assert pil_img is None
        assert "empty" in err.lower()

    def test_N04_reject_oversized_file(self):
        """Files exceeding the 15 MB threshold are rejected before decoding."""
        oversized_bytes = b"0" * (MAX_UPLOAD_BYTES + 1024)
        is_valid, pil_img, err = validate_uploaded_image(oversized_bytes, "giant_scan.jpg")
        assert is_valid is False
        assert pil_img is None
        assert "exceeds maximum limit of 15 MB" in err

    def test_N05_reject_unsupported_extensions(self):
        """Non-image extensions (.pdf, .dcm, .txt, .exe) are rejected."""
        dummy_bytes = b"test payload"
        for bad_ext in ["scan.pdf", "image.dcm", "notes.txt", "tool.exe"]:
            is_valid, pil_img, err = validate_uploaded_image(dummy_bytes, bad_ext)
            assert is_valid is False
            assert pil_img is None
            assert "Unsupported file extension" in err

    def test_N06_reject_corrupted_image_file(self):
        """Corrupted image bytes pretending to be JPEG are safely caught by PIL verify."""
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"CORRUPTED_NON_IMAGE_DATA_PADDING" * 100
        is_valid, pil_img, err = validate_uploaded_image(fake_jpeg, "corrupt.jpg")
        assert is_valid is False
        assert pil_img is None
        assert "Failed to decode image file" in err

    def test_N07_cxr_inference_on_valid_image(self):
        """CXREncoder executes deterministic forward pass with dual thresholds & disclaimer."""
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        result = run_session_cxr_inference(img)

        assert result["status"] == "success"
        assert result["model_name"] == "Unimodal CXREncoder (ResNet-18 Backbone)"
        assert isinstance(result["raw_logit"], float)
        assert 0.0 <= result["raw_probability"] <= 1.0
        assert 0.0 <= result["platt_calibrated_probability"] <= 1.0

        # Check dual operating points
        thrs = result["thresholds"]
        assert "original_threshold_010" in thrs
        assert "experimental_balanced_threshold_05237" in thrs
        assert thrs["original_threshold_010"]["threshold"] == 0.10
        assert thrs["experimental_balanced_threshold_05237"]["threshold"] == 0.5237

        # Check disclaimer
        assert "EXPERIMENTAL RESEARCH DEMO ONLY" in result["disclaimer"]
        assert "ROC-AUC: 0.6131" in result["disclaimer"]

    def test_N08_cxr_inference_rejects_none_image(self):
        """CXREncoder cleanly returns error status when image is None."""
        result = run_session_cxr_inference(None)
        assert result["status"] == "error"
        assert "No valid image" in result["message"]


class TestClinicalModelExclusionSafeguard:
    """Group O01-O03: Strict Policy on Clinical Transformer Unavailability for Sensor Sessions."""

    def test_O01_clinical_transformer_requires_49_features(self):
        """Clinical Transformer inference requires the full 49 EHR features."""
        # Sensor data has only 3 fields: HR, SpO2, IR PPG. Cannot run Transformer.
        invalid_feature_dict = {"heart_rate": 80.0, "spo2": 98.0, "ir_ppg": 45000.0}
        with pytest.raises(Exception):
            run_clinical_transformer_inference(invalid_feature_dict)

    def test_O02_no_synthetic_feature_fabrication_for_live_session(self):
        """Verifies that live session functions do not invent or impute clinical features."""
        import inspect
        from src.sensor import live_session
        source = inspect.getsource(live_session)
        # Verify no fake clinical feature dictionaries or random feature generations
        assert "creatinine" not in source.lower()
        assert "bilirubin" not in source.lower()
        assert "platelets" not in source.lower()
        assert "synthetic" not in source.lower()

    def test_O03_no_unified_ai_risk_score_in_session_output(self):
        """Verify that live session CXR inference and sensor telemetry do not produce unified risk."""
        img = Image.new("RGB", (224, 224), color=(120, 120, 120))
        cxr_res = run_session_cxr_inference(img)
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)

        # Neither produces a "multimodal_risk_score" or "combined_sepsis_score"
        assert "multimodal_risk_score" not in cxr_res
        assert "combined_risk_score" not in cxr_res
        assert "multimodal_risk_score" not in telemetry
        assert "combined_risk_score" not in telemetry
