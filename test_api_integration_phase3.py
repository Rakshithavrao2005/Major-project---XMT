"""
tests/test_api_integration_phase3.py
─────────────────────────────────────────────────────────────────────────────
Automated Test Suite for Step 6B Phase 3: REST API Integration.

Validates the 4 REST endpoints added in Phase 3:
  1. Multimodal Fusion inference (/api/patient/<pid>/stay/<sid>/infer/fusion)
  2. Clinical SHAP explanation (/api/patient/<pid>/stay/<sid>/explain/clinical/shap)
  3. CXR Grad-CAM explanation (/api/patient/<pid>/stay/<sid>/explain/cxr/gradcam)
  4. qSOFA clinical screening assessment (/api/patient/<pid>/stay/<sid>/qsofa)

Test Groups:
  - Group P3_1: Authentication & Authorization Enforcement (401 vs Authenticated)
  - Group P3_2: Input Parameter Validation (nsamples bounds/type, alpha bounds/type)
  - Group P3_3: Patient/Stay Isolation & Not Found Handling (404)
  - Group P3_4: Multimodal Fusion Response Schema & Calibration (0.18, 0.20 thresholds)
  - Group P3_5: Clinical SHAP Response Schema (prediction_logit, additivity)
  - Group P3_6: CXR Grad-CAM Response Schema (overlay_file, layer4, disclaimer)
  - Group P3_7: qSOFA Clinical Semantics & Non-Diagnosis Disclaimers
"""

import os
import sys
import time
import pytest
import json

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.app import app
from src.auth.auth_manager import (
    provision_doctor,
    authenticate_doctor,
    create_session_token,
)


@pytest.fixture(scope="module")
def doctor_token():
    """Create a verified and active doctor session token for test execution."""
    unique_id = int(time.time() * 1000)
    username = f"dr_p3_{unique_id}"
    email = f"p3.{unique_id}@hospital.org"
    password = "ApiSecurityPass@2026#"

    provision_doctor(
        username=username,
        email=email,
        full_name="Dr. Phase 3 Clinician",
        password=password,
        role="Attending Intensivist",
    )

    success, doctor, _ = authenticate_doctor(username, password)
    assert success is True, "Doctor authentication failed in test fixture"
    return create_session_token(doctor)


class TestPhase3AuthEnforcement:
    """Group P3_1: Verifies all newly exposed endpoints enforce @require_doctor_auth."""

    @classmethod
    def setup_class(cls):
        cls.client = app.test_client()
        cls.patient_id = 10213765
        cls.stay_id = 36628477

    def test_unauthenticated_requests_rejected_with_401(self):
        """Endpoints must reject unauthenticated requests with HTTP 401 when REQUIRE_AUTH=True."""
        original_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = True
        try:
            endpoints = [
                ("GET", f"/api/patient/{self.patient_id}/stay/{self.stay_id}/infer/fusion"),
                ("POST", f"/api/patient/{self.patient_id}/stay/{self.stay_id}/infer/fusion"),
                ("GET", f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/clinical/shap"),
                ("POST", f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/clinical/shap"),
                ("GET", f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/cxr/gradcam"),
                ("POST", f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/cxr/gradcam"),
                ("GET", f"/api/patient/{self.patient_id}/stay/{self.stay_id}/qsofa"),
            ]
            for method, url in endpoints:
                if method == "GET":
                    resp = self.client.get(url)
                else:
                    resp = self.client.post(url)
                assert resp.status_code == 401, f"{method} {url} should return 401 when unauthenticated"
                data = resp.get_json()
                assert data["status"] == "error"
                assert "Authentication required" in data.get("error", "")
        finally:
            app.config["REQUIRE_AUTH"] = original_auth

    def test_authenticated_requests_succeed_with_bearer_token(self, doctor_token):
        """Endpoints must accept valid Bearer token and return 200."""
        original_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = True
        try:
            headers = {"Authorization": f"Bearer {doctor_token}"}
            # Test qSOFA endpoint with token
            resp = self.client.get(
                f"/api/patient/{self.patient_id}/stay/{self.stay_id}/qsofa",
                headers=headers,
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["patient_id"] == self.patient_id
            assert data["stay_id"] == self.stay_id
        finally:
            app.config["REQUIRE_AUTH"] = original_auth


class TestPhase3InputValidation:
    """Group P3_2: Input parameter validation on nsamples and alpha."""

    @classmethod
    def setup_class(cls):
        cls.client = app.test_client()
        cls.patient_id = 10213765
        cls.stay_id = 36628477
        cls._orig_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = cls._orig_auth

    def test_shap_invalid_nsamples_query_param(self):
        """SHAP rejects out-of-range or non-numeric nsamples via query parameters."""
        for bad_val in [-10, 0, 5, 501, 1000, "abc", "fifty"]:
            resp = self.client.get(
                f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/clinical/shap?nsamples={bad_val}"
            )
            assert resp.status_code == 400
            data = resp.get_json()
            assert data["status"] == "error"
            assert "Invalid parameter 'nsamples'" in data["message"]

    def test_shap_invalid_nsamples_json_body(self):
        """SHAP rejects out-of-range or non-numeric nsamples via JSON payload."""
        for bad_val in [-1, 2, 600, "invalid"]:
            resp = self.client.post(
                f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/clinical/shap",
                json={"nsamples": bad_val},
            )
            assert resp.status_code == 400
            data = resp.get_json()
            assert data["status"] == "error"
            assert "Invalid parameter 'nsamples'" in data["message"]

    def test_gradcam_invalid_alpha_query_param(self):
        """Grad-CAM rejects alpha values outside [0.0, 1.0] or non-numeric."""
        for bad_alpha in [-0.1, -1.0, 1.05, 2.0, "abc"]:
            resp = self.client.get(
                f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/cxr/gradcam?alpha={bad_alpha}"
            )
            assert resp.status_code == 400
            data = resp.get_json()
            assert data["status"] == "error"
            assert "Invalid parameter 'alpha'" in data["message"]

    def test_gradcam_invalid_alpha_json_body(self):
        """Grad-CAM rejects invalid alpha values in JSON payload."""
        for bad_alpha in [-0.5, 1.1, "bad"]:
            resp = self.client.post(
                f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/cxr/gradcam",
                json={"alpha": bad_alpha},
            )
            assert resp.status_code == 400
            data = resp.get_json()
            assert data["status"] == "error"
            assert "Invalid parameter 'alpha'" in data["message"]


class TestPhase3NotFoundHandling:
    """Group P3_3: Verifies 404 response for unknown patient/stay across all endpoints."""

    @classmethod
    def setup_class(cls):
        cls.client = app.test_client()
        cls._orig_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = cls._orig_auth

    def test_nonexistent_stay_returns_404(self):
        """All 4 endpoints must return 404 when requested patient or stay does not exist."""
        unknown_pid = 99999999
        unknown_sid = 88888888
        endpoints = [
            f"/api/patient/{unknown_pid}/stay/{unknown_sid}/infer/fusion",
            f"/api/patient/{unknown_pid}/stay/{unknown_sid}/explain/clinical/shap",
            f"/api/patient/{unknown_pid}/stay/{unknown_sid}/explain/cxr/gradcam",
            f"/api/patient/{unknown_pid}/stay/{unknown_sid}/qsofa",
        ]
        for url in endpoints:
            resp = self.client.get(url)
            assert resp.status_code == 404, f"{url} should return 404 for non-existent stay"
            data = resp.get_json()
            assert data["status"] == "error"
            assert "not found" in data["message"].lower()


class TestPhase3FusionResponseSchema:
    """Group P3_4: Multimodal Fusion inference schema and calibration alignment."""

    @classmethod
    def setup_class(cls):
        cls.client = app.test_client()
        cls.patient_id = 10213765
        cls.stay_id = 36628477
        cls._orig_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = cls._orig_auth

    def test_fusion_response_schema_and_probabilities(self):
        """Verifies fusion endpoint returns Model A & B with calibrated probabilities and thresholds."""
        resp = self.client.get(f"/api/patient/{self.patient_id}/stay/{self.stay_id}/infer/fusion")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["patient_id"] == self.patient_id
        assert data["stay_id"] == self.stay_id
        assert "model_a" in data
        assert "model_b" in data

        # Model A verification
        ma = data["model_a"]
        assert "raw_logit" in ma
        assert "raw_probability" in ma
        assert "platt_calibrated_probability" in ma
        assert ma["calibrated_decision_threshold"] == 0.18
        assert 0.0 <= ma["raw_probability"] <= 1.0
        assert 0.0 <= ma["platt_calibrated_probability"] <= 1.0
        assert ma["thresholds"]["calibrated_operating_threshold_018"]["threshold"] == 0.18
        assert ma["governing_calibrated_rule"]["threshold"] == 0.18

        # Model B verification
        mb = data["model_b"]
        assert "raw_logit" in mb
        assert "raw_probability" in mb
        assert "platt_calibrated_probability" in mb
        assert mb["calibrated_decision_threshold"] == 0.20
        assert 0.0 <= mb["raw_probability"] <= 1.0
        assert 0.0 <= mb["platt_calibrated_probability"] <= 1.0
        assert mb["thresholds"]["calibrated_operating_threshold_020"]["threshold"] == 0.20
        assert mb["governing_calibrated_rule"]["threshold"] == 0.20
        assert "top_attended_features" in mb

        # Zero target leakage
        assert "sepsis_label" not in data
        assert "sepsis_onset_time" not in data


class TestPhase3ClinicalSHAPResponseSchema:
    """Group P3_5: Clinical SHAP explanation response schema and additivity."""

    @classmethod
    def setup_class(cls):
        cls.client = app.test_client()
        cls.patient_id = 10213765
        cls.stay_id = 36628477
        cls._orig_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = cls._orig_auth

    def test_clinical_shap_response_fields(self):
        """Verifies SHAP endpoint returns prediction_logit, base_value, and additive top features."""
        resp = self.client.get(
            f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/clinical/shap?nsamples=30"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["patient_id"] == self.patient_id
        assert data["stay_id"] == self.stay_id
        assert "prediction_logit" in data
        assert "base_value" in data
        assert "raw_probability" in data
        assert "calibrated_probability" in data
        assert "top_features" in data
        assert "all_features" in data
        assert data["is_additive"] is True
        assert len(data["top_features"]) == 10

        # Does NOT return ambiguous model_output
        assert "model_output" not in data


class TestPhase3CXRGradCAMResponseSchema:
    """Group P3_6: CXR Grad-CAM activation heatmap and overlay response schema."""

    @classmethod
    def setup_class(cls):
        cls.client = app.test_client()
        cls.patient_id = 10213765
        cls.stay_id = 36628477
        cls._orig_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = cls._orig_auth

    def test_gradcam_response_fields_and_overlay(self):
        """Verifies Grad-CAM endpoint returns overlay file, target layer, and disclaimer."""
        resp = self.client.get(
            f"/api/patient/{self.patient_id}/stay/{self.stay_id}/explain/cxr/gradcam?alpha=0.45"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["patient_id"] == self.patient_id
        assert data["stay_id"] == self.stay_id
        assert "overlay_file" in data
        assert os.path.isfile(data["overlay_file"])
        assert data["target_layer"] == "layer4 (ResNet-18)"
        assert "raw_logit" in data
        assert "probability" in data
        assert "disclaimer" in data
        assert "GRAD-CAM EXPLAINABILITY DISCLAIMER" in data["disclaimer"]


class TestPhase3QSOFAClinicalSemantics:
    """Group P3_7: qSOFA screening semantics and non-diagnostic disclaimers."""

    @classmethod
    def setup_class(cls):
        cls.client = app.test_client()
        cls.patient_id = 10213765
        cls.stay_id = 36628477
        cls._orig_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = cls._orig_auth

    def test_qsofa_assessment_and_disclaimers(self):
        """Verifies qSOFA reports unavailable SBP and clear non-diagnosis disclaimer."""
        resp = self.client.get(f"/api/patient/{self.patient_id}/stay/{self.stay_id}/qsofa")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["patient_id"] == self.patient_id
        assert data["stay_id"] == self.stay_id
        assert data["status"] == "unavailable"
        assert data["is_complete"] is False

        # Criteria breakdown
        assert "criteria" in data
        criteria = data["criteria"]
        assert "systolic_blood_pressure" in criteria
        assert criteria["systolic_blood_pressure"]["available"] is False
        assert "MAP is NOT substituted for SBP" in criteria["systolic_blood_pressure"]["note"]

        # Clinical disclaimers
        assert "clinical_target_divergence" in data
        assert "data_integrity_notice" in data
        assert "bedside screening prompt" in data["clinical_target_divergence"].lower()
        assert "1-to-1 substitute for the ml prediction" in data["clinical_target_divergence"].lower()
