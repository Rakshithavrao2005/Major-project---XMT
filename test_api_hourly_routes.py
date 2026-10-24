"""
tests/test_api_hourly_routes.py
─────────────────────────────────────────────────────────────────────────────
Unit Test Suite for Phase 7 Hourly Status and API Prediction Routes.
Verifies that:
1. The hourly endpoint is explicitly marked unavailable (HTTP 503 with descriptive payload).
2. Unauthenticated requests are rejected with 401.
3. Unknown patients return 404.
4. System-wide hourly status returns 200 with prerequisite verification.
"""

import os
import sys
import time
import pytest
from flask import Flask

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.app import app
from src.auth.auth_manager import (
    provision_doctor,
    authenticate_doctor,
    create_session_token,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["REQUIRE_AUTH"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def valid_auth_headers():
    unique_id = int(time.time() * 1000)
    username = f"dr_hourly_{unique_id}"
    email = f"hourly.{unique_id}@hospital.org"
    password = "HourlySecurityPass@2026#"

    provision_doctor(
        username=username,
        email=email,
        full_name="Dr. Hourly Clinician",
        password=password,
        role="Attending Intensivist",
    )

    success, doctor, _ = authenticate_doctor(username, password)
    assert success is True
    token = create_session_token(doctor)
    return {"Authorization": f"Bearer {token}"}



def test_hourly_system_status_endpoint(client):
    """GET /api/hourly-status returns 200 with explicit unavailable status and prerequisites."""
    resp = client.get("/api/hourly-status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["feature_available"] is False
    assert "prerequisites" in data
    assert data["prerequisites"]["all_files_present"] is False


def test_hourly_prediction_unauthenticated_rejected(client):
    """Unauthenticated GET /api/patient/13269859/stay/30000001/hourly-prediction is rejected with 401."""
    resp = client.get("/api/patient/13269859/stay/30000001/hourly-prediction")
    assert resp.status_code == 401


def test_hourly_prediction_unknown_stay_returns_404(client, valid_auth_headers):
    """Unknown stay returns 404."""
    resp = client.get("/api/patient/99999999/stay/99999999/hourly-prediction", headers=valid_auth_headers)
    assert resp.status_code == 404


def test_hourly_prediction_returns_explicit_503_unavailable(client, valid_auth_headers):
    """
    Authenticated request for valid stay returns HTTP 503 Service Unavailable
    with explicit payload documenting that genuine hourly prediction is pending.
    """
    # Use known test stay
    pid = 13269859
    sid = 39782194  # Valid stay for 13269859

    resp = client.get(f"/api/patient/{pid}/stay/{sid}/hourly-prediction", headers=valid_auth_headers)
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["feature_available"] is False
    assert "Hourly next-hour sepsis prediction is currently UNAVAILABLE" in data["message"]
    assert data["governance"]["static_admission_model_available"] is True
    assert data["governance"]["static_admission_window_hours"] == 6.0


from unittest.mock import patch
from src.hourly.feature_extractor import CANONICAL_FEATURE_COLUMNS
from src.backend.patient_service import (
    get_hourly_predictor,
    reset_hourly_predictor_cache,
)


def _generate_synthetic_sequence(hours: int = 3, reverse_order: bool = False) -> list:
    """Helper to generate a valid synthetic canonical 184-feature hourly sequence."""
    sequence = []
    hour_range = list(range(hours, 0, -1)) if reverse_order else list(range(1, hours + 1))
    for h in hour_range:
        row = {col: 1.0 for col in CANONICAL_FEATURE_COLUMNS}
        row["hour_index"] = h
        sequence.append(row)
    return sequence


def test_hourly_predictor_singleton_lifecycle():
    """Verify singleton loads once per process and reuses the instance."""
    reset_hourly_predictor_cache()
    p1 = get_hourly_predictor()
    p2 = get_hourly_predictor()
    assert p1 is p2
    assert id(p1) == id(p2)

    reset_hourly_predictor_cache()
    p3 = get_hourly_predictor()
    assert p3 is not p1
    assert id(p3) != id(p1)


def test_successful_hourly_inference_valid_synthetic_sequence(client, valid_auth_headers):
    """
    POST /api/patient/<pid>/stay/<sid>/hourly-prediction with valid 184-feature sequence
    returns 200 with structured calibrated risk score, trajectory, and disclaimer.
    """
    pid = 13269859
    sid = 39782194
    seq = _generate_synthetic_sequence(hours=3)

    resp = client.post(
        f"/api/patient/{pid}/stay/{sid}/hourly-prediction",
        headers=valid_auth_headers,
        json={"sequence": seq},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["patient_id"] == pid
    assert data["stay_id"] == sid

    pred = data["prediction"]
    assert pred["sequence_length"] == 3
    assert pred["current_hour_index"] == 3
    assert 0.0 <= pred["risk_score"] <= 1.0
    assert len(pred["hourly_risk_trajectory"]) == 3
    assert pred["calibrator_type"] == "temperature"
    assert pred["decision_threshold"] == 0.90
    assert isinstance(pred["flagged_for_surveillance"], bool)
    assert "RESEARCH PROTOTYPE PREDICTION ONLY" in pred["disclaimer"]

    gov = data["governance"]
    assert gov["model_type"] == "HourlyTemporalTransformer"
    assert gov["parameter_count"] == 81153
    assert gov["calibrator_type"] == "temperature"
    assert gov["decision_threshold"] == 0.90


def test_incomplete_features_isolated_vitals_rejected(client, valid_auth_headers):
    """
    POST with isolated vitals (e.g. only HR/SpO2) without canonical 184 features
    is safely rejected with HTTP 400 Bad Request.
    """
    pid = 13269859
    sid = 39782194
    isolated_seq = [
        {"hour_index": 1, "hr_mean": 85.0, "spo2_mean": 97.0},
        {"hour_index": 2, "hr_mean": 92.0, "spo2_mean": 95.0},
    ]

    resp = client.post(
        f"/api/patient/{pid}/stay/{sid}/hourly-prediction",
        headers=valid_auth_headers,
        json={"sequence": isolated_seq},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "invalid_input"
    assert "Incomplete feature set rejected" in data["message"]
    assert "Isolated vitals" in data["message"]
    assert data["missing_column_count"] > 100


def test_empty_sequence_rejected(client, valid_auth_headers):
    """Empty sequence payload is rejected with HTTP 400."""
    pid = 13269859
    sid = 39782194

    resp = client.post(
        f"/api/patient/{pid}/stay/{sid}/hourly-prediction",
        headers=valid_auth_headers,
        json={"sequence": []},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "invalid_input"
    assert "empty" in data["message"].lower()


def test_non_chronological_sequence_rejected(client, valid_auth_headers):
    """Non-chronological sequence ordering is rejected with HTTP 400."""
    pid = 13269859
    sid = 39782194
    bad_seq = _generate_synthetic_sequence(hours=3, reverse_order=True)

    resp = client.post(
        f"/api/patient/{pid}/stay/{sid}/hourly-prediction",
        headers=valid_auth_headers,
        json={"sequence": bad_seq},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "invalid_input"
    assert "Non-chronological" in data["message"]


def test_sequence_length_exceeding_bound_rejected(client, valid_auth_headers):
    """Sequence length exceeding 120 hours is rejected with HTTP 400."""
    pid = 13269859
    sid = 39782194
    long_seq = _generate_synthetic_sequence(hours=121)

    resp = client.post(
        f"/api/patient/{pid}/stay/{sid}/hourly-prediction",
        headers=valid_auth_headers,
        json={"sequence": long_seq},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "invalid_input"
    assert "120" in data["message"]


def test_unauthorized_unknown_stay_rejected_with_payload(client, valid_auth_headers):
    """Unknown stay with sequence payload returns 404."""
    seq = _generate_synthetic_sequence(hours=2)
    resp = client.post(
        "/api/patient/99999999/stay/99999999/hourly-prediction",
        headers=valid_auth_headers,
        json={"sequence": seq},
    )
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["status"] == "error"


def test_model_initialization_failure_safe_503(client, valid_auth_headers):
    """
    When predictor initialization fails, endpoint returns HTTP 503
    without leaking internal file paths or stack traces.
    """
    pid = 13269859
    sid = 39782194
    seq = _generate_synthetic_sequence(hours=2)

    with patch("src.backend.patient_service.get_hourly_predictor", side_effect=RuntimeError("Simulated checkpoint load failure")):
        resp = client.post(
            f"/api/patient/{pid}/stay/{sid}/hourly-prediction",
            headers=valid_auth_headers,
            json={"sequence": seq},
        )
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "service_unavailable"
        assert "temporarily unavailable" in data["message"]
        # Ensure no leak of local system paths or traceback details
        assert "C:\\" not in data["message"]
        assert "Traceback" not in data["message"]
        assert "Simulated checkpoint load failure" not in data["message"]


def test_regression_existing_six_hour_models_unaffected(client, valid_auth_headers):
    """
    Regression check: Calling the 6-hour clinical transformer endpoint is 100% operational.
    """
    pid = 13269859
    sid = 39782194

    resp = client.get(
        f"/api/patient/{pid}/stay/{sid}/infer/clinical",
        headers=valid_auth_headers,
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["patient_id"] == pid
    assert data["stay_id"] == sid
    assert "raw_probability" in data
    assert "calibrated_probability" in data
    assert "top_attended_features" in data

