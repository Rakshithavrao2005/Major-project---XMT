"""
tests/test_dashboard_hourly.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Unit & Integration Tests for Streamlit Dashboard Longitudinal Integration.

Verifies:
1. Correct rendering of valid backend responses (risk score, metrics, trajectory chart).
2. Hourly trajectory data alignment, timestamps, and 0.90 threshold line.
3. Unavailable-stream responses (HTTP 503) with zero data fabrication.
4. HTTP 400, 401/403, 404, and 503 handling without internal error leakage.
5. Defensive handling of missing or malformed response fields.
6. Research disclaimer visibility across all prediction and status states.
7. Preservation of existing six-hour model functionality and test suites.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dashboard.hourly_prediction_view import (
    RESEARCH_DISCLAIMER_TEXT,
    fetch_backend_hourly_prediction,
    render_hourly_prediction_section,
)
from backend.patient_service import (
    run_clinical_transformer_inference,
    resolve_patient_stay,
    predict_stay_hourly_sepsis,
)
from src.hourly.retrospective_adapter import DEFAULT_CACHE_DIR, _resolve_stay_partition

# Synthetic anonymous IDs for tests (NO real patient identifiers)
MOCK_PID = 12345
MOCK_SID = 67890


def _build_synthetic_200_response(
    seq_len: int = 5,
    risk_score: float = 0.125,
    flagged: bool = False,
    threshold: float = 0.90,
    trajectory: list | None = None,
) -> Dict[str, Any]:
    """Helper to build a valid canonical Stage 3.5.3 backend response."""
    if trajectory is None:
        trajectory = [0.05, 0.07, 0.09, 0.11, risk_score]
    return {
        "status": "success",
        "patient_id": MOCK_PID,
        "stay_id": MOCK_SID,
        "prediction": {
            "sequence_length": seq_len,
            "current_hour_index": seq_len,
            "risk_score": risk_score,
            "hourly_risk_trajectory": trajectory,
            "raw_logits": [-2.5, -2.2, -2.0, -1.8, -1.5],
            "calibrator_type": "temperature",
            "decision_threshold": threshold,
            "flagged_for_surveillance": flagged,
            "disclaimer": RESEARCH_DISCLAIMER_TEXT,
        },
        "governance": {
            "model_type": "HourlyTemporalTransformer",
            "parameter_count": 81153,
            "calibrator_type": "temperature",
            "decision_threshold": threshold,
            "research_disclaimer": RESEARCH_DISCLAIMER_TEXT,
        },
    }


# ── Test Suite 1: Backend Client Fetching ────────────────────────────────────

def test_fetch_backend_inprocess_fallback():
    """Verify in-process fallback when local REST server is not running."""
    # Stay without longitudinal sequence returns 503 unavailable
    code, data = fetch_backend_hourly_prediction(
        patient_id=13269859,
        stay_id=39782194,
        backend_url="http://127.0.0.1:59999",  # unreachable port
    )
    assert code == 503
    assert data["status"] == "unavailable"
    assert "UNAVAILABLE" in data["message"]


def test_fetch_backend_rest_success_mocked():
    """Verify REST fetch when backend returns HTTP 200."""
    synthetic_payload = _build_synthetic_200_response()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = synthetic_payload

    with patch("requests.get", return_value=mock_resp) as mock_get:
        code, data = fetch_backend_hourly_prediction(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            session_token="mock_doc_token",
            backend_url="http://127.0.0.1:5000",
        )
        assert code == 200
        assert data["status"] == "success"
        assert data["prediction"]["risk_score"] == 0.125
        mock_get.assert_called_once()
        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer mock_doc_token"


def test_fetch_backend_rest_error_codes():
    """Verify REST fetch handles 400, 401, 404, and 503 cleanly."""
    for err_code, err_status in [
        (400, "invalid_input"),
        (401, "unauthorized"),
        (404, "error"),
        (503, "unavailable"),
    ]:
        mock_resp = MagicMock()
        mock_resp.status_code = err_code
        mock_resp.json.return_value = {"status": err_status, "message": f"Simulated {err_code}"}

        with patch("requests.get", return_value=mock_resp):
            code, data = fetch_backend_hourly_prediction(
                patient_id=MOCK_PID,
                stay_id=MOCK_SID,
                backend_url="http://127.0.0.1:5000",
            )
            assert code == err_code
            assert data["status"] == err_status


# ── Test Suite 2: Rendering Valid 200 Prediction Responses ───────────────────

def test_render_valid_prediction_and_trajectory_chart():
    """Verify rendering of valid 200 response with trajectory line chart and metric cards."""
    synth_data = _build_synthetic_200_response(seq_len=4, risk_score=0.184, trajectory=[0.05, 0.09, 0.14, 0.184])

    rendered_markdown = []
    rendered_charts = []
    rendered_captions = []

    def mock_markdown(text, **kwargs):
        rendered_markdown.append(str(text))

    def mock_line_chart(df, **kwargs):
        rendered_charts.append(df)

    def mock_caption(text, **kwargs):
        rendered_captions.append(str(text))

    with patch("streamlit.markdown", side_effect=mock_markdown), \
         patch("streamlit.line_chart", side_effect=mock_line_chart), \
         patch("streamlit.caption", side_effect=mock_caption), \
         patch("streamlit.columns", return_value=[MagicMock() for _ in range(4)]), \
         patch("streamlit.expander", return_value=MagicMock()):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(200, synth_data),
        )

    # 1. Trajectory chart was invoked with correct pandas DataFrame
    assert len(rendered_charts) == 1
    chart_df = rendered_charts[0]
    assert isinstance(chart_df, pd.DataFrame)
    assert len(chart_df) == 4
    assert "Calibrated Sepsis Risk" in chart_df.columns
    assert "Decision Threshold (0.90)" in chart_df.columns
    assert list(chart_df["Calibrated Sepsis Risk"]) == [0.05, 0.09, 0.14, 0.184]
    assert list(chart_df["Decision Threshold (0.90)"]) == [0.90, 0.90, 0.90, 0.90]

    # 2. Disclaimer is present in rendered markdown
    all_md = " ".join(rendered_markdown)
    assert "RESEARCH PROTOTYPE PREDICTION ONLY" in all_md
    assert "Temperature (T=0.9576)" in all_md or "temperature" in all_md.lower()
    assert "18.40%" in all_md

    # 3. Caption explains causal trajectory
    assert any("Historical risk progression" in c or "Longitudinal risk progression" in c or "Causal" in c for c in rendered_captions)


def test_render_surveillance_flag_when_risk_exceeds_threshold():
    """Verify surveillance flag is prominently displayed when risk exceeds 0.90."""
    synth_data = _build_synthetic_200_response(seq_len=3, risk_score=0.925, flagged=True, trajectory=[0.70, 0.85, 0.925])

    rendered_markdown = []

    with patch("streamlit.markdown", side_effect=lambda t, **kw: rendered_markdown.append(str(t))), \
         patch("streamlit.line_chart"), \
         patch("streamlit.caption"), \
         patch("streamlit.columns", return_value=[MagicMock() for _ in range(4)]), \
         patch("streamlit.expander", return_value=MagicMock()):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(200, synth_data),
        )

    all_md = " ".join(rendered_markdown)
    assert "SURVEILLANCE FLAG" in all_md
    assert "CRITICAL" in all_md
    assert "92.50%" in all_md


# ── Test Suite 3: Rendering Safe Unavailable State (HTTP 503) ─────────────────

def test_render_unavailable_stream_no_data_fabrication():
    """
    Verify HTTP 503 response renders transparent unavailable status,
    vital clinical warning that data absence is NOT low risk,
    and ABSOLUTELY ZERO fabricated charts or risk scores.
    """
    unavail_data = {
        "status": "unavailable",
        "feature_available": False,
        "patient_id": MOCK_PID,
        "stay_id": MOCK_SID,
        "message": "Longitudinal hourly sequence unavailable for stay.",
    }

    rendered_markdown = []
    rendered_warnings = []
    rendered_charts = []

    with patch("streamlit.markdown", side_effect=lambda t, **kw: rendered_markdown.append(str(t))), \
         patch("streamlit.warning", side_effect=lambda t, **kw: rendered_warnings.append(str(t))), \
         patch("streamlit.line_chart", side_effect=lambda df, **kw: rendered_charts.append(df)), \
         patch("streamlit.expander", return_value=MagicMock()):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(503, unavail_data),
        )

    # Zero data fabrication: no line chart should EVER be rendered
    assert len(rendered_charts) == 0

    all_md = " ".join(rendered_markdown)
    assert "LONGITUDINAL HOURLY DATA STREAM UNAVAILABLE" in all_md
    assert "RESEARCH PROTOTYPE NOTICE" in all_md
    assert "RESEARCH PROTOTYPE PREDICTION ONLY" in all_md

    # Clinical warning must explicitly state this is NOT a low-risk result
    assert len(rendered_warnings) >= 1
    warn_text = " ".join(rendered_warnings).lower()
    assert "low-risk result" in warn_text
    assert "do not interpret data absence as low risk" in warn_text


# ── Test Suite 4: Handling HTTP Errors (400, 401, 404, 500) ───────────────────

def test_render_http_400_isolated_vitals_rejection():
    """Verify HTTP 400 displays incomplete input rejection and MAX30102 safeguard policy."""
    bad_data = {
        "status": "invalid_input",
        "message": "Incomplete feature set rejected: Input contains 2 columns, but 184 canonical features are required.",
    }

    rendered_errors = []
    rendered_markdown = []

    with patch("streamlit.markdown", side_effect=lambda t, **kw: rendered_markdown.append(str(t))), \
         patch("streamlit.error", side_effect=lambda t, **kw: rendered_errors.append(str(t))):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(400, bad_data),
        )

    all_err = " ".join(rendered_errors)
    assert "Incomplete Feature Set Rejected (HTTP 400)" in all_err
    assert "Safe Telemetry Policy" in all_err
    assert "MAX30102" in all_err
    assert "184" in all_err


def test_render_http_401_auth_required():
    """Verify HTTP 401 displays doctor session authentication required."""
    rendered_warnings = []

    with patch("streamlit.markdown"), \
         patch("streamlit.warning", side_effect=lambda t, **kw: rendered_warnings.append(str(t))):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(401, {"status": "unauthorized"}),
        )

    warn_text = " ".join(rendered_warnings)
    assert "Authentication Required (HTTP 401/403)" in warn_text
    assert "session token is missing or expired" in warn_text


def test_render_http_404_stay_not_found():
    """Verify HTTP 404 displays stay not found notice without leaking internals."""
    rendered_infos = []

    with patch("streamlit.markdown"), \
         patch("streamlit.info", side_effect=lambda t, **kw: rendered_infos.append(str(t))):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(404, {"status": "error", "message": "Stay not found"}),
        )

    info_text = " ".join(rendered_infos)
    assert "Stay Not Found (HTTP 404)" in info_text
    assert str(MOCK_PID) in info_text
    assert str(MOCK_SID) in info_text


def test_render_http_500_service_error():
    """Verify HTTP 500 displays service error banner without leaking internal stack traces."""
    rendered_errors = []
    rendered_markdown = []

    with patch("streamlit.markdown", side_effect=lambda t, **kw: rendered_markdown.append(str(t))), \
         patch("streamlit.error", side_effect=lambda t, **kw: rendered_errors.append(str(t))):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(500, {"status": "error", "message": "Database timeout"}),
        )

    all_md = " ".join(rendered_markdown)
    all_err = " ".join(rendered_errors)
    assert "Longitudinal Hourly Sepsis Prediction (Service Error)" in all_md
    assert "Service Communication Error" in all_err
    assert "Database timeout" not in all_err  # internal message not leaked


def test_render_malformed_response_fields_handled_safely():
    """Verify missing or malformed fields in 200 payload do not cause crashes."""
    # Missing hourly_risk_trajectory
    malformed_payload = {
        "status": "success",
        "prediction": {
            "risk_score": 0.15,
            # 'hourly_risk_trajectory' missing!
        },
    }

    rendered_errors = []

    with patch("streamlit.markdown"), \
         patch("streamlit.error", side_effect=lambda t, **kw: rendered_errors.append(str(t))):

        render_hourly_prediction_section(
            patient_id=MOCK_PID,
            stay_id=MOCK_SID,
            backend_response_override=(200, malformed_payload),
        )

    assert len(rendered_errors) >= 1
    assert "Malformed prediction response" in rendered_errors[0]


# ── Test Suite 5: Invariance & Preservation of 6-Hour Static Models ───────────

def test_preservation_existing_six_hour_models():
    """Verify that existing 6-hour admission model inference remains 100% functional."""
    res = run_clinical_transformer_inference(13269859, 39782194)
    assert res["status"] == "success"
    assert res["patient_id"] == 13269859
    assert res["stay_id"] == 39782194
    assert "raw_probability" in res
    assert "calibrated_probability" in res
    assert res["calibrated_decision_threshold"] == 0.21
    assert "top_attended_features" in res
    assert len(res["top_attended_features"]) > 0


# ── Test Suite 6: Phase 7.22 Status Banners, Quarantine & Shortcut Tests ──────

def test_render_quarantined_test_partition_banner():
    """Verify quarantined test stay renders distinct test quarantine banners."""
    quarantine_payload = {
        "status": "unavailable",
        "feature_available": False,
        "patient_id": 10023239,
        "stay_id": 33846653,
        "message": (
            "Hourly next-hour prediction is UNAVAILABLE for Stay 33846653: "
            "This stay belongs to the frozen test evaluation partition and is strictly "
            "quarantined from interactive retrospective inference to prevent evaluation leakage."
        ),
        "governance": {
            "static_admission_model_available": True,
            "static_admission_window_hours": 6.0,
            "partition": "test",
            "quarantined": True,
        },
    }

    rendered_markdown = []
    rendered_warnings = []
    rendered_charts = []

    with patch("streamlit.markdown", side_effect=lambda t, **kw: rendered_markdown.append(str(t))), \
         patch("streamlit.warning", side_effect=lambda t, **kw: rendered_warnings.append(str(t))), \
         patch("streamlit.line_chart", side_effect=lambda df, **kw: rendered_charts.append(df)), \
         patch("streamlit.expander", return_value=MagicMock()):

        render_hourly_prediction_section(
            patient_id=10023239,
            stay_id=33846653,
            backend_response_override=(503, quarantine_payload),
        )

    # Zero data fabrication
    assert len(rendered_charts) == 0

    all_md = " ".join(rendered_markdown)
    assert "Longitudinal Hourly Sepsis Prediction (Test Cohort Quarantined)" in all_md
    assert "TEST PARTITION QUARANTINE — LEAKAGE PREVENTION" in all_md
    assert "ids_test.csv" in all_md
    assert "RESEARCH PROTOTYPE NOTICE" in all_md

    all_warn = " ".join(rendered_warnings).lower()
    assert "test-partition stay quarantined" in all_warn
    assert "low-risk result" in all_warn
    assert "do not interpret data absence as low risk" in all_warn


def test_render_unmapped_hourly_sequence_banner():
    """Verify unmapped stay renders distinct unmapped sequence banner."""
    unmapped_payload = {
        "status": "unavailable",
        "feature_available": False,
        "patient_id": 10213765,
        "stay_id": 36628477,
        "message": (
            "Hourly next-hour sepsis prediction is currently UNAVAILABLE. "
            "The current validated models operate exclusively on the 6-hour admission baseline."
        ),
        "governance": {
            "static_admission_model_available": True,
            "static_admission_window_hours": 6.0,
            "static_target": "Sepsis-3 onset within 1-24 hours following T=intime+6h",
            "hourly_data_exported": False,
        },
    }

    rendered_markdown = []
    rendered_warnings = []
    rendered_charts = []

    with patch("streamlit.markdown", side_effect=lambda t, **kw: rendered_markdown.append(str(t))), \
         patch("streamlit.warning", side_effect=lambda t, **kw: rendered_warnings.append(str(t))), \
         patch("streamlit.line_chart", side_effect=lambda df, **kw: rendered_charts.append(df)), \
         patch("streamlit.expander", return_value=MagicMock()):

        render_hourly_prediction_section(
            patient_id=10213765,
            stay_id=36628477,
            backend_response_override=(503, unmapped_payload),
        )

    assert len(rendered_charts) == 0

    all_md = " ".join(rendered_markdown)
    assert "Longitudinal Hourly Sepsis Prediction (Hourly Sequence Unmapped)" in all_md
    assert "LONGITUDINAL HOURLY DATA STREAM UNAVAILABLE" in all_md
    assert "Stage 3.4A" in all_md

    all_warn = " ".join(rendered_warnings).lower()
    assert "data stream is currently unavailable" in all_warn
    assert "low-risk result" in all_warn
    assert "do not interpret data absence as low risk" in all_warn


def test_shortcut_patient_10007928_resolution_and_validation_partition():
    """Verify Patient 10007928 resolves deterministically to Stay 35128235 in validation partition."""
    pid, sid, stays, err = resolve_patient_stay("10007928", None)
    assert err is None
    assert pid == 10007928
    assert sid == 35128235
    assert len(stays) >= 1

    part_name, part_path = _resolve_stay_partition(sid, DEFAULT_CACHE_DIR)
    assert part_name == "val"
    assert part_path is not None
    assert "val_features.parquet" in part_path


def test_shortcut_patient_10007928_successful_hourly_inference():
    """Verify in-process prediction for Patient 10007928 succeeds with 67 trajectory steps."""
    res = predict_stay_hourly_sepsis(10007928, 35128235)
    assert res["status"] == "success"
    assert res["patient_id"] == 10007928
    assert res["stay_id"] == 35128235
    assert res["partition"] == "val"

    prediction = res["prediction"]
    assert prediction["sequence_length"] == 67
    assert 0.0 <= prediction["risk_score"] <= 1.0
    assert len(prediction["hourly_risk_trajectory"]) == 67
    assert prediction["decision_threshold"] == 0.90
    assert prediction["calibrator_type"] == "temperature"


def test_test_partition_quarantine_enforcement():
    """Verify test cohort stay is strictly quarantined and cannot return hourly prediction."""
    res = predict_stay_hourly_sepsis(10023239, 33846653)
    assert res["status"] == "unavailable"
    assert res["governance"]["quarantined"] is True
    assert res["governance"]["partition"] == "test"
    assert "quarantined" in res["message"].lower()
    assert "prediction" not in res
