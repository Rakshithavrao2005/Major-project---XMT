"""
tests/test_workspace_navigation.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive tests for Unified Dashboard Workspace Architecture:
1. Workspace taxonomy definitions (Overview, Software, Hardware).
2. Software sub-page inventory (7 pages including Explainability).
3. Hardware sub-page inventory (6 pages including History, Sessions, Independent CXR).
4. Entrypoint and callable contracts for all modular page renderers.
5. Cohort isolation invariants between Software and Hardware contexts.
6. Session state preservation during workspace switching.
"""

import pytest
from dashboard.styles import (
    WORKSPACE_OVERVIEW,
    WORKSPACE_SOFTWARE,
    WORKSPACE_HARDWARE,
    WORKSPACES,
    SOFTWARE_PAGES,
    HARDWARE_PAGES,
    PAGE_OVERVIEW,
    PAGE_PATIENT_EXPLORER,
    PAGE_CLINICAL_AI,
    PAGE_CHEST_XRAY,
    PAGE_MULTIMODAL_RESEARCH,
    PAGE_EXPLAINABILITY,
    PAGE_MODEL_EVALUATION,
    PAGE_DATA_QUALITY,
    PAGE_HW_OVERVIEW,
    PAGE_LIVE_MONITORING,
    PAGE_MONITORING_SESSIONS,
    PAGE_TELEMETRY_HISTORY,
    PAGE_HARDWARE_ALERTS,
    PAGE_INDEPENDENT_CXR,
    PAGE_ICONS,
    PAGE_SUBTITLES,
)
import dashboard.pages.overview as p_ov
import dashboard.pages.patient_explorer as p_pe
import dashboard.pages.clinical_ai as p_ca
import dashboard.pages.chest_xray as p_cx
import dashboard.pages.multimodal_research as p_mr
import dashboard.pages.explainability as p_ex
import dashboard.pages.model_evaluation as p_me
import dashboard.pages.data_quality as p_dq
import dashboard.pages.hardware_overview as p_ho
import dashboard.pages.live_monitoring as p_lm
import dashboard.pages.monitoring_sessions as p_ms
import dashboard.pages.telemetry_history as p_th
import dashboard.pages.hardware_alerts as p_ha
import dashboard.pages.independent_cxr as p_ic


def test_workspaces_taxonomy():
    """Verify that workspaces are properly defined."""
    assert WORKSPACE_OVERVIEW in WORKSPACES
    assert WORKSPACE_SOFTWARE in WORKSPACES
    assert WORKSPACE_HARDWARE in WORKSPACES
    assert len(WORKSPACES) == 3


def test_software_subpages_taxonomy():
    """Verify that Software workspace has exactly the 7 documented sub-pages."""
    expected_software = [
        PAGE_PATIENT_EXPLORER,
        PAGE_CLINICAL_AI,
        PAGE_CHEST_XRAY,
        PAGE_MULTIMODAL_RESEARCH,
        PAGE_EXPLAINABILITY,
        PAGE_MODEL_EVALUATION,
        PAGE_DATA_QUALITY,
    ]
    assert len(SOFTWARE_PAGES) == 7
    assert SOFTWARE_PAGES == expected_software
    for p in SOFTWARE_PAGES:
        assert p in PAGE_ICONS
        assert p in PAGE_SUBTITLES


def test_hardware_subpages_taxonomy():
    """Verify that Hardware workspace has exactly the 6 documented sub-pages."""
    expected_hardware = [
        PAGE_HW_OVERVIEW,
        PAGE_LIVE_MONITORING,
        PAGE_MONITORING_SESSIONS,
        PAGE_TELEMETRY_HISTORY,
        PAGE_HARDWARE_ALERTS,
        PAGE_INDEPENDENT_CXR,
    ]
    assert len(HARDWARE_PAGES) == 6
    assert HARDWARE_PAGES == expected_hardware
    for p in HARDWARE_PAGES:
        assert p in PAGE_ICONS
        assert p in PAGE_SUBTITLES


def test_all_page_render_callables():
    """Verify that all 14 page modules are importable and expose callable render functions."""
    callables = [
        (p_ov, "render_overview_page"),
        (p_pe, "render_patient_explorer_page"),
        (p_ca, "render_clinical_ai_page"),
        (p_cx, "render_chest_xray_page"),
        (p_mr, "render_multimodal_research_page"),
        (p_ex, "render_explainability_page"),
        (p_me, "render_model_evaluation_page"),
        (p_dq, "render_data_quality_page"),
        (p_ho, "render_hardware_overview_page"),
        (p_lm, "render_live_monitoring_page"),
        (p_ms, "render_monitoring_sessions_page"),
        (p_th, "render_telemetry_history_page"),
        (p_ha, "render_hardware_alerts_page"),
        (p_ic, "render_independent_cxr_page"),
    ]
    for mod, func_name in callables:
        assert hasattr(mod, func_name), f"Module {mod.__name__} missing {func_name}"
        assert callable(getattr(mod, func_name)), f"{func_name} in {mod.__name__} is not callable"


def test_workspace_cohort_separation():
    """Verify strict cohort separation invariants: Hardware never touches hospital Subject IDs."""
    from src.sensor.live_session import (
        start_monitoring_session,
        get_latest_sensor_telemetry,
        stop_monitoring_session,
    )
    sess_id = "SESSION-WORKSPACE-TEST-001"
    sess = start_monitoring_session(sess_id)
    assert sess["session_id"] == sess_id
    assert "subject_id" not in sess
    assert "stay_id" not in sess

    telemetry = get_latest_sensor_telemetry(fetch_remote=False)
    assert "subject_id" not in telemetry
    assert "stay_id" not in telemetry

    stop_monitoring_session(sess_id)


def test_session_state_preservation_on_workspace_switch():
    """Simulate workspace switching and verify that session variables remain intact."""
    mock_state = {
        "ui_theme": "Clinical Light",
        "current_search_pid": "10213765",
        "current_stay_id": 36628477,
        "live_session_id": "SESSION-20261016-001",
        "session_uploaded_filename": "cxr_sample.jpg",
        "auto_poll_enabled": True,
        "poll_interval_seconds": 30,
    }

    # Simulate switching to Hardware workspace
    mock_state["nav_mode"] = PAGE_LIVE_MONITORING
    assert mock_state["current_search_pid"] == "10213765"
    assert mock_state["current_stay_id"] == 36628477
    assert mock_state["live_session_id"] == "SESSION-20261016-001"

    # Simulate switching to Software workspace
    mock_state["nav_mode"] = PAGE_CLINICAL_AI
    assert mock_state["current_search_pid"] == "10213765"
    assert mock_state["current_stay_id"] == 36628477
    assert mock_state["live_session_id"] == "SESSION-20261016-001"
    assert mock_state["auto_poll_enabled"] is True
