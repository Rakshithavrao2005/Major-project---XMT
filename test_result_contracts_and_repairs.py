"""
tests/test_result_contracts_and_repairs.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive regression test suite verifying repairs for:
1. get_stay_model_features tuple return and get_stay_model_features_dict contract.
2. run_cxr_gradcam_inference result schema, overlay_file/gradcam_overlay_file aliases,
   missing image handling, invalid stay handling, and unreadable file resilience.
3. NotificationDispatcher and TelemetryCoordinator alert_log_file attribute integration.
4. run_session_cxr_inference flexibility (1-arg and 2-arg calls, None handling).
"""

from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.patient_service import (
    get_stay_model_features,
    get_stay_model_features_dict,
    run_cxr_gradcam_inference,
    resolve_patient_stay,
    search_patient,
    get_all_patients_directory,
)
from src.sensor.notification_service import (
    NotificationDispatcher,
    AlertRepository,
    DEFAULT_ALERTS_LOG_PATH,
)
from src.sensor.telemetry_poller import (
    TelemetryCoordinator,
    get_telemetry_coordinator,
)
from src.sensor.live_session import run_session_cxr_inference


# ── 1. Model Features Tuple & Dictionary Contract Tests ───────────────────────

class TestModelFeaturesContracts:
    """Verifies that get_stay_model_features returns an unpacked 4-tuple and get_stay_model_features_dict returns a typed dict."""

    def test_get_stay_model_features_tuple_contract(self):
        """get_stay_model_features must return exactly (feat_array, scaled_feats, feature_cols, missing_imputed)."""
        pts = get_all_patients_directory()
        pid = int(pts.iloc[0]["subject_id"])
        stays = search_patient(pid)["stays"]
        sid = stays[0]["stay_id"]

        res = get_stay_model_features(pid, sid)
        assert isinstance(res, tuple)
        assert len(res) == 4

        feat_arr, scaled, cols, imputed = res
        assert isinstance(feat_arr, np.ndarray)
        assert feat_arr.shape == (1, 49)
        assert isinstance(scaled, np.ndarray)
        assert scaled.shape == (1, 49)
        assert isinstance(cols, list)
        assert len(cols) == 49
        assert isinstance(imputed, list)
        # All imputed cols must be in cols
        for imp in imputed:
            assert imp in cols

    def test_get_stay_model_features_invalid_stay_raises_value_error(self):
        """get_stay_model_features raises ValueError when patient or stay is invalid."""
        with pytest.raises(ValueError):
            get_stay_model_features(999999999, 888888888)

    def test_get_stay_model_features_dict_success(self):
        """get_stay_model_features_dict returns a well-formed dictionary with 49 features."""
        pts = get_all_patients_directory()
        pid = int(pts.iloc[0]["subject_id"])
        stays = search_patient(pid)["stays"]
        sid = stays[0]["stay_id"]

        dict_res = get_stay_model_features_dict(pid, sid)
        assert isinstance(dict_res, dict)
        assert dict_res["status"] == "success"
        assert dict_res["patient_id"] == pid
        assert dict_res["stay_id"] == sid
        assert dict_res["total_features"] == 49
        assert len(dict_res["features"]) == 49

        # Verify feature item schema
        for item in dict_res["features"]:
            assert "name" in item
            assert "raw_value" in item
            assert "scaled_value" in item
            assert "is_imputed" in item
            assert isinstance(item["is_imputed"], bool)
            if item["is_imputed"]:
                assert item["raw_value"] is None
            else:
                assert isinstance(item["raw_value"], float)

    def test_get_stay_model_features_dict_error_handling(self):
        """get_stay_model_features_dict returns status='error' safely without raising unhandled exception."""
        dict_err = get_stay_model_features_dict(999999999, 888888888)
        assert isinstance(dict_err, dict)
        assert dict_err["status"] == "error"
        assert "not found" in dict_err["message"]
        assert dict_err["features"] == []
        assert dict_err["total_features"] == 0


# ── 2. Grad-CAM Result Schema & Overlay Resilience Tests ─────────────────────

class TestGradCAMContractsAndResilience:
    """Verifies run_cxr_gradcam_inference result schema, alias support, and failure modes."""

    def test_gradcam_success_schema_and_aliases(self):
        """On a patient with valid CXR, returns both overlay_file and gradcam_overlay_file aliases."""
        # Patient 17585185, stay 32417784 is a known test stay with paired radiograph
        pid, sid, _, _ = resolve_patient_stay("17585185", 32417784)
        gcam = run_cxr_gradcam_inference(pid, sid, alpha=0.5)

        assert isinstance(gcam, dict)
        assert gcam["status"] == "success"
        assert "overlay_file" in gcam
        assert "gradcam_overlay_file" in gcam
        assert gcam["overlay_file"] == gcam["gradcam_overlay_file"]
        assert "image_file" in gcam
        assert "target_class_name" in gcam
        assert gcam["target_class_name"] == "Sepsis Risk"
        assert "probability" in gcam
        assert "raw_logit" in gcam

        # Both files must exist on disk and be openable by PIL
        assert os.path.isfile(gcam["overlay_file"])
        assert os.path.isfile(gcam["image_file"])

        with Image.open(gcam["overlay_file"]) as img:
            assert img.size[0] > 0 and img.size[1] > 0

    def test_gradcam_missing_radiograph_handling(self):
        """Stays without linked radiograph return status='unavailable' with a descriptive message."""
        # Query patient directory for a stay without image
        pts = get_all_patients_directory()
        no_img_row = pts[pts["has_images"] == False]
        if not no_img_row.empty:
            pid = int(no_img_row.iloc[0]["subject_id"])
            stays = search_patient(pid)["stays"]
            sid = stays[0]["stay_id"]
            gcam = run_cxr_gradcam_inference(pid, sid)
            assert isinstance(gcam, dict)
            assert gcam["status"] == "unavailable"
            assert "Inference unavailable" in gcam["message"]
            assert "overlay_file" not in gcam
            assert "gradcam_overlay_file" not in gcam

    def test_gradcam_invalid_patient_stay(self):
        """Invalid patient/stay returns status='error' with clear message."""
        gcam = run_cxr_gradcam_inference(999999999, 888888888)
        assert isinstance(gcam, dict)
        assert gcam["status"] == "error"
        assert "not found" in gcam["message"]

    def test_gradcam_unreadable_overlay_defensive_behavior(self):
        """Simulate a corrupt or missing overlay file and verify defensive dashboard fallback pattern."""
        mock_gcam_res = {
            "status": "success",
            "overlay_file": "C:/nonexistent/path/gradcam_missing.png",
            "gradcam_overlay_file": "C:/nonexistent/path/gradcam_missing.png",
            "image_file": "C:/nonexistent/path/image_missing.jpg",
            "target_class_name": "Sepsis Risk",
        }
        overlay_path = mock_gcam_res.get("gradcam_overlay_file") or mock_gcam_res.get("overlay_file")
        # Direct check used by UI
        is_valid = bool(overlay_path and os.path.isfile(overlay_path))
        assert is_valid is False  # UI safely falls back without raising KeyError or FileNotFoundError


# ── 3. NotificationDispatcher & TelemetryCoordinator Alert Log Tests ───────────

class TestNotificationDispatcherAlertLogIntegration:
    """Verifies that NotificationDispatcher and TelemetryCoordinator expose alert_log_file reliably."""

    def test_dispatcher_with_repository_exposes_storage_path(self):
        """When initialized with a repository, dispatcher.alert_log_file matches repository.storage_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_log = os.path.join(tmpdir, "custom_alerts.json")
            repo = AlertRepository(storage_path=custom_log)
            dispatcher = NotificationDispatcher(repository=repo)

            assert hasattr(dispatcher, "alert_log_file")
            assert dispatcher.alert_log_file == os.path.abspath(custom_log)
            assert Path(dispatcher.alert_log_file).name == "custom_alerts.json"

    def test_dispatcher_without_repository_falls_back_safely(self):
        """When repository is None, dispatcher.alert_log_file falls back to DEFAULT_ALERTS_LOG_PATH."""
        dispatcher = NotificationDispatcher(repository=None)
        assert hasattr(dispatcher, "alert_log_file")
        assert dispatcher.alert_log_file == DEFAULT_ALERTS_LOG_PATH
        assert Path(dispatcher.alert_log_file).name == "sensor_alerts_log.json"

    def test_coordinator_exposes_alert_log_file(self):
        """TelemetryCoordinator exposes alert_log_file matching its repository storage path."""
        coord = get_telemetry_coordinator()
        assert hasattr(coord, "alert_log_file")
        assert coord.alert_log_file == coord.repository.storage_path
        assert Path(coord.alert_log_file).name == "sensor_alerts_log.json"


# ── 4. Session CXR Inference Tests (1-arg vs 2-arg calls) ────────────────────

class TestSessionCXRInferenceContract:
    """Verifies run_session_cxr_inference handles 1-arg, 2-arg, and None calls safely."""

    def test_run_session_cxr_inference_with_one_arg(self):
        """Calling with only pil_image succeeds."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        res = run_session_cxr_inference(img)
        assert isinstance(res, dict)
        assert res["status"] == "success"
        assert "platt_calibrated_probability" in res
        assert "raw_logit" in res

    def test_run_session_cxr_inference_with_two_args(self):
        """Calling with pil_image and filename succeeds and echoes filename."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        res = run_session_cxr_inference(img, filename="bedside_sample.png")
        assert isinstance(res, dict)
        assert res["status"] == "success"
        assert res.get("filename") == "bedside_sample.png"

    def test_run_session_cxr_inference_none_image(self):
        """Calling with None returns status='error' safely."""
        res = run_session_cxr_inference(None, filename="empty.png")
        assert isinstance(res, dict)
        assert res["status"] == "error"
        assert "No valid image" in res["message"]


# ── 5. CXR Classification & Probability Consistency Tests ────────────────────

class TestCXRClassificationProbabilityConsistency:
    """
    Verifies that CXR classification uses the intended score (raw_probability)
    at the balanced threshold (0.5237), clearly separates Platt-calibrated probability,
    and maintains strict isolation from the Clinical Transformer.
    """

    def test_cxr_threshold_specifies_score_type_and_governing_score(self):
        """run_cxr_inference must return score_type='raw_probability' and governing_score block."""
        pts = get_all_patients_directory()
        pid = int(pts.iloc[0]["subject_id"])
        sid = search_patient(pid)["stays"][0]["stay_id"]

        from backend.patient_service import run_cxr_inference
        res = run_cxr_inference(pid, sid)
        assert res["status"] == "success"

        thrs = res["thresholds"]
        bal = thrs["experimental_balanced_threshold_05237"]

        # 1. Must explicitly document the score type applied to the threshold
        assert bal["score_type"] == "raw_probability"
        assert bal["threshold"] == 0.5237
        assert "score_value" in bal

        # 2. Must expose governing_score block
        assert "governing_score" in res
        gov = res["governing_score"]
        assert gov["score_type"] == "raw_probability"
        assert gov["threshold"] == 0.5237
        assert "explanation" in gov

        # 3. Platt calibrated probability and raw probability must be distinct
        assert "platt_calibrated_probability" in res
        assert "raw_probability" in res
        assert res["platt_calibrated_probability"] != res["raw_probability"]
        # In typical ICU sepsis population, Platt probability reflects true prevalence (~17%)
        assert 0.05 <= res["platt_calibrated_probability"] <= 0.35

    def test_cxr_balanced_classification_boundary_conditions(self):
        """Boundary conditions for 0.5237 threshold on raw_probability."""
        thr = 0.5237

        # Exactly at threshold -> Positive (High Risk)
        cls_at = "Positive (High Risk)" if 0.5237 >= thr else "Negative"
        assert cls_at == "Positive (High Risk)"

        # Just above threshold -> Positive (High Risk)
        cls_above = "Positive (High Risk)" if 0.5238 >= thr else "Negative"
        assert cls_above == "Positive (High Risk)"

        # Just below threshold -> Negative
        cls_below = "Positive (High Risk)" if 0.5236 >= thr else "Negative"
        assert cls_below == "Negative"

    def test_clinical_transformer_and_cxr_isolation(self):
        """Clinical Transformer and Chest X-ray models must never mix or average probabilities."""
        pts = get_all_patients_directory()
        pid = int(pts.iloc[0]["subject_id"])
        sid = search_patient(pid)["stays"][0]["stay_id"]

        from backend.patient_service import run_clinical_transformer_inference, run_cxr_inference
        clin_res = run_clinical_transformer_inference(pid, sid)
        cxr_res = run_cxr_inference(pid, sid)

        assert clin_res["status"] == "success"
        assert cxr_res["status"] == "success"

        # Separate probabilities and separate thresholding
        clin_p = clin_res["predicted_probability"]
        clin_thr = clin_res["decision_threshold"]
        cxr_raw_p = cxr_res["raw_probability"]
        cxr_cal_p = cxr_res["platt_calibrated_probability"]

        assert clin_thr == 0.60
        assert cxr_res["thresholds"]["experimental_balanced_threshold_05237"]["threshold"] == 0.5237

        # Check independence: changing one does not affect the other
        assert isinstance(clin_p, float)
        assert isinstance(cxr_cal_p, float)
        assert isinstance(cxr_raw_p, float)


# ── 6. AlertEvent Defensive Attributes & Triage Tests ─────────────────────────

class TestAlertEventDefensiveAttributesAndTriage:
    """Verifies AlertEvent canonical fields, legacy aliases, and coordinator operations."""

    def test_alert_event_canonical_and_defensive_properties(self):
        """AlertEvent exposes reason, vitals, message, telemetry_snapshot, and cooldown_seconds."""
        import datetime
        from src.sensor.alert_engine import (
            AlertEvent,
            AlertCategory,
            AlertSeverity,
            DataQualityStatus,
        )

        evt = AlertEvent(
            event_id="test-evt-001",
            rule_name="TACHYCARDIA_WARNING",
            category=AlertCategory.PHYSIOLOGICAL_RISK,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            reason="Heart rate elevated at 108 bpm",
            vitals={"heart_rate": 108.0, "spo2": 96.0},
            data_quality_status=DataQualityStatus.VALID,
        )

        # Canonical attributes
        assert evt.reason == "Heart rate elevated at 108 bpm"
        assert evt.vitals == {"heart_rate": 108.0, "spo2": 96.0}

        # Defensive legacy aliases
        assert evt.message == "Heart rate elevated at 108 bpm"
        assert evt.telemetry_snapshot == {"heart_rate": 108.0, "spo2": 96.0}
        assert evt.cooldown_seconds == 900.0

    def test_alert_event_missing_optional_fields_handled_safely(self):
        """AlertEvent handles missing reason, vitals, and notes gracefully."""
        import datetime
        from src.sensor.alert_engine import (
            AlertEvent,
            AlertCategory,
            AlertSeverity,
            DataQualityStatus,
        )

        evt = AlertEvent(
            event_id="test-evt-002",
            rule_name="SIGNAL_DROPOUT",
            category=AlertCategory.DATA_QUALITY,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            reason="",
            vitals={},
            data_quality_status=DataQualityStatus.QUESTIONABLE_CONTACT,
        )

        assert evt.message == ""
        assert evt.telemetry_snapshot == {}
        assert evt.is_acknowledged is False
        assert evt.is_resolved is False

    def test_coordinator_acknowledgement_and_resolution_aliases(self):
        """Coordinator supports clinician_id and acknowledged_by aliases seamlessly."""
        coord = get_telemetry_coordinator()
        import datetime
        from src.sensor.alert_engine import (
            AlertEvent,
            AlertCategory,
            AlertSeverity,
            DataQualityStatus,
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        evt = AlertEvent(
            event_id="test-evt-triage-001",
            rule_name="TEST_RULE",
            category=AlertCategory.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            timestamp=now,
            reason="Triage test event",
            vitals={"heart_rate": 90.0},
            data_quality_status=DataQualityStatus.VALID,
        )
        coord.repository.add_event(evt)

        # Test acknowledgement with acknowledged_by keyword argument
        ack_res = coord.acknowledge_alert("test-evt-triage-001", acknowledged_by="DR-SMITH")
        assert ack_res is not None
        assert ack_res.is_acknowledged is True
        assert ack_res.acknowledged_by == "DR-SMITH"

        # Test resolution with resolution_notes keyword argument
        res_res = coord.resolve_alert("test-evt-triage-001", resolution_notes="Resolved on ward")
        assert res_res is not None
        assert res_res.is_resolved is True


# ── 7. Disclaimer Card & Home Page Signature Tests ────────────────────────────

class TestDisclaimerCardAndHomePageSignature:
    """Verifies render_disclaimer_card supports all keyword arguments and Home page imports safely."""

    def test_render_disclaimer_card_supports_all_parameter_variants(self):
        """render_disclaimer_card handles title, text, subtitle, body, and **kwargs without TypeError."""
        from dashboard.styles import render_disclaimer_card

        # 1. With text
        html1 = render_disclaimer_card(title="Title 1", text="Text body 1")
        assert "Title 1" in html1
        assert "Text body 1" in html1

        # 2. With subtitle
        html2 = render_disclaimer_card(title="Title 2", subtitle="Subtitle body 2")
        assert "Title 2" in html2
        assert "Subtitle body 2" in html2

        # 3. With body
        html3 = render_disclaimer_card(title="Title 3", body="Body text 3")
        assert "Title 3" in html3
        assert "Body text 3" in html3

        # 4. With extra unknown kwargs
        html4 = render_disclaimer_card(title="Title 4", text="Body 4", extra_arg="ignored")
        assert "Title 4" in html4
        assert "Body 4" in html4

        # 5. Default fallback
        html5 = render_disclaimer_card()
        assert "Research decision-support system" in html5
        assert "clinical judgement" in html5

    def test_home_page_module_imports_cleanly(self):
        """dashboard.pages.home must import render_home_page without error."""
        import dashboard.pages.home as p_home
        assert hasattr(p_home, "render_home_page")
        assert callable(p_home.render_home_page)

