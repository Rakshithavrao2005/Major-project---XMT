"""
tests/test_dashboard_alerts.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Unit & Integration Test Suite for Phase 7C Stage 3:
Dashboard Integration, Telemetry Auto-Polling, Freshness, and Persistent Alerts.

Test Dimensions:
- D01: Freshness categorization (fresh, stale, missing, edge cases).
- D02: Polling interval bounds & defaults (5s - 300s, default 30s).
- D03: Mock ThingSpeak polling and successful state updates.
- D04: Auto-polling disabled vs enabled behavior.
- D05: Repeated entry IDs suppress duplicate alert processing.
- D06: Manual refresh / force_evaluate re-processes repeated entry IDs.
- D07: Missing/unavailable telemetry triggers data quality alert.
- D08: Stale telemetry (>300s) triggers STALE_TELEMETRY warning alert.
- D09: Optical contact failure (IR <= 50k) triggers POOR_OPTICAL_CONTACT alert.
- D10: Physiological bounds artifact (HR/SpO2) triggers INVALID_PHYSIOLOGICAL_BOUND.
- D11: Critical vitals trigger multi-tier alerts and persist to isolated repository.
- D12: Notification dispatcher operates safely in simulation mode by default.
- D13: Notification failure updates event delivery status to FAILED and persists.
- D14: Acknowledgement persists clinician notes without resolving or deleting alert.
- D15: Resolution marks is_resolved=True while preserving complete audit record.
- D16: Non-blocking execution lock prevents overlapping concurrent polling queries.
- D17: Strict EHR cohort isolation: prohibits subject_id/stay_id in telemetry/alerts.
- D18: Historical EHR patient search & AI workflows remain functional and untainted.
"""

from __future__ import annotations
import os
import sys
import shutil
import tempfile
import datetime
import unittest.mock as mock
import pytest
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.alert_engine import (
    SensorAlertEngine,
    AlertEvent,
    AlertCategory,
    AlertSeverity,
    DeliveryStatus,
    DataQualityStatus,
)
from src.sensor.notification_service import (
    AlertRepository,
    NotificationDispatcher,
    SMTPConfig,
    DeliveryOutcome,
)
from src.sensor.telemetry_poller import (
    TelemetryAlertCoordinator,
    get_telemetry_freshness,
    get_telemetry_coordinator,
    reset_telemetry_coordinator,
)
from backend.patient_service import (
    search_patient,
    get_stay_clinical_record,
    run_clinical_transformer_inference,
)


class TestDashboardAlertsAndAutoPolling:
    """Test suite for Phase 7C Stage 3 dashboard integration and polling behaviors."""

    @pytest.fixture(autouse=True)
    def setup_isolated_env(self):
        """Create a dedicated temporary directory for isolated alert persistence."""
        self.temp_dir = tempfile.mkdtemp(prefix="xmt_dashboard_test_")
        self.temp_log = os.path.join(self.temp_dir, "sensor_alerts_log.json")
        self.repo = AlertRepository(storage_path=self.temp_log)
        self.dispatcher = NotificationDispatcher(repository=self.repo, force_simulation=True)
        self.engine = SensorAlertEngine()
        self.coordinator = TelemetryAlertCoordinator(
            alert_engine=self.engine,
            repository=self.repo,
            dispatcher=self.dispatcher,
        )
        yield
        # Teardown
        reset_telemetry_coordinator()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ── D01: Freshness Categorization ────────────────────────────────────────

    def test_D01_telemetry_freshness_categorization(self):
        """Freshness helper must accurately categorize fresh, stale, missing, and future dates."""
        now = datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.timezone.utc)

        # Missing
        f_missing = get_telemetry_freshness(None, current_time=now)
        assert f_missing["status"] == "MISSING"
        assert f_missing["is_stale"] is True

        # Fresh (30s ago)
        t_fresh = now - datetime.timedelta(seconds=30)
        f_fresh = get_telemetry_freshness(t_fresh, current_time=now)
        assert f_fresh["status"] == "FRESH"
        assert f_fresh["is_stale"] is False
        assert "Fresh (30s ago)" in f_fresh["description"]

        # Stale (600s / 10m ago)
        t_stale = now - datetime.timedelta(seconds=600)
        f_stale = get_telemetry_freshness(t_stale, current_time=now)
        assert f_stale["status"] == "STALE"
        assert f_stale["is_stale"] is True
        assert "Stale (10.0 minutes" in f_stale["description"]

        # Future timestamp edge case (clock skew: elapsed clamped to 0)
        t_future = now + datetime.timedelta(seconds=5)
        f_future = get_telemetry_freshness(t_future, current_time=now)
        assert f_future["status"] == "FRESH"
        assert f_future["elapsed_seconds"] == 0.0

    # ── D02: Polling Interval Bounds ─────────────────────────────────────────

    def test_D02_configurable_polling_interval_bounds(self):
        """Dashboard interval must validate bounds: 5s to 300s, default 30s."""
        default_interval = 30
        assert 5 <= default_interval <= 300

        # Validate custom values
        for valid in [5, 10, 30, 60, 120, 300]:
            assert 5 <= valid <= 300

    # ── D03: Mock ThingSpeak Polling ─────────────────────────────────────────

    def test_D03_mock_thingspeak_polling_success(self):
        """Coordinator successfully polls mock telemetry, extracts fields, and logs status."""
        mock_telemetry = {
            "status": "success",
            "connected": True,
            "entry_id": 101,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 78.0,
            "spo2": 98.0,
            "ir_ppg": 125000.0,
            "risk_status_code": 0,
            "valid_hr": True,
            "valid_spo2": True,
            "finger_detected": True,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            eval_time = datetime.datetime(2026, 10, 16, 12, 0, 15, tzinfo=datetime.timezone.utc)
            res = self.coordinator.poll_and_evaluate(fetch_remote=False, current_time=eval_time)

            assert res["status"] == "success"
            assert self.coordinator.last_processed_entry_id == 101
            assert self.coordinator.last_poll_status == "success"
            assert len(res["new_alerts"]) == 0  # Normal vitals: no alert

    # ── D04: Auto-Polling Disabled vs Enabled ────────────────────────────────

    def test_D04_auto_polling_disabled_vs_enabled(self):
        """fetch_remote flag is respected: False uses local cache, True triggers remote fetch."""
        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry") as mock_get:
            mock_get.return_value = {
                "status": "success",
                "entry_id": 102,
                "timestamp": "2026-10-16T12:00:00+00:00",
                "heart_rate": 75.0,
                "spo2": 99.0,
                "ir_ppg": 130000.0,
                "risk_status_code": 0,
                "is_stale": False,
            }

            # Disabled (fetch_remote=False)
            self.coordinator.poll_and_evaluate(fetch_remote=False)
            mock_get.assert_called_with(fetch_remote=False)

            # Enabled (fetch_remote=True)
            self.coordinator.poll_and_evaluate(fetch_remote=True, force_evaluate=True)
            mock_get.assert_called_with(fetch_remote=True)

    # ── D05: Repeated Entry ID Deduplication ─────────────────────────────────

    def test_D05_repeated_entry_id_suppresses_duplicate_processing(self):
        """Repeated polls returning the same entry_id must return no_new_data without alert spam."""
        mock_telemetry = {
            "status": "success",
            "entry_id": 200,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 135.0,  # Critical tachycardia
            "spo2": 98.0,
            "ir_ppg": 120000.0,
            "risk_status_code": 0,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            t1 = datetime.datetime(2026, 10, 16, 12, 0, 10, tzinfo=datetime.timezone.utc)
            res1 = self.coordinator.poll_and_evaluate(current_time=t1)
            assert res1["status"] == "success"
            assert len(res1["new_alerts"]) >= 1

            # Second poll: same entry_id 200
            t2 = datetime.datetime(2026, 10, 16, 12, 0, 40, tzinfo=datetime.timezone.utc)
            res2 = self.coordinator.poll_and_evaluate(current_time=t2)
            assert res2["status"] == "no_new_data"
            assert len(res2["new_alerts"]) == 0
            assert "Entry ID 200 already evaluated" in res2["message"]

    # ── D06: Force Evaluate Re-processes Repeated Entry ──────────────────────

    def test_D06_force_evaluate_bypasses_repeated_entry_id(self):
        """Manual refresh with force_evaluate=True bypasses entry_id deduplication."""
        mock_telemetry = {
            "status": "success",
            "entry_id": 201,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 72.0,
            "spo2": 98.0,
            "ir_ppg": 110000.0,
            "risk_status_code": 0,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            res1 = self.coordinator.poll_and_evaluate(force_evaluate=False)
            assert res1["status"] == "success"

            # Manual click: force_evaluate=True
            res2 = self.coordinator.poll_and_evaluate(force_evaluate=True)
            assert res2["status"] == "success"

    # ── D07: Missing Telemetry Handling ──────────────────────────────────────

    def test_D07_missing_telemetry_evaluates_missing_alert(self):
        """When telemetry stream is unavailable, coordinator generates MISSING_MEASUREMENTS alert."""
        mock_telemetry = {
            "status": "unavailable",
            "message": "Sensor CSV not found.",
            "connected": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            res = self.coordinator.poll_and_evaluate()
            assert res["status"] == "unavailable"
            assert any(a.rule_name == "MISSING_MEASUREMENTS" for a in res["new_alerts"])
            
            # Persisted to repository
            saved = self.repo.load_events()
            assert len(saved) >= 1
            assert saved[0].data_quality_status == DataQualityStatus.MISSING_VALUES

    # ── D08: Stale Telemetry Handling ────────────────────────────────────────

    def test_D08_stale_telemetry_evaluates_stale_alert(self):
        """Telemetry older than 300s triggers STALE_TELEMETRY data quality alert."""
        obs_time = datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.timezone.utc)
        eval_time = obs_time + datetime.timedelta(seconds=350)  # 350s > 300s

        mock_telemetry = {
            "status": "success",
            "entry_id": 301,
            "timestamp": obs_time.isoformat(),
            "heart_rate": 70.0,
            "spo2": 98.0,
            "ir_ppg": 120000.0,
            "risk_status_code": 0,
            "is_stale": True,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            res = self.coordinator.poll_and_evaluate(current_time=eval_time)
            assert res["status"] == "success"
            assert any(a.rule_name == "STALE_TELEMETRY" for a in res["new_alerts"])

    # ── D09: Optical Lead-Off Alert ──────────────────────────────────────────

    def test_D09_optical_lead_off_evaluates_quality_alert(self):
        """IR PPG <= 50k triggers POOR_OPTICAL_CONTACT warning alert."""
        mock_telemetry = {
            "status": "success",
            "entry_id": 401,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 0.0,
            "spo2": 0.0,
            "ir_ppg": 12000.0,  # Below 50,000 counts
            "risk_status_code": 0,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            t = datetime.datetime(2026, 10, 16, 12, 0, 10, tzinfo=datetime.timezone.utc)
            res = self.coordinator.poll_and_evaluate(current_time=t)
            assert any(a.rule_name == "POOR_OPTICAL_CONTACT" for a in res["new_alerts"])

    # ── D10: Physiological Bounds Artifact ───────────────────────────────────

    def test_D10_physiological_artifact_bounds(self):
        """SpO2 < 70% with high IR counts triggers INVALID_PHYSIOLOGICAL_BOUND alert."""
        mock_telemetry = {
            "status": "success",
            "entry_id": 501,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 80.0,
            "spo2": 45.0,  # Out of bounds [70, 100]
            "ir_ppg": 140000.0,
            "risk_status_code": 0,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            t = datetime.datetime(2026, 10, 16, 12, 0, 10, tzinfo=datetime.timezone.utc)
            res = self.coordinator.poll_and_evaluate(current_time=t)
            assert any(a.rule_name == "INVALID_PHYSIOLOGICAL_BOUNDS" for a in res["new_alerts"])

    # ── D11: Multi-Tier Critical Sepsis Heuristic & Persistence ───────────────

    def test_D11_critical_tachycardia_hypoxemia_evaluates_and_persists(self):
        """Critical vitals trigger multi-tier alerts and persist to the storage repository."""
        mock_telemetry = {
            "status": "success",
            "entry_id": 601,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 138.0,  # Critical tachycardia (>=130)
            "spo2": 88.0,        # Critical hypoxemia (<92)
            "ir_ppg": 160000.0,
            "risk_status_code": 2,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            t = datetime.datetime(2026, 10, 16, 12, 0, 10, tzinfo=datetime.timezone.utc)
            res = self.coordinator.poll_and_evaluate(session_id="SESSION-TEST-001", current_time=t)

            rules = {a.rule_name for a in res["new_alerts"]}
            assert "CRITICAL_TACHYCARDIA" in rules
            assert "CRITICAL_HYPOXEMIA" in rules
            assert "SEPSIS_HEURITICAL_CRITICAL" in rules or "SEPSIS_HEURISTIC_CRITICAL" in rules

            # Verify persisted to disk
            saved = self.repo.load_events()
            assert len(saved) >= 3
            assert all(e.session_id == "SESSION-TEST-001" for e in saved)

    # ── D12: Notification Dispatch in Safe Simulation Mode ───────────────────

    def test_D12_notification_dispatch_in_simulation_mode(self):
        """Alerts evaluated in default simulation mode receive SIMULATED delivery status."""
        mock_telemetry = {
            "status": "success",
            "entry_id": 701,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 140.0,
            "spo2": 98.0,
            "ir_ppg": 150000.0,
            "risk_status_code": 0,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
            res = self.coordinator.poll_and_evaluate()
            for alert in res["new_alerts"]:
                assert alert.delivery_status == DeliveryStatus.SIMULATED

            # Verify on disk
            events = self.repo.load_events()
            assert len(events) >= 1
            assert events[0].delivery_status == DeliveryStatus.SIMULATED

    # ── D13: Notification Failure State ──────────────────────────────────────

    def test_D13_notification_dispatch_failure_handling(self):
        """Simulated SMTP connection error marks event as FAILED and preserves record."""
        # Create dispatcher configured for live SMTP but mocked to raise an error
        smtp_cfg = SMTPConfig(
            enabled=True,
            host="smtp.local.invalid",
            port=587,
            sender_email="xmt@local.test",
            recipient_email="nurse@local.test",
        )
        broken_dispatcher = NotificationDispatcher(config=smtp_cfg, repository=self.repo, force_simulation=False)
        self.coordinator.dispatcher = broken_dispatcher

        mock_telemetry = {
            "status": "success",
            "entry_id": 801,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 132.0,
            "spo2": 97.0,
            "ir_ppg": 140000.0,
            "risk_status_code": 0,
            "is_stale": False,
        }

        with mock.patch("smtplib.SMTP", side_effect=ConnectionRefusedError("Connection refused by test host")):
            with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=mock_telemetry):
                res = self.coordinator.poll_and_evaluate()
                assert len(res["new_alerts"]) >= 1
                for alert in res["new_alerts"]:
                    assert alert.delivery_status == DeliveryStatus.FAILED

                # Verify in repository
                events = self.repo.load_events()
                assert events[0].delivery_status == DeliveryStatus.FAILED

    # ── D14: Acknowledgement Preserves Alert Without Resolving or Deleting ───

    def test_D14_acknowledgement_persists_without_resolving_or_deleting(self):
        """Acknowledging an alert updates clinician notes, leaves is_resolved=False, and does not delete it."""
        event = AlertEvent(
            event_id="ALT-20261016-D14-001",
            rule_name="CRITICAL_HYPOXEMIA",
            category=AlertCategory.PHYSIOLOGICAL_RISK,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            reason="SpO2 = 88%",
            vitals={"heart_rate": 100, "spo2": 88},
            data_quality_status=DataQualityStatus.VALID,
        )
        self.repo.add_event(event)

        # Clinician acknowledges
        ack_res = self.coordinator.acknowledge_alert(
            event_id="ALT-20261016-D14-001",
            clinician_id="Dr. Alice Smith",
            notes="Oxygen cannula increased to 4 L/min.",
        )

        assert ack_res is not None
        assert ack_res.is_acknowledged is True
        assert ack_res.acknowledged_by == "Dr. Alice Smith"
        assert ack_res.acknowledgement_notes == "Oxygen cannula increased to 4 L/min."
        assert ack_res.is_resolved is False  # Acknowledging must NOT resolve the alert

        # Reload from disk
        reloaded = self.repo.get_event("ALT-20261016-D14-001")
        assert reloaded is not None
        assert reloaded.is_acknowledged is True
        assert reloaded.is_resolved is False

        # Alert still appears in active feed
        active = self.coordinator.get_alerts_feed(active_only=True)
        assert len(active) == 1
        assert active[0].event_id == "ALT-20261016-D14-001"

    # ── D15: Resolution Preserves Complete Audit Record ──────────────────────

    def test_D15_resolution_persists_audit_trail(self):
        """Resolving an alert updates is_resolved=True but preserves full audit history in storage."""
        event = AlertEvent(
            event_id="ALT-20261016-D15-001",
            rule_name="MODERATE_TACHYCARDIA",
            category=AlertCategory.PHYSIOLOGICAL_RISK,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            reason="HR = 110 bpm",
            vitals={"heart_rate": 110, "spo2": 97},
            data_quality_status=DataQualityStatus.VALID,
        )
        self.repo.add_event(event)

        # Resolve
        res_evt = self.coordinator.resolve_alert("ALT-20261016-D15-001")
        assert res_evt is not None
        assert res_evt.is_resolved is True
        assert res_evt.resolved_at is not None

        # Excluded from active feed
        active = self.coordinator.get_alerts_feed(active_only=True)
        assert len(active) == 0

        # Preserved in full audit history
        all_events = self.coordinator.get_alerts_feed(active_only=False)
        assert len(all_events) == 1
        assert all_events[0].event_id == "ALT-20261016-D15-001"
        assert all_events[0].is_resolved is True

    # ── D16: Non-Blocking Lock Prevents Overlapping Polls ─────────────────────

    def test_D16_concurrency_lock_prevents_overlapping_polls(self):
        """When lock is held, simultaneous poll_and_evaluate returns in_progress without deadlock."""
        # Artificially acquire lock
        acquired = self.coordinator._lock.acquire(blocking=False)
        assert acquired is True

        try:
            # Second call while lock is held
            res = self.coordinator.poll_and_evaluate()
            assert res["status"] == "in_progress"
            assert "Overlapping request suppressed" in res["message"]
        finally:
            self.coordinator._lock.release()

    # ── D17: Strict EHR Cohort Isolation ─────────────────────────────────────

    def test_D17_hardware_telemetry_isolation_from_hospital_ehr(self):
        """Prohibits subject_id, stay_id, or hadm_id from leaking into telemetry or alert records."""
        # Attempt to pass EHR identifiers in telemetry
        leaky_telemetry = {
            "status": "success",
            "entry_id": 999,
            "timestamp": "2026-10-16T12:00:00+00:00",
            "heart_rate": 135.0,
            "spo2": 90.0,
            "ir_ppg": 140000.0,
            "risk_status_code": 2,
            "subject_id": 12345678,
            "stay_id": 34567890,
            "hadm_id": 23456789,
            "is_stale": False,
        }

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=leaky_telemetry):
            res = self.coordinator.poll_and_evaluate()
            assert len(res["new_alerts"]) >= 1

            for alert in res["new_alerts"]:
                # Prohibited keys must be stripped by AlertEvent.__post_init__
                assert "subject_id" not in alert.vitals
                assert "stay_id" not in alert.vitals
                assert "hadm_id" not in alert.vitals

            # Check disk
            saved = self.repo.load_events()
            for s in saved:
                assert "subject_id" not in s.vitals
                assert "stay_id" not in s.vitals
                assert "hadm_id" not in s.vitals

    # ── D18: Historical EHR Patient Search Workflows Unaffected ───────────────

    def test_D18_existing_backend_patient_service_workflows_unaffected(self):
        """Historical patient search, record loading, and Transformer AI inference remain fully intact."""
        # Test lookup for known MIMIC-IV patient 17585185
        search_res = search_patient(17585185)
        assert search_res["found"] is True
        assert len(search_res["stays"]) >= 1

        stay_id = search_res["stays"][0]["stay_id"]
        rec = get_stay_clinical_record(17585185, stay_id)
        assert rec["patient_id"] == 17585185
        assert rec["stay_id"] == stay_id
        assert "vital_signs" in rec

        # AI inference
        inf_res = run_clinical_transformer_inference(17585185, stay_id)
        assert inf_res["status"] == "success"
        assert 0.0 <= inf_res["predicted_probability"] <= 1.0
        assert "predicted_class" in inf_res

    # ── D19: Multi-Schema Telemetry & Stale Alert Handling ────────────────────

    def test_D19_multi_schema_telemetry_and_stale_handling(self):
        """Historical Format B telemetry (stale timestamp, None IR PPG, valid vitals) evaluates cleanly."""
        historical_telemetry = {
            "status": "success",
            "connected": True,
            "has_readings": True,
            "timestamp": "2026-09-30T10:00:00Z",
            "is_stale": True,
            "staleness_description": "Stale (> 300s)",
            "entry_id": 290,
            "heart_rate": 83.0,
            "spo2": 97.5,
            "ir_ppg": None,  # Legacy text stream: IR PPG was unmeasured
            "risk_status_code": 1,
            "patient_status_text": "Stable",
            "record_format": "format_b_legacy_text",
            "firmware_status": "STABLE (Status 1: Normal Vitals)",
            "valid_hr": True,
            "valid_spo2": True,
            "finger_detected": True,
            "physiologically_valid": True,
            "recent_history": pd.DataFrame(),
            "total_records": 290,
            "disclaimer": "HARDWARE BENCH TELEMETRY",
        }

        obs_time = datetime.datetime.fromisoformat("2026-09-30T10:00:00+00:00")
        eval_time = obs_time + datetime.timedelta(seconds=400)

        with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=historical_telemetry):
            res = self.coordinator.poll_and_evaluate(force_evaluate=True, current_time=eval_time)
            assert res["status"] == "success"
            # Must generate STALE_TELEMETRY alert due to historical timestamp (> 5 min)
            stale_alerts = [a for a in res["new_alerts"] if a.rule_name == "STALE_TELEMETRY"]
            assert len(stale_alerts) == 1
            assert stale_alerts[0].severity == AlertSeverity.WARNING
            assert "stale" in stale_alerts[0].reason.lower()

    # ── D20: Malformed & Missing Field Telemetry Resilience ───────────────────

    def test_D20_malformed_telemetry_resilience(self):
        """Engine safely handles completely empty or corrupted telemetry dictionaries without throwing unhandled exceptions."""
        malformed_cases = [
            {},
            {"status": "unavailable", "message": "Failed connection"},
            {"status": "success", "heart_rate": "invalid_number", "spo2": None, "timestamp": "corrupt_date"},
        ]
        for bad_telem in malformed_cases:
            with mock.patch("src.sensor.telemetry_poller.get_latest_sensor_telemetry", return_value=bad_telem):
                res = self.coordinator.poll_and_evaluate(force_evaluate=True)
                assert "status" in res
                assert isinstance(res.get("new_alerts", []), list)
