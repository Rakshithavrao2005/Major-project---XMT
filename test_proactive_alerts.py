"""
tests/test_proactive_alerts.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive unit test suite for the proactive alert engine (src/sensor/alert_engine.py).

Test Groups:
P01: Valid Normal Telemetry Baseline
P02: Missing and Incomplete Sensor Values
P03: Stale Telemetry Detection
P04: Optical Contact & Finger Detachment (Lead-Off)
P05: Invalid Physiological Bounds Filtering
P06: Heart Rate Threshold Tiers (Tachycardia / Bradycardia)
P07: Oxygen Saturation Threshold Tiers (Hypoxemia)
P08: Combined Sepsis Firmware Heuristic (Code 2)
P09: Deduplication Within Cooldown Window
P10: Independent Alert Categories During Active Cooldown
P11: Worsening Condition Cooldown Bypass (Escalation)
P12: Condition Normalization & Recovery Events
P13: Recovery Followed by Rapid Recurrence
P14: Timezone-Aware UTC Timestamp Enforcement
P15: Data Leakage Prevention (Rejects Hospital Patient Identifiers)
P16: Robustness on Malformed / Empty Dictionaries
P17: Clinician Acknowledgement Workflow
P18: Event Serialization & Deserialization Round-Trip
"""

import os
import sys
import datetime
import pytest

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.alert_engine import (
    SensorAlertEngine,
    AlertThresholdConfig,
    AlertEvent,
    AlertCategory,
    AlertSeverity,
    DataQualityStatus,
    DeliveryStatus,
    RESEARCH_ALERT_DISCLAIMER,
)


@pytest.fixture
def base_time():
    """Anchor reference UTC timestamp for deterministic tests."""
    return datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture
def engine():
    """Fresh alert engine instance with standard test configuration."""
    cfg = AlertThresholdConfig(
        cooldown_seconds=900.0,  # 15 minutes
        stale_seconds=300.0,     # 5 minutes
        hr_critical_high=130.0,
        hr_warning_high=100.0,
        hr_warning_low=50.0,
        hr_critical_low=40.0,
        spo2_critical_low=92.0,
        spo2_warning_low=95.0,
        worsening_hr_delta=15.0,
        worsening_spo2_delta=3.0,
        emit_recovery_events=True,
    )
    return SensorAlertEngine(config=cfg)


class TestProactiveAlertEngine:

    def test_P01_valid_normal_reading_produces_no_alerts(self, engine, base_time):
        """A normal healthy reading within all bounds triggers zero alerts."""
        normal_reading = {
            "heart_rate": 75.0,
            "spo2": 98.0,
            "ir_ppg": 95000.0,
            "risk_status": 0,
        }
        events = engine.evaluate_reading(normal_reading, observation_time=base_time, current_time=base_time)
        assert len(events) == 0, f"Expected 0 alerts for normal reading, got: {events}"
        assert len(engine.get_active_alerts()) == 0

    def test_P02_missing_measurements_handled_safely(self, engine, base_time):
        """Missing HR and SpO2 generate DATA_QUALITY warning and no clinical crash."""
        missing_reading = {
            "heart_rate": None,
            "spo2": None,
            "ir_ppg": 60000.0,
        }
        events = engine.evaluate_reading(missing_reading, observation_time=base_time, current_time=base_time)
        assert len(events) == 1
        evt = events[0]
        assert evt.rule_name == "MISSING_MEASUREMENTS"
        assert evt.category == AlertCategory.DATA_QUALITY
        assert evt.severity == AlertSeverity.WARNING
        assert evt.data_quality_status == DataQualityStatus.MISSING_VALUES
        assert "Missing sensor parameter" in evt.reason

    def test_P03_stale_telemetry_detection(self, engine, base_time):
        """Observations older than stale_seconds trigger STALE_TELEMETRY warning."""
        stale_reading = {
            "heart_rate": 78.0,
            "spo2": 98.0,
            "ir_ppg": 85000.0,
        }
        # Observation was 10 minutes ago (> 5 min threshold)
        obs_time = base_time - datetime.timedelta(minutes=10)
        events = engine.evaluate_reading(stale_reading, observation_time=obs_time, current_time=base_time)
        
        stale_events = [e for e in events if e.rule_name == "STALE_TELEMETRY"]
        assert len(stale_events) == 1
        assert stale_events[0].category == AlertCategory.DATA_QUALITY
        assert stale_events[0].data_quality_status == DataQualityStatus.STALE_TELEMETRY
        assert "no new observation received" in stale_events[0].reason

    def test_P04_poor_optical_contact_or_finger_detached(self, engine, base_time):
        """Low IR PPG amplitude (<= 50,000 counts) triggers POOR_OPTICAL_CONTACT."""
        detached_reading = {
            "heart_rate": 80.0,
            "spo2": 97.0,
            "ir_ppg": 25000.0, # Below 50,000 contact threshold
        }
        events = engine.evaluate_reading(detached_reading, observation_time=base_time, current_time=base_time)
        contact_events = [e for e in events if e.rule_name == "POOR_OPTICAL_CONTACT"]
        assert len(contact_events) == 1
        assert contact_events[0].category == AlertCategory.DATA_QUALITY
        assert contact_events[0].data_quality_status == DataQualityStatus.QUESTIONABLE_CONTACT
        # Crucial safety check: optical dropout does not trigger false physiological alarms
        clinical_events = [e for e in events if e.category == AlertCategory.PHYSIOLOGICAL_RISK]
        assert len(clinical_events) == 0

    def test_P05_invalid_physiological_bounds_filtering(self, engine, base_time):
        """Physiological artifacts (e.g. HR=250 or SpO2=45%) are flagged as data quality artifacts."""
        artifact_reading = {
            "heart_rate": 240.0, # Beyond max valid limit 220
            "spo2": 45.0,        # Below min valid limit 70
            "ir_ppg": 80000.0,
        }
        events = engine.evaluate_reading(artifact_reading, observation_time=base_time, current_time=base_time)
        bounds_events = [e for e in events if e.rule_name == "INVALID_PHYSIOLOGICAL_BOUNDS"]
        assert len(bounds_events) == 1
        assert bounds_events[0].category == AlertCategory.DATA_QUALITY
        assert bounds_events[0].data_quality_status == DataQualityStatus.INVALID_BOUNDS

    def test_P06_heart_rate_threshold_tiers(self, engine, base_time):
        """Verify moderate tachycardia, critical tachycardia, and bradycardia tiers."""
        # 1. Moderate Tachycardia (110 bpm)
        events_mod = engine.evaluate_reading(
            {"heart_rate": 110.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time,
        )
        assert any(e.rule_name == "MODERATE_TACHYCARDIA" and e.severity == AlertSeverity.WARNING for e in events_mod)

        # Clear state for next test
        engine.clear_state()

        # 2. Critical Tachycardia (135 bpm >= 130)
        events_crit = engine.evaluate_reading(
            {"heart_rate": 135.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time,
        )
        assert any(e.rule_name == "CRITICAL_TACHYCARDIA" and e.severity == AlertSeverity.CRITICAL for e in events_crit)

        engine.clear_state()

        # 3. Moderate Bradycardia (45 bpm < 50)
        events_brady = engine.evaluate_reading(
            {"heart_rate": 45.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time,
        )
        assert any(e.rule_name == "MODERATE_BRADYCARDIA" and e.severity == AlertSeverity.WARNING for e in events_brady)

    def test_P07_oxygen_saturation_threshold_tiers(self, engine, base_time):
        """Verify moderate (<95%) and critical (<92%) hypoxemia thresholds."""
        # 1. Moderate Hypoxemia (93%)
        events_mod = engine.evaluate_reading(
            {"heart_rate": 78.0, "spo2": 93.5, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time,
        )
        assert any(e.rule_name == "MODERATE_HYPOXEMIA" and e.severity == AlertSeverity.WARNING for e in events_mod)

        engine.clear_state()

        # 2. Critical Hypoxemia (89% < 92%)
        events_crit = engine.evaluate_reading(
            {"heart_rate": 78.0, "spo2": 89.0, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time,
        )
        assert any(e.rule_name == "CRITICAL_HYPOXEMIA" and e.severity == AlertSeverity.CRITICAL for e in events_crit)

    def test_P08_combined_sepsis_firmware_heuristic(self, engine, base_time):
        """HR >= 130 and SpO2 < 92% (or firmware status 2) triggers combined sepsis alert."""
        shock_reading = {
            "heart_rate": 138.0,
            "spo2": 88.0,
            "ir_ppg": 85000.0,
            "risk_status": 2,
        }
        events = engine.evaluate_reading(shock_reading, observation_time=base_time, current_time=base_time)
        sepsis_events = [e for e in events if e.rule_name == "SEPSIS_HEURISTIC_CRITICAL"]
        assert len(sepsis_events) == 1
        assert sepsis_events[0].severity == AlertSeverity.CRITICAL
        assert sepsis_events[0].category == AlertCategory.PHYSIOLOGICAL_RISK
        assert "Combined Sepsis Screening Alert" in sepsis_events[0].reason

    def test_P09_deduplication_suppresses_repeat_alerts_within_cooldown(self, engine, base_time):
        """Identical abnormal readings within the 15-minute cooldown do not re-alert."""
        abnormal = {"heart_rate": 135.0, "spo2": 98.0, "ir_ppg": 90000.0}
        
        # Poll 1 at T=0
        t0 = base_time
        events_t0 = engine.evaluate_reading(abnormal, observation_time=t0, current_time=t0)
        assert len(events_t0) == 1
        assert events_t0[0].rule_name == "CRITICAL_TACHYCARDIA"

        # Poll 2 at T=30s (identical reading) -> must be suppressed
        t1 = t0 + datetime.timedelta(seconds=30)
        events_t1 = engine.evaluate_reading(abnormal, observation_time=t1, current_time=t1)
        assert len(events_t1) == 0, "Duplicate alert within cooldown was not suppressed"

        # Poll 3 at T=14 min (within 15 min cooldown) -> must be suppressed
        t2 = t0 + datetime.timedelta(minutes=14)
        events_t2 = engine.evaluate_reading(abnormal, observation_time=t2, current_time=t2)
        assert len(events_t2) == 0, "Alert at 14 minutes should still be suppressed"

        # Poll 4 at T=16 min (cooldown expired) -> must re-alert
        t3 = t0 + datetime.timedelta(minutes=16)
        events_t3 = engine.evaluate_reading(abnormal, observation_time=t3, current_time=t3)
        assert len(events_t3) == 1
        assert events_t3[0].rule_name == "CRITICAL_TACHYCARDIA"

    def test_P10_independent_alert_categories_fire_during_cooldown(self, engine, base_time):
        """An active tachycardia cooldown does not block a newly occurring hypoxemia alert."""
        t0 = base_time
        # Tachycardia fires at T=0
        e0 = engine.evaluate_reading(
            {"heart_rate": 132.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=t0, current_time=t0,
        )
        assert len(e0) == 1
        assert e0[0].rule_name == "CRITICAL_TACHYCARDIA"

        # At T=2 min, patient additionally develops hypoxemia (SpO2=88%)
        t1 = t0 + datetime.timedelta(minutes=2)
        e1 = engine.evaluate_reading(
            {"heart_rate": 132.0, "spo2": 88.0, "ir_ppg": 90000.0},
            observation_time=t1, current_time=t1,
        )
        # CRITICAL_TACHYCARDIA is in cooldown, but CRITICAL_HYPOXEMIA must fire independently!
        hypox_events = [e for e in e1 if e.rule_name == "CRITICAL_HYPOXEMIA"]
        assert len(hypox_events) == 1
        assert hypox_events[0].severity == AlertSeverity.CRITICAL

    def test_P11_worsening_condition_bypasses_cooldown(self, engine, base_time):
        """A significant physiological deterioration or severity escalation bypasses cooldown."""
        t0 = base_time
        # Moderate Tachycardia (105 bpm, WARNING)
        e0 = engine.evaluate_reading(
            {"heart_rate": 105.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=t0, current_time=t0,
        )
        assert any(e.rule_name == "MODERATE_TACHYCARDIA" and e.severity == AlertSeverity.WARNING for e in e0)

        # 3 minutes later: HR spikes to 135 bpm (escalates to CRITICAL_TACHYCARDIA and jumps by 30 bpm)
        t1 = t0 + datetime.timedelta(minutes=3)
        e1 = engine.evaluate_reading(
            {"heart_rate": 135.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=t1, current_time=t1,
        )
        # Must bypass cooldown and alert immediately
        crit_events = [e for e in e1 if e.rule_name == "CRITICAL_TACHYCARDIA"]
        assert len(crit_events) == 1
        assert crit_events[0].severity == AlertSeverity.CRITICAL

    def test_P12_condition_normalization_and_recovery_events(self, engine, base_time):
        """When an abnormal condition resolves, emit an INFO recovery event and update state."""
        t0 = base_time
        # Patient is tachycardic at T=0
        engine.evaluate_reading(
            {"heart_rate": 135.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=t0, current_time=t0,
        )
        assert len(engine.get_active_alerts()) == 1

        # At T=5 min, patient heart rate normalizes to 75 bpm
        t1 = t0 + datetime.timedelta(minutes=5)
        e1 = engine.evaluate_reading(
            {"heart_rate": 75.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=t1, current_time=t1,
        )
        rec_events = [e for e in e1 if "RESOLVED" in e.rule_name]
        assert len(rec_events) >= 1
        rec = rec_events[0]
        assert rec.category == AlertCategory.SYSTEM_RECOVERY
        assert rec.severity == AlertSeverity.INFO
        assert rec.is_resolved is True
        assert rec.resolved_at is not None
        # Active alerts list should now be cleared
        assert len(engine.get_active_alerts()) == 0

    def test_P13_recovery_followed_by_rapid_recurrence_fires_immediately(self, engine, base_time):
        """If a condition resolves and then recurs within 15 minutes, it alerts immediately without suppression."""
        t0 = base_time
        # 1. Tachycardia at T=0
        engine.evaluate_reading({"heart_rate": 135.0, "spo2": 98.0, "ir_ppg": 90000.0}, observation_time=t0, current_time=t0)
        
        # 2. Normalizes at T=3 min
        t1 = t0 + datetime.timedelta(minutes=3)
        engine.evaluate_reading({"heart_rate": 75.0, "spo2": 98.0, "ir_ppg": 90000.0}, observation_time=t1, current_time=t1)

        # 3. Recurrence at T=6 min (well within original 15 min cooldown window)
        t2 = t0 + datetime.timedelta(minutes=6)
        e2 = engine.evaluate_reading({"heart_rate": 136.0, "spo2": 98.0, "ir_ppg": 90000.0}, observation_time=t2, current_time=t2)
        
        # Must alert immediately because prior event was resolved
        tach_events = [e for e in e2 if e.rule_name == "CRITICAL_TACHYCARDIA"]
        assert len(tach_events) == 1, "Recurrent alert was improperly suppressed"

    def test_P14_timezone_aware_utc_timestamp_enforcement(self, engine, base_time):
        """All alert event timestamps are strictly timezone-aware UTC."""
        events = engine.evaluate_reading(
            {"heart_rate": 135.0, "spo2": 88.0, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time,
        )
        assert len(events) > 0
        for e in events:
            assert e.timestamp.tzinfo is not None
            assert e.timestamp.utcoffset() == datetime.timedelta(0)

    def test_P15_data_leakage_prevention_rejects_hospital_patient_ids(self, engine, base_time):
        """Historical EHR identifiers (subject_id, stay_id) are scrubbed from vitals dict."""
        compromised_reading = {
            "heart_rate": 135.0,
            "spo2": 98.0,
            "ir_ppg": 90000.0,
            "subject_id": 10002930,  # Simulated illegal EHR leak
            "stay_id": 35629889,
        }
        events = engine.evaluate_reading(compromised_reading, observation_time=base_time, current_time=base_time)
        assert len(events) > 0
        for e in events:
            assert "subject_id" not in e.vitals
            assert "stay_id" not in e.vitals
            # Verify serialization as well
            d = e.to_dict()
            assert "subject_id" not in d["vitals"]
            assert "stay_id" not in d["vitals"]

    def test_P16_robustness_on_malformed_empty_payloads(self, engine, base_time):
        """Engine handles empty, corrupt, or non-numeric dictionaries gracefully."""
        # Empty dict
        e1 = engine.evaluate_reading({}, observation_time=base_time, current_time=base_time)
        assert any(e.rule_name == "MISSING_MEASUREMENTS" for e in e1)

        engine.clear_state()

        # String-typed corrupt values
        corrupt = {"heart_rate": "INVALID_STR", "spo2": "ERROR", "ir_ppg": None}
        e2 = engine.evaluate_reading(corrupt, observation_time=base_time, current_time=base_time)
        assert any(e.rule_name == "MISSING_MEASUREMENTS" for e in e2)

    def test_P17_clinician_acknowledgement_workflow(self, engine, base_time):
        """Clinician acknowledgement updates acknowledgement status, notes, and timestamp."""
        events = engine.evaluate_reading(
            {"heart_rate": 135.0, "spo2": 98.0, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time,
        )
        assert len(events) == 1
        target_event = events[0]
        assert target_event.is_acknowledged is False

        # Clinician acknowledges the alert
        ack_time = base_time + datetime.timedelta(minutes=2)
        success = engine.acknowledge_alert(
            event_id=target_event.event_id,
            clinician_id="DR_SMITH_ICU_TELEMED",
            notes="Patient assessed via video feed. Fluid bolus initiated.",
            ack_time=ack_time,
        )
        assert success is True

        # Verify updated event in history
        history_evt = [e for e in engine.alert_history if e.event_id == target_event.event_id][0]
        assert history_evt.is_acknowledged is True
        assert history_evt.acknowledged_by == "DR_SMITH_ICU_TELEMED"
        assert history_evt.acknowledged_at == ack_time
        assert "Fluid bolus initiated" in history_evt.acknowledgement_notes

    def test_P18_event_serialization_round_trip(self, engine, base_time):
        """AlertEvent serializes to dict and deserializes back faithfully."""
        events = engine.evaluate_reading(
            {"heart_rate": 135.0, "spo2": 88.0, "ir_ppg": 90000.0},
            observation_time=base_time, current_time=base_time, session_id="SESSION-20261016-TEST",
        )
        assert len(events) > 0
        evt = events[0]
        d = evt.to_dict()
        assert isinstance(d, dict)
        assert d["event_id"] == evt.event_id
        assert d["session_id"] == "SESSION-20261016-TEST"
        assert d["disclaimer"] == RESEARCH_ALERT_DISCLAIMER

        # Round-trip deserialization
        reconstructed = AlertEvent.from_dict(d)
        assert reconstructed.event_id == evt.event_id
        assert reconstructed.rule_name == evt.rule_name
        assert reconstructed.category == evt.category
        assert reconstructed.severity == evt.severity
        assert reconstructed.timestamp == evt.timestamp
        assert reconstructed.vitals == evt.vitals
        assert reconstructed.data_quality_status == evt.data_quality_status
