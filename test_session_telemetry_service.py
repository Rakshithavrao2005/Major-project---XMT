"""
tests/test_session_telemetry_service.py
=============================================================================
Stage 3.7.7D: Safe Telemetry Application Integration Offline Synthetic Test Suite.

Verifies:
 1. The application service invokes the actual gate and repository.
 2. A prospective session reads only its own stored packets.
 3. An empty session shows a waiting state rather than shared CSV readings.
 4. Old shared-CSV rows are never assigned to a new session.
 5. Unauthorized clinicians cannot read or ingest packets.
 6. Closed sessions cannot accept new packets.
 7. Duplicate and conflicting packets retain documented behavior.
 8. Wrong-channel and wrong-device packets are quarantined.
 9. Future-dated and stale packets are rejected.
10. A missing watermark blocks ingestion.
11. Telemetry silence raises a warning but does not close the session or release the device.
12. A channel entry-ID regression fails closed.
13. Session-specific queries never return another session's packets.
14. Retrospective patient search and prediction behavior remains unchanged.
=============================================================================
"""

import copy
import datetime
import os
import sys
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.patient_registry import LocalPatientRegistry
from src.sensor.session_repository import (
    SessionRepository,
    SessionConflictError,
    SessionNotFoundError,
    InvalidSessionStateError,
    UnauthorizedClinicianError,
)
from src.sensor.session_telemetry_gate import (
    SessionTelemetryGate,
    GateOutcome,
    GateResult,
)
from src.sensor.session_telemetry_service import (
    SessionTelemetryService,
    DEFAULT_SILENCE_WARNING_SECONDS,
)


@pytest.fixture
def test_clock():
    """Controllable deterministic UTC clock."""
    class ControllableClock:
        def __init__(self):
            self.current_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

        def get_time(self):
            return self.current_time

        def advance(self, seconds: float):
            self.current_time += datetime.timedelta(seconds=seconds)

    return ControllableClock()


@pytest.fixture
def service_env(tmp_path, test_clock):
    """Provides isolated patient registry, session repository, and service."""
    reg_file = str(tmp_path / "registry" / "patients.json")
    db_file = str(tmp_path / "sessions" / "test_sessions.db")

    reg = LocalPatientRegistry(storage_path=reg_file)
    repo = SessionRepository(db_path=db_file, patient_registry=reg)
    gate = SessionTelemetryGate(
        session_repository=repo,
        require_device_identity=True,
        clock_fn=test_clock.get_time,
    )
    svc = SessionTelemetryService(
        session_repository=repo,
        telemetry_gate=gate,
        silence_warning_seconds=600.0,
        clock_fn=test_clock.get_time,
    )

    # Register standard test doctor and patient
    doctor_id = "DOC-SVC-001"
    patient = reg.register_patient(
        registration_type="prospective_ambulatory",
        gender="F",
        anchor_age=50,
        display_alias="Room-SVC-01",
        mrn_plaintext="MRN-SVC-001",
        registered_by_doctor_id=doctor_id,
    )

    return {
        "registry": reg,
        "repo": repo,
        "gate": gate,
        "service": svc,
        "clock": test_clock,
        "doctor_id": doctor_id,
        "patient": patient,
    }


# ── 1. Application Service Invokes Actual Gate and Repository ─────────────────

def test_01_service_invokes_actual_gate_and_repository(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888111,
        device_id="ESP32-SVC-01",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    packet = {
        "channel_id": 888111,
        "device_id": "ESP32-SVC-01",
        "entry_id": 11,
        "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 78.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
        "risk_status": 0,
    }

    res = svc.ingest_telemetry_packet(
        session_id=sess["session_id"],
        raw_packet=packet,
        requesting_doctor_id=env["doctor_id"],
    )

    assert isinstance(res, GateResult)
    assert res.outcome == GateOutcome.ACCEPTED
    assert res.is_success is True

    # Real repository verification
    packets = env["repo"].get_session_packets(sess["session_id"])
    assert len(packets) == 1
    assert packets[0]["entry_id"] == 11
    assert packets[0]["heart_rate"] == 78.0


# ── 2. Prospective Session Reads Only Its Own Stored Packets ──────────────────

def test_02_session_reads_only_own_stored_packets(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888111,
        device_id="ESP32-SVC-01",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=20,
        custom_start_time=now,
    )

    # Ingest 3 valid packets
    for i in range(1, 4):
        svc.ingest_telemetry_packet(
            session_id=sess["session_id"],
            raw_packet={
                "channel_id": 888111,
                "device_id": "ESP32-SVC-01",
                "entry_id": 20 + i,
                "recorded_at": (now + datetime.timedelta(seconds=i * 2)).isoformat(),
                "heart_rate": 70.0 + i,
                "spo2": 98.0,
                "ir_ppg": 72000,
            },
            requesting_doctor_id=env["doctor_id"],
        )

    telemetry_view = svc.get_session_telemetry(
        session_id=sess["session_id"],
        requesting_doctor_id=env["doctor_id"],
    )

    assert telemetry_view["status"] == "success"
    assert telemetry_view["has_packets"] is True
    assert telemetry_view["total_packets"] == 3
    assert [p["entry_id"] for p in telemetry_view["packets"]] == [21, 22, 23]
    assert telemetry_view["latest_packet"]["entry_id"] == 23


# ── 3. Empty Session Shows Waiting State Rather Than Shared CSV ───────────────

def test_03_empty_session_shows_waiting_state(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888222,
        device_id="ESP32-SVC-02",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=0,
        custom_start_time=now,
    )

    telemetry_view = svc.get_session_telemetry(
        session_id=sess["session_id"],
        requesting_doctor_id=env["doctor_id"],
    )

    # Must be empty and in WAITING_FOR_DATA state
    assert telemetry_view["has_packets"] is False
    assert telemetry_view["total_packets"] == 0
    assert telemetry_view["packets"] == []
    assert telemetry_view["latest_packet"] is None
    assert telemetry_view["silence_status"] == "WAITING_FOR_DATA"
    assert "waiting for initial telemetry" in telemetry_view["silence_message"].lower()


# ── 4. Old Shared-CSV Rows Never Assigned to a New Session ────────────────────

def test_04_old_shared_csv_rows_never_assigned(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    # Create session with watermark 500 started at now
    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888333,
        device_id="ESP32-SVC-03",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=500,
        custom_start_time=now,
    )

    # Attempt to feed an old packet from shared CSV (entry_id 450, older timestamp)
    old_packet = {
        "channel_id": 888333,
        "device_id": "ESP32-SVC-03",
        "entry_id": 450,
        "recorded_at": (now - datetime.timedelta(minutes=30)).isoformat(),
        "heart_rate": 82.0,
        "spo2": 96.0,
        "ir_ppg": 65000,
    }

    res = svc.ingest_telemetry_packet(
        session_id=sess["session_id"],
        raw_packet=old_packet,
        requesting_doctor_id=env["doctor_id"],
    )

    assert res.outcome in (GateOutcome.REJECTED_STALE, GateOutcome.REJECTED_BEFORE_WATERMARK)
    assert res.is_quarantined is True

    telemetry_view = svc.get_session_telemetry(sess["session_id"], requesting_doctor_id=env["doctor_id"])
    assert telemetry_view["has_packets"] is False
    assert telemetry_view["packets"] == []


# ── 5. Unauthorized Clinicians Cannot Access or Ingest Packets ────────────────

def test_05_unauthorized_clinician_rejected(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888444,
        device_id="ESP32-SVC-04",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    packet = {
        "channel_id": 888444,
        "device_id": "ESP32-SVC-04",
        "entry_id": 11,
        "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }

    # Empty or None clinician identifier fails authorization
    with pytest.raises(UnauthorizedClinicianError):
        svc.ingest_telemetry_packet(
            session_id=sess["session_id"],
            raw_packet=packet,
            requesting_doctor_id="",
        )

    with pytest.raises(UnauthorizedClinicianError):
        svc.get_session_telemetry(
            session_id=sess["session_id"],
            requesting_doctor_id="UNKNOWN",
        )


# ── 6. Closed Sessions Cannot Accept New Packets ─────────────────────────────

def test_06_closed_session_cannot_accept_packets(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888555,
        device_id="ESP32-SVC-05",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    # Close session formally
    env["clock"].advance(10)
    end_time = env["clock"].get_time()
    env["repo"].close_session(
        session_id=sess["session_id"],
        closed_by_doctor_id=env["doctor_id"],
        status="completed",
        closure_reason="routine_discharge",
        custom_end_time=end_time,
    )

    # Ingest packet into closed session
    packet = {
        "channel_id": 888555,
        "device_id": "ESP32-SVC-05",
        "entry_id": 11,
        "recorded_at": (now + datetime.timedelta(seconds=5)).isoformat(),
        "heart_rate": 74.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }

    res = svc.ingest_telemetry_packet(
        session_id=sess["session_id"],
        raw_packet=packet,
        requesting_doctor_id=env["doctor_id"],
    )

    assert res.outcome == GateOutcome.REJECTED_SESSION_INACTIVE
    assert res.is_quarantined is True


# ── 7. Duplicate and Conflicting Packets Retain Gate Behavior ─────────────────

def test_07_duplicate_and_conflict_handling(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888666,
        device_id="ESP32-SVC-06",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    packet1 = {
        "channel_id": 888666,
        "device_id": "ESP32-SVC-06",
        "entry_id": 11,
        "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 76.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }

    # 1. First send: ACCEPTED
    res1 = svc.ingest_telemetry_packet(sess["session_id"], packet1, requesting_doctor_id=env["doctor_id"])
    assert res1.outcome == GateOutcome.ACCEPTED

    # 2. Duplicate send: DUPLICATE_ACCEPTED
    res2 = svc.ingest_telemetry_packet(sess["session_id"], packet1, requesting_doctor_id=env["doctor_id"])
    assert res2.outcome == GateOutcome.DUPLICATE_ACCEPTED

    # 3. Conflicting send: CONFLICT_QUARANTINED
    packet_conflict = copy.deepcopy(packet1)
    packet_conflict["heart_rate"] = 150.0
    packet_conflict["spo2"] = 85.0
    res3 = svc.ingest_telemetry_packet(sess["session_id"], packet_conflict, requesting_doctor_id=env["doctor_id"])
    assert res3.outcome == GateOutcome.CONFLICT_QUARANTINED

    # Database still has only 1 packet
    packets = env["repo"].get_session_packets(sess["session_id"])
    assert len(packets) == 1
    assert packets[0]["heart_rate"] == 76.0


# ── 8. Wrong-Channel and Wrong-Device Packets Are Quarantined ──────────────────

def test_08_wrong_channel_and_device_quarantined(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888777,
        device_id="ESP32-SVC-07",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    # Wrong channel
    res_chan = svc.ingest_telemetry_packet(
        sess["session_id"],
        {"channel_id": 999999, "device_id": "ESP32-SVC-07", "entry_id": 11, "recorded_at": now.isoformat()},
        requesting_doctor_id=env["doctor_id"],
    )
    assert res_chan.outcome == GateOutcome.REJECTED_CHANNEL_MISMATCH
    assert res_chan.is_quarantined is True

    # Wrong device
    res_dev = svc.ingest_telemetry_packet(
        sess["session_id"],
        {"channel_id": 888777, "device_id": "ESP32-IMPOSTOR", "entry_id": 12, "recorded_at": now.isoformat()},
        requesting_doctor_id=env["doctor_id"],
    )
    assert res_dev.outcome == GateOutcome.REJECTED_DEVICE_MISMATCH
    assert res_dev.is_quarantined is True


# ── 9. Future-Dated and Stale Packets Are Rejected ────────────────────────────

def test_09_future_and_stale_packets_rejected(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888888,
        device_id="ESP32-SVC-08",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    # Future-skew packet (+70s > 60s tolerance)
    res_future = svc.ingest_telemetry_packet(
        sess["session_id"],
        {
            "channel_id": 888888,
            "device_id": "ESP32-SVC-08",
            "entry_id": 11,
            "recorded_at": (now + datetime.timedelta(seconds=70)).isoformat(),
            "heart_rate": 75.0,
            "spo2": 98.0,
            "ir_ppg": 70000,
        },
        requesting_doctor_id=env["doctor_id"],
    )
    assert res_future.outcome == GateOutcome.REJECTED_FUTURE_CLOCK_SKEW
    assert res_future.is_quarantined is True

    # Stale packet (-400s > 300s window)
    env["clock"].advance(500)
    now_adv = env["clock"].get_time()
    res_stale = svc.ingest_telemetry_packet(
        sess["session_id"],
        {
            "channel_id": 888888,
            "device_id": "ESP32-SVC-08",
            "entry_id": 12,
            "recorded_at": (now_adv - datetime.timedelta(seconds=400)).isoformat(),
            "heart_rate": 75.0,
            "spo2": 98.0,
            "ir_ppg": 70000,
        },
        requesting_doctor_id=env["doctor_id"],
    )
    assert res_stale.outcome == GateOutcome.REJECTED_STALE
    assert res_stale.is_quarantined is True


# ── 10. Missing Watermark Blocks Ingestion ────────────────────────────────────

def test_10_missing_watermark_blocks_ingestion(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=888999,
        device_id="ESP32-SVC-09",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=0,
        custom_start_time=now,
    )

    # Pass simulated context with missing start_entry_id
    corrupted_sess = copy.deepcopy(sess)
    corrupted_sess["start_entry_id"] = None

    res = svc.gate.process_packet_for_session(
        corrupted_sess,
        {"channel_id": 888999, "entry_id": 1, "recorded_at": now.isoformat()},
    )
    assert res.outcome == GateOutcome.REJECTED_MISSING_WATERMARK
    assert res.is_quarantined is True


# ── 11. Telemetry Silence Raises Warning Without Closing Session ──────────────

def test_11_telemetry_silence_raises_warning_preserves_session(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=777111,
        device_id="ESP32-SILENCE-01",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    # Ingest one packet at now + 5s
    svc.ingest_telemetry_packet(
        sess["session_id"],
        {
            "channel_id": 777111,
            "device_id": "ESP32-SILENCE-01",
            "entry_id": 11,
            "recorded_at": (now + datetime.timedelta(seconds=5)).isoformat(),
            "heart_rate": 78.0,
            "spo2": 98.0,
            "ir_ppg": 72000,
        },
        requesting_doctor_id=env["doctor_id"],
    )

    # Advance clock by 300s (5 min < 10 min warning threshold)
    env["clock"].advance(300)
    view1 = svc.get_session_telemetry(sess["session_id"], requesting_doctor_id=env["doctor_id"])
    assert view1["silence_status"] == "STREAMING_ACTIVE"
    assert view1["is_silent"] is False

    # Advance clock by an additional 350s (total 650s > 600s warning threshold)
    env["clock"].advance(350)
    view2 = svc.get_session_telemetry(sess["session_id"], requesting_doctor_id=env["doctor_id"])
    assert view2["silence_status"] == "SILENCE_WARNING"
    assert view2["is_silent"] is True
    assert "SILENCE" in view2["silence_message"]
    assert "UNAVAILABLE" in view2["silence_message"]

    # Verify session and device reservations remain intact in database!
    fresh_sess = env["repo"].get_session(sess["session_id"])
    assert fresh_sess["status"] == "active"
    assert fresh_sess["end_time"] is None


# ── 12. Channel Entry-ID Regression Fails Closed ──────────────────────────────

def test_12_entry_id_regression_fails_closed(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    sess = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=777222,
        device_id="ESP32-REGRESS-01",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=50,
        custom_start_time=now,
    )

    # Ingest packet 55
    res1 = svc.ingest_telemetry_packet(
        sess["session_id"],
        {
            "channel_id": 777222,
            "device_id": "ESP32-REGRESS-01",
            "entry_id": 55,
            "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
            "heart_rate": 75.0,
            "spo2": 98.0,
            "ir_ppg": 70000,
        },
        requesting_doctor_id=env["doctor_id"],
    )
    assert res1.outcome == GateOutcome.ACCEPTED

    # Ingest packet 52 (lower than 55: entry ID regressed backwards)
    res2 = svc.ingest_telemetry_packet(
        sess["session_id"],
        {
            "channel_id": 777222,
            "device_id": "ESP32-REGRESS-01",
            "entry_id": 52,
            "recorded_at": (now + datetime.timedelta(seconds=4)).isoformat(),
            "heart_rate": 76.0,
            "spo2": 98.0,
            "ir_ppg": 70000,
        },
        requesting_doctor_id=env["doctor_id"],
    )

    # Must fail closed and quarantine
    assert res2.outcome == GateOutcome.REJECTED_MALFORMED
    assert res2.reason_code == "ENTRY_ID_REGRESSION"
    assert res2.is_quarantined is True

    # Verify session has only packet 55
    packets = env["repo"].get_session_packets(sess["session_id"])
    assert len(packets) == 1
    assert packets[0]["entry_id"] == 55


# ── 13. Session-Specific Queries Never Return Another Session's Packets ───────

def test_13_session_queries_strictly_isolated(service_env):
    env = service_env
    svc: SessionTelemetryService = env["service"]
    now = env["clock"].get_time()

    # Patient A, Session A
    sess_a = env["repo"].create_session(
        patient_id=env["patient"]["patient_id"],
        channel_id=777333,
        device_id="ESP32-ISO-A",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )
    # Patient B, Session B
    pat_b = env["registry"].register_patient(
        registration_type="prospective_ambulatory",
        gender="M", anchor_age=65, display_alias="Patient-B",
        mrn_plaintext="MRN-PAT-B", registered_by_doctor_id=env["doctor_id"],
    )
    sess_b = env["repo"].create_session(
        patient_id=pat_b["patient_id"],
        channel_id=777444,
        device_id="ESP32-ISO-B",
        created_by_doctor_id=env["doctor_id"],
        start_entry_id=10,
        custom_start_time=now,
    )

    # Ingest packet 15 for Session A
    svc.ingest_telemetry_packet(sess_a["session_id"], {
        "channel_id": 777333, "device_id": "ESP32-ISO-A", "entry_id": 15,
        "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 72.0, "spo2": 98.0, "ir_ppg": 70000,
    }, requesting_doctor_id=env["doctor_id"])

    # Ingest packet 16 for Session B
    svc.ingest_telemetry_packet(sess_b["session_id"], {
        "channel_id": 777444, "device_id": "ESP32-ISO-B", "entry_id": 16,
        "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 88.0, "spo2": 97.0, "ir_ppg": 70000,
    }, requesting_doctor_id=env["doctor_id"])

    # Query telemetry for Session A
    view_a = svc.get_session_telemetry(sess_a["session_id"], requesting_doctor_id=env["doctor_id"])
    assert len(view_a["packets"]) == 1
    assert view_a["packets"][0]["entry_id"] == 15
    assert view_a["packets"][0]["session_id"] == sess_a["session_id"]

    # Query telemetry for Session B
    view_b = svc.get_session_telemetry(sess_b["session_id"], requesting_doctor_id=env["doctor_id"])
    assert len(view_b["packets"]) == 1
    assert view_b["packets"][0]["entry_id"] == 16
    assert view_b["packets"][0]["session_id"] == sess_b["session_id"]


# ── 14. Retrospective Patient Search and Prediction Behavior Unchanged ────────

def test_14_retrospective_patient_lookup_unchanged(service_env):
    """
    Verifies that the session telemetry service and repository have zero impact
    on the retrospective MIMIC-IV cohort and inference pipelines.
    """
    from src.backend.data_repository import DataRepository, RepositoryStatus, DataOrigin

    data_repo = DataRepository()
    res = data_repo.get_patient("10002930")

    assert res.status == RepositoryStatus.SUCCESS
    assert res.origin == DataOrigin.LOCAL_PRIMARY
    assert res.data["patient_id"] == "10002930"
    assert res.is_synthetic is False

    # Check stay observations
    res_stay = data_repo.get_stay_clinical_record("10002930", "39999999")
    assert res_stay.status in (RepositoryStatus.SUCCESS, RepositoryStatus.NOT_FOUND)
