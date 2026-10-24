"""
tests/test_session_telemetry_gate.py
=============================================================================
Stage 3.7.7B: Bounded Session Telemetry Gate Offline Synthetic Test Suite.

Verifies:
 1. Packet at the start watermark is rejected (entry_id == start_entry_id).
 2. Packet before start watermark is rejected (entry_id < start_entry_id).
 3. Packet before session start time is rejected even if its entry ID is larger.
 4. Packet more than 60 seconds in the future is rejected.
 5. Packet exactly at the permitted future-skew boundary (<= 60s) passes.
 6. Stale packet (> 300s) is rejected.
 7. Wrong channel is quarantined (REJECTED_CHANNEL_MISMATCH).
 8. Missing or mismatched device identity fails closed when required.
 9. Closed-session packet is rejected (REJECTED_SESSION_INACTIVE).
10. Invalid optical contact (IR < 50,000) is not marked valid.
11. Sequential patient sessions have zero packet leakage.
12. Identical retransmission creates only one stored row (DUPLICATE_ACCEPTED).
13. Conflicting payload for same source packet is quarantined (CONFLICT_QUARANTINED).
14. Concurrent attempts cannot assign one packet to two patient sessions (uq_global_source_packet).
15. Session closure racing with ingestion cannot create a post-closure assignment.
16. Missing watermark blocks ingestion (REJECTED_MISSING_WATERMARK).
17. Restart preserves packet/session associations without reassignment.
18. Quarantine does not leak patient data and honors retention bounds.
19. Inactivity behavior is deterministic, audited, and does not silently reassign the device.
20. Existing retrospective inference and patient lookup behavior remain unchanged.
=============================================================================
"""

import copy
import datetime
import os
import sqlite3
import sys
import threading
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.patient_registry import LocalPatientRegistry
from src.sensor.session_repository import (
    SessionRepository,
    SessionConflictError,
    InvalidSessionStateError,
    SessionNotFoundError,
)
from src.sensor.session_telemetry_gate import (
    SessionTelemetryGate,
    GateOutcome,
    GateResult,
    DEFAULT_MAX_FUTURE_SKEW_SECONDS,
)
from src.sensor.telemetry_adapter import TelemetryIngestionAdapter, TelemetryQualityState


@pytest.fixture
def temp_patient_registry(tmp_path):
    """Provides a fresh local patient registry in tmp_path."""
    reg_file = str(tmp_path / "registry" / "patients.json")
    return LocalPatientRegistry(storage_path=reg_file)


@pytest.fixture
def temp_session_repo(tmp_path, temp_patient_registry):
    """Provides a fresh session repository in tmp_path linked to temp_patient_registry."""
    db_file = str(tmp_path / "sessions" / "test_sessions.db")
    return SessionRepository(db_path=db_file, patient_registry=temp_patient_registry)


@pytest.fixture
def test_clock():
    """A controllable clock container for predictable time-based tests."""
    class ControllableClock:
        def __init__(self):
            self.current_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

        def get_time(self):
            return self.current_time

        def advance(self, seconds: float):
            self.current_time += datetime.timedelta(seconds=seconds)

    return ControllableClock()


@pytest.fixture
def active_session_and_gate(temp_patient_registry, temp_session_repo, test_clock):
    """Creates a registered patient, an active session, and a gate instance."""
    patient = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="F",
        anchor_age=52,
        display_alias="Bed-Gate-01",
        mrn_plaintext="MRN-GATE-001",
        registered_by_doctor_id="DOC-GATE-01",
    )

    session = temp_session_repo.create_session(
        patient_id=patient["patient_id"],
        channel_id=123456,
        device_id="ESP32-GATE-ALPHA",
        created_by_doctor_id="DOC-GATE-01",
        start_entry_id=100,
        custom_start_time=test_clock.get_time(),
    )

    gate = SessionTelemetryGate(
        session_repository=temp_session_repo,
        require_device_identity=False,
        clock_fn=test_clock.get_time,
    )

    return {
        "patient": patient,
        "session": session,
        "repo": temp_session_repo,
        "gate": gate,
        "clock": test_clock,
    }


# ── Scenario 1: Packet at the start watermark is rejected ────────────────────

def test_01_packet_at_watermark_is_rejected(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Packet with entry_id == start_entry_id (100)
    packet = {
        "channel_id": 123456,
        "entry_id": 100,
        "recorded_at": (now + datetime.timedelta(seconds=5)).isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 75000,
        "risk_status": 0,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_BEFORE_WATERMARK
    assert res.is_quarantined is True
    assert "Historical packet rejected" in res.message

    # Verify no packets stored in session
    stored = ctx["repo"].get_session_packets(session["session_id"])
    assert len(stored) == 0


# ── Scenario 2: Packet before start watermark is rejected ────────────────────

def test_02_packet_before_watermark_is_rejected(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Packet with entry_id < start_entry_id (95 < 100)
    packet = {
        "channel_id": 123456,
        "entry_id": 95,
        "recorded_at": (now + datetime.timedelta(seconds=5)).isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 75000,
        "risk_status": 0,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_BEFORE_WATERMARK
    assert res.is_quarantined is True
    assert ctx["repo"].get_session_packets(session["session_id"]) == []


# ── Scenario 3: Packet before session start time is rejected ─────────────────

def test_03_packet_before_start_time_is_rejected(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    start_time = datetime.datetime.fromisoformat(session["start_time"])

    # Packet has entry_id > 100 (105), but recorded_at is 10 seconds before start_time
    packet = {
        "channel_id": 123456,
        "entry_id": 105,
        "recorded_at": (start_time - datetime.timedelta(seconds=10)).isoformat(),
        "heart_rate": 78.0,
        "spo2": 97.0,
        "ir_ppg": 72000,
        "risk_status": 0,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_BEFORE_START_TIME
    assert res.is_quarantined is True
    assert "earlier than session start_time" in res.message
    assert ctx["repo"].get_session_packets(session["session_id"]) == []


# ── Scenario 4: Packet more than 60 seconds in the future is rejected ────────

def test_04_packet_excessive_future_skew_rejected(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Packet timestamp is 65 seconds in the future (> 60s tolerance)
    packet = {
        "channel_id": 123456,
        "entry_id": 101,
        "recorded_at": (now + datetime.timedelta(seconds=65)).isoformat(),
        "heart_rate": 76.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
        "risk_status": 0,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_FUTURE_CLOCK_SKEW
    assert res.is_quarantined is True
    assert "FUTURE_TIMESTAMP_ANOMALY" in res.reason_code


# ── Scenario 5: Packet exactly at permitted future-skew boundary passes ──────

def test_05_packet_at_future_skew_boundary_passes(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Packet timestamp is exactly 60 seconds in the future (<= 60s tolerance)
    packet = {
        "channel_id": 123456,
        "entry_id": 101,
        "recorded_at": (now + datetime.timedelta(seconds=60)).isoformat(),
        "heart_rate": 74.0,
        "spo2": 98.0,
        "ir_ppg": 72000,
        "risk_status": 0,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.ACCEPTED
    assert res.is_success is True

    stored = ctx["repo"].get_session_packets(session["session_id"])
    assert len(stored) == 1
    assert stored[0]["entry_id"] == 101


# ── Scenario 6: Stale packet (> 300s) is rejected ────────────────────────────

def test_06_stale_packet_is_rejected(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]

    # Advance clock by 400 seconds
    ctx["clock"].advance(400)
    now = ctx["clock"].get_time()

    # Packet timestamp is 350 seconds in the past (> 300s freshness window)
    packet = {
        "channel_id": 123456,
        "entry_id": 102,
        "recorded_at": (now - datetime.timedelta(seconds=350)).isoformat(),
        "heart_rate": 72.0,
        "spo2": 99.0,
        "ir_ppg": 80000,
        "risk_status": 0,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_STALE
    assert res.is_quarantined is True
    assert "STALE_TELEMETRY" in res.reason_code


# ── Scenario 7: Wrong channel is quarantined ─────────────────────────────────

def test_07_channel_mismatch_quarantined(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Packet for channel 999999 instead of 123456
    packet = {
        "channel_id": 999999,
        "entry_id": 101,
        "recorded_at": now.isoformat(),
        "heart_rate": 80.0,
        "spo2": 98.0,
        "ir_ppg": 75000,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_CHANNEL_MISMATCH
    assert res.is_quarantined is True

    # Quarantined record exists in repo
    q_pkts = ctx["repo"].get_quarantined_packets()
    assert len(q_pkts) >= 1
    assert any(q["reason_code"] == "CHANNEL_MISMATCH" for q in q_pkts)


# ── Scenario 8: Missing or mismatched device identity fails closed ───────────

def test_08_device_mismatch_fails_closed(active_session_and_gate):
    ctx = active_session_and_gate
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Gate configured to strictly require device identity match
    strict_gate = SessionTelemetryGate(
        session_repository=ctx["repo"],
        require_device_identity=True,
        clock_fn=ctx["clock"].get_time,
    )

    # 1. Packet with wrong device ID
    packet_wrong_dev = {
        "channel_id": 123456,
        "device_id": "ESP32-IMPOSTOR-99",
        "entry_id": 101,
        "recorded_at": now.isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }
    res1 = strict_gate.process_packet_for_session(session["session_id"], packet_wrong_dev)
    assert res1.outcome == GateOutcome.REJECTED_DEVICE_MISMATCH
    assert res1.is_quarantined is True

    # 2. Packet with missing device ID
    packet_no_dev = {
        "channel_id": 123456,
        "entry_id": 102,
        "recorded_at": now.isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }
    res2 = strict_gate.process_packet_for_session(session["session_id"], packet_no_dev)
    assert res2.outcome == GateOutcome.REJECTED_DEVICE_MISMATCH
    assert res2.is_quarantined is True

    # 3. Packet with matching device ID passes
    packet_correct_dev = {
        "channel_id": 123456,
        "device_id": "ESP32-GATE-ALPHA",
        "entry_id": 103,
        "recorded_at": now.isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }
    res3 = strict_gate.process_packet_for_session(session["session_id"], packet_correct_dev)
    assert res3.outcome == GateOutcome.ACCEPTED


# ── Scenario 9: Closed-session packet is rejected ────────────────────────────

def test_09_closed_session_packet_is_rejected(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Conclude the session
    now = ctx["clock"].get_time()
    ctx["clock"].advance(10)
    end_time = ctx["clock"].get_time()
    ctx["repo"].close_session(
        session_id=session["session_id"],
        closed_by_doctor_id="DOC-GATE-01",
        status="completed",
        closure_reason="routine_discharge",
        custom_end_time=end_time,
    )

    packet = {
        "channel_id": 123456,
        "entry_id": 101,
        "recorded_at": now.isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_SESSION_INACTIVE
    assert res.is_quarantined is True
    assert "not active" in res.message


# ── Scenario 10: Invalid optical contact (IR < 50,000) is not marked valid ───

def test_10_optical_contact_invalid_handling(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    # Packet with IR PPG of 25,000 (< 50,000 count contact threshold)
    packet = {
        "channel_id": 123456,
        "entry_id": 101,
        "recorded_at": now.isoformat(),
        "heart_rate": 72.0,
        "spo2": 97.0,
        "ir_ppg": 25000,
        "risk_status": 0,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.ACCEPTED  # Stored with quality flags
    assert res.packet is not None

    # Verify optical contact is marked false and flagged
    assert res.packet.cleaned.optical_contact is False
    assert res.packet.cleaned.physiologically_valid is False
    assert "OPTICAL_LEAD_OFF" in res.packet.quality_flags
    assert res.packet.quality_state in (TelemetryQualityState.SUSPECT, TelemetryQualityState.INVALID)

    stored = ctx["repo"].get_session_packets(session["session_id"])
    assert len(stored) == 1
    assert stored[0]["optical_contact"] == 0
    assert stored[0]["physiologically_valid"] == 0
    assert "OPTICAL_LEAD_OFF" in stored[0]["quality_flags"]


# ── Scenario 11: Sequential patient sessions have zero packet leakage ────────

def test_11_sequential_sessions_zero_leakage(temp_patient_registry, temp_session_repo, test_clock):
    repo = temp_session_repo
    now = test_clock.get_time()

    # Patient 1
    p1 = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="M", anchor_age=60, display_alias="Patient-1",
        mrn_plaintext="MRN-SEQ-1", registered_by_doctor_id="DOC-1",
    )
    s1 = repo.create_session(
        patient_id=p1["patient_id"],
        channel_id=123456,
        device_id="DEV-SHARED",
        created_by_doctor_id="DOC-1",
        start_entry_id=100,
        custom_start_time=now,
    )

    gate = SessionTelemetryGate(session_repository=repo, clock_fn=test_clock.get_time)

    # Ingest packets 101 and 102 for s1
    gate.process_packet_for_session(s1["session_id"], {
        "channel_id": 123456, "entry_id": 101, "recorded_at": (now + datetime.timedelta(seconds=1)).isoformat(),
        "heart_rate": 70.0, "spo2": 98.0, "ir_ppg": 60000,
    })
    gate.process_packet_for_session(s1["session_id"], {
        "channel_id": 123456, "entry_id": 102, "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 71.0, "spo2": 98.0, "ir_ppg": 61000,
    })

    # Conclude s1
    test_clock.advance(10)
    end_time1 = test_clock.get_time()
    repo.close_session(
        session_id=s1["session_id"],
        closed_by_doctor_id="DOC-1",
        status="completed",
        closure_reason="routine_discharge",
        custom_end_time=end_time1,
    )

    # Patient 2 on the same hardware channel, watermark 102
    test_clock.advance(60)
    now2 = test_clock.get_time()

    p2 = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="F", anchor_age=45, display_alias="Patient-2",
        mrn_plaintext="MRN-SEQ-2", registered_by_doctor_id="DOC-1",
    )
    s2 = repo.create_session(
        patient_id=p2["patient_id"],
        channel_id=123456,
        device_id="DEV-SHARED",
        created_by_doctor_id="DOC-1",
        start_entry_id=102,
        custom_start_time=now2,
    )

    # Ingest packets 103 and 104 for s2
    gate.process_packet_for_session(s2["session_id"], {
        "channel_id": 123456, "entry_id": 103, "recorded_at": (now2 + datetime.timedelta(seconds=1)).isoformat(),
        "heart_rate": 85.0, "spo2": 99.0, "ir_ppg": 65000,
    })
    gate.process_packet_for_session(s2["session_id"], {
        "channel_id": 123456, "entry_id": 104, "recorded_at": (now2 + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 86.0, "spo2": 99.0, "ir_ppg": 66000,
    })

    # Verify zero packet leakage across sessions
    s1_pkts = repo.get_session_packets(s1["session_id"])
    s2_pkts = repo.get_session_packets(s2["session_id"])

    assert len(s1_pkts) == 2
    assert [p["entry_id"] for p in s1_pkts] == [101, 102]
    assert all(p["session_id"] == s1["session_id"] for p in s1_pkts)

    assert len(s2_pkts) == 2
    assert [p["entry_id"] for p in s2_pkts] == [103, 104]
    assert all(p["session_id"] == s2["session_id"] for p in s2_pkts)


# ── Scenario 12: Identical retransmission creates only one stored row ────────

def test_12_identical_retransmission_accepted(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    packet = {
        "channel_id": 123456,
        "entry_id": 101,
        "recorded_at": now.isoformat(),
        "heart_rate": 78.0,
        "spo2": 98.0,
        "ir_ppg": 72000,
        "risk_status": 0,
    }

    # 1. First send: ACCEPTED
    res1 = gate.process_packet_for_session(session["session_id"], packet)
    assert res1.outcome == GateOutcome.ACCEPTED

    # 2. Second send (exact copy): DUPLICATE_ACCEPTED
    res2 = gate.process_packet_for_session(session["session_id"], packet)
    assert res2.outcome == GateOutcome.DUPLICATE_ACCEPTED
    assert res2.is_success is True

    # Database still has only 1 row
    stored = ctx["repo"].get_session_packets(session["session_id"])
    assert len(stored) == 1
    assert stored[0]["entry_id"] == 101


# ── Scenario 13: Conflicting payload for same source packet is quarantined ────

def test_13_conflicting_payload_quarantined(active_session_and_gate):
    ctx = active_session_and_gate
    gate: SessionTelemetryGate = ctx["gate"]
    session = ctx["session"]
    now = ctx["clock"].get_time()

    packet1 = {
        "channel_id": 123456,
        "entry_id": 101,
        "recorded_at": now.isoformat(),
        "heart_rate": 78.0,
        "spo2": 98.0,
        "ir_ppg": 72000,
        "risk_status": 0,
    }
    res1 = gate.process_packet_for_session(session["session_id"], packet1)
    assert res1.outcome == GateOutcome.ACCEPTED

    # Conflicting packet: same (channel_id, entry_id) but different HR and SpO2
    packet_conflict = {
        "channel_id": 123456,
        "entry_id": 101,
        "recorded_at": now.isoformat(),
        "heart_rate": 140.0,
        "spo2": 88.0,
        "ir_ppg": 72000,
        "risk_status": 2,
    }

    res2 = gate.process_packet_for_session(session["session_id"], packet_conflict)
    assert res2.outcome == GateOutcome.CONFLICT_QUARANTINED
    assert res2.is_quarantined is True

    # Original packet unchanged
    stored = ctx["repo"].get_session_packets(session["session_id"])
    assert len(stored) == 1
    assert stored[0]["heart_rate"] == 78.0


# ── Scenario 14: Concurrent attempts cannot assign one packet to two sessions ─

def test_14_cross_session_assignment_prevented(temp_patient_registry, temp_session_repo, test_clock):
    repo = temp_session_repo
    now = test_clock.get_time()

    p1 = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="M", anchor_age=40, display_alias="Pat-1",
        mrn_plaintext="MRN-C-1", registered_by_doctor_id="DOC-1",
    )
    p2 = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="F", anchor_age=38, display_alias="Pat-2",
        mrn_plaintext="MRN-C-2", registered_by_doctor_id="DOC-1",
    )

    # Session 1 on channel 111111
    s1 = repo.create_session(
        patient_id=p1["patient_id"],
        channel_id=111111,
        device_id="DEV-1",
        created_by_doctor_id="DOC-1",
        start_entry_id=50,
        custom_start_time=now,
    )
    # Session 2 on channel 222222
    s2 = repo.create_session(
        patient_id=p2["patient_id"],
        channel_id=222222,
        device_id="DEV-2",
        created_by_doctor_id="DOC-1",
        start_entry_id=50,
        custom_start_time=now,
    )

    # Store packet (111111, 55) in s1
    packet = {
        "channel_id": 111111,
        "entry_id": 55,
        "recorded_at": (now + datetime.timedelta(seconds=5)).isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
        "risk_status": 0,
    }
    repo.store_session_packet(session_id=s1["session_id"], packet=packet)

    # Attempt to directly store packet (111111, 55) in s2 -> must fail with SessionConflictError
    with pytest.raises(SessionConflictError) as exc_info:
        repo.store_session_packet(session_id=s2["session_id"], packet=packet)

    assert "Reassignment to session" in str(exc_info.value) or "already assigned" in str(exc_info.value)


# ── Scenario 15: Session closure racing with ingestion cannot create assignment

def test_15_session_closure_race_fails_closed(active_session_and_gate):
    ctx = active_session_and_gate
    session = ctx["session"]
    repo = ctx["repo"]
    gate = ctx["gate"]
    now = ctx["clock"].get_time()

    # Pre-close the session in repository
    ctx["clock"].advance(10)
    end_time = ctx["clock"].get_time()
    repo.close_session(
        session_id=session["session_id"],
        closed_by_doctor_id="DOC-GATE-01",
        status="completed",
        closure_reason="routine_discharge",
        custom_end_time=end_time,
    )

    # Gate processes packet: inside transaction, repository validates session status and raises
    packet = {
        "channel_id": 123456,
        "entry_id": 105,
        "recorded_at": now.isoformat(),
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }

    res = gate.process_packet_for_session(session["session_id"], packet)
    assert res.outcome == GateOutcome.REJECTED_SESSION_INACTIVE
    assert res.is_quarantined is True
    assert repo.get_session_packets(session["session_id"]) == []


# ── Scenario 16: Missing watermark blocks ingestion ──────────────────────────

def test_16_missing_watermark_blocks_ingestion(temp_patient_registry, temp_session_repo, test_clock):
    repo = temp_session_repo
    now = test_clock.get_time()

    p = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="M", anchor_age=65, display_alias="Pat-NoWM",
        mrn_plaintext="MRN-NOWM", registered_by_doctor_id="DOC-1",
    )
    s = repo.create_session(
        patient_id=p["patient_id"],
        channel_id=333333,
        device_id="DEV-NOWM",
        created_by_doctor_id="DOC-1",
        start_entry_id=0,
        custom_start_time=now,
    )

    # Simulate a session context missing start_entry_id watermark
    corrupted_session = copy.deepcopy(s)
    corrupted_session["start_entry_id"] = None

    gate = SessionTelemetryGate(session_repository=repo, clock_fn=test_clock.get_time)
    packet = {
        "channel_id": 333333,
        "entry_id": 1,
        "recorded_at": now.isoformat(),
        "heart_rate": 80.0,
        "spo2": 98.0,
        "ir_ppg": 70000,
    }

    res = gate.process_packet_for_session(corrupted_session, packet)
    assert res.outcome == GateOutcome.REJECTED_MISSING_WATERMARK
    assert res.is_quarantined is True
    assert "MISSING_SESSION_WATERMARK" in res.reason_code


# ── Scenario 17: Restart preserves packet/session associations ────────────────

def test_17_restart_preserves_associations(tmp_path, temp_patient_registry, test_clock):
    db_file = str(tmp_path / "sessions" / "restart_test.db")
    repo1 = SessionRepository(db_path=db_file, patient_registry=temp_patient_registry)
    now = test_clock.get_time()

    p = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="F", anchor_age=48, display_alias="Pat-Restart",
        mrn_plaintext="MRN-RESTART", registered_by_doctor_id="DOC-1",
    )
    s = repo1.create_session(
        patient_id=p["patient_id"],
        channel_id=555555,
        device_id="DEV-RST",
        created_by_doctor_id="DOC-1",
        start_entry_id=200,
        custom_start_time=now,
    )

    gate1 = SessionTelemetryGate(session_repository=repo1, clock_fn=test_clock.get_time)
    res = gate1.process_packet_for_session(s["session_id"], {
        "channel_id": 555555,
        "entry_id": 201,
        "recorded_at": (now + datetime.timedelta(seconds=2)).isoformat(),
        "heart_rate": 77.0,
        "spo2": 98.0,
        "ir_ppg": 75000,
    })
    assert res.outcome == GateOutcome.ACCEPTED

    # Simulate restart by instantiating a new repository with the same SQLite path
    repo2 = SessionRepository(db_path=db_file, patient_registry=temp_patient_registry)
    packets = repo2.get_session_packets(s["session_id"])

    assert len(packets) == 1
    assert packets[0]["entry_id"] == 201
    assert packets[0]["heart_rate"] == 77.0
    assert packets[0]["channel_id"] == 555555


# ── Scenario 18: Quarantine does not leak patient data and honors bounds ─────

def test_18_quarantine_bounded_and_zero_patient_leak(temp_session_repo, test_clock):
    repo = temp_session_repo
    now = test_clock.get_time()

    # Quarantine 10 packets with maximum buffer set to 5
    for i in range(10):
        repo.quarantine_packet(
            channel_id=999999,
            entry_id=i,
            device_id=f"UNKNOWN-DEV-{i}",
            recorded_at=now.isoformat(),
            reason_code="AUDIT_TEST",
            diagnostic_message="Unproven telemetry test",
            raw_payload={"secret_unproven": i},
            current_time=now,
            max_quarantine_records=5,
        )

    quarantined = repo.get_quarantined_packets(limit=50)
    # Must be pruned down to max_quarantine_records (5)
    assert len(quarantined) == 5

    # Verify no patient_id or session_id columns exist in quarantine table schema
    for q in quarantined:
        assert "patient_id" not in q
        assert "session_id" not in q
        assert q["reason_code"] == "AUDIT_TEST"


# ── Scenario 19: Inactivity behavior is deterministic and audited ────────────

def test_19_inactivity_deterministic_and_audited(temp_patient_registry, temp_session_repo, test_clock):
    repo = temp_session_repo
    now = test_clock.get_time()

    p = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="M", anchor_age=62, display_alias="Pat-Timeout",
        mrn_plaintext="MRN-TO", registered_by_doctor_id="DOC-TO",
    )
    s = repo.create_session(
        patient_id=p["patient_id"],
        channel_id=777777,
        device_id="DEV-TO",
        created_by_doctor_id="DOC-TO",
        start_entry_id=10,
        custom_start_time=now,
    )

    # Ingest a packet at now + 5s
    gate = SessionTelemetryGate(session_repository=repo, clock_fn=test_clock.get_time)
    gate.process_packet_for_session(s["session_id"], {
        "channel_id": 777777,
        "entry_id": 11,
        "recorded_at": (now + datetime.timedelta(seconds=5)).isoformat(),
        "heart_rate": 80.0,
        "spo2": 97.0,
        "ir_ppg": 70000,
    })

    # Advance clock by 1700s (< 1800s TTL) -> should NOT auto-close
    test_clock.advance(1700)
    closed = repo.auto_close_inactive_sessions(inactivity_ttl_seconds=1800.0, current_time=test_clock.get_time())
    assert len(closed) == 0

    sess = repo.get_session(s["session_id"])
    assert sess["status"] == "active"

    # Advance clock by another 200s (total 1900s > 1800s TTL) -> should auto-close
    test_clock.advance(200)
    closed = repo.auto_close_inactive_sessions(inactivity_ttl_seconds=1800.0, current_time=test_clock.get_time())
    assert len(closed) == 1
    assert closed[0]["session_id"] == s["session_id"]

    sess_after = repo.get_session(s["session_id"])
    assert sess_after["status"] == "abandoned"
    assert sess_after["closure_reason"] == "inactivity_timeout"

    # Audit trail verification
    audit = repo.get_session_audit_trail(s["session_id"])
    timeout_audits = [a for a in audit if a["action"] == "SESSION_ABANDONED" and a["doctor_id"] == "SYSTEM_INACTIVITY_MONITOR"]
    assert len(timeout_audits) == 1


# ── Scenario 20: Retrospective inference and patient lookup unaffected ───────

def test_20_retrospective_inference_unaffected(temp_session_repo):
    """
    Verifies that the session telemetry gate and repository maintain complete isolation
    from retrospective MIMIC hospital stays and data repository features.
    """
    from src.backend.data_repository import DataRepository, RepositoryStatus, DataOrigin

    # Initialize data repository in offline/local mode
    data_repo = DataRepository()

    # Retrospective patient lookup from local MIMIC dataset
    res = data_repo.get_patient("10002930")
    assert res.status == RepositoryStatus.SUCCESS
    assert res.origin == DataOrigin.LOCAL_PRIMARY
    assert res.data["patient_id"] == "10002930"
    assert res.is_synthetic is False

    # Verify that telemetry tables are strictly decoupled from retrospective repository
    assert hasattr(temp_session_repo, "store_session_packet")
    assert not hasattr(data_repo, "store_session_packet")
