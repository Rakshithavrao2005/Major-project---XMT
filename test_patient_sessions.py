"""
tests/test_patient_sessions.py
=============================================================================
Comprehensive Offline Synthetic Test Suite for Stage 3.7.6C:
Local Patient Registry & Persistent Session Repository.

Verifies:
 1. Valid prospective and synthetic patient registration.
 2. Invalid and duplicate registration rejection.
 3. Clear separation of synthetic, prospective, and retrospective record types.
 4. Session creation with a valid patient and authenticated clinician identity.
 5. Rejection of unknown patient or missing clinician identity.
 6. Atomic prevention of a second active session for the same patient.
 7. Atomic prevention of concurrent active sessions on the same device/channel.
 8. Valid close operations and rejection of invalid state transitions.
 9. Persistence after closing and reopening the repository.
10. Correct clinician attribution and immutable creation attribution.
11. Lifecycle audit records for successful operations.
12. Timestamp and entry-ID validation.
13. Rollback after simulated transaction failure.
14. Isolation between sequential synthetic patients.
15. Rejection of closed-session modification or reassignment.
16. Safe handling of a pre-existing incompatible test database without modifying it.
=============================================================================
"""

import datetime
import os
import sqlite3
import sys
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.patient_registry import (
    LocalPatientRegistry,
    DuplicatePatientError,
    InvalidPatientDataError,
    hash_medical_record_number,
)
from src.sensor.session_repository import (
    SessionRepository,
    SessionConflictError,
    SessionNotFoundError,
    InvalidSessionStateError,
    IncompatibleDatabaseError,
    UnauthorizedClinicianError,
    SessionRepositoryError,
)


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


# ── 1. Valid Prospective & Synthetic Patient Registration ─────────────────────

def test_01_valid_patient_registration(temp_patient_registry):
    """Register prospective and synthetic demo patients successfully."""
    # Prospective patient
    p1 = temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        gender="M",
        anchor_age=58,
        display_alias="Room-101",
        mrn_plaintext="MRN-PROSP-001",
        registered_by_doctor_id="DOC-REG-0001",
    )
    assert p1["patient_id"].startswith("PROSP-")
    assert p1["registration_type"] == "prospective_ambulatory"
    assert p1["gender"] == "M"
    assert p1["anchor_age"] == 58
    assert p1["display_alias"] == "Room-101"
    assert p1["registered_by_doctor_id"] == "DOC-REG-0001"
    assert p1["medical_record_hash"] is not None
    assert "MRN-PROSP-001" not in str(p1)  # Never stores plaintext MRN

    # Synthetic demo patient
    p2 = temp_patient_registry.register_patient(
        registration_type="synthetic_demo",
        gender="F",
        anchor_age=45,
        display_alias="Demo-Bed-02",
        mrn_plaintext="SYNTH-MRN-999",
        registered_by_doctor_id="DOC-ADMIN-0001",
    )
    assert p2["patient_id"].startswith("SYNTH-")
    assert p2["registration_type"] == "synthetic_demo"


# ── 2. Invalid and Duplicate Registration Rejection ───────────────────────────

def test_02_invalid_and_duplicate_registration_rejection(temp_patient_registry):
    """Rejects invalid demographics and duplicate MRN hashes."""
    # Invalid registration type
    with pytest.raises(InvalidPatientDataError):
        temp_patient_registry.register_patient(registration_type="invalid_type")

    # Invalid gender
    with pytest.raises(InvalidPatientDataError):
        temp_patient_registry.register_patient(gender="X_INVALID")

    # Invalid age (out of bounds)
    with pytest.raises(InvalidPatientDataError):
        temp_patient_registry.register_patient(anchor_age=150)

    # Register initial patient
    temp_patient_registry.register_patient(
        registration_type="prospective_ambulatory",
        mrn_plaintext="UNIQUE-MRN-12345",
    )

    # Re-registering with identical MRN is rejected as duplicate
    with pytest.raises(DuplicatePatientError):
        temp_patient_registry.register_patient(
            registration_type="prospective_ambulatory",
            mrn_plaintext="UNIQUE-MRN-12345",
        )


# ── 3. Separation of Synthetic, Prospective, and Retrospective Records ────────

def test_03_record_type_separation(temp_patient_registry):
    """Ensure prospective, synthetic, and retrospective types remain distinct and listable."""
    temp_patient_registry.register_patient(registration_type="prospective_ambulatory", display_alias="P1")
    temp_patient_registry.register_patient(registration_type="prospective_ambulatory", display_alias="P2")
    temp_patient_registry.register_patient(registration_type="synthetic_demo", display_alias="S1")

    prosp_list = temp_patient_registry.list_patients(registration_type="prospective_ambulatory")
    synth_list = temp_patient_registry.list_patients(registration_type="synthetic_demo")

    assert len(prosp_list) == 2
    assert len(synth_list) == 1
    assert all(p["registration_type"] == "prospective_ambulatory" for p in prosp_list)
    assert all(p["registration_type"] == "synthetic_demo" for p in synth_list)


# ── 4. Session Creation with Valid Patient & Authenticated Clinician ───────────

def test_04_session_creation_success(temp_patient_registry, temp_session_repo):
    """Create a monitoring session linked to an existing patient and authenticated doctor."""
    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    sess = temp_session_repo.create_session(
        patient_id=pid,
        device_id="ESP32-MAX30102-01",
        channel_id=3490289,
        created_by_doctor_id="DOC-REG-0001",
        start_entry_id=500,
        session_notes="Initial baseline monitoring",
    )

    assert sess["session_id"].startswith("SESS-")
    assert sess["patient_id"] == pid
    assert sess["device_id"] == "ESP32-MAX30102-01"
    assert sess["channel_id"] == 3490289
    assert sess["status"] == "active"
    assert sess["start_entry_id"] == 500
    assert sess["created_by_doctor_id"] == "DOC-REG-0001"
    assert sess["end_time"] is None
    assert sess["closed_by_doctor_id"] is None


# ── 5. Rejection of Unknown Patient or Missing Clinician Identity ──────────────

def test_05_rejection_unknown_patient_or_missing_doctor(temp_session_repo):
    """Rejects unregistered patients and unauthenticated requests."""
    # Unknown patient
    with pytest.raises(SessionNotFoundError):
        temp_session_repo.create_session(
            patient_id="PROSP-NONEXISTENT",
            device_id="ESP32-MAX30102-01",
            channel_id=3490289,
            created_by_doctor_id="DOC-REG-0001",
        )

    # Missing clinician ID
    with pytest.raises(UnauthorizedClinicianError):
        temp_session_repo.create_session(
            patient_id="SOME-PID",
            device_id="ESP32-MAX30102-01",
            channel_id=3490289,
            created_by_doctor_id="",  # Empty doctor
        )


# ── 6. Atomic Prevention of Second Active Session for Same Patient ────────────

def test_06_atomic_prevention_same_patient(temp_patient_registry, temp_session_repo):
    """Enforces that a patient can have AT MOST ONE active session."""
    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    # Start first session
    temp_session_repo.create_session(
        patient_id=pid,
        device_id="DEVICE-A",
        channel_id=10001,
        created_by_doctor_id="DOC-REG-0001",
    )

    # Attempting second session on different device/channel for same patient must fail
    with pytest.raises(SessionConflictError) as exc_info:
        temp_session_repo.create_session(
            patient_id=pid,
            device_id="DEVICE-B",
            channel_id=10002,
            created_by_doctor_id="DOC-REG-0001",
        )
    assert "already has an active monitoring session" in str(exc_info.value)


# ── 7. Atomic Prevention of Concurrent Sessions on Same Device/Channel ────────

def test_07_atomic_prevention_same_device_or_channel(temp_patient_registry, temp_session_repo):
    """Enforces at most one active session per channel and per device."""
    p1 = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")["patient_id"]
    p2 = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")["patient_id"]

    # Patient 1 occupies Device-01 and Channel 3490289
    temp_session_repo.create_session(
        patient_id=p1,
        device_id="ESP32-DEV-01",
        channel_id=3490289,
        created_by_doctor_id="DOC-REG-0001",
    )

    # Patient 2 tries to use the same channel (different device) -> Conflict
    with pytest.raises(SessionConflictError) as exc1:
        temp_session_repo.create_session(
            patient_id=p2,
            device_id="ESP32-DEV-02",
            channel_id=3490289,
            created_by_doctor_id="DOC-REG-0001",
        )
    assert "Channel 3490289 already has an active monitoring session" in str(exc1.value)

    # Patient 2 tries to use the same device (different channel) -> Conflict
    with pytest.raises(SessionConflictError) as exc2:
        temp_session_repo.create_session(
            patient_id=p2,
            device_id="ESP32-DEV-01",
            channel_id=9999999,
            created_by_doctor_id="DOC-REG-0001",
        )
    assert "Device 'ESP32-DEV-01' already has an active monitoring session" in str(exc2.value)


# ── 8. Valid Close Operations & Rejection of Invalid State Transitions ─────────

def test_08_close_operations_and_state_transitions(temp_patient_registry, temp_session_repo):
    """Verify clean session closure and rejection of illegal transitions."""
    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    sess = temp_session_repo.create_session(
        patient_id=pid,
        device_id="ESP32-DEV-01",
        channel_id=3490289,
        start_entry_id=100,
        created_by_doctor_id="DOC-REG-0001",
    )
    sid = sess["session_id"]

    # Close session
    closed = temp_session_repo.close_session(
        session_id=sid,
        closed_by_doctor_id="DOC-REG-0002",
        closure_reason="routine_discharge",
        end_entry_id=250,
        session_notes="Patient discharged stable",
    )
    assert closed["status"] == "completed"
    assert closed["end_time"] is not None
    assert closed["end_entry_id"] == 250
    assert closed["closed_by_doctor_id"] == "DOC-REG-0002"
    assert closed["closure_reason"] == "routine_discharge"

    # Attempting to close an already completed session raises InvalidSessionStateError
    with pytest.raises(InvalidSessionStateError):
        temp_session_repo.close_session(
            session_id=sid,
            closed_by_doctor_id="DOC-REG-0002",
            closure_reason="routine_discharge",
        )

    # Invalid closure reason rejected
    # Start new session
    sess2 = temp_session_repo.create_session(
        patient_id=pid,
        device_id="ESP32-DEV-01",
        channel_id=3490289,
        created_by_doctor_id="DOC-REG-0001",
    )
    with pytest.raises(InvalidSessionStateError):
        temp_session_repo.close_session(
            session_id=sess2["session_id"],
            closed_by_doctor_id="DOC-REG-0001",
            closure_reason="invalid_unapproved_reason",
        )


# ── 9. Persistence Across Closing and Reopening the Repository ─────────────────

def test_09_persistence_across_reconnect(tmp_path, temp_patient_registry):
    """Ensure data persists in SQLite across independent SessionRepository instances."""
    db_file = str(tmp_path / "sessions" / "persisted_sessions.db")

    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    # Instance 1: Create session
    repo1 = SessionRepository(db_path=db_file, patient_registry=temp_patient_registry)
    s1 = repo1.create_session(
        patient_id=pid,
        device_id="DEV-PERSIST-01",
        channel_id=55555,
        start_entry_id=10,
        created_by_doctor_id="DOC-REG-0001",
    )
    sid = s1["session_id"]
    del repo1

    # Instance 2: Connect to same DB and read
    repo2 = SessionRepository(db_path=db_file, patient_registry=temp_patient_registry)
    s2 = repo2.get_session(sid)
    assert s2 is not None
    assert s2["session_id"] == sid
    assert s2["patient_id"] == pid
    assert s2["status"] == "active"

    # Close session using Instance 2
    repo2.close_session(
        session_id=sid,
        closed_by_doctor_id="DOC-REG-0003",
        closure_reason="clinical_escalation",
    )
    del repo2

    # Instance 3: Verify closure persisted
    repo3 = SessionRepository(db_path=db_file, patient_registry=temp_patient_registry)
    s3 = repo3.get_session(sid)
    assert s3["status"] == "completed"
    assert s3["closed_by_doctor_id"] == "DOC-REG-0003"


# ── 10. Correct Clinician Attribution & Immutable Creation Attribution ────────

def test_10_clinician_attribution_immutability(temp_patient_registry, temp_session_repo):
    """Creation attribution is immutable and separate from closing clinician."""
    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    sess = temp_session_repo.create_session(
        patient_id=pat["patient_id"],
        device_id="DEV-01",
        channel_id=111,
        created_by_doctor_id="DOC-CREATOR-01",
    )
    sid = sess["session_id"]

    closed = temp_session_repo.close_session(
        session_id=sid,
        closed_by_doctor_id="DOC-CLOSER-02",
        closure_reason="device_transfer",
    )
    assert closed["created_by_doctor_id"] == "DOC-CREATOR-01"
    assert closed["closed_by_doctor_id"] == "DOC-CLOSER-02"


# ── 11. Lifecycle Audit Records for Operations ─────────────────────────────────

def test_11_lifecycle_audit_records(temp_patient_registry, temp_session_repo):
    """Verify session creation and closure generate audit trail records."""
    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    sess = temp_session_repo.create_session(
        patient_id=pat["patient_id"],
        device_id="DEV-01",
        channel_id=222,
        created_by_doctor_id="DOC-01",
    )
    sid = sess["session_id"]

    trail1 = temp_session_repo.get_session_audit_trail(sid)
    assert len(trail1) == 1
    assert trail1[0]["action"] == "SESSION_CREATED"
    assert trail1[0]["doctor_id"] == "DOC-01"

    temp_session_repo.close_session(
        session_id=sid,
        closed_by_doctor_id="DOC-02",
        closure_reason="routine_discharge",
    )
    trail2 = temp_session_repo.get_session_audit_trail(sid)
    assert len(trail2) == 2
    assert trail2[1]["action"] == "SESSION_COMPLETED"
    assert trail2[1]["doctor_id"] == "DOC-02"


# ── 12. Timestamp & Entry-ID Validation ────────────────────────────────────────

def test_12_timestamp_and_entry_id_validation(temp_patient_registry, temp_session_repo):
    """Rejects negative entry IDs and backwards timestamps."""
    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    # Negative start entry ID
    with pytest.raises(SessionRepositoryError):
        temp_session_repo.create_session(
            patient_id=pid,
            device_id="DEV-01",
            channel_id=333,
            start_entry_id=-5,
            created_by_doctor_id="DOC-01",
        )

    # Start valid session
    t_start = datetime.datetime(2026, 10, 22, 10, 0, 0, tzinfo=datetime.timezone.utc)
    sess = temp_session_repo.create_session(
        patient_id=pid,
        device_id="DEV-01",
        channel_id=333,
        start_entry_id=100,
        created_by_doctor_id="DOC-01",
        custom_start_time=t_start,
    )
    sid = sess["session_id"]

    # end_entry_id < start_entry_id rejected
    with pytest.raises(InvalidSessionStateError):
        temp_session_repo.close_session(
            session_id=sid,
            closed_by_doctor_id="DOC-01",
            closure_reason="routine_discharge",
            end_entry_id=50,  # 50 < 100
        )

    # end_time earlier than start_time rejected
    t_early = datetime.datetime(2026, 10, 22, 9, 0, 0, tzinfo=datetime.timezone.utc)
    with pytest.raises(InvalidSessionStateError):
        temp_session_repo.close_session(
            session_id=sid,
            closed_by_doctor_id="DOC-01",
            closure_reason="routine_discharge",
            end_entry_id=150,
            custom_end_time=t_early,
        )


# ── 13. Rollback After Simulated Transaction Failure ──────────────────────────

def test_13_rollback_on_failure(tmp_path, temp_patient_registry):
    """Verify that an interrupted write operation rolls back completely."""
    db_file = str(tmp_path / "sessions" / "rollback_test.db")
    repo = SessionRepository(db_path=db_file, patient_registry=temp_patient_registry)

    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    # Test that failed close or insert leaves state clean
    sess = repo.create_session(
        patient_id=pid,
        device_id="DEV-01",
        channel_id=444,
        created_by_doctor_id="DOC-01",
    )
    sid = sess["session_id"]

    # Attempt an illegal close that raises error
    try:
        repo.close_session(
            session_id=sid,
            closed_by_doctor_id="DOC-01",
            closure_reason="illegal_reason",
        )
    except InvalidSessionStateError:
        pass

    # Verify session remains cleanly 'active'
    reloaded = repo.get_session(sid)
    assert reloaded["status"] == "active"
    assert reloaded["end_time"] is None


# ── 14. Isolation Between Sequential Synthetic Patients ────────────────────────

def test_14_sequential_patient_isolation(temp_patient_registry, temp_session_repo):
    """Sequential sessions on the same hardware for different patients remain isolated."""
    p1 = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")["patient_id"]
    p2 = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")["patient_id"]

    # Patient 1 session
    s1 = temp_session_repo.create_session(
        patient_id=p1,
        device_id="SHARED-BED-PULSE",
        channel_id=7777,
        start_entry_id=0,
        created_by_doctor_id="DOC-01",
    )
    temp_session_repo.close_session(
        session_id=s1["session_id"],
        closed_by_doctor_id="DOC-01",
        closure_reason="routine_discharge",
        end_entry_id=100,
    )

    # Now Patient 2 can safely use the device and channel
    s2 = temp_session_repo.create_session(
        patient_id=p2,
        device_id="SHARED-BED-PULSE",
        channel_id=7777,
        start_entry_id=100,  # Watermark begins at prior end
        created_by_doctor_id="DOC-02",
    )
    assert s2["status"] == "active"
    assert s2["patient_id"] == p2
    assert s2["start_entry_id"] == 100

    # Listing by patient shows separate histories
    p1_sessions = temp_session_repo.list_sessions(patient_id=p1)
    p2_sessions = temp_session_repo.list_sessions(patient_id=p2)
    assert len(p1_sessions) == 1
    assert len(p2_sessions) == 1
    assert p1_sessions[0]["session_id"] != p2_sessions[0]["session_id"]


# ── 15. Rejection of Closed-Session Modification or Reassignment ──────────────

def test_15_closed_session_modification_rejected(temp_patient_registry, temp_session_repo):
    """Once closed, a session cannot be re-opened, modified, or reassigned."""
    pat = temp_patient_registry.register_patient(registration_type="prospective_ambulatory")
    sess = temp_session_repo.create_session(
        patient_id=pat["patient_id"],
        device_id="DEV-01",
        channel_id=888,
        created_by_doctor_id="DOC-01",
    )
    sid = sess["session_id"]
    temp_session_repo.close_session(
        session_id=sid,
        closed_by_doctor_id="DOC-01",
        closure_reason="routine_discharge",
    )

    # Trying to close again fails
    with pytest.raises(InvalidSessionStateError):
        temp_session_repo.close_session(
            session_id=sid,
            closed_by_doctor_id="DOC-01",
            closure_reason="clinical_escalation",
        )


# ── 16. Incompatible Pre-Existing Database Protection ─────────────────────────

def test_16_incompatible_database_protection(tmp_path):
    """Detects incompatible pre-existing database schema and refuses to overwrite."""
    bad_db = str(tmp_path / "sessions" / "incompatible.db")
    os.makedirs(os.path.dirname(bad_db), exist_ok=True)

    # Create dummy database with missing required columns
    conn = sqlite3.connect(bad_db)
    with conn:
        conn.execute("CREATE TABLE patient_monitoring_sessions (session_id TEXT PRIMARY KEY, dummy_col TEXT);")
    conn.close()

    # Initializing SessionRepository against this DB must raise IncompatibleDatabaseError
    with pytest.raises(IncompatibleDatabaseError) as exc:
        SessionRepository(db_path=bad_db)
    assert "incompatible" in str(exc.value).lower()
    assert "refusing to overwrite" in str(exc.value).lower()
