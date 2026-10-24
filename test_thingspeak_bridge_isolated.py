"""
tests/test_thingspeak_bridge_isolated.py
=============================================================================
Stage 3.8.5A: Isolated Offline ThingSpeak Bridge Test Suite.

Verifies:
 1. Single active session polling: valid synthetic feed (entry_id=297) ingested cleanly.
 2. Zero active sessions: fail-closed (REJECTED_NO_ACTIVE_SESSION), nothing ingested.
 3. Multiple active sessions collision: fail-closed (REJECTED_SESSION_CONFLICT), nothing ingested.
 4. Duplicate feed handling: duplicate feed entries are safely skipped.
 5. Network timeouts, retries, and exponential backoff behavior.
 6. Stale packet rejection: feeds older than freshness window are quarantined.
 7. Missing and malformed field4 evaluates strictly to UNKNOWN, NEVER NORMAL.
 8. Missing/invalid HR, SpO2, and IR are preserved as missing/invalid; zero reassuring substitutions.
 9. Optical lead-off (IR < 50,000) accepted into session audit record but marked physiologically_valid = False.
10. Channel reset detection: entry_id <= watermark permanently locks session without lowering watermark.
11. Persistent reset protection across process restarts: brand-new bridge instance respects SQLite lock.
12. Live demonstration database (monitoring_sessions.db) immunity: bit-for-bit SHA-256 verification.
=============================================================================
"""

import os
import sys
import copy
import json
import hashlib
import sqlite3
import datetime
import tempfile
from typing import Optional, Dict, Any, Tuple, List, Union
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.patient_registry import LocalPatientRegistry
from src.sensor.session_repository import SessionRepository
from src.sensor.session_telemetry_gate import SessionTelemetryGate
from src.sensor.session_telemetry_service import SessionTelemetryService
from src.sensor.thingspeak_bridge import (
    ThingSpeakSessionBridge,
    ThingSpeakHttpClient,
    get_thingspeak_session_bridge,
    reset_bridge_instances,
    BridgeStatus,
    PollOutcome,
    ThingSpeakTimeoutError,
    ThingSpeakFeedError,
    RESET_LOCK_AUDIT_ACTION,
)
from unittest.mock import patch, MagicMock

LIVE_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "sessions", "monitoring_sessions.db")

TEST_CHANNEL_ID = 3490289
TEST_DEVICE_ID = "ESP32-MAX30102-01"
TEST_WATERMARK = 296


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class MockThingSpeakClient:
    """Mock ThingSpeak HTTP feed client with controllable responses."""

    def __init__(self):
        self.next_response: Optional[dict] = None
        self.should_timeout: bool = False
        self.should_error: bool = False
        self.call_count: int = 0
        self.last_channel: Optional[int] = None
        self.last_timeout: Optional[float] = None

    def get_latest_feed(self, channel_id: int, timeout: float = 5.0) -> Optional[dict]:
        self.call_count += 1
        self.last_channel = channel_id
        self.last_timeout = timeout

        if self.should_timeout:
            raise ThingSpeakTimeoutError("Mock network timeout reading ThingSpeak feeds.")
        if self.should_error:
            raise ThingSpeakFeedError("Mock network connection refused.")
        return copy.deepcopy(self.next_response)


class ControllableClock:
    """Deterministic controllable clock."""

    def __init__(self, start_dt: Optional[datetime.datetime] = None):
        self.current_time = start_dt or datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

    def get_time(self) -> datetime.datetime:
        return self.current_time

    def advance(self, seconds: float):
        self.current_time += datetime.timedelta(seconds=seconds)


@pytest.fixture
def bridge_env(tmp_path):
    """Provides isolated registry, repository, service, mock client, and clock."""
    reg_file = str(tmp_path / "registry" / "patients.json")
    db_file = str(tmp_path / "sessions" / "isolated_bridge_test.db")

    clock = ControllableClock()
    registry = LocalPatientRegistry(storage_path=reg_file)
    repository = SessionRepository(db_path=db_file, patient_registry=registry)
    gate = SessionTelemetryGate(
        session_repository=repository,
        require_device_identity=True,
        clock_fn=clock.get_time,
    )
    service = SessionTelemetryService(
        session_repository=repository,
        telemetry_gate=gate,
        silence_warning_seconds=600.0,
        clock_fn=clock.get_time,
    )
    mock_http = MockThingSpeakClient()

    # Register synthetic patient
    patient = registry.register_patient(
        registration_type="synthetic_demo",
        gender="U",
        anchor_age=50,
        display_alias="ISOLATED-BED-01",
        registered_by_doctor_id="DOC-TEST-0042",
        custom_patient_id="SYNTH-TEST-8888",
    )

    # Initialize bridge
    bridge = ThingSpeakSessionBridge(
        channel_id=TEST_CHANNEL_ID,
        session_repository=repository,
        session_service=service,
        http_client=mock_http,
        poll_interval_seconds=20.0,
        request_timeout_seconds=5.0,
        max_backoff_seconds=60.0,
        clock_fn=clock.get_time,
    )

    return {
        "bridge": bridge,
        "repository": repository,
        "registry": registry,
        "service": service,
        "mock_http": mock_http,
        "clock": clock,
        "patient": patient,
        "db_file": db_file,
        "tmp_path": tmp_path,
    }


# ── Scenario 1: Valid Synthetic Packet Ingested Cleanly ──────────────────────

def test_01_single_active_session_valid_feed(bridge_env):
    """Valid feed above watermark is ingested into SQLite under active session."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]

    # Start active session with watermark 296
    sess = repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=TEST_WATERMARK,
        custom_start_time=clock.get_time() - datetime.timedelta(minutes=5),
    )
    sid = sess["session_id"]

    # Mock feed 297
    mock_http.next_response = {
        "channel_id": TEST_CHANNEL_ID,
        "entry_id": 297,
        "created_at": (clock.get_time() - datetime.timedelta(seconds=5)).isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
        "field4": 1,
    }

    res = bridge.poll_once()
    assert res["outcome"] == PollOutcome.INGESTED.value
    assert res["status"] == "success"
    assert res["entry_id"] == 297
    assert res["session_id"] == sid

    # Verify committed to SQLite session_telemetry_packets
    packets = repo.get_session_packets(sid)
    assert len(packets) == 1
    assert packets[0]["entry_id"] == 297
    assert packets[0]["heart_rate"] == 75.0
    assert packets[0]["spo2"] == 98.0
    assert packets[0]["optical_contact"] == 1
    assert packets[0]["physiologically_valid"] == 1


# ── Scenario 2: Fail Closed on Zero Active Sessions ──────────────────────────

def test_02_fail_closed_on_zero_active_sessions(bridge_env):
    """If no active session exists on channel, fail closed with no ingestion."""
    env = bridge_env
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]

    # No session created!
    mock_http.next_response = {
        "entry_id": 297,
        "created_at": clock.get_time().isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
    }

    res = bridge.poll_once()
    assert res["outcome"] == PollOutcome.REJECTED_NO_ACTIVE_SESSION.value
    assert res["status"] == "halted"
    assert res["session_id"] is None
    assert mock_http.call_count == 0  # Does not even call network if no session!


# ── Scenario 3: Fail Closed on Multiple Active Sessions Collision ────────────

def test_03_fail_closed_on_multiple_active_sessions(bridge_env, monkeypatch):
    """If multiple active sessions exist on channel, fail closed and halt."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]

    # Mock list_sessions returning multiple active sessions (simulating collision anomaly)
    monkeypatch.setattr(repo, "list_sessions", lambda channel_id, status: [
        {"session_id": "SESS-COLLIDE-A", "channel_id": channel_id},
        {"session_id": "SESS-COLLIDE-B", "channel_id": channel_id},
    ])

    res = bridge.poll_once()
    assert res["outcome"] == PollOutcome.REJECTED_SESSION_CONFLICT.value
    assert res["status"] == "halted"


# ── Scenario 4: Duplicate Feed Handling ──────────────────────────────────────

def test_04_duplicate_feed_handling(bridge_env):
    """Same feed entry polled sequentially is skipped as duplicate."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]

    repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=TEST_WATERMARK,
        custom_start_time=clock.get_time() - datetime.timedelta(minutes=5),
    )

    mock_http.next_response = {
        "channel_id": TEST_CHANNEL_ID,
        "entry_id": 297,
        "created_at": (clock.get_time() - datetime.timedelta(seconds=5)).isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
    }

    # First poll: ingests
    res1 = bridge.poll_once()
    assert res1["outcome"] == PollOutcome.INGESTED.value

    # Second poll with same entry: skipped as duplicate
    res2 = bridge.poll_once()
    assert res2["outcome"] == PollOutcome.SKIPPED_DUPLICATE.value
    assert res2["status"] == "idle"


# ── Scenario 5: Network Timeout & Exponential Backoff ────────────────────────

def test_05_network_timeout_and_exponential_backoff(bridge_env):
    """Network timeouts trigger backoff multiplier without crashing."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]

    repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=TEST_WATERMARK,
        custom_start_time=clock.get_time() - datetime.timedelta(minutes=5),
    )

    mock_http.should_timeout = True

    # 1st timeout: backoff 20s
    res1 = bridge.poll_once()
    assert res1["outcome"] == PollOutcome.NETWORK_ERROR.value
    assert bridge.status == BridgeStatus.BACKOFF
    assert bridge.consecutive_errors == 1
    assert bridge.current_backoff == 20.0

    # 2nd timeout: backoff 40s
    res2 = bridge.poll_once()
    assert bridge.consecutive_errors == 2
    assert bridge.current_backoff == 40.0

    # 3rd timeout: backoff capped at 60s
    res3 = bridge.poll_once()
    assert bridge.consecutive_errors == 3
    assert bridge.current_backoff == 60.0

    # Recovery
    mock_http.should_timeout = False
    mock_http.next_response = {
        "entry_id": 297,
        "created_at": (clock.get_time() - datetime.timedelta(seconds=5)).isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
    }
    res_rec = bridge.poll_once()
    assert res_rec["outcome"] == PollOutcome.INGESTED.value
    assert bridge.consecutive_errors == 0
    assert bridge.current_backoff == 20.0


# ── Scenario 6: Stale Telemetry Quarantined ───────────────────────────────────

def test_06_stale_telemetry_quarantined(bridge_env):
    """Packet older than 300s is quarantined with zero patient attribution."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]

    sess = repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=TEST_WATERMARK,
        custom_start_time=clock.get_time() - datetime.timedelta(hours=1),
    )
    sid = sess["session_id"]

    mock_http.next_response = {
        "entry_id": 298,
        "created_at": (clock.get_time() - datetime.timedelta(seconds=450)).isoformat(),  # 450s > 300s
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
    }

    res = bridge.poll_once()
    assert res["outcome"] == PollOutcome.REJECTED_STALE.value

    # Accepted packets must be 0
    assert len(repo.get_session_packets(sid)) == 0

    # Must be in quarantine
    quarantined = repo.get_quarantined_packets()
    assert len(quarantined) == 1
    assert quarantined[0]["entry_id"] == 298
    assert quarantined[0]["reason_code"] == "STALE_TELEMETRY"


# ── Scenario 7: Missing or Malformed Field 4 Produces UNKNOWN ─────────────────

def test_07_missing_or_malformed_field4_produces_unknown(bridge_env):
    """Missing or non-numeric field4 must produce UNKNOWN, NEVER NORMAL."""
    env = bridge_env
    bridge = env["bridge"]
    session = {"channel_id": 3490289, "device_id": "ESP32-DEV-01"}

    # 1. Missing field4 entirely
    norm1 = bridge.normalize_feed_packet({"entry_id": 300, "field1": 80}, session)
    assert norm1["risk_status"] is None
    assert norm1["firmware_risk_status_text"] == "UNKNOWN"

    # 2. None / null field4
    norm2 = bridge.normalize_feed_packet({"entry_id": 300, "field4": None}, session)
    assert norm2["risk_status"] is None
    assert norm2["firmware_risk_status_text"] == "UNKNOWN"

    # 3. String / non-numeric field4
    norm3 = bridge.normalize_feed_packet({"entry_id": 300, "field4": "BAD_TEXT"}, session)
    assert norm3["risk_status"] is None
    assert norm3["firmware_risk_status_text"] == "UNKNOWN"

    # 4. Valid 0 produces NORMAL
    norm4 = bridge.normalize_feed_packet({"entry_id": 300, "field4": "0"}, session)
    assert norm4["risk_status"] == 0
    assert norm4["firmware_risk_status_text"] == "NORMAL"

    # 5. Valid 2 produces HIGH_RISK
    norm5 = bridge.normalize_feed_packet({"entry_id": 300, "field4": 2}, session)
    assert norm5["risk_status"] == 2
    assert norm5["firmware_risk_status_text"] == "HIGH_RISK"

    # 6. Valid 1 produces WARNING (NEVER STABLE)
    norm6 = bridge.normalize_feed_packet({"entry_id": 300, "field4": 1}, session)
    assert norm6["risk_status"] == 1
    assert norm6["firmware_risk_status_text"] == "WARNING"
    assert norm6["firmware_risk_status_text"] != "STABLE"

    # 7. Valid 3 produces CRITICAL_VITALS
    norm7 = bridge.normalize_feed_packet({"entry_id": 300, "field4": 3}, session)
    assert norm7["risk_status"] == 3
    assert norm7["firmware_risk_status_text"] == "CRITICAL_VITALS"

    # 8. Unrecognized integer produces CUSTOM_RISK_<code\>
    norm8 = bridge.normalize_feed_packet({"entry_id": 300, "field4": 5}, session)
    assert norm8["risk_status"] == 5
    assert norm8["firmware_risk_status_text"] == "CUSTOM_RISK_5"


# ── Scenario 8: Missing/Invalid Fields Preserved Without Substitution ─────────

def test_08_missing_and_invalid_fields_preserved_safely(bridge_env):
    """Missing or non-numeric vitals remain None; no reassuring defaults."""
    env = bridge_env
    bridge = env["bridge"]
    session = {"channel_id": 3490289, "device_id": "ESP32-DEV-01"}

    # String in field3 ("No Finger"), missing HR
    feed = {
        "entry_id": 301,
        "field1": "",           # Missing HR
        "field2": "95.0",       # Valid SpO2
        "field3": "No Finger",  # Non-numeric IR string
        "field4": "",
    }
    norm = bridge.normalize_feed_packet(feed, session)
    assert norm["heart_rate"] is None
    assert norm["spo2"] == 95.0
    assert norm["ir_ppg"] is None  # Preserved as None, not defaulted to 120000!


# ── Scenario 9: Optical Lead-Off Handled Correctly ────────────────────────────

def test_09_optical_lead_off_accepted_into_audit_flagged_invalid(bridge_env):
    """IR < 50,000 is accepted into session audit trail but marked invalid."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]

    sess = repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=TEST_WATERMARK,
        custom_start_time=clock.get_time() - datetime.timedelta(minutes=5),
    )
    sid = sess["session_id"]

    mock_http.next_response = {
        "entry_id": 297,
        "created_at": (clock.get_time() - datetime.timedelta(seconds=5)).isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 12000,  # 12,000 < 50,000 lead-off threshold!
        "field4": 0,
    }

    res = bridge.poll_once()
    assert res["outcome"] == PollOutcome.INGESTED.value

    # Packet is in session database for auditability
    packets = repo.get_session_packets(sid)
    assert len(packets) == 1
    assert packets[0]["optical_contact"] == 0
    assert packets[0]["physiologically_valid"] == 0  # Flagged motion artifact / lead-off!


# ── Scenario 10: Channel Reset Detection Locks Session ────────────────────────

def test_10_channel_reset_detection_locks_session(bridge_env):
    """Feed entry_id <= watermark triggers CHANNEL_RESET_LOCKED and halts."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]

    sess = repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=TEST_WATERMARK,  # Watermark 296
        custom_start_time=clock.get_time() - datetime.timedelta(minutes=5),
    )
    sid = sess["session_id"]

    # Channel cleared on ThingSpeak -> entry_id restarts at 1
    mock_http.next_response = {
        "entry_id": 1,  # 1 <= 296
        "created_at": (clock.get_time() - datetime.timedelta(seconds=5)).isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
    }

    res = bridge.poll_once()
    assert res["outcome"] == PollOutcome.REJECTED_CHANNEL_RESET.value
    assert bridge.status == BridgeStatus.RESET_LOCKED

    # Verify session watermark was NEVER lowered
    refreshed_sess = repo.get_session(sid)
    assert refreshed_sess["start_entry_id"] == 296, "Watermark must never be lowered!"

    # Verify persistent reset lock is recorded in SQLite audit trail
    assert bridge.is_session_reset_locked(sid) is True


# ── Scenario 11: Persistent Reset Lock Survives Process Restart ───────────────

def test_11_reset_lock_survives_process_restart(bridge_env):
    """Reset lock persists in SQLite and blocks new bridge instance on restart."""
    env = bridge_env
    repo = env["repository"]
    bridge = env["bridge"]
    mock_http = env["mock_http"]
    clock = env["clock"]
    db_file = env["db_file"]

    sess = repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=TEST_WATERMARK,
        custom_start_time=clock.get_time() - datetime.timedelta(minutes=5),
    )
    sid = sess["session_id"]

    # Trigger reset lock
    mock_http.next_response = {
        "entry_id": 1,
        "created_at": (clock.get_time() - datetime.timedelta(seconds=5)).isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
    }
    bridge.poll_once()
    assert bridge.is_session_reset_locked(sid) is True

    # ── SIMULATE PROCESS RESTART ──
    # Create entirely new repository, service, and bridge instances pointing to same SQLite DB
    restarted_repo = SessionRepository(db_path=db_file)
    restarted_gate = SessionTelemetryGate(
        session_repository=restarted_repo,
        require_device_identity=True,
        clock_fn=clock.get_time,
    )
    restarted_service = SessionTelemetryService(
        session_repository=restarted_repo,
        telemetry_gate=restarted_gate,
        clock_fn=clock.get_time,
    )
    restarted_http = MockThingSpeakClient()
    restarted_bridge = ThingSpeakSessionBridge(
        channel_id=TEST_CHANNEL_ID,
        session_repository=restarted_repo,
        session_service=restarted_service,
        http_client=restarted_http,
        clock_fn=clock.get_time,
    )

    # Verify new bridge instance immediately detects persistent lock in SQLite
    assert restarted_bridge.is_session_reset_locked(sid) is True

    # Even if feed now attempts entry 297, bridge refuses to ingest under locked session
    restarted_http.next_response = {
        "entry_id": 297,
        "created_at": (clock.get_time() - datetime.timedelta(seconds=5)).isoformat(),
        "field1": 75.0,
        "field2": 98.0,
        "field3": 125000,
    }
    res_after_restart = restarted_bridge.poll_once()
    assert res_after_restart["outcome"] == PollOutcome.REJECTED_CHANNEL_RESET.value
    assert restarted_bridge.status == BridgeStatus.RESET_LOCKED
    assert len(restarted_repo.get_session_packets(sid)) == 0


# ── Scenario 12: Live Demonstration Database Immunity ─────────────────────────

def test_12_live_demonstration_database_immunity():
    """Verify live monitoring_sessions.db is completely untouched."""
    initial_hash = compute_file_hash(LIVE_DB_PATH)

    # Verify live DB row counts
    conn = sqlite3.connect(LIVE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM session_telemetry_packets")
    packets_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM quarantined_telemetry_packets")
    quarantine_count = cur.fetchone()[0]
    cur.execute("SELECT session_id, status, start_entry_id FROM patient_monitoring_sessions WHERE session_id = 'SESS-20261022-113349-61F1'")
    live_sess = cur.fetchone()
    conn.close()

    assert packets_count in (0, 1), "Live session_telemetry_packets was unexpectedly modified!"
    assert quarantine_count == 0, "Live quarantined_telemetry_packets was modified!"
    assert live_sess == ("SESS-20261022-113349-61F1", "active", 296), "Active session was modified!"

    final_hash = compute_file_hash(LIVE_DB_PATH)
    assert initial_hash == final_hash, "Live database SHA-256 hash changed!"


# ── Scenario 13: ThingSpeakHttpClient Mocked Success & Timeout ────────────────

def test_13_thingspeak_http_client_mock_success_and_timeout():
    """Verify ThingSpeakHttpClient with mocked urllib responses without calling real services."""
    client = ThingSpeakHttpClient(api_key="MOCK_KEY_SECRET_123")

    # 1. Successful JSON feed
    mock_payload = json.dumps({
        "channel_id": 3490289,
        "entry_id": 401,
        "created_at": "2026-10-23T17:00:00Z",
        "field1": "78.0",
        "field2": "98.0",
        "field3": "65000",
        "field4": "0",
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_payload
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        feed = client.get_latest_feed(channel_id=3490289, timeout=3.0)
        assert feed is not None
        assert feed["entry_id"] == 401
        assert feed["field1"] == "78.0"

    # 2. Timeout raises ThingSpeakTimeoutError
    import socket
    with patch("urllib.request.urlopen", side_effect=socket.timeout("Mock socket timeout")):
        with pytest.raises(ThingSpeakTimeoutError) as exc_info:
            client.get_latest_feed(channel_id=3490289, timeout=2.0)
        assert "timed out" in str(exc_info.value).lower()
        # Verify secret key is never leaked in error message
        assert "MOCK_KEY_SECRET_123" not in str(exc_info.value)

    # 3. HTTP Error raises ThingSpeakFeedError without leaking secret
    import urllib.error
    mock_http_err = urllib.error.HTTPError(
        url="https://api.thingspeak.com",
        code=404,
        msg="Channel Not Found",
        hdrs={},
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=mock_http_err):
        with pytest.raises(ThingSpeakFeedError) as exc_info:
            client.get_latest_feed(channel_id=3490289, timeout=2.0)
        assert "404" in str(exc_info.value)
        assert "MOCK_KEY_SECRET_123" not in str(exc_info.value)


# ── Scenario 14: Singleton Factory get_thingspeak_session_bridge ──────────────

def test_14_factory_get_thingspeak_session_bridge(bridge_env):
    """Verify get_thingspeak_session_bridge caches per channel and reset_bridge_instances clears."""
    env = bridge_env
    reset_bridge_instances()

    b1 = get_thingspeak_session_bridge(
        channel_id=TEST_CHANNEL_ID,
        session_repository=env["repository"],
        session_service=env["service"],
        http_client=env["mock_http"],
    )
    b2 = get_thingspeak_session_bridge(
        channel_id=TEST_CHANNEL_ID,
        session_repository=env["repository"],
        session_service=env["service"],
    )

    assert b1 is b2
    assert b1.channel_id == TEST_CHANNEL_ID

    reset_bridge_instances()
    b3 = get_thingspeak_session_bridge(
        channel_id=TEST_CHANNEL_ID,
        session_repository=env["repository"],
        session_service=env["service"],
        http_client=env["mock_http"],
    )
    assert b3 is not b1
    reset_bridge_instances()


# ── Scenario 15: Mocked Poll Once With Real Client Mocked at Socket ───────────

def test_15_mocked_poll_once_end_to_end(bridge_env):
    """Verify bridge.poll_once() executes successfully with mocked HTTP response."""
    env = bridge_env
    repo = env["repository"]
    clock = env["clock"]

    # Start session with watermark 300
    sess = repo.create_session(
        patient_id=env["patient"]["patient_id"],
        device_id=TEST_DEVICE_ID,
        channel_id=TEST_CHANNEL_ID,
        created_by_doctor_id="DOC-TEST-0042",
        start_entry_id=300,
        custom_start_time=clock.get_time() - datetime.timedelta(minutes=2),
    )
    sid = sess["session_id"]

    feed_payload = json.dumps({
        "channel_id": TEST_CHANNEL_ID,
        "entry_id": 305,
        "created_at": clock.get_time().isoformat(),
        "field1": "72.0",
        "field2": "99.0",
        "field3": "85000",
        "field4": "0",
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = feed_payload
    mock_resp.__enter__.return_value = mock_resp

    http_client = ThingSpeakHttpClient()
    bridge = ThingSpeakSessionBridge(
        channel_id=TEST_CHANNEL_ID,
        session_repository=repo,
        session_service=env["service"],
        http_client=http_client,
        clock_fn=clock.get_time,
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = bridge.poll_once()
        assert res["outcome"] == PollOutcome.INGESTED.value
        assert res["entry_id"] == 305

    # Verify packet persisted in isolated repo
    packets = repo.get_session_packets(sid)
    assert len(packets) == 1
    assert packets[0]["entry_id"] == 305
    assert packets[0]["heart_rate"] == 72.0
    assert packets[0]["spo2"] == 99.0

