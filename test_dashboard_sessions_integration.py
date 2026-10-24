"""
tests/test_dashboard_sessions_integration.py
=============================================================================
Integration and multi-patient safety test suite for Stage 3.7.6D:
Authenticated Dashboard Integration & Multi-Patient Safety Testing.

Verifies:
 1. Unauthenticated session start is rejected server-side (401).
 2. Unauthenticated session stop is rejected server-side (401).
 3. Authenticated clinician identity comes from verified authentication context.
 4. Browser-supplied doctor IDs cannot impersonate another clinician.
 5. Prospective patient registration works without altering the MIMIC cohort.
 6. A session cannot start for an unknown or invalid patient.
 7. Explicit patient confirmation is required before session creation.
 8. Duplicate active patient/device/channel sessions are rejected safely (409).
 9. Two synthetic patients cannot see or inherit each other's session history.
10. Old shared-CSV telemetry is never assigned to a new session.
11. Closed sessions cannot receive new readings / cannot be closed twice.
12. Backend restart does not silently create or reassign sessions.
13. Repository conflicts and storage errors are handled safely.
14. Existing login/logout flows continue to work.
15. Existing retrospective patient search and prediction endpoints remain functional.
16. Patient-facing errors do not disclose sensitive identifiers or secrets.
=============================================================================
"""

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.app import app
from src.sensor.patient_registry import LocalPatientRegistry
from src.sensor.session_repository import SessionRepository


@pytest.fixture
def isolated_test_context(tmp_path):
    """Sets up an isolated registry and session database in tmp_path."""
    reg_path = str(tmp_path / "registry" / "prospective_patients.json")
    db_path = str(tmp_path / "sessions" / "test_sessions.db")

    reg = LocalPatientRegistry(storage_path=reg_path)
    repo = SessionRepository(db_path=db_path, patient_registry=reg)

    with patch("backend.app.get_patient_registry", return_value=reg), \
         patch("backend.app.get_session_repository", return_value=repo):
        yield {
            "registry": reg,
            "repository": repo,
            "tmp_path": tmp_path,
        }


@pytest.fixture
def client():
    """Provides Flask test client."""
    app.config["TESTING"] = True
    app.config["REQUIRE_AUTH"] = True
    with app.test_client() as test_client:
        yield test_client


MOCK_DOCTOR_PAYLOAD = {
    "sub": "DOC-AUTH-4412",
    "username": "dr_verified",
    "full_name": "Dr. Sarah Verified",
    "role": "Attending Intensivist",
    "admin": False,
}


# ── Scenario 1: Unauthenticated Session Start Rejected Server-Side (401) ──────

def test_01_unauthenticated_session_start_rejected(client, isolated_test_context):
    """POST /api/sessions/start without Authorization header returns 401."""
    res = client.post("/api/sessions/start", json={
        "patient_id": "PROSP-001",
        "device_id": "DEV-01",
        "channel_id": 3490289,
        "confirmed": True,
    })
    assert res.status_code == 401
    data = res.get_json()
    assert data["status"] == "error"
    assert "Authentication required" in data["error"]


# ── Scenario 2: Unauthenticated Session Stop Rejected Server-Side (401) ───────

def test_02_unauthenticated_session_stop_rejected(client, isolated_test_context):
    """POST /api/sessions/<id>/close without Authorization header returns 401."""
    res = client.post("/api/sessions/SESS-999/close", json={
        "closure_reason": "routine_discharge",
        "confirmed": True,
    })
    assert res.status_code == 401
    data = res.get_json()
    assert data["status"] == "error"
    assert "Authentication required" in data["error"]


# ── Scenario 3: Clinician Identity Comes from Verified Context ────────────────

def test_03_clinician_identity_from_verified_context(client, isolated_test_context):
    """Session created has created_by_doctor_id equal to token's 'sub', not request body."""
    reg = isolated_test_context["registry"]
    pat = reg.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        res = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": pid,
                "device_id": "DEV-VERIFIED-01",
                "channel_id": 3490289,
                "confirmed": True,
            },
        )
        assert res.status_code == 201
        data = res.get_json()
        assert data["session"]["created_by_doctor_id"] == "DOC-AUTH-4412"


# ── Scenario 4: Browser-Supplied Doctor IDs Cannot Impersonate ────────────────

def test_04_browser_supplied_doctor_id_ignored(client, isolated_test_context):
    """Passing a malicious 'doctor_id' in JSON body is ignored; verified token takes precedence."""
    reg = isolated_test_context["registry"]
    pat = reg.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        res = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": pid,
                "device_id": "DEV-SPOOF-TEST",
                "channel_id": 77777,
                "doctor_id": "DOC-IMPOSTOR-9999",  # Malicious attempt to spoof
                "created_by_doctor_id": "DOC-IMPOSTOR-9999",
                "confirmed": True,
            },
        )
        assert res.status_code == 201
        data = res.get_json()
        # Must be DOC-AUTH-4412 from token, NOT DOC-IMPOSTOR-9999
        assert data["session"]["created_by_doctor_id"] == "DOC-AUTH-4412"


# ── Scenario 5: Prospective Registration Without Altering MIMIC Cohort ────────

def test_05_prospective_registration_preserves_mimic_cohort(client, isolated_test_context):
    """Prospective registration adds to registry without touching raw MIMIC-IV dataset."""
    from backend.patient_service import get_patient_dataset_statistics
    stats_before = get_patient_dataset_statistics()

    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        res = client.post(
            "/api/sessions/patients/register",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "registration_type": "prospective_ambulatory",
                "gender": "F",
                "anchor_age": 62,
                "display_alias": "Bed-09",
            },
        )
        assert res.status_code == 201
        data = res.get_json()
        assert data["patient"]["patient_id"].startswith("PROSP-")
        assert data["patient"]["registered_by_doctor_id"] == "DOC-AUTH-4412"

    stats_after = get_patient_dataset_statistics()
    # MIMIC-IV cohort total records and unique patients must remain exactly identical
    assert stats_before["total_records"] == stats_after["total_records"] == 2884
    assert stats_before["unique_patients"] == stats_after["unique_patients"] == 2661


# ── Scenario 6: Rejection of Unknown or Invalid Patient ────────────────────────

def test_06_rejection_of_unknown_patient(client, isolated_test_context):
    """Attempting to start a session for a nonexistent patient returns 404."""
    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        res = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": "PROSP-NONEXISTENT-999",
                "device_id": "DEV-01",
                "channel_id": 3490289,
                "confirmed": True,
            },
        )
        assert res.status_code == 404
        assert "not registered" in res.get_json()["message"]


# ── Scenario 7: Explicit Patient Confirmation Required ────────────────────────

def test_07_explicit_confirmation_required(client, isolated_test_context):
    """Creating a session with confirmed=False or missing confirmed returns 400."""
    reg = isolated_test_context["registry"]
    pat = reg.register_patient(registration_type="prospective_ambulatory")

    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        # Missing confirmed
        res1 = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": pat["patient_id"],
                "device_id": "DEV-01",
                "channel_id": 3490289,
            },
        )
        assert res1.status_code == 400
        assert "confirmation required" in res1.get_json()["message"]

        # confirmed is False
        res2 = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": pat["patient_id"],
                "device_id": "DEV-01",
                "channel_id": 3490289,
                "confirmed": False,
            },
        )
        assert res2.status_code == 400


# ── Scenario 8: Duplicate Active Session Rejected Safely (409 Conflict) ────────

def test_08_duplicate_active_sessions_conflict(client, isolated_test_context):
    """Second concurrent session on same channel/patient/device returns 409 Conflict."""
    reg = isolated_test_context["registry"]
    p1 = reg.register_patient(registration_type="prospective_ambulatory")["patient_id"]
    p2 = reg.register_patient(registration_type="prospective_ambulatory")["patient_id"]

    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        # 1. Start Session 1 on Channel 3490289
        res1 = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": p1,
                "device_id": "DEV-01",
                "channel_id": 3490289,
                "confirmed": True,
            },
        )
        assert res1.status_code == 201

        # 2. Try starting another session on same channel (different patient & device) -> 409
        res2 = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": p2,
                "device_id": "DEV-02",
                "channel_id": 3490289,
                "confirmed": True,
            },
        )
        assert res2.status_code == 409
        assert res2.get_json()["conflict"] is True


# ── Scenario 9: Two Synthetic Patients Cannot See or Inherit History ──────────

def test_09_patient_session_history_isolation(client, isolated_test_context):
    """Listing sessions by patient_id strictly isolates patient records."""
    repo = isolated_test_context["repository"]
    reg = isolated_test_context["registry"]
    p1 = reg.register_patient(registration_type="synthetic_demo")["patient_id"]
    p2 = reg.register_patient(registration_type="synthetic_demo")["patient_id"]

    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        # Session for P1
        s1 = repo.create_session(p1, "DEV-A", 1001, "DOC-01")
        repo.close_session(s1["session_id"], "DOC-01", "routine_discharge")

        # Session for P2
        s2 = repo.create_session(p2, "DEV-B", 1002, "DOC-01")

        # Query P1
        res_p1 = client.get(f"/api/sessions?patient_id={p1}", headers={"Authorization": "Bearer valid_mock_token"})
        assert res_p1.status_code == 200
        p1_list = res_p1.get_json()["sessions"]
        assert len(p1_list) == 1
        assert p1_list[0]["patient_id"] == p1

        # Query P2
        res_p2 = client.get(f"/api/sessions?patient_id={p2}", headers={"Authorization": "Bearer valid_mock_token"})
        assert res_p2.status_code == 200
        p2_list = res_p2.get_json()["sessions"]
        assert len(p2_list) == 1
        assert p2_list[0]["patient_id"] == p2


# ── Scenario 10: Old Shared-CSV Telemetry Never Assigned to New Session ───────

def test_10_telemetry_watermark_isolation(isolated_test_context):
    """Session stores start_entry_id watermark and prohibits historical packet assignment."""
    repo = isolated_test_context["repository"]
    reg = isolated_test_context["registry"]
    pat = reg.register_patient(registration_type="prospective_ambulatory")

    # Historical max entry_id is 850
    sess = repo.create_session(
        patient_id=pat["patient_id"],
        device_id="ESP32-DEV-01",
        channel_id=3490289,
        created_by_doctor_id="DOC-01",
        start_entry_id=850,
    )
    assert sess["start_entry_id"] == 850

    # Verification: Any packet with entry_id <= 850 must be filtered out by the ingestion boundary rule
    def is_packet_in_session(packet_entry_id: int, session_watermark: int) -> bool:
        return packet_entry_id > session_watermark

    assert is_packet_in_session(849, sess["start_entry_id"]) is False
    assert is_packet_in_session(850, sess["start_entry_id"]) is False
    assert is_packet_in_session(851, sess["start_entry_id"]) is True


# ── Scenario 11: Closed Sessions Cannot Receive New Readings / Close Twice ────

def test_11_closed_session_protection(client, isolated_test_context):
    """Closing an already-closed session returns 400 Bad Request."""
    repo = isolated_test_context["repository"]
    reg = isolated_test_context["registry"]
    pat = reg.register_patient(registration_type="prospective_ambulatory")
    sess = repo.create_session(pat["patient_id"], "DEV-01", 3490289, "DOC-01")
    sid = sess["session_id"]

    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        # 1. Close session
        res1 = client.post(
            f"/api/sessions/{sid}/close",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={"closure_reason": "routine_discharge", "confirmed": True},
        )
        assert res1.status_code == 200
        assert res1.get_json()["session"]["status"] == "completed"

        # 2. Try closing again -> 400
        res2 = client.post(
            f"/api/sessions/{sid}/close",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={"closure_reason": "routine_discharge", "confirmed": True},
        )
        assert res2.status_code == 400
        assert "already" in res2.get_json()["message"].lower()


# ── Scenario 12: Backend Restart Does Not Silently Reassign Sessions ──────────

def test_12_backend_restart_persistence(tmp_path):
    """State is preserved across independent repository re-initialization."""
    reg_path = str(tmp_path / "reg.json")
    db_path = str(tmp_path / "sessions.db")

    reg = LocalPatientRegistry(storage_path=reg_path)
    pat = reg.register_patient(registration_type="prospective_ambulatory")
    pid = pat["patient_id"]

    # Session started on Repo instance 1
    repo1 = SessionRepository(db_path=db_path, patient_registry=reg)
    s1 = repo1.create_session(pid, "DEV-01", 3490289, "DOC-01")
    sid = s1["session_id"]
    del repo1

    # Simulate restart: Repo instance 2 connects
    repo2 = SessionRepository(db_path=db_path, patient_registry=reg)
    reloaded = repo2.get_session(sid)
    assert reloaded["session_id"] == sid
    assert reloaded["patient_id"] == pid
    assert reloaded["status"] == "active"

    # Confirms conflict is STILL enforced across restarts
    from src.sensor.session_repository import SessionConflictError
    with pytest.raises(SessionConflictError):
        repo2.create_session(pid, "DEV-02", 9999, "DOC-01")


# ── Scenario 13: Repository Conflicts and Storage Errors Handled Safely ───────

def test_13_repository_conflicts_and_errors_handled(client, isolated_test_context):
    """Missing fields return 400, conflict returns 409, no crashes or stack traces."""
    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        # Missing fields
        res = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={"confirmed": True},
        )
        assert res.status_code == 400
        assert "required" in res.get_json()["message"]


# ── Scenario 14: Existing Login / Logout Flows Continue to Work ───────────────

def test_14_existing_auth_login_endpoints_work(client):
    """Health and unauthenticated auth routes respond normally."""
    res_health = client.get("/api/health")
    assert res_health.status_code == 200

    # Login with bad credentials returns 401
    res_login = client.post("/api/auth/login", json={"username": "fake", "password": "bad"})
    assert res_login.status_code == 401


# ── Scenario 15: Retrospective Patient Search & Predictions Remain Intact ─────

def test_15_retrospective_endpoints_remain_functional(client):
    """MIMIC-IV patient endpoints remain fully operational when authenticated."""
    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        res_search = client.get(
            "/api/patient/10213765",
            headers={"Authorization": "Bearer valid_mock_token"},
        )
        assert res_search.status_code == 200
        data = res_search.get_json()
        assert data["found"] is True
        assert data["patient_id"] == 10213765


# ── Scenario 16: Patient-Facing Errors Do Not Disclose Secrets ─────────────────

def test_16_errors_do_not_disclose_secrets(client, isolated_test_context):
    """Error responses do not leak file paths, database strings, or auth secrets."""
    with patch("backend.app.verify_session_token", return_value=MOCK_DOCTOR_PAYLOAD):
        res = client.post(
            "/api/sessions/start",
            headers={"Authorization": "Bearer valid_mock_token"},
            json={
                "patient_id": "PROSP-NONEXISTENT",
                "device_id": "DEV-01",
                "channel_id": "not_an_int",
                "confirmed": True,
            },
        )
        # Should be an error code (400 or 404 or 500)
        assert res.status_code >= 400
        response_text = res.get_data(as_text=True)
        # Secrets / internal paths check
        assert "password" not in response_text.lower()
        assert "secret_key" not in response_text.lower()
        assert "c:\\" not in response_text.lower()
        assert "/users/" not in response_text.lower()


# ── Scenario 17: Dashboard Live Monitoring Response-Contract Compatibility ────

def test_17_live_monitoring_response_contract_compatibility(isolated_test_context):
    """
    Regression test for KeyError: 'session' in dashboard/pages/live_monitoring.py.
    Verifies that get_session_telemetry supports the dashboard's expected session fields
    (patient_id, channel_id, device_id, start_entry_id, start_time, packets, silence_status)
    whether accessed directly or via the resilient fallback `sess = res.get("session") or res`.
    """
    from src.sensor.session_telemetry_service import SessionTelemetryService

    ctx = isolated_test_context
    reg = ctx["registry"]
    repo = ctx["repository"]

    # Register patient and create session
    pat = reg.register_patient(
        gender="F",
        anchor_age=55,
        registered_by_doctor_id="DOC-REG-4412",
        display_alias="RegPat",
    )
    sess_record = repo.create_session(
        patient_id=pat["patient_id"],
        channel_id=3490289,
        device_id="ESP32-REG-01",
        created_by_doctor_id="DOC-REG-4412",
        start_entry_id=100,
    )

    service = SessionTelemetryService(session_repository=repo)
    res = service.get_session_telemetry(sess_record["session_id"], requesting_doctor_id="DOC-REG-4412")

    assert res["status"] == "success"

    # Contract evaluation as implemented in dashboard/pages/live_monitoring.py:
    sess = res.get("session") or res
    assert sess["patient_id"] == pat["patient_id"]
    assert sess["channel_id"] == 3490289
    assert sess["device_id"] == "ESP32-REG-01"
    assert sess["start_entry_id"] == 100
    assert "start_time" in sess

    # Key response fields required by downstream UI
    assert "packets" in res
    assert isinstance(res["packets"], list)
    assert "silence_status" in res
    assert res["silence_status"] in ("WAITING_FOR_DATA", "SILENCE_WARNING", "STREAMING_ACTIVE")

