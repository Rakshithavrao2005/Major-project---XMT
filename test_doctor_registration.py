"""
tests/test_doctor_registration.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Test Suite for Administrator-Controlled Doctor Registration.

Verifies:
1. Strict server-side authorization: clinical roles never grant admin rights.
2. Only authorized administrators can register doctor accounts.
3. Cryptographic token claims and tamper resistance (admin claim validation against registry).
4. Validation rules: uniqueness of username & email, password strength, clinical role whitelist.
5. Registered doctors are strictly initialized with is_admin=False.
6. Backend API endpoint /api/admin/register-doctor (401 unauth, 403 forbidden, 400 bad data, 201 created).
7. End-to-end flow: admin registers doctor -> doctor signs in with PBKDF2 hash verification.
8. UI gatekeeper enforcement in dashboard/pages/register_doctor.py.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from src.auth.auth_manager import (
    ALLOWED_ROLES,
    authenticate_doctor,
    create_session_token,
    verify_session_token,
    is_administrator,
    provision_doctor,
    provision_administrator,
    register_doctor,
    validate_password_strength,
)
import src.auth.auth_manager as auth_mod
from backend.app import app


@pytest.fixture(autouse=True)
def isolated_auth_registry(tmp_path, monkeypatch):
    """Isolate auth storage in a temporary directory for each test."""
    temp_dir = str(tmp_path / "auth")
    os.makedirs(temp_dir, exist_ok=True)
    registry_file = os.path.join(temp_dir, "doctors.json")
    secret_key_file = os.path.join(temp_dir, ".secret_key")
    revoked_file = os.path.join(temp_dir, "revoked_tokens.json")

    monkeypatch.setattr(auth_mod, "AUTH_DIR", temp_dir)
    monkeypatch.setattr(auth_mod, "REGISTRY_FILE", registry_file)
    monkeypatch.setattr(auth_mod, "SECRET_KEY_FILE", secret_key_file)
    monkeypatch.setattr(auth_mod, "REVOKED_TOKENS_FILE", revoked_file)

    # Initialize a base test registry with one doctor and one administrator
    admin = provision_administrator(
        username="admin_test",
        email="admin@hospital.org",
        full_name="Lead System Administrator",
        password="AdminSecure@2026!",
    )
    doctor = provision_doctor(
        username="dr_base",
        email="base@hospital.org",
        full_name="Dr. Base Physician, MD",
        password="ClinicalDoctor@2026!",
        role="Attending Intensivist",
    )
    return {"admin": admin, "doctor": doctor}


# ── 1. Administrator Authorization Model Tests ─────────────────────────────────

class TestAdministratorAuthorizationModel:
    def test_01_is_administrator_logic(self):
        """Clinical roles and arbitrary dicts never grant administrator privileges."""
        assert is_administrator(None) is False
        assert is_administrator({}) is False
        assert is_administrator({"role": "Attending Intensivist"}) is False
        assert is_administrator({"role": "Chief of Medicine"}) is False
        assert is_administrator({"is_admin": False}) is False
        assert is_administrator({"admin": False}) is False

        # True only when is_admin or admin claim is explicitly True
        assert is_administrator({"is_admin": True}) is True
        assert is_administrator({"admin": True}) is True

    def test_02_provision_doctor_defaults_to_non_admin(self):
        """Provisioned doctors are strictly is_admin=False."""
        doc = provision_doctor(
            username="dr_intern",
            email="intern@hospital.org",
            full_name="Dr. Intern Doe, MD",
            password="StrongPass@2026!",
            role="Resident Physician",
        )
        assert doc.get("is_admin") is False

        # Verify persisted record in registry
        reg = auth_mod.load_doctor_registry()
        persisted = reg["doctors"]["dr_intern"]
        assert persisted.get("is_admin") is False

    def test_03_provision_administrator_sets_admin_flag(self):
        """Administrator provisioning properly assigns is_admin=True and admin role."""
        admin = provision_administrator(
            username="admin_sec",
            email="sec_admin@hospital.org",
            full_name="Security Admin",
            password="StrongAdmin@2026!",
        )
        assert admin.get("is_admin") is True
        assert admin.get("role") == "System Administrator"

        reg = auth_mod.load_doctor_registry()
        persisted = reg["doctors"]["admin_sec"]
        assert persisted.get("is_admin") is True


# ── 2. Session Token Admin Claims and Integrity ───────────────────────────────

class TestSessionTokenAdminClaims:
    def test_04_token_contains_admin_claim(self, isolated_auth_registry):
        """Tokens carry accurate admin claim matching account privileges."""
        admin = isolated_auth_registry["admin"]
        doctor = isolated_auth_registry["doctor"]

        admin_token = create_session_token(admin)
        doctor_token = create_session_token(doctor)

        admin_payload = verify_session_token(admin_token)
        doctor_payload = verify_session_token(doctor_token)

        assert admin_payload is not None
        assert admin_payload.get("admin") is True

        assert doctor_payload is not None
        assert doctor_payload.get("admin") is False

    def test_05_tampered_token_admin_escalation_rejected(self, isolated_auth_registry):
        """If a doctor token claims admin=True, verify_session_token cross-checks registry and rejects."""
        doctor = isolated_auth_registry["doctor"]

        # Legitimate doctor token
        doc_token = create_session_token(doctor)
        payload_b64, sig_b64 = doc_token.split(".")

        # Decode, tamper payload to set admin=True, and resign with legitimate key (simulating key leak or forged claim)
        payload = json.loads(auth_mod._b64decode_url(payload_b64).decode("utf-8"))
        payload["admin"] = True
        tampered_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        tampered_b64 = auth_mod._b64encode_url(tampered_json)

        secret = auth_mod.get_auth_secret_key()
        import hmac, hashlib
        tampered_sig = auth_mod._b64encode_url(hmac.new(secret, tampered_b64.encode("ascii"), hashlib.sha256).digest())
        tampered_token = f"{tampered_b64}.{tampered_sig}"

        # Must be rejected because dr_base in registry has is_admin=False
        verified = verify_session_token(tampered_token)
        assert verified is None, "Tampered admin claim must be rejected when user in registry is not admin"


# ── 3. Doctor Registration Service Logic ──────────────────────────────────────

class TestDoctorRegistrationService:
    def test_06_admin_can_register_doctor_successfully(self, isolated_auth_registry):
        """Authorized administrator can register a new doctor."""
        admin = isolated_auth_registry["admin"]
        doc_data = {
            "username": "dr_new_cln",
            "email": "new.cln@hospital.org",
            "full_name": "Dr. New Clinician, MD",
            "password": "ValidDoctorPass@2026!",
            "role": "ICU Fellow",
            "department": "Trauma ICU",
        }

        registered = register_doctor(admin, doc_data)
        assert registered["username"] == "dr_new_cln"
        assert registered["email"] == "new.cln@hospital.org"
        assert registered["role"] == "ICU Fellow"
        assert registered["is_admin"] is False
        assert "password_hash" not in registered

    def test_07_clinician_cannot_register_doctor(self, isolated_auth_registry):
        """Ordinary clinician account attempting to register a doctor raises PermissionError."""
        doctor = isolated_auth_registry["doctor"]
        doc_data = {
            "username": "dr_unauthorized",
            "email": "unauth@hospital.org",
            "full_name": "Dr. Unauthorized, MD",
            "password": "ValidDoctorPass@2026!",
            "role": "Medical Officer",
        }

        with pytest.raises(PermissionError, match="Access Denied"):
            register_doctor(doctor, doc_data)

    def test_08_unauthenticated_entity_cannot_register_doctor(self):
        """Unauthenticated entity (None or empty dict) raises PermissionError."""
        with pytest.raises(PermissionError, match="Access Denied"):
            register_doctor({}, {"username": "foo"})

    def test_09_registration_rejects_duplicate_username(self, isolated_auth_registry):
        """Duplicate username is rejected with ValueError."""
        admin = isolated_auth_registry["admin"]
        doc_data = {
            "username": "dr_base",  # Already exists
            "email": "unique@hospital.org",
            "full_name": "Dr. Duplicate, MD",
            "password": "ValidPass@2026!",
            "role": "Attending Intensivist",
        }
        with pytest.raises(ValueError, match="already exists"):
            register_doctor(admin, doc_data)

    def test_10_registration_rejects_duplicate_email(self, isolated_auth_registry):
        """Duplicate email is rejected with ValueError."""
        admin = isolated_auth_registry["admin"]
        doc_data = {
            "username": "dr_unique_user",
            "email": "base@hospital.org",  # Already exists
            "full_name": "Dr. Duplicate Email, MD",
            "password": "ValidPass@2026!",
            "role": "Attending Intensivist",
        }
        with pytest.raises(ValueError, match="already exists"):
            register_doctor(admin, doc_data)

    def test_11_registration_rejects_weak_password(self, isolated_auth_registry):
        """Password failing complexity policy is rejected."""
        admin = isolated_auth_registry["admin"]
        doc_data = {
            "username": "dr_weak",
            "email": "weak@hospital.org",
            "full_name": "Dr. Weak Pass, MD",
            "password": "short",
            "role": "Attending Intensivist",
        }
        with pytest.raises(ValueError, match="at least 10 characters"):
            register_doctor(admin, doc_data)

    def test_12_registration_rejects_invalid_clinical_role(self, isolated_auth_registry):
        """Non-whitelisted clinical roles are rejected."""
        admin = isolated_auth_registry["admin"]
        doc_data = {
            "username": "dr_hacker",
            "email": "hacker@hospital.org",
            "full_name": "Dr. Hacker, MD",
            "password": "ValidPass@2026!",
            "role": "Superuser / Root",
        }
        with pytest.raises(ValueError, match="not a permitted clinical role"):
            register_doctor(admin, doc_data)

    def test_13_registration_cannot_grant_admin_via_role_or_payload(self, isolated_auth_registry):
        """System Administrator role is blocked in doctor registration; is_admin is strictly False."""
        admin = isolated_auth_registry["admin"]
        doc_data = {
            "username": "dr_attempt_admin",
            "email": "attempt@hospital.org",
            "full_name": "Dr. Attempt Admin, MD",
            "password": "ValidPass@2026!",
            "role": "System Administrator",  # Not in ALLOWED_ROLES
        }
        with pytest.raises(ValueError):
            register_doctor(admin, doc_data)

    def test_14_registered_doctor_can_authenticate(self, isolated_auth_registry):
        """A registered doctor can immediately authenticate using username or email."""
        admin = isolated_auth_registry["admin"]
        reg_data = {
            "username": "dr_logintest",
            "email": "logintest@hospital.org",
            "full_name": "Dr. Login Test, MD",
            "password": "LoginTestPass@2026!",
            "role": "Consulting Pulmonologist",
        }
        register_doctor(admin, reg_data)

        # Authenticate by username
        ok, profile, msg = authenticate_doctor("dr_logintest", "LoginTestPass@2026!")
        assert ok is True
        assert profile["full_name"] == "Dr. Login Test, MD"
        assert profile["is_admin"] is False

        # Authenticate by email
        ok2, profile2, msg2 = authenticate_doctor("logintest@hospital.org", "LoginTestPass@2026!")
        assert ok2 is True
        assert profile2["username"] == "dr_logintest"


# ── 4. Backend REST API Registration Endpoint Tests ───────────────────────────

class TestBackendRegistrationAPI:
    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_15_api_register_unauthenticated_returns_401(self, client):
        """POST /api/admin/register-doctor without token returns 401."""
        res = client.post("/api/admin/register-doctor", json={"username": "dr_test"})
        assert res.status_code == 401
        data = res.get_json()
        assert data["status"] == "error"

    def test_16_api_register_clinician_token_returns_403(self, client, isolated_auth_registry):
        """POST /api/admin/register-doctor with regular clinician token returns 403 Forbidden."""
        doctor = isolated_auth_registry["doctor"]
        doc_token = create_session_token(doctor)

        headers = {"Authorization": f"Bearer {doc_token}"}
        res = client.post(
            "/api/admin/register-doctor",
            headers=headers,
            json={
                "username": "dr_forbidden",
                "email": "forbidden@hospital.org",
                "full_name": "Dr. Forbidden",
                "password": "ValidPass@2026!",
                "role": "Medical Officer",
            },
        )
        assert res.status_code == 403
        data = res.get_json()
        assert data["error"] == "Forbidden"
        assert "Administrative privileges required" in data["message"]

    def test_17_api_register_admin_token_invalid_data_returns_400(self, client, isolated_auth_registry):
        """POST /api/admin/register-doctor with admin token but missing fields returns 400 Bad Request."""
        admin = isolated_auth_registry["admin"]
        admin_token = create_session_token(admin)

        headers = {"Authorization": f"Bearer {admin_token}"}
        res = client.post(
            "/api/admin/register-doctor",
            headers=headers,
            json={
                "username": "dr_incomplete",
                # missing email, password, full_name
            },
        )
        assert res.status_code == 400
        data = res.get_json()
        assert data["status"] == "error"

    def test_18_api_register_admin_token_success_returns_201(self, client, isolated_auth_registry):
        """POST /api/admin/register-doctor with admin token and valid data returns 201 Created."""
        admin = isolated_auth_registry["admin"]
        admin_token = create_session_token(admin)

        headers = {"Authorization": f"Bearer {admin_token}"}
        payload = {
            "username": "dr_api_success",
            "email": "api.success@hospital.org",
            "full_name": "Dr. API Success, MD",
            "password": "ValidPassword@2026!",
            "role": "Attending Cardiologist",
            "department": "Cardiology ICU",
        }
        res = client.post("/api/admin/register-doctor", headers=headers, json=payload)
        assert res.status_code == 201
        data = res.get_json()
        assert data["status"] == "success"
        assert data["doctor"]["username"] == "dr_api_success"
        assert data["doctor"]["is_admin"] is False

        # Verify doctor can now log in via /api/auth/login
        login_res = client.post(
            "/api/auth/login",
            json={"username": "dr_api_success", "password": "ValidPassword@2026!"},
        )
        assert login_res.status_code == 200
        login_data = login_res.get_json()
        assert login_data["doctor"]["full_name"] == "Dr. API Success, MD"
        assert login_data["doctor"]["is_admin"] is False


# ── 5. UI Gatekeeper and Navigation Safeguards ─────────────────────────────────

class TestUIGatekeeperAndNavigation:
    def test_19_non_admin_blocked_from_register_doctor_page(self, isolated_auth_registry):
        """When an ordinary clinician visits register_doctor page, Access Denied is shown."""
        from dashboard.pages.register_doctor import render_register_doctor_page

        doctor = isolated_auth_registry["doctor"]

        mock_session_state = {
            "authenticated_doctor": doctor,
            "doctor_session_token": create_session_token(doctor),
        }

        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.error") as mock_error, \
             patch("streamlit.form") as mock_form:

            render_register_doctor_page()

            # Error should have been called with Access Denied message
            mock_error.assert_called_once()
            err_msg = mock_error.call_args[0][0]
            assert "Access Denied" in err_msg or "Administrative Privileges Required" in err_msg
            # Form should NOT be rendered for non-admin
            mock_form.assert_not_called()

    def test_20_admin_permitted_on_register_doctor_page(self, isolated_auth_registry):
        """When an administrator visits register_doctor page, the registration form is rendered."""
        from dashboard.pages.register_doctor import render_register_doctor_page

        admin = isolated_auth_registry["admin"]

        mock_session_state = {
            "authenticated_doctor": admin,
            "doctor_session_token": create_session_token(admin),
        }

        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.error") as mock_error, \
             patch("streamlit.form") as mock_form:

            # Mock form context manager
            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_register_doctor_page()

            # Error should NOT be called
            mock_error.assert_not_called()
            # Form should be rendered
            mock_form.assert_called_with("admin_register_doctor_form", clear_on_submit=False)
