"""
tests/test_first_time_setup.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Test Suite for First-Time Administrator Setup in XMT-Sepsis-System.

Covers:
1. First-time setup is shown when the registry is valid and contains no administrator.
2. The initial administrator can be created successfully.
3. The new administrator's password is stored only as a secure hash (PBKDF2 600k iterations).
4. Missing fields and weak passwords are rejected.
5. Password-confirmation mismatches are rejected.
6. Duplicate usernames and emails are rejected.
7. A non-administrator cannot assign themselves administrator privileges through client input.
8. First-time setup is rejected after an administrator exists.
9. Direct API calls cannot bypass the one-time setup restriction.
10. Concurrent setup requests cannot create multiple initial administrators.
11. Missing or corrupted registry state fails safely.
12. Normal Doctor Sign In is displayed after initial setup.
13. An administrator can access the existing Register Doctor workflow.
14. An ordinary doctor cannot access administrator functions.
15. End-to-end integration: setup initial admin -> login -> access doctor registration -> register doctor.
"""

from __future__ import annotations

import json
import os
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from src.auth.auth_manager import (
    ALLOWED_ROLES,
    authenticate_doctor,
    create_initial_administrator,
    create_session_token,
    get_registry_status,
    has_administrator,
    is_administrator,
    is_first_time_setup_allowed,
    load_doctor_registry,
    provision_doctor,
    register_doctor,
    save_doctor_registry,
    verify_session_token,
)
import src.auth.auth_manager as auth_mod
from backend.app import app


@pytest.fixture
def isolated_clean_env(tmp_path, monkeypatch):
    """Provide an isolated auth directory with a valid clinician-only registry (no admin)."""
    temp_dir = str(tmp_path / "auth")
    os.makedirs(temp_dir, exist_ok=True)
    registry_file = os.path.join(temp_dir, "doctors.json")
    secret_key_file = os.path.join(temp_dir, ".secret_key")
    revoked_file = os.path.join(temp_dir, "revoked_tokens.json")

    monkeypatch.setattr(auth_mod, "AUTH_DIR", temp_dir)
    monkeypatch.setattr(auth_mod, "REGISTRY_FILE", registry_file)
    monkeypatch.setattr(auth_mod, "SECRET_KEY_FILE", secret_key_file)
    monkeypatch.setattr(auth_mod, "REVOKED_TOKENS_FILE", revoked_file)

    # Initialize a valid registry containing only a regular clinician (no admin)
    initial_registry = {
        "version": 2,
        "description": "XMT-Sepsis-System Authorized Clinician Credentials Registry",
        "last_updated": "2026-10-18T00:00:00Z",
        "doctors": {
            "dr.sarah.chen": {
                "doctor_id": "DOC-MICU-01",
                "username": "dr.sarah.chen",
                "email": "dr.sarah.chen@hospital.org",
                "full_name": "Dr. Sarah Chen, MD",
                "role": "Attending Intensivist / Clinical Researcher",
                "department": "Medical Intensive Care Unit (MICU)",
                "initials": "SC",
                "password_hash": auth_mod.hash_password("ClinicalSepsis@2026!", auth_mod.PBKDF2_ITERATIONS),
                "token_version": 1,
                "is_active": True,
                "is_admin": False,
                "created_at": "2026-10-18T00:00:00Z",
                "last_login": None,
                "login_count": 0,
            }
        },
    }
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(initial_registry, f, indent=2)

    return {
        "dir": temp_dir,
        "registry_file": registry_file,
    }


class TestFirstTimeAdministratorSetup:
    """Verifies core first-time setup functionality, security invariants, and edge cases."""

    def test_01_setup_allowed_when_valid_registry_has_no_admin(self, isolated_clean_env):
        """Requirement 1: First-time setup is allowed when registry is valid and has no admin."""
        status, err = get_registry_status()
        assert status == "VALID"
        assert err is None

        assert has_administrator() is False
        allowed, reason = is_first_time_setup_allowed()
        assert allowed is True
        assert "available" in reason.lower()

    def test_02_create_initial_administrator_success(self, isolated_clean_env):
        """Requirement 2: Initial administrator is created successfully with required attributes."""
        admin = create_initial_administrator(
            full_name="Lead System Administrator",
            email="sysadmin@hospital.org",
            username="admin.lead",
            password="AdminPassword@2026!",
            confirm_password="AdminPassword@2026!",
        )

        assert admin["username"] == "admin.lead"
        assert admin["email"] == "sysadmin@hospital.org"
        assert admin["full_name"] == "Lead System Administrator"
        assert admin["role"] == "System Administrator"
        assert admin["is_admin"] is True
        assert admin["doctor_id"].startswith("DOC-ADM-")

        # Now an administrator exists
        assert has_administrator() is True
        allowed, _ = is_first_time_setup_allowed()
        assert allowed is False

    def test_03_administrator_password_stored_only_as_pbkdf2_hash(self, isolated_clean_env):
        """Requirement 3: Plaintext password is never stored; only 600k-iteration PBKDF2 hash exists."""
        admin = create_initial_administrator(
            full_name="Dr. Alex Smith, MD",
            email="alex.smith@hospital.org",
            username="admin.alex",
            password="Crypt0Password@2026!",
            confirm_password="Crypt0Password@2026!",
        )
        assert "password_hash" not in admin  # Sanitized return

        reg = load_doctor_registry()
        persisted = reg["doctors"]["admin.alex"]
        assert "password" not in persisted
        assert "password_hash" in persisted
        hash_val = persisted["password_hash"]
        assert hash_val.startswith("pbkdf2:sha256:600000$")

        # Verification passes with correct password, fails with wrong password
        assert auth_mod.verify_password("Crypt0Password@2026!", hash_val) is True
        assert auth_mod.verify_password("WrongPassword@2026!", hash_val) is False

    def test_04_missing_fields_and_weak_passwords_rejected(self, isolated_clean_env):
        """Requirement 4: Missing required fields and weak passwords are appropriately rejected."""
        # Missing full name
        with pytest.raises(ValueError, match="full name"):
            create_initial_administrator(
                full_name="",
                email="sysadmin@hospital.org",
                username="admin.valid",
                password="AdminPassword@2026!",
            )

        # Missing email
        with pytest.raises(ValueError, match="institutional email"):
            create_initial_administrator(
                full_name="Valid Name",
                email="not-an-email",
                username="admin.valid",
                password="AdminPassword@2026!",
            )

        # Invalid username
        with pytest.raises(ValueError, match="username"):
            create_initial_administrator(
                full_name="Valid Name",
                email="sysadmin@hospital.org",
                username="a",  # too short
                password="AdminPassword@2026!",
            )

        # Weak password (fails complexity)
        with pytest.raises(ValueError, match="at least 10 characters"):
            create_initial_administrator(
                full_name="Valid Name",
                email="sysadmin@hospital.org",
                username="admin.valid",
                password="short",
            )

    def test_05_password_confirmation_mismatch_rejected(self, isolated_clean_env):
        """Requirement 5: Password confirmation mismatch raises ValueError."""
        with pytest.raises(ValueError, match="do not match"):
            create_initial_administrator(
                full_name="System Admin",
                email="sysadmin@hospital.org",
                username="admin.mismatch",
                password="ValidPassword@2026!",
                confirm_password="DifferentPassword@2026!",
            )

    def test_06_duplicate_username_and_email_rejected(self, isolated_clean_env):
        """Requirement 6: Duplicate usernames and institutional emails are rejected."""
        # dr.sarah.chen already exists in isolated_clean_env
        with pytest.raises(ValueError, match="already exists"):
            create_initial_administrator(
                full_name="Impostor Admin",
                email="unique@hospital.org",
                username="dr.sarah.chen",
                password="AdminPassword@2026!",
            )

        with pytest.raises(ValueError, match="already exists"):
            create_initial_administrator(
                full_name="Impostor Admin",
                email="dr.sarah.chen@hospital.org",
                username="admin.unique",
                password="AdminPassword@2026!",
            )

    def test_07_privilege_flags_cannot_be_manipulated_by_client_input(self, isolated_clean_env):
        """Requirement 7: Admin privileges are strictly assigned by trusted server-side code."""
        admin = create_initial_administrator(
            full_name="Fixed Admin",
            email="fixed@hospital.org",
            username="admin.fixed",
            password="AdminPassword@2026!",
        )
        assert admin["is_admin"] is True
        assert admin["role"] == "System Administrator"

        # Check subsequent ordinary clinician registration cannot receive is_admin=True
        cln = register_doctor(
            admin_actor=admin,
            doctor_data={
                "username": "dr.sneaky",
                "email": "sneaky@hospital.org",
                "full_name": "Dr. Sneaky, MD",
                "password": "SneakyPass@2026!",
                "role": "Attending Intensivist",
                "is_admin": True,  # Untrusted client attempt to self-elevate
            },
        )
        assert cln["is_admin"] is False
        reg = load_doctor_registry()
        assert reg["doctors"]["dr.sneaky"]["is_admin"] is False

    def test_08_setup_rejected_after_administrator_exists(self, isolated_clean_env):
        """Requirement 8: Once initial admin exists, subsequent setup attempts are rejected."""
        create_initial_administrator(
            full_name="First Admin",
            email="first@hospital.org",
            username="admin.first",
            password="AdminPassword@2026!",
        )

        # Attempting second setup must raise PermissionError
        with pytest.raises(PermissionError, match="First-time setup is closed"):
            create_initial_administrator(
                full_name="Second Admin",
                email="second@hospital.org",
                username="admin.second",
                password="AdminPassword@2026!",
            )

    def test_09_direct_api_calls_cannot_bypass_setup_restriction(self, isolated_clean_env):
        """Requirement 9: POST /api/auth/setup-admin rejects setup once admin exists."""
        client = app.test_client()

        payload = {
            "full_name": "Initial Admin",
            "email": "api.admin@hospital.org",
            "username": "admin.api",
            "password": "ValidAdminPass@2026!",
            "confirm_password": "ValidAdminPass@2026!",
        }

        # First call: succeeds with 201 Created
        res1 = client.post("/api/auth/setup-admin", json=payload)
        assert res1.status_code == 201
        data1 = res1.get_json()
        assert data1["status"] == "success"
        assert data1["administrator"]["username"] == "admin.api"

        # Second call (replay or attacker): rejected with 403 Forbidden
        res2 = client.post(
            "/api/auth/setup-admin",
            json={
                "full_name": "Second Rogue Admin",
                "email": "rogue@hospital.org",
                "username": "admin.rogue",
                "password": "ValidAdminPass@2026!",
                "confirm_password": "ValidAdminPass@2026!",
            },
        )
        assert res2.status_code == 403
        data2 = res2.get_json()
        assert data2["status"] == "error"
        assert "already exists" in data2["message"]

    def test_10_concurrent_setup_requests_race_condition_protection(self, isolated_clean_env):
        """Requirement 10: Concurrent setup requests cannot create multiple administrators."""
        results = []
        errors = []

        def worker(idx: int):
            try:
                admin = create_initial_administrator(
                    full_name=f"Admin Thread {idx}",
                    email=f"admin_{idx}@hospital.org",
                    username=f"admin_{idx}",
                    password=f"StrongPass#{idx}@2026!",
                    confirm_password=f"StrongPass#{idx}@2026!",
                )
                results.append(admin)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly ONE thread must succeed; all others must fail safely
        assert len(results) == 1, f"Expected 1 winner, got {len(results)}"
        assert len(errors) == 4
        for err in errors:
            assert isinstance(err, (PermissionError, ValueError))

        # Authoritative registry has exactly 1 administrator
        reg = load_doctor_registry()
        admin_count = sum(1 for d in reg["doctors"].values() if d.get("is_admin") is True)
        assert admin_count == 1

    def test_11_missing_or_corrupted_registry_state_fails_safely(self, tmp_path, monkeypatch):
        """Requirement 11: Missing or corrupted registry files fail safely and refuse setup."""
        temp_dir = str(tmp_path / "auth_corrupt")
        os.makedirs(temp_dir, exist_ok=True)
        missing_file = os.path.join(temp_dir, "nonexistent.json")

        monkeypatch.setattr(auth_mod, "AUTH_DIR", temp_dir)
        monkeypatch.setattr(auth_mod, "REGISTRY_FILE", missing_file)

        # 1. Missing file: status is MISSING, setup is disallowed
        status, _ = get_registry_status()
        assert status == "MISSING"
        allowed, reason = is_first_time_setup_allowed()
        assert allowed is False
        assert "missing" in reason.lower()
        assert has_administrator() is True  # Fails safely

        with pytest.raises(RuntimeError, match="missing on disk"):
            create_initial_administrator(
                full_name="Admin",
                email="admin@hospital.org",
                username="admin",
                password="ValidPass@2026!",
            )

        # 2. Corrupted JSON file: status is CORRUPTED, setup is disallowed
        corrupt_file = os.path.join(temp_dir, "corrupted.json")
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json content :::")

        monkeypatch.setattr(auth_mod, "REGISTRY_FILE", corrupt_file)

        status, _ = get_registry_status()
        assert status == "CORRUPTED"
        allowed, reason = is_first_time_setup_allowed()
        assert allowed is False
        assert "corrupted" in reason.lower()
        assert has_administrator() is True  # Fails safely

        with pytest.raises(RuntimeError, match="corrupted or unreadable"):
            create_initial_administrator(
                full_name="Admin",
                email="admin@hospital.org",
                username="admin",
                password="ValidPass@2026!",
            )

    def test_12_normal_doctor_sign_in_after_setup_complete(self, isolated_clean_env):
        """Requirement 12: Normal Doctor Sign In is used after setup; both admin and doctor can authenticate."""
        # Setup initial admin
        create_initial_administrator(
            full_name="Lead Administrator",
            email="sysadmin@hospital.org",
            username="admin.lead",
            password="AdminPassword@2026!",
        )

        # Authenticate newly created administrator
        ok_admin, admin_doc, _ = authenticate_doctor("admin.lead", "AdminPassword@2026!")
        assert ok_admin is True
        assert admin_doc["is_admin"] is True

        # Existing doctor (dr.sarah.chen) can also authenticate
        ok_doc, doc_profile, _ = authenticate_doctor("dr.sarah.chen", "ClinicalSepsis@2026!")
        assert ok_doc is True
        assert doc_profile["is_admin"] is False

        # Unregistered user cannot authenticate
        ok_bad, _, err_msg = authenticate_doctor("unregistered.user", "AnyPassword@2026!")
        assert ok_bad is False
        assert "Invalid credentials" in err_msg

    def test_13_administrator_can_access_register_doctor_workflow(self, isolated_clean_env):
        """Requirement 13: Authorized administrator can access the Register Doctor workflow."""
        admin = create_initial_administrator(
            full_name="Chief Admin",
            email="chief.admin@hospital.org",
            username="admin.chief",
            password="ChiefAdminPass@2026!",
        )

        # Register doctor as administrator
        new_doc = register_doctor(
            admin_actor=admin,
            doctor_data={
                "username": "dr.onboarded",
                "email": "onboarded@hospital.org",
                "full_name": "Dr. Onboarded Physician, MD",
                "password": "OnboardPass@2026!",
                "role": "Attending Cardiologist",
            },
        )
        assert new_doc["username"] == "dr.onboarded"
        assert new_doc["is_admin"] is False

        # Doctor can now log in
        ok, profile, _ = authenticate_doctor("dr.onboarded", "OnboardPass@2026!")
        assert ok is True
        assert profile["is_admin"] is False

    def test_14_ordinary_doctor_cannot_access_administrator_functions(self, isolated_clean_env):
        """Requirement 14: Ordinary clinicians cannot access doctor registration or admin endpoints."""
        admin = create_initial_administrator(
            full_name="Security Admin",
            email="sec.admin@hospital.org",
            username="admin.sec",
            password="SecAdminPass@2026!",
        )

        # Clinician tries to register another doctor
        ok, clinician, _ = authenticate_doctor("dr.sarah.chen", "ClinicalSepsis@2026!")
        assert ok is True
        assert is_administrator(clinician) is False

        with pytest.raises(PermissionError, match="Access Denied"):
            register_doctor(
                admin_actor=clinician,
                doctor_data={
                    "username": "dr.unauthorized",
                    "email": "unauth@hospital.org",
                    "full_name": "Dr. Unauthorized, MD",
                    "password": "ValidPass@2026!",
                    "role": "Medical Officer",
                },
            )

        # Clinician token against backend admin endpoint returns 403 Forbidden
        clinician_token = create_session_token(clinician)
        client = app.test_client()
        res = client.post(
            "/api/admin/register-doctor",
            headers={"Authorization": f"Bearer {clinician_token}"},
            json={
                "username": "dr.unauthorized",
                "email": "unauth@hospital.org",
                "full_name": "Dr. Unauthorized",
                "password": "ValidPass@2026!",
                "role": "Medical Officer",
            },
        )
        assert res.status_code == 403
        data = res.get_json()
        assert "Administrative privileges required" in data["message"]

    def test_15_ui_flow_renders_setup_form_then_sign_in_form(self, isolated_clean_env):
        """Requirement 15: UI renders First-Time Setup when no admin exists, and Doctor Sign In once created."""
        from dashboard.pages.login import render_login_page

        mock_session_state = {}

        # 1. Before setup: no admin exists -> setup form rendered
        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.form") as mock_form, \
             patch("streamlit.error") as mock_error:

            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_login_page()

            # First-time setup form is called
            mock_form.assert_called_with("initial_admin_setup_form", clear_on_submit=False)
            mock_error.assert_not_called()

        # 2. Complete setup
        create_initial_administrator(
            full_name="Main Admin",
            email="main.admin@hospital.org",
            username="admin.main",
            password="MainAdminPass@2026!",
        )

        # 3. After setup: admin exists -> role selection page rendered; setup form is NOT called
        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.form") as mock_form, \
             patch("streamlit.error") as mock_error:

            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_login_page()

            # First-time setup form is NOT called
            assert not any(call[0][0] == "initial_admin_setup_form" for call in mock_form.call_args_list)
            mock_error.assert_not_called()

        # 4. When clinician portal is selected, doctor_login_form is rendered
        mock_session_state["login_portal_mode"] = "DOCTOR_LOGIN"
        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.form") as mock_form, \
             patch("streamlit.error") as mock_error:

            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_login_page()

            mock_form.assert_called_with("doctor_login_form", clear_on_submit=False)
            mock_error.assert_not_called()

        # 5. When administrator portal is selected, admin_login_form is rendered
        mock_session_state["login_portal_mode"] = "ADMIN_LOGIN"
        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.form") as mock_form, \
             patch("streamlit.error") as mock_error:

            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_login_page()

            mock_form.assert_called_with("admin_login_form", clear_on_submit=False)
            mock_error.assert_not_called()
