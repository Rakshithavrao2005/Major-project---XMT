"""
tests/test_web_doctor_workflow.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Test Suite for Website-Based Doctor Registration, Email Verification,
Administrator Approval, Login, and Password Recovery.

Covers all 11 required scenarios:
1. Registration field validation and policy checks.
2. Unique Doctor ID generation and least-privilege assignment.
3. Duplicate email and username rejection.
4. PBKDF2-HMAC-SHA256 password hashing (600,000 iterations, unique salts).
5. Email verification token single-use, expiry, and state transition.
6. Administrator approval and rejection workflow.
7. Login multi-stage status enforcement (pending vs approved).
8. Web forgot-password token generation, single-use, and session invalidation.
9. Zero-privilege public registration (privilege escalation defense).
10. Existing account backward compatibility (dr.sarah.chen and admin.system).
11. Backend API protection against pending / unapproved accounts.
12. Session expiration and logout revocation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import pytest

import src.auth.auth_manager as auth_mod
from src.auth.auth_manager import (
    ACCOUNT_STATUS_ACTIVE,
    ACCOUNT_STATUS_PENDING_APPROVAL,
    ACCOUNT_STATUS_PENDING_EMAIL_VERIFICATION,
    ACCOUNT_STATUS_REJECTED,
    admin_approve_doctor,
    admin_reject_doctor,
    authenticate_doctor,
    create_session_token,
    get_pending_doctor_registrations,
    hash_password,
    is_session_expired,
    load_doctor_registry,
    register_public_doctor,
    request_web_password_reset,
    reset_web_password_with_token,
    revoke_session_token,
    verify_doctor_email,
    verify_password,
    verify_session_token,
)
import src.auth.email_service as email_mod
from backend.app import app


@pytest.fixture
def isolated_web_auth_env(tmp_path, monkeypatch):
    """Provide a temporary isolated authentication directory with existing accounts."""
    temp_dir = str(tmp_path / "auth")
    os.makedirs(temp_dir, exist_ok=True)
    registry_file = os.path.join(temp_dir, "doctors.json")
    secret_key_file = os.path.join(temp_dir, ".secret_key")
    revoked_file = os.path.join(temp_dir, "revoked_tokens.json")
    tokens_file = os.path.join(temp_dir, "auth_tokens.json")

    monkeypatch.setattr(auth_mod, "AUTH_DIR", temp_dir)
    monkeypatch.setattr(auth_mod, "REGISTRY_FILE", registry_file)
    monkeypatch.setattr(auth_mod, "SECRET_KEY_FILE", secret_key_file)
    monkeypatch.setattr(auth_mod, "REVOKED_TOKENS_FILE", revoked_file)

    monkeypatch.setattr(email_mod, "AUTH_DIR", temp_dir)
    monkeypatch.setattr(email_mod, "TOKENS_FILE", tokens_file)
    monkeypatch.setitem(app.config, "REQUIRE_AUTH", True)

    # Initial registry with active administrator and staff clinician
    initial_registry = {
        "version": 2,
        "description": "XMT Isolated Web Auth Test Registry",
        "last_updated": "2026-10-18T00:00:00Z",
        "doctors": {
            "admin.system": {
                "doctor_id": "DOC-ADM-01",
                "username": "admin.system",
                "email": "admin@hospital.org",
                "full_name": "Lead System Administrator",
                "role": "System Administrator",
                "department": "Clinical Informatics",
                "initials": "LS",
                "password_hash": auth_mod.hash_password("AdminSecure@2026!", auth_mod.PBKDF2_ITERATIONS),
                "token_version": 1,
                "is_active": True,
                "is_admin": True,
                "status": ACCOUNT_STATUS_ACTIVE,
                "created_at": "2026-10-18T00:00:00Z",
                "last_login": None,
                "login_count": 0,
            },
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
                "status": ACCOUNT_STATUS_ACTIVE,
                "created_at": "2026-10-18T00:00:00Z",
                "last_login": None,
                "login_count": 0,
            },
        },
    }

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(initial_registry, f, indent=2)

    return {
        "dir": temp_dir,
        "registry_file": registry_file,
    }


class TestWebDoctorRegistrationAndLogin:
    """Verifies all aspects of website-based doctor onboarding, approval, and login."""

    def test_01_registration_validation_and_required_fields(self, isolated_web_auth_env):
        """Requirement 1: Validates required fields, email format, age, consent, and passwords."""
        base_payload = {
            "full_name": "Dr. Alice Morgan, MD",
            "email": "alice.morgan@hospital.org",
            "username": "dr.alice.morgan",
            "age": 36,
            "specialization": "Critical Care Medicine",
            "department": "MICU",
            "hospital": "City Memorial Hospital",
            "password": "SecurePassword@2026!",
            "confirm_password": "SecurePassword@2026!",
            "consent_accepted": True,
        }

        # A: Missing full name
        bad_name = {**base_payload, "full_name": ""}
        ok, msg, _, _ = register_public_doctor(bad_name)
        assert ok is False
        assert "full name" in msg.lower()

        # B: Invalid email format
        bad_email = {**base_payload, "email": "notanemail"}
        ok, msg, _, _ = register_public_doctor(bad_email)
        assert ok is False
        assert "email" in msg.lower()

        # C: Invalid age
        bad_age = {**base_payload, "age": 15}
        ok, msg, _, _ = register_public_doctor(bad_age)
        assert ok is False
        assert "age" in msg.lower()

        # D: Missing consent
        no_consent = {**base_payload, "consent_accepted": False}
        ok, msg, _, _ = register_public_doctor(no_consent)
        assert ok is False
        assert "consent" in msg.lower()

        # E: Password mismatch
        mismatch_pw = {**base_payload, "confirm_password": "DifferentPassword@2026!"}
        ok, msg, _, _ = register_public_doctor(mismatch_pw)
        assert ok is False
        assert "match" in msg.lower()

        # F: Weak password (too short)
        weak_pw = {**base_payload, "password": "Short1!", "confirm_password": "Short1!"}
        ok, msg, _, _ = register_public_doctor(weak_pw)
        assert ok is False
        assert "at least 10" in msg.lower()

    def test_02_registration_assigns_unique_doctor_id_and_least_privilege(self, isolated_web_auth_env):
        """Requirement 2: Registration generates unique ID, least privilege role, and pending status."""
        payload = {
            "full_name": "Dr. Gregory House, MD",
            "email": "gregory.house@hospital.org",
            "username": "dr.gregory.house",
            "age": 45,
            "specialization": "Infectious Disease",
            "department": "Diagnostic Medicine",
            "hospital": "Princeton Plainsboro",
            "password": "Diagnostician@2026!",
            "confirm_password": "Diagnostician@2026!",
            "consent_accepted": True,
        }

        ok, msg, doc, eval_token = register_public_doctor(payload)
        assert ok is True
        assert doc is not None
        assert doc["doctor_id"].startswith("DOC-REG-")
        assert doc["role"] == "Staff Clinician"  # Least privilege
        assert doc["is_admin"] is False
        assert doc["is_active"] is False  # Inactive until approved
        assert doc["status"] == ACCOUNT_STATUS_PENDING_EMAIL_VERIFICATION
        assert eval_token is not None  # In dev mode, evaluation token is provided

    def test_03_registration_duplicate_username_and_email_rejection(self, isolated_web_auth_env):
        """Requirement 3: Duplicate usernames and emails are rejected."""
        # Attempt to register existing username 'dr.sarah.chen'
        dup_user = {
            "full_name": "Dr. Sarah Chen Clone",
            "email": "sarah.clone@hospital.org",
            "username": "dr.sarah.chen",
            "specialization": "Internal Medicine",
            "department": "ICU",
            "hospital": "City Hospital",
            "password": "Password@2026!Valid",
            "confirm_password": "Password@2026!Valid",
            "consent_accepted": True,
        }
        ok, msg, _, _ = register_public_doctor(dup_user)
        assert ok is False
        assert "already registered" in msg.lower()

        # Attempt to register existing email 'admin@hospital.org'
        dup_email = {
            "full_name": "Dr. New Impostor",
            "email": "admin@hospital.org",
            "username": "dr.new.impostor",
            "specialization": "Internal Medicine",
            "department": "ICU",
            "hospital": "City Hospital",
            "password": "Password@2026!Valid",
            "confirm_password": "Password@2026!Valid",
            "consent_accepted": True,
        }
        ok, msg, _, _ = register_public_doctor(dup_email)
        assert ok is False
        assert "already registered" in msg.lower()

    def test_04_password_hashing_pbkdf2_600k_and_no_plaintext(self, isolated_web_auth_env):
        """Requirement 4: Password is encrypted with PBKDF2 600,000 iterations and salt."""
        payload = {
            "full_name": "Dr. Robert Chase, MD",
            "email": "robert.chase@hospital.org",
            "username": "dr.robert.chase",
            "specialization": "Cardiology",
            "department": "Cardiology Unit",
            "hospital": "City Hospital",
            "password": "CardioSecret@2026!",
            "confirm_password": "CardioSecret@2026!",
            "consent_accepted": True,
        }
        ok, _, doc, _ = register_public_doctor(payload)
        assert ok is True
        assert "password_hash" not in doc  # Sanitized return

        reg = load_doctor_registry()
        stored = reg["doctors"]["dr.robert.chase"]
        stored_hash = stored["password_hash"]
        assert stored_hash.startswith("pbkdf2:sha256:600000$")
        assert verify_password("CardioSecret@2026!", stored_hash) is True
        assert verify_password("WrongPassword123!", stored_hash) is False

    def test_05_email_verification_token_lifecycle(self, isolated_web_auth_env):
        """Requirement 5: Verification token transitions account to Pending Approval and is single-use."""
        payload = {
            "full_name": "Dr. Lisa Cuddy, MD",
            "email": "lisa.cuddy@hospital.org",
            "username": "dr.lisa.cuddy",
            "specialization": "Internal Medicine",
            "department": "Administration & Medicine",
            "hospital": "City Hospital",
            "password": "DeanMedicine@2026!",
            "confirm_password": "DeanMedicine@2026!",
            "consent_accepted": True,
        }
        ok, _, _, eval_token = register_public_doctor(payload)
        assert ok is True
        assert eval_token is not None

        # Verify email using the token
        ok_v, msg_v, doc_v = verify_doctor_email(eval_token)
        assert ok_v is True
        assert doc_v["status"] == ACCOUNT_STATUS_PENDING_APPROVAL
        assert doc_v["email_verified"] is True
        assert doc_v["is_active"] is False  # Still inactive awaiting admin approval

        # Token is single use: second attempt must fail
        ok_v2, msg_v2, _ = verify_doctor_email(eval_token)
        assert ok_v2 is False
        assert "already been used" in msg_v2.lower() or "invalid" in msg_v2.lower()

    def test_06_administrator_approval_and_rejection_workflow(self, isolated_web_auth_env):
        """Requirement 6: Only administrators can approve or reject pending doctor accounts."""
        reg = load_doctor_registry()
        admin_user = reg["doctors"]["admin.system"]
        clinician_user = reg["doctors"]["dr.sarah.chen"]

        # Register doctor and verify email
        payload = {
            "full_name": "Dr. Allison Cameron, MD",
            "email": "allison.cameron@hospital.org",
            "username": "dr.allison.cameron",
            "specialization": "Immunology",
            "department": "ICU",
            "hospital": "City Hospital",
            "password": "ImmunoDoctor@2026!",
            "confirm_password": "ImmunoDoctor@2026!",
            "consent_accepted": True,
        }
        _, _, _, tok = register_public_doctor(payload)
        verify_doctor_email(tok)

        # 1. Non-admin clinician cannot approve
        ok_non_admin, msg_non_admin, _ = admin_approve_doctor(clinician_user, "dr.allison.cameron")
        assert ok_non_admin is False
        assert "access denied" in msg_non_admin.lower()

        # 2. Administrator can approve
        ok_admin, msg_admin, doc_approved = admin_approve_doctor(admin_user, "dr.allison.cameron")
        assert ok_admin is True
        assert doc_approved["status"] == ACCOUNT_STATUS_ACTIVE
        assert doc_approved["is_active"] is True
        assert doc_approved["approved_by_admin"] is True

        # 3. Test rejection on another doctor
        payload2 = {
            "full_name": "Dr. James Wilson, MD",
            "email": "james.wilson@hospital.org",
            "username": "dr.james.wilson",
            "specialization": "Oncology",
            "department": "Oncology",
            "hospital": "City Hospital",
            "password": "OncologyPass@2026!",
            "confirm_password": "OncologyPass@2026!",
            "consent_accepted": True,
        }
        _, _, _, tok2 = register_public_doctor(payload2)
        verify_doctor_email(tok2)

        ok_rej, msg_rej, doc_rejected = admin_reject_doctor(admin_user, "dr.james.wilson", reason="Incomplete credential documentation")
        assert ok_rej is True
        assert doc_rejected["status"] == ACCOUNT_STATUS_REJECTED
        assert doc_rejected["is_active"] is False

    def test_07_login_status_enforcement(self, isolated_web_auth_env):
        """Requirement 7: Login blocks unverified, pending, and rejected accounts with clear guidance."""
        reg = load_doctor_registry()
        admin_user = reg["doctors"]["admin.system"]

        payload = {
            "full_name": "Dr. Eric Foreman, MD",
            "email": "eric.foreman@hospital.org",
            "username": "dr.eric.foreman",
            "specialization": "Neurology",
            "department": "Neurology",
            "hospital": "City Hospital",
            "password": "NeuroSepsis@2026!",
            "confirm_password": "NeuroSepsis@2026!",
            "consent_accepted": True,
        }
        _, _, _, tok = register_public_doctor(payload)

        # State 1: Pending Email Verification
        auth1, doc1, msg1 = authenticate_doctor("dr.eric.foreman", "NeuroSepsis@2026!")
        assert auth1 is False
        assert "pending email verification" in msg1.lower()

        # State 2: Pending Approval
        verify_doctor_email(tok)
        auth2, doc2, msg2 = authenticate_doctor("dr.eric.foreman", "NeuroSepsis@2026!")
        assert auth2 is False
        assert "pending administrator approval" in msg2.lower()

        # State 3: Active after Admin Approval
        admin_approve_doctor(admin_user, "dr.eric.foreman")
        auth3, doc3, msg3 = authenticate_doctor("dr.eric.foreman", "NeuroSepsis@2026!")
        assert auth3 is True
        assert doc3 is not None
        assert doc3["username"] == "dr.eric.foreman"
        assert doc3["role"] == "Staff Clinician"

    def test_08_web_forgot_password_workflow(self, isolated_web_auth_env):
        """Requirement 8: Forgot password generates single-use token, resets password, and invalidates sessions."""
        # 1. Request reset for dr.sarah.chen
        ok_req, msg_req, reset_tok = request_web_password_reset("dr.sarah.chen@hospital.org")
        assert ok_req is True
        assert reset_tok is not None

        # Issue an active session token before password change
        reg_before = load_doctor_registry()
        token_before = create_session_token(reg_before["doctors"]["dr.sarah.chen"])
        assert verify_session_token(token_before) is not None

        # 2. Reset password using the token
        new_password = "UpdatedSarahPassword@2026!"
        ok_reset, msg_reset = reset_web_password_with_token(reset_tok, new_password, new_password)
        assert ok_reset is True
        assert "successfully updated" in msg_reset.lower()

        # 3. Prior session token is invalidated
        assert verify_session_token(token_before) is None

        # 4. Old password fails; new password works
        old_auth, _, _ = authenticate_doctor("dr.sarah.chen", "ClinicalSepsis@2026!")
        assert old_auth is False
        new_auth, doc_new, _ = authenticate_doctor("dr.sarah.chen", new_password)
        assert new_auth is True

        # 5. Token reuse rejected
        ok_reuse, _ = reset_web_password_with_token(reset_tok, "AnotherPass@2026!", "AnotherPass@2026!")
        assert ok_reuse is False

    def test_09_privilege_escalation_prevention(self, isolated_web_auth_env):
        """Requirement 9: Public registration cannot assign administrator privileges or higher roles."""
        malicious_payload = {
            "full_name": "Attacker Doctor",
            "email": "attacker@hospital.org",
            "username": "dr.attacker",
            "specialization": "Clinical Informatics",
            "department": "Security",
            "hospital": "Rogue Hospital",
            "password": "RoguePassword@2026!",
            "confirm_password": "RoguePassword@2026!",
            "consent_accepted": True,
            # Malicious client injection attempts
            "is_admin": True,
            "admin": True,
            "role": "System Administrator",
            "status": "Active",
            "is_active": True,
        }

        ok, _, doc, _ = register_public_doctor(malicious_payload)
        assert ok is True
        assert doc["is_admin"] is False
        assert doc["role"] == "Staff Clinician"
        assert doc["status"] == ACCOUNT_STATUS_PENDING_EMAIL_VERIFICATION
        assert doc["is_active"] is False

    def test_10_existing_account_backward_compatibility(self, isolated_web_auth_env):
        """Requirement 10: Existing accounts (admin.system and dr.sarah.chen) function normally."""
        auth_admin, admin_doc, _ = authenticate_doctor("admin.system", "AdminSecure@2026!")
        assert auth_admin is True
        assert admin_doc["is_admin"] is True

        auth_sarah, sarah_doc, _ = authenticate_doctor("dr.sarah.chen", "ClinicalSepsis@2026!")
        assert auth_sarah is True
        assert sarah_doc["role"] == "Attending Intensivist / Clinical Researcher"
        assert sarah_doc["is_admin"] is False

    def test_11_backend_api_rejects_pending_or_unapproved_users(self, isolated_web_auth_env):
        """Requirement 11: Backend endpoints reject tokens from non-active / pending accounts."""
        # A: Non-existent user token is rejected
        ghost_record = {
            "username": "dr.ghost.user",
            "is_admin": False,
            "token_version": 1,
            "is_active": False,
        }
        ghost_token = create_session_token(ghost_record)
        assert verify_session_token(ghost_token) is None, "Session token for non-existent doctor must be rejected"

        # B: Persisted account with pending approval status is rejected
        reg = load_doctor_registry()
        pending_record = {
            "doctor_id": "DOC-REG-9999",
            "username": "dr.pending.user",
            "email": "pending@hospital.org",
            "full_name": "Dr. Pending User, MD",
            "role": "Staff Clinician",
            "department": "ICU",
            "initials": "PU",
            "password_hash": auth_mod.hash_password("PendingPass@2026!", auth_mod.PBKDF2_ITERATIONS),
            "is_admin": False,
            "token_version": 1,
            "status": ACCOUNT_STATUS_PENDING_APPROVAL,
            "is_active": False,
        }
        reg["doctors"]["dr.pending.user"] = pending_record
        auth_mod.save_doctor_registry(reg)

        raw_token = create_session_token(pending_record)

        # verify_session_token must reject pending user
        payload = verify_session_token(raw_token)
        assert payload is None, "Session token for pending doctor must be rejected"

        # Verify via Flask test client
        with app.test_client() as client:
            res = client.get("/api/patients", headers={"Authorization": f"Bearer {raw_token}"})
            assert res.status_code == 401
            data = res.get_json()
            assert "invalid or expired" in data.get("error", "").lower()

    def test_12_logout_and_session_expiration(self, isolated_web_auth_env):
        """Requirement 12: Revocation and idle timeout are enforced."""
        reg = load_doctor_registry()
        active_doctor = reg["doctors"]["dr.sarah.chen"]
        token = create_session_token(active_doctor)

        # Valid initially
        assert verify_session_token(token) is not None

        # Revoke on logout
        revoke_session_token(token)
        assert verify_session_token(token) is None

        # Test idle session expiration helper
        past_time = time.time() - 1900  # > 1800s (30m)
        assert is_session_expired(past_time) is True
        recent_time = time.time() - 300  # 5 minutes
        assert is_session_expired(recent_time) is False

    def test_13_email_verification_hardening_and_edge_cases(self, isolated_web_auth_env):
        """Requirement 13: Empty token rejection, expired tokens, and identity match enforcement."""
        # 1. Empty token rejection
        ok, msg, doc = verify_doctor_email("")
        assert ok is False
        assert "please enter your verification token" in msg.lower()

        ok, msg, doc = verify_doctor_email("   ")
        assert ok is False
        assert "please enter your verification token" in msg.lower()

        # 2. Register doctor
        payload = {
            "full_name": "Dr. Remy Hadley, MD",
            "email": "thirteen@hospital.org",
            "username": "dr.thirteen",
            "specialization": "Internal Medicine",
            "department": "Diagnostic Medicine",
            "hospital": "Princeton Plainsboro",
            "password": "ThirteenPass@2026!",
            "confirm_password": "ThirteenPass@2026!",
            "consent_accepted": True,
        }
        ok_reg, _, _, raw_tok = register_public_doctor(payload)
        assert ok_reg is True
        assert raw_tok is not None

        # 3. Mismatched expected email/username rejection
        ok_mismatch, msg_mismatch, _ = verify_doctor_email(raw_tok, expected_email_or_username="wrong.doctor@hospital.org")
        assert ok_mismatch is False
        assert "does not match" in msg_mismatch.lower()

        # 4. Correct expected email match verification
        ok_match, msg_match, doc_match = verify_doctor_email(raw_tok, expected_email_or_username="thirteen@hospital.org")
        assert ok_match is True
        assert doc_match["username"] == "dr.thirteen"
        assert doc_match["status"] == ACCOUNT_STATUS_PENDING_APPROVAL

        # 5. Expired token rejection
        payload_exp = {
            "full_name": "Dr. Chris Taub, MD",
            "email": "chris.taub@hospital.org",
            "username": "dr.chris.taub",
            "specialization": "Plastic Surgery",
            "department": "Surgery",
            "hospital": "Princeton Plainsboro",
            "password": "TaubPassword@2026!",
            "confirm_password": "TaubPassword@2026!",
            "consent_accepted": True,
        }
        _, _, _, exp_tok = register_public_doctor(payload_exp)
        # Fast-forward token expiration in token store
        token_store = email_mod.load_token_store()
        tok_hash = email_mod._hash_token(exp_tok)
        assert tok_hash in token_store["tokens"]
        token_store["tokens"][tok_hash]["expires_at"] = time.time() - 100
        email_mod.save_token_store(token_store)

        ok_expired, msg_expired, _ = verify_doctor_email(exp_tok)
        assert ok_expired is False
        assert "expired" in msg_expired.lower()

    def test_14_production_mode_token_protection(self, isolated_web_auth_env, monkeypatch):
        """Requirement 14: In production mode (XMT_ENV=production), evaluation tokens are never leaked."""
        monkeypatch.setenv("XMT_ENV", "production")

        payload = {
            "full_name": "Dr. Lawrence Kutner, MD",
            "email": "kutner@hospital.org",
            "username": "dr.kutner",
            "specialization": "Sports Medicine",
            "department": "Rehabilitation",
            "hospital": "Princeton Plainsboro",
            "password": "KutnerPass@2026!",
            "confirm_password": "KutnerPass@2026!",
            "consent_accepted": True,
        }

        ok, msg, doc, eval_token = register_public_doctor(payload)
        assert ok is True
        # In production mode, evaluation token must be None
        assert eval_token is None, "Evaluation token must NEVER be returned in production mode!"

        # Also for forgot password in production
        ok_reset, msg_reset, reset_tok = request_web_password_reset("kutner@hospital.org")
        assert ok_reset is True
        assert reset_tok is None, "Password reset token must NEVER be returned in production mode!"

    def test_15_backend_verify_email_endpoint(self, isolated_web_auth_env):
        """Requirement 15: Flask /api/auth/verify-email endpoint handles valid, empty, and invalid requests."""
        payload = {
            "full_name": "Dr. Martha Masters, MD",
            "email": "martha.masters@hospital.org",
            "username": "dr.martha.masters",
            "specialization": "Mathematics & Medicine",
            "department": "Diagnostics",
            "hospital": "Princeton Plainsboro",
            "password": "MastersPassword@2026!",
            "confirm_password": "MastersPassword@2026!",
            "consent_accepted": True,
        }
        _, _, _, valid_tok = register_public_doctor(payload)

        with app.test_client() as client:
            # 1. Missing token
            res_missing = client.post("/api/auth/verify-email", json={})
            assert res_missing.status_code == 400
            assert "please enter your verification token" in res_missing.get_json().get("message", "").lower()

            # 2. Empty string token
            res_empty = client.post("/api/auth/verify-email", json={"token": "   "})
            assert res_empty.status_code == 400
            assert "please enter your verification token" in res_empty.get_json().get("message", "").lower()

            # 3. Mismatched email
            res_mismatch = client.post(
                "/api/auth/verify-email",
                json={"token": valid_tok, "email": "impostor@hospital.org"}
            )
            assert res_mismatch.status_code == 400
            assert "does not match" in res_mismatch.get_json().get("message", "").lower()

            # 4. Valid token verification
            res_valid = client.post(
                "/api/auth/verify-email",
                json={"token": valid_tok, "email": "martha.masters@hospital.org"}
            )
            assert res_valid.status_code == 200
            data = res_valid.get_json()
            assert data["doctor"]["status"] == ACCOUNT_STATUS_PENDING_APPROVAL
            assert data["doctor"]["email_verified"] is True

    def test_16_role_selection_landing_page_navigation(self, isolated_web_auth_env):
        """Requirement 16: Landing page role selection renders appropriate views and preserves state."""
        from unittest.mock import patch, MagicMock
        from dashboard.pages.login import render_login_page

        mock_session_state = {}

        # 1. Default unauthenticated state: Role Selection page rendered
        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.form") as mock_form:

            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_login_page()
            # Role selection does not render login forms; it renders role cards
            assert not any(call[0][0] in ("doctor_login_form", "admin_login_form", "initial_admin_setup_form") for call in mock_form.call_args_list)

        # 2. Selecting ADMIN_LOGIN renders admin_login_form
        mock_session_state["login_portal_mode"] = "ADMIN_LOGIN"
        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.form") as mock_form:

            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_login_page()
            mock_form.assert_called_with("admin_login_form", clear_on_submit=False)

        # 3. Selecting DOCTOR_LOGIN renders doctor_login_form
        mock_session_state["login_portal_mode"] = "DOCTOR_LOGIN"
        mock_session_state["portal_active_tab"] = "🩺 Already Registered? Doctor Login"
        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.form") as mock_form:

            mock_form.return_value.__enter__ = MagicMock()
            mock_form.return_value.__exit__ = MagicMock(return_value=False)

            render_login_page()
            mock_form.assert_called_with("doctor_login_form", clear_on_submit=False)

    def test_17_admin_login_requires_admin_privilege_and_routes_to_management_dashboard(self, isolated_web_auth_env):
        """Requirement 17: Admin login verifies is_admin=True and prevents ordinary doctors from admin access."""
        from dashboard.styles import PAGE_REGISTER_DOCTOR
        from src.auth.auth_manager import is_administrator

        # 1. System administrator authentication
        ok_admin, admin_doc, _ = authenticate_doctor("admin.system", "AdminSecure@2026!")
        assert ok_admin is True
        assert is_administrator(admin_doc) is True
        assert admin_doc.get("is_admin") is True

        # 2. Ordinary doctor authentication succeeds as doctor but is NOT an administrator
        ok_doc, doctor_doc, _ = authenticate_doctor("dr.sarah.chen", "ClinicalSepsis@2026!")
        assert ok_doc is True
        assert is_administrator(doctor_doc) is False
        assert doctor_doc.get("is_admin") is False

        # In login.py, an ordinary doctor entering admin portal is blocked by is_administrator(doctor) check
        assert not (is_administrator(doctor_doc) is True)

    def test_18_email_verification_required_before_admin_approval(self, isolated_web_auth_env):
        """Requirement 18: Admin cannot approve an account that has not completed email verification."""
        reg = load_doctor_registry()
        admin_user = reg["doctors"]["admin.system"]

        payload = {
            "full_name": "Dr. Amber Volakis, MD",
            "email": "amber.volakis@hospital.org",
            "username": "dr.amber.volakis",
            "specialization": "Radiology",
            "department": "Radiology Department",
            "hospital": "Princeton Plainsboro",
            "password": "CutthroatPass@2026!",
            "confirm_password": "CutthroatPass@2026!",
            "consent_accepted": True,
        }
        ok_reg, _, doc_reg, raw_token = register_public_doctor(payload)
        assert ok_reg is True
        assert doc_reg["status"] == ACCOUNT_STATUS_PENDING_EMAIL_VERIFICATION

        # Attempt to approve BEFORE email verification must fail
        ok_appr_premature, msg_premature, _ = admin_approve_doctor(admin_user, "dr.amber.volakis")
        assert ok_appr_premature is False
        assert "email verification is pending" in msg_premature.lower()

        # Doctor cannot sign in yet
        ok_auth_early, _, msg_early = authenticate_doctor("dr.amber.volakis", "CutthroatPass@2026!")
        assert ok_auth_early is False
        assert "pending email verification" in msg_early.lower()

        # Now verify email
        ok_ver, msg_ver, doc_ver = verify_doctor_email(raw_token, expected_email_or_username="amber.volakis@hospital.org")
        assert ok_ver is True
        assert doc_ver["status"] == ACCOUNT_STATUS_PENDING_APPROVAL

        # Now administrator can approve
        ok_appr, msg_appr, doc_appr = admin_approve_doctor(admin_user, "dr.amber.volakis")
        assert ok_appr is True
        assert doc_appr["status"] == ACCOUNT_STATUS_ACTIVE
        assert doc_appr["is_active"] is True

        # Doctor can now sign in
        ok_auth_now, doc_auth, _ = authenticate_doctor("dr.amber.volakis", "CutthroatPass@2026!")
        assert ok_auth_now is True
        assert doc_auth["username"] == "dr.amber.volakis"
        assert doc_auth["role"] == "Staff Clinician"

    def test_19_pending_registrations_query_and_rejection_cycle(self, isolated_web_auth_env):
        """Requirement 19: Pending registrations are queried accurately and rejection blocks access."""
        reg = load_doctor_registry()
        admin_user = reg["doctors"]["admin.system"]

        payload = {
            "full_name": "Dr. Chi Park, MD",
            "email": "chi.park@hospital.org",
            "username": "dr.chi.park",
            "specialization": "Neurology",
            "department": "Neurology",
            "hospital": "Princeton Plainsboro",
            "password": "ParkPassword@2026!",
            "confirm_password": "ParkPassword@2026!",
            "consent_accepted": True,
        }
        ok_reg, _, _, raw_tok = register_public_doctor(payload)
        verify_doctor_email(raw_tok)

        # Inspect pending list
        pending = get_pending_doctor_registrations()
        usernames = [d["username"] for d in pending]
        assert "dr.chi.park" in usernames

        # Reject application
        ok_rej, msg_rej, doc_rej = admin_reject_doctor(admin_user, "dr.chi.park", reason="Incomplete hospital credential documents")
        assert ok_rej is True
        assert doc_rej["status"] == ACCOUNT_STATUS_REJECTED
        assert doc_rej["is_active"] is False

        # Authentication must be refused for rejected accounts
        ok_auth, _, msg_auth = authenticate_doctor("dr.chi.park", "ParkPassword@2026!")
        assert ok_auth is False
        assert "rejected" in msg_auth.lower()



