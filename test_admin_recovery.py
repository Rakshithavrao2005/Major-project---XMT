"""
tests/test_admin_recovery.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Test Suite for Secure Administrator Account Recovery.

Verifies:
1. Password policy enforcement during recovery (rejects weak/short/simple passwords).
2. Password hash verification (PBKDF2-HMAC-SHA256 600k iterations with unique salts).
3. Targeted administrator password reset by username or email.
4. Rejection of invalid account targets (non-existent users, ordinary clinicians, inactive admins).
5. Invalidation of previous sessions after recovery (token_version increment).
6. Continued authentication and token validity for unaffected accounts.
7. CLI recover-admin command execution, confirmation guard, and abort behavior.

All tests run in isolated sandboxed directories using pytest's tmp_path fixture.
The production/live registry (data/auth/doctors.json) is never modified.
"""

from __future__ import annotations

import io
import json
import os
import sys
from unittest.mock import patch
import pytest

import src.auth.auth_manager as auth_mod
from src.auth.auth_manager import (
    authenticate_doctor,
    create_session_token,
    hash_password,
    load_doctor_registry,
    recover_administrator_password,
    validate_password_strength,
    verify_password,
    verify_session_token,
)
import src.auth.provision_doctor as cli_mod


@pytest.fixture
def isolated_recovery_env(tmp_path, monkeypatch):
    """
    Creates an isolated auth environment with:
    1. An active administrator ('admin.system')
    2. An active ordinary clinician ('dr.sarah.chen')
    3. An inactive administrator ('admin.disabled')
    """
    temp_dir = str(tmp_path / "auth")
    os.makedirs(temp_dir, exist_ok=True)
    registry_file = os.path.join(temp_dir, "doctors.json")
    secret_key_file = os.path.join(temp_dir, ".secret_key")
    revoked_file = os.path.join(temp_dir, "revoked_tokens.json")

    monkeypatch.setattr(auth_mod, "AUTH_DIR", temp_dir)
    monkeypatch.setattr(auth_mod, "REGISTRY_FILE", registry_file)
    monkeypatch.setattr(auth_mod, "SECRET_KEY_FILE", secret_key_file)
    monkeypatch.setattr(auth_mod, "REVOKED_TOKENS_FILE", revoked_file)

    test_registry = {
        "version": 2,
        "description": "XMT Isolated Recovery Test Registry",
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
                "password_hash": auth_mod.hash_password("OldAdminPass@2026!", auth_mod.PBKDF2_ITERATIONS),
                "token_version": 1,
                "is_active": True,
                "is_admin": True,
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
                "password_hash": auth_mod.hash_password("ClinicianSecret@2026!", auth_mod.PBKDF2_ITERATIONS),
                "token_version": 1,
                "is_active": True,
                "is_admin": False,
                "created_at": "2026-10-18T00:00:00Z",
                "last_login": None,
                "login_count": 0,
            },
            "admin.disabled": {
                "doctor_id": "DOC-ADM-99",
                "username": "admin.disabled",
                "email": "disabled.admin@hospital.org",
                "full_name": "Deactivated Administrator",
                "role": "System Administrator",
                "department": "IT",
                "initials": "DA",
                "password_hash": auth_mod.hash_password("DisabledAdmin@2026!", auth_mod.PBKDF2_ITERATIONS),
                "token_version": 1,
                "is_active": False,
                "is_admin": True,
                "created_at": "2026-10-18T00:00:00Z",
                "last_login": None,
                "login_count": 0,
            },
        },
    }

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(test_registry, f, indent=2)

    return {
        "dir": temp_dir,
        "registry_file": registry_file,
    }


class TestSecureAdministratorAccountRecovery:
    """Verifies all six required recovery invariants in an isolated test harness."""

    def test_01_password_policy_enforcement(self, isolated_recovery_env):
        """Requirement 1: Enforces password complexity policy on recovery."""
        weak_passwords = [
            ("", "at least 10 characters"),
            ("Short1!", "at least 10 characters"),
            ("alllowercase1234!", "at least one uppercase"),
            ("ALLUPPERCASE1234!", "at least one lowercase"),
            ("NoDigitsSpecialHere!", "at least one numerical"),
            ("NoSpecialChars12345", "at least one special character"),
        ]

        for weak_pw, expected_snippet in weak_passwords:
            success, msg, record = recover_administrator_password("admin.system", weak_pw)
            assert success is False
            assert record is None
            assert expected_snippet.lower() in msg.lower(), f"Expected '{expected_snippet}' in error '{msg}'"

    def test_02_password_hash_verification(self, isolated_recovery_env):
        """Requirement 2: Uses PBKDF2-HMAC-SHA256 600,000 iterations and verifies new hash."""
        new_password = "RecoveredAdminPassword@2026!"
        success, msg, record = recover_administrator_password("admin.system", new_password)

        assert success is True
        assert record is not None
        assert "password_hash" not in record  # Plaintext / hash never leaked in return

        # Verify old password no longer works
        auth_old, _, _ = authenticate_doctor("admin.system", "OldAdminPass@2026!")
        assert auth_old is False

        # Verify new password successfully authenticates
        auth_new, doc, auth_msg = authenticate_doctor("admin.system", new_password)
        assert auth_new is True
        assert doc is not None
        assert doc["username"] == "admin.system"
        assert doc["is_admin"] is True

        # Inspect on-disk stored hash structure
        reg = load_doctor_registry()
        stored_hash = reg["doctors"]["admin.system"]["password_hash"]
        assert stored_hash.startswith("pbkdf2:sha256:600000$")

        # Verify cryptographic verification helper
        assert verify_password(new_password, stored_hash) is True
        assert verify_password("WrongPassword123!", stored_hash) is False

    def test_03_targeted_administrator_password_reset(self, isolated_recovery_env):
        """Requirement 3: Targets administrator explicitly by username or institutional email."""
        # Reset via institutional email
        new_pw_email = "EmailResetPassword@2026!"
        success, msg, record = recover_administrator_password("admin@hospital.org", new_pw_email)
        assert success is True
        assert record["username"] == "admin.system"

        auth_ok, doc, _ = authenticate_doctor("admin.system", new_pw_email)
        assert auth_ok is True
        assert doc["is_admin"] is True

    def test_04_rejection_of_invalid_account_targets(self, isolated_recovery_env):
        """Requirement 4: Rejects non-existent accounts, non-admins, and inactive accounts."""
        valid_new_pw = "NewTargetPassword@2026!"

        # A: Non-existent account
        ok_fake, msg_fake, rec_fake = recover_administrator_password("nonexistent.user", valid_new_pw)
        assert ok_fake is False
        assert "not found" in msg_fake.lower()
        assert rec_fake is None

        # B: Non-administrator clinician (dr.sarah.chen)
        ok_clinician, msg_clinician, rec_clinician = recover_administrator_password("dr.sarah.chen", valid_new_pw)
        assert ok_clinician is False
        assert "not an administrator" in msg_clinician.lower()
        assert rec_clinician is None

        # C: Inactive administrator (admin.disabled)
        ok_inactive, msg_inactive, rec_inactive = recover_administrator_password("admin.disabled", valid_new_pw)
        assert ok_inactive is False
        assert "disabled" in msg_inactive.lower() or "inactive" in msg_inactive.lower()
        assert rec_inactive is None

    def test_05_invalidation_of_previous_sessions(self, isolated_recovery_env):
        """Requirement 5: Existing session tokens are invalidated immediately upon recovery."""
        reg = load_doctor_registry()
        admin_doc = reg["doctors"]["admin.system"]

        # Issue an active session token before password recovery
        token_before = create_session_token(admin_doc)
        payload_before = verify_session_token(token_before)
        assert payload_before is not None
        assert payload_before["username"] == "admin.system"

        # Execute recovery
        new_pw = "BrandNewAdminSecret@2026!"
        success, _, record = recover_administrator_password("admin.system", new_pw)
        assert success is True
        assert record["token_version"] == 2

        # The prior token MUST now be invalidated because token_version was incremented
        payload_after = verify_session_token(token_before)
        assert payload_after is None, "Prior session token must be rejected after password recovery"

        # Newly issued token for the administrator MUST be accepted
        updated_doc = {**admin_doc, "token_version": record["token_version"]}
        token_new = create_session_token(updated_doc)
        payload_new = verify_session_token(token_new)
        assert payload_new is not None
        assert payload_new["username"] == "admin.system"
        assert payload_new["admin"] is True

    def test_06_continued_authentication_of_unaffected_accounts(self, isolated_recovery_env):
        """Requirement 6: Other accounts remain fully intact and operational."""
        reg_before = load_doctor_registry()
        clinician_doc = reg_before["doctors"]["dr.sarah.chen"]
        clinician_token = create_session_token(clinician_doc)

        # Clinician token works before admin recovery
        assert verify_session_token(clinician_token) is not None

        # Execute admin recovery
        recover_administrator_password("admin.system", "AdminResetPW@2026!")

        # Verify clinician account is completely unaffected
        auth_clinician, doc_clinician, _ = authenticate_doctor("dr.sarah.chen", "ClinicianSecret@2026!")
        assert auth_clinician is True
        assert doc_clinician["role"] == "Attending Intensivist / Clinical Researcher"
        assert doc_clinician["is_admin"] is False

        # Verify clinician's prior session token remains valid (its token_version was not incremented)
        assert verify_session_token(clinician_token) is not None

    def test_07_cli_recover_admin_with_confirmation_and_bypass(self, isolated_recovery_env, monkeypatch):
        """Requirement 7: CLI command requires confirmation and supports non-interactive execution."""
        # 1. Test CLI with explicit -y and --new-password
        test_args = [
            "provision_doctor.py",
            "recover-admin",
            "--identifier", "admin.system",
            "--new-password", "CliRecoveredPassword@2026!",
            "-y",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cli_mod.main()
            output = mock_out.getvalue()
            assert "[SUCCESS]" in output
            assert "admin.system" in output

        # Verify credentials work
        auth_ok, _, _ = authenticate_doctor("admin.system", "CliRecoveredPassword@2026!")
        assert auth_ok is True

        # 2. Test CLI user confirmation denial aborts safely without changes
        abort_args = [
            "provision_doctor.py",
            "recover-admin",
            "--identifier", "admin.system",
        ]
        monkeypatch.setattr(sys, "argv", abort_args)
        monkeypatch.setattr("builtins.input", lambda prompt: "no")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out, pytest.raises(SystemExit) as exc_info:
            cli_mod.main()

        assert exc_info.value.code == 0
        output = mock_out.getvalue()
        assert "[ABORTED]" in output

        # Password should still be the one set in part 1
        auth_still_ok, _, _ = authenticate_doctor("admin.system", "CliRecoveredPassword@2026!")
        assert auth_still_ok is True
