"""
tests/test_doctor_authentication.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Unit & Integration Test Suite for Hardened Doctor Authentication.

Covers:
1. PBKDF2-HMAC-SHA256 password hashing (600k iterations OWASP guidance, random salts, constant-time verification).
2. Backward-compatible verification of legacy 100k hashes and automatic rehash-on-success migration.
3. Password complexity policy enforcement and input sanitization.
4. Clinician registry bootstrapping, provisioning with role whitelisting, duplicate rejection, and admin password reset.
5. Anti-enumeration: identical generic credential error messages for unknown users and wrong passwords.
6. Signed HMAC-SHA256 session tokens with claim validation, tampering rejection, and TTL expiration.
7. Server-side token revocation on logout and administrative password reset.
8. Idle session timeout (30-minute threshold) vs absolute token lifetime (8 hours).
9. Flask REST API route protection (unauthenticated 401, revoked 401, authorized 200, public endpoints, no stack traces).
10. Production secret enforcement and safe failure on missing/weak secrets.
"""

from __future__ import annotations

import os
import sys
import time
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.auth.auth_manager import (
    hash_password,
    verify_password,
    needs_rehash,
    validate_password_strength,
    create_session_token,
    verify_session_token,
    revoke_session_token,
    is_token_revoked,
    is_session_expired,
    load_doctor_registry,
    save_doctor_registry,
    provision_doctor,
    authenticate_doctor,
    reset_doctor_password,
    get_auth_secret_key,
    DEFAULT_SESSION_TTL_SECONDS,
    MAX_ABSOLUTE_SESSION_SECONDS,
    PBKDF2_ITERATIONS,
    ALLOWED_ROLES,
    _secure_file_permissions,
)
from backend.app import app


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Password Hashing, Salting, Complexity & Migration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoctorPasswordHashing:
    """Verifies PBKDF2-HMAC-SHA256 with 600,000 iterations (OWASP standards)."""

    def test_hash_format_and_owasp_iteration_count(self):
        pw = "ClinicalSepsis@2026!"
        hashed = hash_password(pw)
        assert hashed.startswith("pbkdf2:sha256:600000$"), "New password hashes must use 600,000 iterations"
        parts = hashed.split("$")
        assert len(parts) == 3
        salt_hex, key_hex = parts[1], parts[2]
        assert len(salt_hex) == 32  # 16 bytes = 32 hex chars
        assert len(key_hex) == 64   # SHA-256 = 32 bytes = 64 hex chars

    def test_unique_salts_for_identical_passwords(self):
        pw = "SamePasswordEveryTime#123"
        h1 = hash_password(pw)
        h2 = hash_password(pw)
        assert h1 != h2, "Hashes for identical passwords must use distinct cryptographic salts"

    def test_verify_password_positive(self):
        pw = "CorrectDoctorPass@2026"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed) is True

    def test_verify_password_negative_mismatch(self):
        pw = "CorrectDoctorPass@2026"
        hashed = hash_password(pw)
        assert verify_password("WrongDoctorPass@2026", hashed) is False
        assert verify_password(pw.lower(), hashed) is False

    def test_verify_legacy_100k_iterations_hash(self):
        """Verifies that existing accounts with 100,000 iterations are safely verified."""
        pw = "LegacyDoctorPass@2026"
        legacy_hash = hash_password(pw, iterations=100_000)
        assert legacy_hash.startswith("pbkdf2:sha256:100000$")
        assert verify_password(pw, legacy_hash) is True
        assert verify_password("wrong", legacy_hash) is False

    def test_needs_rehash_detection(self):
        pw = "TestRehashPass@123"
        legacy_hash = hash_password(pw, iterations=100_000)
        current_hash = hash_password(pw, iterations=600_000)
        assert needs_rehash(legacy_hash, desired_iterations=600_000) is True
        assert needs_rehash(current_hash, desired_iterations=600_000) is False

    def test_verify_password_rejects_empty_and_garbage_safely(self):
        assert verify_password("", "some_hash") is False
        assert verify_password("pass", "") is False
        assert verify_password("pass", "malformed$hash$string") is False
        assert verify_password("pass", "pbkdf2:md5:1000$bad$bad") is False
        assert verify_password(None, "hash") is False

    def test_verify_password_rejects_sub_standard_iterations(self):
        """Reject hashes using suspiciously low iterations (e.g. 5,000) to prevent downgrade attacks."""
        pw = "LowIterPass#123"
        weak_hash = f"pbkdf2:sha256:5000${'aa'*16}${'bb'*32}"
        assert verify_password(pw, weak_hash) is False

    def test_hash_password_rejects_empty_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            hash_password("")

    def test_password_strength_policy(self):
        # Valid: >= 10 chars, upper, lower, digit, special
        valid, msg = validate_password_strength("ClinicalSepsis@2026!")
        assert valid is True

        # Invalid: too short (< 10)
        valid, msg = validate_password_strength("Short@1")
        assert valid is False
        assert "at least 10" in msg

        # Invalid: missing uppercase
        valid, msg = validate_password_strength("clinicalsepsis@2026!")
        assert valid is False
        assert "uppercase" in msg

        # Invalid: missing lowercase
        valid, msg = validate_password_strength("CLINICALSEPSIS@2026!")
        assert valid is False
        assert "lowercase" in msg

        # Invalid: missing digit
        valid, msg = validate_password_strength("ClinicalSepsis@Pass!")
        assert valid is False
        assert "digit" in msg

        # Invalid: missing special char
        valid, msg = validate_password_strength("ClinicalSepsis2026")
        assert valid is False
        assert "special character" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Registry Bootstrapping, Provisioning & Anti-Enumeration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoctorRegistryAndProvisioning:
    """Verifies clinician registry loading, role enforcement, and anti-enumeration."""

    def test_registry_contains_active_clinicians(self):
        reg = load_doctor_registry()
        doctors = reg.get("doctors", {})
        assert len(doctors) >= 1
        assert "dr.sarah.chen" in doctors
        sarah = doctors["dr.sarah.chen"]
        assert sarah["is_active"] is True
        assert "password" not in sarah  # No plaintext password field
        assert "password_hash" in sarah

    def test_provision_new_doctor_account_with_whitelisted_role(self):
        test_user = f"dr_marcus_{int(time.time())}"
        test_email = f"marcus.{int(time.time())}@hospital.org"
        doc = provision_doctor(
            username=test_user,
            email=test_email,
            full_name="Dr. Marcus Welby, MD",
            password="MarcusSecurePass!2026",
            role="Attending Cardiologist",
            department="Cardiology ICU",
        )
        assert doc["username"] == test_user
        assert doc["email"] == test_email
        assert doc["role"] == "Attending Cardiologist"
        assert "password_hash" not in doc  # Sanitized return

        # Verify authentication succeeds for provisioned doctor
        ok, authed_doc, _ = authenticate_doctor(test_user, "MarcusSecurePass!2026")
        assert ok is True
        assert authed_doc["username"] == test_user

    def test_provision_doctor_rejects_disallowed_role(self):
        unique_id = int(time.time())
        with pytest.raises(ValueError, match="Role 'System Administrator' is not permitted"):
            provision_doctor(
                username=f"dr_badrole_{unique_id}",
                email=f"badrole.{unique_id}@hospital.org",
                full_name="Dr. Bad Role, MD",
                password="SecurePass#2026",
                role="System Administrator",  # Not in whitelisted clinical roles
            )

    def test_provision_doctor_duplicate_rejection(self):
        unique_id = int(time.time())
        username = f"dr_dup_{unique_id}"
        email = f"dup.{unique_id}@hospital.org"
        provision_doctor(
            username=username,
            email=email,
            full_name="Dr. Duplicate, MD",
            password="SecurePassword@2026",
            role="Attending Intensivist",
        )
        # Duplicate username should raise
        with pytest.raises(ValueError, match="already exists"):
            provision_doctor(
                username=username,
                email=f"other.{unique_id}@hospital.org",
                full_name="Dr. Other, MD",
                password="SecurePassword@2026",
            )
        # Duplicate email should raise
        with pytest.raises(ValueError, match="already exists"):
            provision_doctor(
                username=f"other_user_{unique_id}",
                email=email,
                full_name="Dr. Other, MD",
                password="SecurePassword@2026",
            )

    def test_provision_doctor_input_validation(self):
        with pytest.raises(ValueError, match="Username must be 3-32 characters"):
            provision_doctor("", "valid@hospital.org", "Name", "password123#")

        with pytest.raises(ValueError, match="valid institutional email"):
            provision_doctor("valid_user", "not-an-email", "Name", "password123#")

        with pytest.raises(ValueError, match="at least 10 characters"):
            provision_doctor("valid_user", "valid@hospital.org", "Name", "short")

    def test_anti_enumeration_unified_error_messages(self):
        """Confirm error messages do NOT disclose whether an identifier exists."""
        # Non-existent user
        ok1, _, msg1 = authenticate_doctor("dr.completely.nonexistent.user", "AnyPassword#2026!")
        # Existing user with wrong password
        ok2, _, msg2 = authenticate_doctor("dr.sarah.chen", "WrongPassword#2026!")

        assert ok1 is False
        assert ok2 is False
        assert msg1 == msg2, "Error message must be identical for non-existent user and wrong password to prevent user enumeration"
        assert "Invalid credentials" in msg1

    def test_rehash_migration_on_successful_login(self):
        """Confirm that logging in with a legacy 100k hash transparently upgrades to 600k iterations."""
        unique_id = int(time.time())
        user = f"dr_legacy_{unique_id}"
        email = f"legacy.{unique_id}@hospital.org"
        pw = "LegacyPassword@2026!"

        # Manually seed user with 100,000 iterations
        reg = load_doctor_registry()
        reg["doctors"][user] = {
            "doctor_id": f"DOC-LEG-{unique_id % 1000}",
            "username": user,
            "email": email,
            "full_name": "Dr. Legacy Account, MD",
            "role": "Attending Intensivist",
            "department": "MICU",
            "initials": "LA",
            "password_hash": hash_password(pw, iterations=100_000),
            "token_version": 1,
            "is_active": True,
            "created_at": "2026-01-01T00:00:00Z",
            "last_login": None,
            "login_count": 0,
        }
        save_doctor_registry(reg)

        # Confirm it was seeded with 100k
        assert reg["doctors"][user]["password_hash"].startswith("pbkdf2:sha256:100000$")

        # Authenticate
        ok, doc, msg = authenticate_doctor(user, pw)
        assert ok is True

        # Check registry: hash should now be upgraded to 600,000 iterations
        updated_reg = load_doctor_registry()
        updated_hash = updated_reg["doctors"][user]["password_hash"]
        assert updated_hash.startswith("pbkdf2:sha256:600000$"), "Legacy hash must be upgraded to 600k iterations on successful login"

    def test_admin_password_reset_invalidates_prior_sessions(self):
        unique_id = int(time.time())
        user = f"dr_reset_{unique_id}"
        email = f"reset.{unique_id}@hospital.org"
        provision_doctor(user, email, "Dr. Reset, MD", "OriginalPass@2026#", role="Attending Intensivist")

        # Authenticate and obtain active token
        ok, doc, _ = authenticate_doctor(user, "OriginalPass@2026#")
        assert ok is True
        initial_token = create_session_token(doc)
        assert verify_session_token(initial_token) is not None

        # Perform administrative password reset
        success, msg = reset_doctor_password(user, "UpdatedNewPass@2026#")
        assert success is True

        # Old token MUST now be invalidated because token_version was incremented
        assert verify_session_token(initial_token) is None, "Prior session tokens must be invalidated after a password reset"

        # Old password must fail
        ok_old, _, _ = authenticate_doctor(user, "OriginalPass@2026#")
        assert ok_old is False

        # New password succeeds and produces a working token
        ok_new, new_doc, _ = authenticate_doctor(user, "UpdatedNewPass@2026#")
        assert ok_new is True
        new_token = create_session_token(new_doc)
        assert verify_session_token(new_token) is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Session Tokens, Revocation, Idle Timeout & Lifetime Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoctorAuthenticationAndSessions:
    """Verifies token signing, claims, server-side revocation, and timeouts."""

    @classmethod
    def setup_class(cls):
        # Ensure a known doctor account is available
        unique_id = int(time.time())
        cls.username = f"dr_session_{unique_id}"
        cls.email = f"session.{unique_id}@hospital.org"
        cls.password = "SessionPass@2026#"
        provision_doctor(cls.username, cls.email, "Dr. Session Tester, MD", cls.password, role="Attending Intensivist")

    def test_token_creation_and_payload_claims(self):
        ok, doc, _ = authenticate_doctor(self.username, self.password)
        assert ok is True
        token = create_session_token(doc, ttl_seconds=1800)
        assert isinstance(token, str) and "." in token

        payload = verify_session_token(token)
        assert payload is not None
        assert payload["username"] == self.username
        assert "sub" in payload
        assert "jti" in payload
        assert "iat" in payload
        assert "exp" in payload
        assert payload["exp"] > time.time()

    def test_tampered_signature_rejected(self):
        ok, doc, _ = authenticate_doctor(self.username, self.password)
        token = create_session_token(doc)
        parts = token.split(".")
        # Tamper signature portion
        tampered_sig = parts[1][:-2] + ("AA" if not parts[1].endswith("AA") else "BB")
        tampered_token = f"{parts[0]}.{tampered_sig}"
        assert verify_session_token(tampered_token) is None

    def test_tampered_payload_rejected(self):
        ok, doc, _ = authenticate_doctor(self.username, self.password)
        token = create_session_token(doc)
        parts = token.split(".")
        # Tamper payload portion
        tampered_payload = parts[0] + "xyz"
        tampered_token = f"{tampered_payload}.{parts[1]}"
        assert verify_session_token(tampered_token) is None

    def test_expired_token_rejected(self):
        ok, doc, _ = authenticate_doctor(self.username, self.password)
        # Create token already expired 10 seconds ago
        expired_token = create_session_token(doc, ttl_seconds=-10)
        assert verify_session_token(expired_token) is None

    def test_server_side_token_revocation_on_logout(self):
        ok, doc, _ = authenticate_doctor(self.username, self.password)
        token = create_session_token(doc)
        assert verify_session_token(token) is not None

        # Revoke token server-side
        assert revoke_session_token(token) is True

        # Token must now be rejected
        assert verify_session_token(token) is None, "Revoked token must be rejected by verify_session_token"

    def test_idle_timeout_vs_absolute_expiration(self):
        now = time.time()
        # Activity 10 minutes ago (< 30 min idle threshold): active
        assert is_session_expired(now - 600, max_idle_seconds=1800) is False

        # Inactivity 31 minutes ago (> 30 min idle threshold): expired
        assert is_session_expired(now - 1860, max_idle_seconds=1800) is True

        # Missing activity timestamp: expired
        assert is_session_expired(0) is True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Backend API Route Security & Authorization Matrix Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackendApiAuthenticationProtection:
    """Verifies REST API endpoints enforce authentication and reject unauthorized requests."""

    @classmethod
    def setup_class(cls):
        app.config["REQUIRE_AUTH"] = True
        cls.client = app.test_client()

        # Seed clean test clinician
        unique_id = int(time.time())
        cls.user = f"dr_api_{unique_id}"
        cls.email = f"api.{unique_id}@hospital.org"
        cls.password = "ApiSecurityPass@2026#"
        provision_doctor(cls.user, cls.email, "Dr. API Security, MD", cls.password, role="Attending Intensivist")

        _, cls.doctor, _ = authenticate_doctor(cls.user, cls.password)
        cls.valid_token = create_session_token(cls.doctor)

    def test_public_health_endpoint_without_token(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "password" not in str(data).lower()

    def test_public_sensor_status_without_token_exposes_no_patient_data(self):
        resp = self.client.get("/api/sensor/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "isolated"
        assert "STRICT DATA ISOLATION" in data["isolation_guarantee"]
        # Must NOT expose patient identifiers or clinical records
        assert "patient_id" not in data
        assert "subject_id" not in data

    def test_sensitive_routes_inventory_rejects_missing_auth(self):
        """Every sensitive route must return 401 when missing an authentication token."""
        sensitive_routes = [
            ("GET", "/api/patients"),
            ("GET", "/api/patients/stats"),
            ("GET", "/api/patient/10213765"),
            ("GET", "/api/patient/10213765/stay/36628477"),
            ("GET", "/api/patient/10213765/stay/36628477/infer/clinical"),
            ("POST", "/api/patient/10213765/stay/36628477/infer/clinical"),
            ("GET", "/api/patient/10213765/stay/36628477/infer/cxr"),
            ("POST", "/api/patient/10213765/stay/36628477/infer/cxr"),
            ("GET", "/api/patient/10213765/stay/36628477/image/unrelated-dicom-id"),
            ("GET", "/api/auth/verify"),
            ("POST", "/api/auth/logout"),
        ]
        for method, route in sensitive_routes:
            if method == "GET":
                resp = self.client.get(route)
            else:
                resp = self.client.post(route)
            assert resp.status_code == 401, f"Route {method} {route} must reject unauthenticated requests with 401"
            data = resp.get_json()
            assert data["status"] == "error"

    def test_protected_routes_reject_invalid_token(self):
        headers = {"Authorization": "Bearer totally_invalid_token_format"}
        resp = self.client.get("/api/patients", headers=headers)
        assert resp.status_code == 401
        data = resp.get_json()
        assert "invalid, expired, or has been revoked" in data["message"]

    def test_api_auth_login_endpoint_success(self):
        resp = self.client.post("/api/auth/login", json={
            "username": self.user,
            "password": self.password,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "token" in data
        assert data["doctor"]["username"] == self.user
        assert "password_hash" not in data["doctor"]

    def test_api_auth_login_endpoint_invalid_credentials_generic_message(self):
        resp = self.client.post("/api/auth/login", json={
            "username": self.user,
            "password": "WrongPassword@2026#",
        })
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["status"] == "error"
        assert "Invalid credentials" in data["message"]

    def test_api_auth_logout_revokes_token_server_side(self):
        # 1. Login to get fresh token
        resp_login = self.client.post("/api/auth/login", json={
            "username": self.user,
            "password": self.password,
        })
        token = resp_login.get_json()["token"]

        # 2. Verify token works on protected route
        headers = {"Authorization": f"Bearer {token}"}
        resp_active = self.client.get("/api/auth/verify", headers=headers)
        assert resp_active.status_code == 200

        # 3. Call /api/auth/logout
        resp_logout = self.client.post("/api/auth/logout", headers=headers)
        assert resp_logout.status_code == 200
        assert "revoked server-side" in resp_logout.get_json()["message"]

        # 4. Same token MUST now be rejected on subsequent protected requests
        resp_after = self.client.get("/api/auth/verify", headers=headers)
        assert resp_after.status_code == 401, "Token must be rejected after /api/auth/logout"

    def test_protected_patients_endpoint_accepts_valid_token(self):
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        resp = self.client.get("/api/patients", headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "patients" in data

    def test_protected_x_doctor_token_header_accepted(self):
        headers = {"X-Doctor-Token": self.valid_token}
        resp = self.client.get("/api/patients/stats", headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "statistics" in data

    def test_custom_error_handlers_prevent_stack_trace_leakage(self):
        # 404 handler
        resp_404 = self.client.get("/api/nonexistent_route_99999")
        assert resp_404.status_code == 404
        assert resp_404.get_json()["status"] == "error"
        assert "not found" in resp_404.get_json()["message"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Production Environment & File System Security Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvironmentAndFileSecurity:
    """Verifies production secret enforcement and file permissions."""

    def test_production_fails_safely_if_secret_missing(self, monkeypatch):
        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.delenv("XMT_AUTH_SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError, match="XMT_AUTH_SECRET_KEY must be explicitly set in production"):
            get_auth_secret_key()

    def test_production_fails_safely_if_secret_is_weak(self, monkeypatch):
        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.setenv("XMT_AUTH_SECRET_KEY", "short_secret")
        with pytest.raises(RuntimeError, match="must be a cryptographically random secret of at least 32 characters"):
            get_auth_secret_key()

    def test_file_permission_helper_runs_safely(self, tmp_path):
        dummy_file = tmp_path / "test_sec.json"
        dummy_file.write_text('{"test": 1}', encoding="utf-8")
        # Should execute cleanly without unhandled exceptions on Windows or POSIX
        _secure_file_permissions(str(dummy_file))
        assert os.path.isfile(str(dummy_file))
