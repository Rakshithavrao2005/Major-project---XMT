"""
tests/test_doctor_repository.py
=============================================================================
Unit tests for src/auth/doctor_repository.py.

All tests operate strictly offline using temporary directories (tmp_path)
and synthetic doctor records.
ZERO modifications or reads to the real data/auth/doctors.json registry file.
ZERO cloud network calls.
=============================================================================
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.auth.doctor_repository import (
    DoctorRepository,
    DoctorRepositoryStatus,
    DoctorResult,
    sanitize_doctor_record,
)


@pytest.fixture
def synthetic_registry(tmp_path):
    """
    Create an isolated synthetic doctor registry in a temporary directory.
    Guarantees no interaction with production or local user files.
    """
    reg_file = tmp_path / "mock_doctors.json"
    data = {
        "version": 2,
        "description": "Synthetic Test Clinician Registry",
        "last_updated": "2026-10-22T00:00:00Z",
        "doctors": {
            "dr.sarah.chen": {
                "doctor_id": "DOC-REG-0001",
                "username": "dr.sarah.chen",
                "email": "sarah.chen@hospital.org",
                "full_name": "Dr. Sarah Chen, MD",
                "role": "Attending Intensivist",
                "department": "Intensive Care Unit",
                "hospital": "Metropolitan Health",
                "specialization": "Critical Care",
                "initials": "SC",
                "password_hash": "pbkdf2:sha256:600000$mock_salt$mock_hash_sc",
                "token_version": 1,
                "is_admin": False,
                "is_active": True,
                "status": "Active",
                "created_at": "2026-01-10T10:00:00Z",
            },
            "dr.alex.rivera": {
                "doctor_id": "DOC-ADMIN-0001",
                "username": "dr.alex.rivera",
                "email": "alex.rivera@hospital.org",
                "full_name": "Dr. Alex Rivera, MD",
                "role": "System Administrator",
                "department": "Clinical Informatics",
                "hospital": "Metropolitan Health",
                "specialization": "Informatics",
                "initials": "AR",
                "password_hash": "pbkdf2:sha256:600000$mock_salt$mock_hash_ar",
                "token_version": 2,
                "is_admin": True,
                "is_active": True,
                "status": "Active",
                "created_at": "2026-01-01T08:00:00Z",
            },
            "dr.pending.user": {
                "doctor_id": "DOC-REG-0002",
                "username": "dr.pending.user",
                "email": "pending.user@hospital.org",
                "full_name": "Dr. Pending User, MD",
                "role": "Staff Clinician",
                "department": "Pulmonology",
                "hospital": "Metropolitan Health",
                "specialization": "Pulmonology",
                "initials": "PU",
                "password_hash": "pbkdf2:sha256:600000$mock_salt$mock_hash_pu",
                "token_version": 1,
                "is_admin": False,
                "is_active": False,
                "status": "Pending Approval",
                "created_at": "2026-10-20T14:00:00Z",
            },
        },
    }
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(reg_file)


# -----------------------------------------------------------------------------
# 1. Lookup Operations
# -----------------------------------------------------------------------------
def test_get_by_id_success(synthetic_registry):
    """Verify lookup by stable doctor ID returns correct sanitized data."""
    repo = DoctorRepository(registry_file=synthetic_registry)
    res = repo.get_by_id("DOC-REG-0001")
    assert res.is_success is True
    assert res.status == DoctorRepositoryStatus.SUCCESS
    assert res.data["username"] == "dr.sarah.chen"
    assert res.data["doctor_id"] == "DOC-REG-0001"
    # Verify sanitization
    assert "password_hash" not in res.data
    assert res.is_sanitized is True


def test_get_by_id_unknown(synthetic_registry):
    """Verify unknown doctor ID returns NOT_FOUND status."""
    repo = DoctorRepository(registry_file=synthetic_registry)
    res = repo.get_by_id("DOC-UNKNOWN-9999")
    assert res.is_success is False
    assert res.status == DoctorRepositoryStatus.NOT_FOUND
    assert res.data is None
    assert "not found" in res.message.lower()


def test_get_by_identifier_username_and_email(synthetic_registry):
    """Verify lookup by username and email respects case-insensitivity and whitespace trimming."""
    repo = DoctorRepository(registry_file=synthetic_registry)

    # By username with uppercase and spaces
    res1 = repo.get_by_identifier("  DR.SARAH.CHEN  ")
    assert res1.is_success is True
    assert res1.data["doctor_id"] == "DOC-REG-0001"

    # By email with mixed case
    res2 = repo.get_by_identifier("Alex.Rivera@Hospital.ORG")
    assert res2.is_success is True
    assert res2.data["doctor_id"] == "DOC-ADMIN-0001"
    assert res2.data["is_admin"] is True


def test_get_by_identifier_unknown(synthetic_registry):
    """Verify unknown username or email returns NOT_FOUND."""
    repo = DoctorRepository(registry_file=synthetic_registry)
    res = repo.get_by_identifier("unknown.doctor@nowhere.org")
    assert res.is_success is False
    assert res.status == DoctorRepositoryStatus.NOT_FOUND
    assert res.data is None


def test_get_sanitized_profile_resolves_both_keys(synthetic_registry):
    """Verify get_sanitized_profile resolves either doctor ID or username/email."""
    repo = DoctorRepository(registry_file=synthetic_registry)

    # Via ID
    p1 = repo.get_sanitized_profile("DOC-REG-0001")
    assert p1.is_success is True
    assert p1.data["username"] == "dr.sarah.chen"
    assert "password_hash" not in p1.data

    # Via username
    p2 = repo.get_sanitized_profile("dr.sarah.chen")
    assert p2.is_success is True
    assert p2.data["doctor_id"] == "DOC-REG-0001"
    assert "password_hash" not in p2.data


# -----------------------------------------------------------------------------
# 2. Defensive Immutability & Sanitization
# -----------------------------------------------------------------------------
def test_returned_record_mutation_does_not_corrupt_registry(synthetic_registry):
    """Verify mutating a returned dictionary does not mutate subsequent repository reads."""
    repo = DoctorRepository(registry_file=synthetic_registry)

    res1 = repo.get_by_id("DOC-REG-0001")
    assert res1.data["role"] == "Attending Intensivist"

    # Maliciously or inadvertently mutate the returned record
    res1.data["role"] = "HACKED_ROLE"
    res1.data["is_admin"] = True

    # Read again: must remain untouched
    res2 = repo.get_by_id("DOC-REG-0001")
    assert res2.data["role"] == "Attending Intensivist"
    assert res2.data["is_admin"] is False


def test_sanitize_doctor_record_utility():
    """Verify all sensitive fields are stripped by sanitize_doctor_record."""
    dirty = {
        "doctor_id": "DOC-123",
        "username": "doc",
        "password_hash": "pbkdf2:sha256:600000$salt$hash",
        "verification_token": "123456",
        "reset_token": "abc-reset",
        "secret_key": "raw_secret",
        "role": "Staff",
    }
    cleaned = sanitize_doctor_record(dirty)
    assert "password_hash" not in cleaned
    assert "verification_token" not in cleaned
    assert "reset_token" not in cleaned
    assert "secret_key" not in cleaned
    assert cleaned["doctor_id"] == "DOC-123"
    assert cleaned["role"] == "Staff"


# -----------------------------------------------------------------------------
# 3. Missing, Malformed, and Error States
# -----------------------------------------------------------------------------
def test_missing_registry_file_returns_unavailable(tmp_path):
    """Verify missing file returns UNAVAILABLE without creating dummy files."""
    missing_path = tmp_path / "non_existent_registry.json"
    repo = DoctorRepository(registry_file=str(missing_path))

    res = repo.get_by_id("DOC-REG-0001")
    assert res.is_success is False
    assert res.status == DoctorRepositoryStatus.UNAVAILABLE
    assert "not found" in res.message.lower()

    status = repo.get_status()
    assert status["file_exists"] is False
    assert status["status"] == "UNAVAILABLE"


def test_malformed_json_returns_malformed_status(tmp_path):
    """Verify corrupted JSON returns MALFORMED_REGISTRY without crashing."""
    bad_file = tmp_path / "corrupted.json"
    bad_file.write_text("{ this is not valid json : [", encoding="utf-8")
    repo = DoctorRepository(registry_file=str(bad_file))

    res = repo.get_by_id("DOC-REG-0001")
    assert res.is_success is False
    assert res.status == DoctorRepositoryStatus.MALFORMED_REGISTRY

    status = repo.get_status()
    assert status["status"] == "MALFORMED_REGISTRY"


def test_invalid_schema_missing_doctors_key(tmp_path):
    """Verify registry missing the 'doctors' mapping returns MALFORMED_REGISTRY."""
    invalid_schema = tmp_path / "invalid_schema.json"
    invalid_schema.write_text(json.dumps({"version": 2, "not_doctors": {}}), encoding="utf-8")
    repo = DoctorRepository(registry_file=str(invalid_schema))

    res = repo.get_by_id("DOC-REG-0001")
    assert res.is_success is False
    assert res.status == DoctorRepositoryStatus.MALFORMED_REGISTRY


# -----------------------------------------------------------------------------
# 4. List Filtering and Concurrency
# -----------------------------------------------------------------------------
def test_list_doctors_filtering(synthetic_registry):
    """Verify list_doctors filters correctly by status and role."""
    repo = DoctorRepository(registry_file=synthetic_registry)

    # All active
    active = repo.list_doctors(status_filter="Active")
    assert active.is_success is True
    assert len(active.data) == 2

    # Pending approval
    pending = repo.list_doctors(status_filter="Pending Approval")
    assert pending.is_success is True
    assert len(pending.data) == 1
    assert pending.data[0]["doctor_id"] == "DOC-REG-0002"

    # By role
    admins = repo.list_doctors(role_filter="System Administrator")
    assert admins.is_success is True
    assert len(admins.data) == 1
    assert admins.data[0]["doctor_id"] == "DOC-ADMIN-0001"


def test_concurrent_read_safety(synthetic_registry):
    """Verify multi-threaded reads execute safely under file and thread locking."""
    repo = DoctorRepository(registry_file=synthetic_registry)

    def read_worker(doc_id: str):
        res = repo.get_by_id(doc_id)
        assert res.is_success is True
        return res.data["doctor_id"]

    ids = ["DOC-REG-0001", "DOC-ADMIN-0001", "DOC-REG-0002"] * 10
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(read_worker, ids))

    assert len(results) == 30
    assert "DOC-REG-0001" in results
