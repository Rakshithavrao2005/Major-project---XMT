"""
tests/test_cxr_storage_adapter.py
=============================================================================
Comprehensive Offline Synthetic Test Suite for Stage 3.7.5B:
Local-First CXR Storage Adapter and Secure Access Abstraction.

Verifies:
 1. Valid local image resolution from synthetic fixture.
 2. Unknown image handling (NOT_FOUND).
 3. Path traversal rejection (.. and path separators).
 4. Absolute-path and symlink-escape rejection.
 5. DICOM identifier validation (schema, length, characters).
 6. Signed-URL expiration boundaries at exactly 1s and 300s.
 7. Rejection of expiry values < 1s or > 300s.
 8. Mock signed-URL request behavior with mock Supabase storage client.
 9. Cloud-disabled local fallback when authorized.
10. Cloud failure fallback without bypassing authorization.
11. Rejection when is_authorized=False (UNAUTHORIZED).
12. Verification of security response headers.
13. Verification that secrets, keys, and image bytes are absent from dicts/logs.
=============================================================================
"""

import io
import os
import sys
from unittest.mock import MagicMock
import pytest
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.imaging.storage_adapter import (
    CXRStorageAdapter,
    CXRImageAccessResult,
    StorageAccessStatus,
    StorageOrigin,
    validate_dicom_id,
    get_secure_image_headers,
    MIN_SIGNED_URL_EXPIRY_SECONDS,
    MAX_SIGNED_URL_EXPIRY_SECONDS,
)


@pytest.fixture
def synthetic_images_dir(tmp_path):
    """Create a temporary images directory populated with synthetic test radiographs."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic 64x64 grayscale test image
    dicom_id = "test-dicom-synthetic-001"
    img_path = img_dir / f"{dicom_id}.jpg"
    img = Image.new("L", (64, 64), color=128)
    img.save(str(img_path), format="JPEG")

    # Generate second synthetic image with 44-char MIMIC-style hex format
    mimic_dicom_id = "84c6ecc4-fd81e31d-1e879d18-13024211-35de54ad"
    mimic_path = img_dir / f"{mimic_dicom_id}.jpg"
    img2 = Image.new("L", (64, 64), color=200)
    img2.save(str(mimic_path), format="JPEG")

    return img_dir


@pytest.fixture
def mock_storage_client():
    """Create a mock Supabase storage client for offline signed URL testing."""
    client = MagicMock()
    storage_bucket = MagicMock()
    client.storage.from_.return_value = storage_bucket

    def _mock_create_signed_url(path, expires_in):
        return {
            "signedURL": f"https://mock-supabase.test/storage/v1/object/sign/chest-xrays/{path}?token=mock_test_token_123&expires={expires_in}"
        }

    storage_bucket.create_signed_url.side_effect = _mock_create_signed_url
    return client


# ── Scenario 1: Valid Local Image Resolution ──────────────────────────────────

def test_01_valid_local_image_resolution(synthetic_images_dir):
    """Resolve a valid synthetic image into an in-memory stream from local disk."""
    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        enable_cloud=False,
    )
    res = adapter.resolve_image_stream("test-dicom-synthetic-001")

    assert res.status == StorageAccessStatus.SUCCESS
    assert res.origin == StorageOrigin.LOCAL_DISK
    assert res.dicom_id == "test-dicom-synthetic-001"
    assert res.image_stream is not None
    assert isinstance(res.image_stream, io.BytesIO)
    assert len(res.image_stream.getvalue()) > 0
    assert res.content_type == "image/jpeg"
    assert res.local_path is not None
    assert os.path.isfile(res.local_path)


# ── Scenario 2: Unknown Image Handling ────────────────────────────────────────

def test_02_unknown_image_handling(synthetic_images_dir):
    """Attempting to resolve a non-existent DICOM returns NOT_FOUND with no stream."""
    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        enable_cloud=False,
    )
    res = adapter.resolve_image_stream("nonexistent-dicom-999")

    assert res.status == StorageAccessStatus.NOT_FOUND
    assert res.origin == StorageOrigin.NONE
    assert res.image_stream is None
    assert res.signed_url is None
    assert "not found" in res.message.lower()


# ── Scenario 3: Path Traversal Rejection ───────────────────────────────────────

@pytest.mark.parametrize("malicious_id", [
    "../secret",
    "../../etc/passwd",
    "..\\windows\\system32",
    "foo/bar",
    "foo\\bar",
    "test/../traversal",
    "test\x00escape",
])
def test_03_path_traversal_rejection(synthetic_images_dir, malicious_id):
    """Paths containing directory traversal or slashes are strictly rejected."""
    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        enable_cloud=False,
    )
    res = adapter.resolve_image_stream(malicious_id)

    assert res.status == StorageAccessStatus.INVALID_IDENTIFIER
    assert res.origin == StorageOrigin.NONE
    assert res.image_stream is None


# ── Scenario 4: Absolute-Path and Symlink-Escape Rejection ─────────────────────

def test_04_symlink_escape_rejection(tmp_path, synthetic_images_dir):
    """Symlink pointing outside the local images directory is rejected by canonical path check."""
    # Create external secret file outside synthetic_images_dir
    outside_dir = tmp_path / "outside_secure_vault"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_file = outside_dir / "secret.jpg"
    outside_file.write_bytes(b"SENSITIVE_DATA_NOT_FOR_CXR")

    # Attempt to create a symlink inside images dir pointing outside
    symlink_id = "symlink-escape-attempt"
    symlink_path = synthetic_images_dir / f"{symlink_id}.jpg"

    try:
        os.symlink(str(outside_file), str(symlink_path))
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in this test environment / privileges.")

    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        enable_cloud=False,
    )
    res = adapter.resolve_image_stream(symlink_id)

    # Must be blocked by canonical realpath verification
    assert res.status == StorageAccessStatus.INVALID_IDENTIFIER
    assert res.image_stream is None
    assert "traversal" in res.message.lower()


# ── Scenario 5: Identifier Validation ─────────────────────────────────────────

@pytest.mark.parametrize("valid_id", [
    "test-dicom-123",
    "84c6ecc4-fd81e31d-1e879d18-13024211-35de54ad",
    "patient_001",
    "ABC-123_xyz",
    "a" * 64,
])
def test_05_identifier_validation_valid(valid_id):
    """Valid DICOM identifiers pass format checks."""
    is_valid, clean_id = validate_dicom_id(valid_id)
    assert is_valid is True
    assert clean_id == valid_id


@pytest.mark.parametrize("invalid_id", [
    "",
    "   ",
    None,
    "has spaces",
    "id@symbol!",
    "id#hash",
    "id$dollar",
    "a" * 65,  # Exceeds max length
    "semi;colon",
    "<xml>tag</xml>",
])
def test_05_identifier_validation_invalid(invalid_id):
    """Malformed or invalid DICOM identifiers fail format checks."""
    is_valid, msg = validate_dicom_id(invalid_id)
    assert is_valid is False
    assert len(msg) > 0


# ── Scenario 6: Expiry Boundaries at Exactly 1s and 300s ──────────────────────

def test_06_signed_url_expiry_boundaries(mock_storage_client):
    """Signed URL generation accepts exactly 1 second and 300 seconds."""
    adapter = CXRStorageAdapter(
        mock_supabase_client=mock_storage_client,
        enable_cloud=True,
    )

    # Test boundary 1s
    res_1 = adapter.create_signed_access_url(
        dicom_id="test-dicom-001",
        expires_in=MIN_SIGNED_URL_EXPIRY_SECONDS,
        is_authorized=True,
    )
    assert res_1.status == StorageAccessStatus.SUCCESS
    assert res_1.expires_in == 1
    assert res_1.signed_url is not None

    # Test boundary 300s
    res_300 = adapter.create_signed_access_url(
        dicom_id="test-dicom-001",
        expires_in=MAX_SIGNED_URL_EXPIRY_SECONDS,
        is_authorized=True,
    )
    assert res_300.status == StorageAccessStatus.SUCCESS
    assert res_300.expires_in == 300
    assert res_300.signed_url is not None


# ── Scenario 7: Rejection of Expiry Values < 1s or > 300s ─────────────────────

@pytest.mark.parametrize("bad_expiry", [
    0,
    -1,
    -100,
    301,
    600,
    3600,
    "300",  # non-int
    None,
])
def test_07_rejection_of_invalid_expiry(mock_storage_client, bad_expiry):
    """Expiration periods outside [1, 300] seconds are rejected with ERROR."""
    adapter = CXRStorageAdapter(
        mock_supabase_client=mock_storage_client,
        enable_cloud=True,
    )
    res = adapter.create_signed_access_url(
        dicom_id="test-dicom-001",
        expires_in=bad_expiry,  # type: ignore
        is_authorized=True,
    )
    assert res.status == StorageAccessStatus.ERROR
    assert res.signed_url is None
    assert "outside allowed bounds" in res.message


# ── Scenario 8: Mock Signed-URL Request Behavior ──────────────────────────────

def test_08_mock_signed_url_behavior(mock_storage_client):
    """Mock storage client correctly yields signed URL when authorized."""
    adapter = CXRStorageAdapter(
        mock_supabase_client=mock_storage_client,
        enable_cloud=True,
        bucket_name="chest-xrays",
    )
    res = adapter.create_signed_access_url(
        dicom_id="test-dicom-001",
        expires_in=120,
        is_authorized=True,
    )
    assert res.status == StorageAccessStatus.SUCCESS
    assert res.origin == StorageOrigin.SUPABASE_SIGNED_URL
    assert res.dicom_id == "test-dicom-001"
    assert res.signed_url is not None
    assert "mock-supabase.test" in res.signed_url
    assert "raw/test-dicom-001.jpg" in res.signed_url
    assert res.expires_in == 120


# ── Scenario 9: Cloud-Disabled Local Fallback ─────────────────────────────────

def test_09_cloud_disabled_local_fallback(synthetic_images_dir):
    """When cloud is disabled and authorized, seamlessly falls back to local disk stream."""
    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        enable_cloud=False,
    )
    res = adapter.create_signed_access_url(
        dicom_id="test-dicom-synthetic-001",
        expires_in=300,
        is_authorized=True,
    )
    assert res.status == StorageAccessStatus.SUCCESS
    assert res.origin == StorageOrigin.LOCAL_DISK
    assert res.image_stream is not None
    assert res.signed_url is None
    assert "local storage fallback" in res.message


# ── Scenario 10: Cloud Failure Fallback Without Bypassing Auth ─────────────────

def test_10_cloud_failure_fallback_authorized(synthetic_images_dir):
    """When cloud client raises an exception, authorized requests fall back to local disk."""
    failing_client = MagicMock()
    failing_client.storage.from_.side_effect = RuntimeError("Supabase storage timeout 504")

    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        mock_supabase_client=failing_client,
        enable_cloud=True,
    )
    res = adapter.create_signed_access_url(
        dicom_id="test-dicom-synthetic-001",
        expires_in=180,
        is_authorized=True,
    )
    assert res.status == StorageAccessStatus.SUCCESS
    assert res.origin == StorageOrigin.LOCAL_DISK
    assert res.image_stream is not None
    assert "local fallback" in res.message


def test_10b_cloud_failure_not_found_locally(synthetic_images_dir):
    """When cloud client fails and image is missing locally, returns CLOUD_UNAVAILABLE."""
    failing_client = MagicMock()
    failing_client.storage.from_.side_effect = RuntimeError("Supabase storage error")

    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        mock_supabase_client=failing_client,
        enable_cloud=True,
    )
    res = adapter.create_signed_access_url(
        dicom_id="nonexistent-dicom-id",
        expires_in=180,
        is_authorized=True,
    )
    assert res.status == StorageAccessStatus.CLOUD_UNAVAILABLE
    assert res.image_stream is None


# ── Scenario 11: Rejection When is_authorized=False ───────────────────────────

def test_11_unauthorized_access_rejected(synthetic_images_dir, mock_storage_client):
    """Access requests with is_authorized=False are rejected immediately with UNAUTHORIZED."""
    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        mock_supabase_client=mock_storage_client,
        enable_cloud=True,
    )
    res = adapter.create_signed_access_url(
        dicom_id="test-dicom-synthetic-001",
        expires_in=300,
        is_authorized=False,
    )
    assert res.status == StorageAccessStatus.UNAUTHORIZED
    assert res.origin == StorageOrigin.NONE
    assert res.signed_url is None
    assert res.image_stream is None
    assert "authentication and patient-stay linkage verification required" in res.message


# ── Scenario 12: Security Response Headers ────────────────────────────────────

def test_12_security_response_headers(synthetic_images_dir, mock_storage_client):
    """All access results must include strict anti-caching and privacy headers."""
    expected_headers = {
        "Cache-Control": "private, no-store, max-age=0",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }
    assert get_secure_image_headers() == expected_headers

    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        mock_supabase_client=mock_storage_client,
        enable_cloud=True,
    )

    # Check across various outcomes:
    # 1. Success
    res_success = adapter.create_signed_access_url("test-dicom-synthetic-001", 300, is_authorized=True)
    assert res_success.headers == expected_headers

    # 2. Unauthorized
    res_unauth = adapter.create_signed_access_url("test-dicom-synthetic-001", 300, is_authorized=False)
    assert res_unauth.headers == expected_headers

    # 3. Invalid identifier
    res_invalid = adapter.resolve_image_stream("bad/id")
    assert res_invalid.headers == expected_headers

    # 4. Not found
    res_nf = adapter.resolve_image_stream("valid-but-missing-id")
    assert res_nf.headers == expected_headers


# ── Scenario 13: Leakage Hardening (No Raw Bytes or Secrets in Dictionaries) ──

def test_13_leakage_hardening_and_sanitization(synthetic_images_dir):
    """to_dict() and representations never leak raw byte streams or secret credentials."""
    adapter = CXRStorageAdapter(
        local_images_dir=str(synthetic_images_dir),
        enable_cloud=False,
    )
    res = adapter.resolve_image_stream("test-dicom-synthetic-001")
    d = res.to_dict()

    # Must contain boolean flag rather than raw stream or bytes
    assert "has_stream" in d
    assert d["has_stream"] is True
    assert "image_stream" not in d

    # Verify no raw bytes appear in dict representation
    serialized = str(d)
    assert "b'\\xff\\xd8" not in serialized
    assert "secret" not in serialized.lower()
