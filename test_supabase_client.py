"""
tests/test_supabase_client.py
=============================================================================
Unit and isolation tests for src/backend/supabase_client.py.

All tests operate strictly offline using mock objects and monkeypatching.
Zero external network calls or cloud connections are made.
=============================================================================
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backend.supabase_client import (
    get_configured_credentials,
    get_supabase_client,
    get_supabase_status,
    is_supabase_enabled,
    reset_supabase_client,
)


@pytest.fixture(autouse=True)
def clean_supabase_environment(monkeypatch):
    """Ensure clean environment and reset singleton state before each test."""
    monkeypatch.delenv("USE_SUPABASE_DB", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    reset_supabase_client()
    yield
    reset_supabase_client()


def test_disabled_by_default():
    """Verify Supabase integration defaults to disabled when unset."""
    assert is_supabase_enabled() is False
    client = get_supabase_client()
    assert client is None
    status = get_supabase_status()
    assert status["enabled"] is False
    assert status["status"] == "disabled"
    assert status["client_ready"] is False


def test_no_network_on_import():
    """Verify that importing supabase_client does not initialize network sockets."""
    from src.backend import supabase_client
    assert supabase_client._SUPABASE_CLIENT_INSTANCE is None


def test_missing_credentials_handled_gracefully(monkeypatch):
    """Verify enabled flag with missing credentials returns None and does not crash."""
    monkeypatch.setenv("USE_SUPABASE_DB", "true")
    # SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are unset
    assert is_supabase_enabled() is True
    client = get_supabase_client()
    assert client is None

    status = get_supabase_status()
    assert status["enabled"] is True
    assert status["configured"] is False
    assert status["status"] == "unconfigured"
    assert "missing" in status["message"].lower()


def test_successful_mocked_client_initialization(monkeypatch):
    """Verify that valid configuration initializes client via mocked create_client."""
    monkeypatch.setenv("USE_SUPABASE_DB", "true")
    monkeypatch.setenv("SUPABASE_URL", "https://xyzcompany.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret-service-role-key-12345")

    mock_client = MagicMock()
    with patch("supabase.create_client", return_value=mock_client) as mock_create:
        client = get_supabase_client()
        assert client is mock_client
        mock_create.assert_called_once_with(
            "https://xyzcompany.supabase.co",
            "secret-service-role-key-12345",
        )

        status = get_supabase_status()
        assert status["enabled"] is True
        assert status["configured"] is True
        assert status["client_ready"] is True
        assert status["status"] == "ready"
        assert status["endpoint_host"] == "https://xyzcompany.supabase.co"


def test_credentials_never_leaked_in_status_or_repr(monkeypatch):
    """Verify that secrets are never present in status dictionaries or error strings."""
    secret_key = "super-secret-service-key-do-not-leak"
    monkeypatch.setenv("USE_SUPABASE_DB", "true")
    monkeypatch.setenv("SUPABASE_URL", "https://xyzcompany.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", secret_key)

    status = get_supabase_status()
    status_str = str(status)
    assert secret_key not in status_str
    assert "service_role" not in status_str or "service-role authorization" in status_str


def test_initialization_exception_sanitized_and_safe(monkeypatch):
    """Verify exceptions during create_client are caught and redacted without crashing."""
    secret_key = "secret-key-to-redact"
    monkeypatch.setenv("USE_SUPABASE_DB", "true")
    monkeypatch.setenv("SUPABASE_URL", "https://xyzcompany.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", secret_key)

    with patch("supabase.create_client", side_effect=RuntimeError(f"Connection error with {secret_key}")):
        client = get_supabase_client()
        assert client is None

        status = get_supabase_status()
        assert status["client_ready"] is False
        assert status["status"] == "initialization_failed"
        # Ensure raw secret is not present in message
        assert secret_key not in status["message"]
        assert "[REDACTED_KEY]" in status["message"]
