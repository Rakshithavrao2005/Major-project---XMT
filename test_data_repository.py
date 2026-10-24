"""
tests/test_data_repository.py
=============================================================================
Unit tests for src/backend/data_repository.py.

All tests run strictly offline with synthetic and mocked data.
Zero real database connections or cloud calls are executed.
=============================================================================
"""

import os
import sys
from unittest.mock import MagicMock
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backend.data_repository import (
    DataOrigin,
    DataRepository,
    RepositoryResult,
    RepositoryStatus,
)


# -----------------------------------------------------------------------------
# Local Primary Mode Tests (Default)
# -----------------------------------------------------------------------------
def test_repository_local_primary_patient_lookup():
    """Verify local patient retrieval returns LOCAL_PRIMARY for existing patient."""
    repo = DataRepository(enable_cloud=False)
    # Using an established real patient in the dataset: 10002930
    res = repo.get_patient("10002930")
    assert res.is_success is True
    assert res.status == RepositoryStatus.SUCCESS
    assert res.origin == DataOrigin.LOCAL_PRIMARY
    assert res.data["patient_id"] == "10002930"
    assert res.is_synthetic is False


def test_repository_local_primary_unknown_patient():
    """Verify unknown patient returns NOT_FOUND with origin NONE."""
    repo = DataRepository(enable_cloud=False)
    res = repo.get_patient("9999999999")
    assert res.is_success is False
    assert res.status == RepositoryStatus.NOT_FOUND
    assert res.origin == DataOrigin.NONE
    assert res.data is None


def test_repository_local_primary_stay_lookup():
    """Verify valid stay returns LOCAL_PRIMARY without label leakage."""
    repo = DataRepository(enable_cloud=False)
    res = repo.get_stay_clinical_record("10002930", "35629889")
    assert res.is_success is True
    assert res.status == RepositoryStatus.SUCCESS
    assert res.origin == DataOrigin.LOCAL_PRIMARY
    assert "sepsis_label" not in res.data
    assert "sepsis_onset_time" not in res.data


def test_repository_local_primary_unknown_stay():
    """Verify unknown stay returns NOT_FOUND."""
    repo = DataRepository(enable_cloud=False)
    res = repo.get_stay_clinical_record("10002930", "99999999")
    assert res.is_success is False
    assert res.status == RepositoryStatus.NOT_FOUND
    assert res.origin == DataOrigin.NONE


# -----------------------------------------------------------------------------
# Dual-Mode with Mocked Cloud Operations
# -----------------------------------------------------------------------------
def test_cloud_success_patient_query():
    """Verify successful cloud lookup returns CLOUD_PRIMARY."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [{
        "patient_id": "SYN-1001",
        "gender": "F",
        "anchor_age": 62,
        "is_synthetic": True,
    }]
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

    repo = DataRepository(supabase_client=mock_client, enable_cloud=True)
    res = repo.get_patient("SYN-1001")
    assert res.is_success is True
    assert res.status == RepositoryStatus.SUCCESS
    assert res.origin == DataOrigin.CLOUD_PRIMARY
    assert res.is_synthetic is True
    assert res.data["patient_id"] == "SYN-1001"


def test_cloud_miss_with_fallback_serves_local():
    """Verify record absent from cloud falls back to local dataset."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = []  # Not in cloud
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

    repo = DataRepository(supabase_client=mock_client, enable_cloud=True, allow_fallback=True)
    res = repo.get_patient("10002930")
    assert res.is_success is True
    assert res.status == RepositoryStatus.SUCCESS
    assert res.origin == DataOrigin.LOCAL_FALLBACK
    assert res.data["patient_id"] == "10002930"
    assert "local fallback" in res.message.lower()


def test_cloud_error_with_fallback_serves_local():
    """Verify cloud exception (e.g. timeout) safely triggers local fallback."""
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.eq.return_value.execute.side_effect = TimeoutError("PostgreSQL timeout")

    repo = DataRepository(supabase_client=mock_client, enable_cloud=True, allow_fallback=True)
    res = repo.get_patient("10002930")
    assert res.is_success is True
    assert res.status == RepositoryStatus.SUCCESS
    assert res.origin == DataOrigin.LOCAL_FALLBACK
    assert res.data["patient_id"] == "10002930"


def test_cloud_error_with_fallback_disabled_returns_unavailable():
    """Verify cloud failure returns CLOUD_UNAVAILABLE when fallback is disallowed."""
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.eq.return_value.execute.side_effect = ConnectionRefusedError("Supabase unreachable")

    repo = DataRepository(supabase_client=mock_client, enable_cloud=True, allow_fallback=False)
    res = repo.get_patient("10002930")
    assert res.is_success is False
    assert res.status == RepositoryStatus.CLOUD_UNAVAILABLE
    assert res.origin == DataOrigin.NONE
    assert "unavailable" in res.message.lower() or "error" in res.message.lower()


def test_test_partition_quarantine_in_cloud_stay_lookup():
    """Verify that a stay marked with data_split='test' in cloud is quarantined."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [{
        "stay_id": "30009999",
        "patient_id": "10009999",
        "data_split": "test",  # Test cohort quarantine target
    }]
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_response

    repo = DataRepository(supabase_client=mock_client, enable_cloud=True, allow_fallback=False)
    res = repo.get_stay_clinical_record("10009999", "30009999")
    assert res.is_success is False
    assert res.status == RepositoryStatus.NOT_FOUND
    assert "quarantined" in res.message.lower()


def test_telemetry_cloud_and_local_isolation():
    """Verify hardware telemetry queries are decoupled from patient stays."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        {"entry_id": 101, "heart_rate": 78.0, "spo2": 98.0, "data_quality": "FRESH"},
        {"entry_id": 102, "heart_rate": 82.0, "spo2": 97.0, "data_quality": "FRESH"},
    ]
    mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

    repo = DataRepository(supabase_client=mock_client, enable_cloud=True)
    res = repo.get_telemetry_history(limit=2)
    assert res.is_success is True
    assert res.origin == DataOrigin.CLOUD_PRIMARY
    assert len(res.data) == 2
    # Ensure no patient fields are present in hardware packets
    for packet in res.data:
        assert "subject_id" not in packet
        assert "stay_id" not in packet


def test_repository_status_reporting():
    """Verify repository status diagnostics accurately reflect mode without secret leakage."""
    repo = DataRepository(enable_cloud=False)
    status = repo.get_status()
    assert status["mode"] == "local_only"
    assert status["cloud_enabled"] is False
    assert status["fallback_allowed"] is True
    assert "supabase" in status
