"""
tests/test_telemetry_adapter.py
=============================================================================
Unit and isolation tests for src/sensor/telemetry_adapter.py and Migration 003.

All tests operate strictly offline using synthetic telemetry payloads.
ZERO cloud network calls to ThingSpeak, Supabase, or external services.
ZERO database executions.
=============================================================================
"""

import datetime
import os
import sys
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.telemetry_adapter import (
    CleanedTelemetryMeasurement,
    IngestedTelemetryPacket,
    IngestionResult,
    IngestionStatus,
    RawTelemetryMeasurement,
    TelemetryIngestionAdapter,
    TelemetryQualityState,
)


@pytest.fixture
def adapter():
    """Provides a clean adapter instance with empty memory buffer."""
    adp = TelemetryIngestionAdapter(freshness_window_seconds=300.0)
    adp.clear_buffer()
    return adp


# -----------------------------------------------------------------------------
# 1. Valid Ingestion & Idempotency Tests
# -----------------------------------------------------------------------------
def test_valid_packet_acceptance(adapter):
    """Verify clean telemetry packet is ingested with VALID quality state."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 101,
        "recorded_at": "2026-10-22T11:58:00Z",  # 2 minutes old -> fresh
        "heart_rate": 78.5,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "risk_status": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.status == IngestionStatus.ACCEPTED
    assert res.packet.quality_state == TelemetryQualityState.VALID
    assert res.packet.cleaned.physiologically_valid is True
    assert res.packet.cleaned.optical_contact is True
    assert res.packet.cleaned.heart_rate == 78.5
    assert res.packet.cleaned.spo2 == 98.0
    assert res.packet.cleaned.ir_ppg == 85000


def test_duplicate_packet_same_channel_entry(adapter):
    """Verify re-transmitting an identical packet returns DUPLICATE_ACCEPTED without mutation."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 102,
        "recorded_at": "2026-10-22T11:58:30Z",
        "heart_rate": 75.0,
        "spo2": 99.0,
        "ir_ppg": 90000,
    }
    res1 = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res1.status == IngestionStatus.ACCEPTED

    # Re-ingest identical packet
    res2 = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res2.status == IngestionStatus.DUPLICATE_ACCEPTED
    assert res2.packet.is_duplicate is True
    assert adapter.buffered_packet_count == 1


def test_identical_entry_id_different_channels(adapter):
    """Verify that identical entry IDs on DIFFERENT channels are both safely accepted."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw1 = {
        "channel_id": 10001,
        "entry_id": 1,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 80.0,
        "spo2": 97.0,
        "ir_ppg": 80000,
    }
    raw2 = {
        "channel_id": 10002,  # Different channel
        "entry_id": 1,        # Same entry ID
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 92.0,
        "spo2": 96.0,
        "ir_ppg": 82000,
    }
    res1 = adapter.ingest_packet(raw1, reference_arrival_time=ref_time)
    res2 = adapter.ingest_packet(raw2, reference_arrival_time=ref_time)

    assert res1.status == IngestionStatus.ACCEPTED
    assert res2.status == IngestionStatus.ACCEPTED
    assert adapter.buffered_packet_count == 2
    assert adapter.get_packet(10001, 1).raw.raw_heart_rate == 80.0
    assert adapter.get_packet(10002, 1).raw.raw_heart_rate == 92.0


def test_conflicting_duplicate_packet_rejected(adapter):
    """Verify conflicting packet for same (channel_id, entry_id) is rejected and original preserved."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw1 = {
        "channel_id": 3490289,
        "entry_id": 105,
        "recorded_at": "2026-10-22T11:58:00Z",
        "heart_rate": 72.0,
        "spo2": 98.0,
    }
    raw2_conflicting = {
        "channel_id": 3490289,
        "entry_id": 105,
        "recorded_at": "2026-10-22T11:58:00Z",
        "heart_rate": 140.0,  # Conflicting HR
        "spo2": 88.0,
    }
    res1 = adapter.ingest_packet(raw1, reference_arrival_time=ref_time)
    assert res1.status == IngestionStatus.ACCEPTED

    res2 = adapter.ingest_packet(raw2_conflicting, reference_arrival_time=ref_time)
    assert res2.status == IngestionStatus.CONFLICT_REJECTED
    assert "conflicting" in res2.error.lower()

    # Verify original record preserved
    saved = adapter.get_packet(3490289, 105)
    assert saved.raw.raw_heart_rate == 72.0


# -----------------------------------------------------------------------------
# 2. Timestamp Normalization & Freshness Tests
# -----------------------------------------------------------------------------
def test_missing_and_malformed_timestamps_rejected(adapter):
    """Verify packets with missing or malformed timestamps are rejected cleanly."""
    bad_ts_1 = {"channel_id": 1, "entry_id": 1, "recorded_at": None, "heart_rate": 70}
    bad_ts_2 = {"channel_id": 1, "entry_id": 2, "recorded_at": "not-a-valid-date", "heart_rate": 70}

    assert adapter.ingest_packet(bad_ts_1).status == IngestionStatus.REJECTED
    assert adapter.ingest_packet(bad_ts_2).status == IngestionStatus.REJECTED


def test_utc_and_offset_aware_normalization(adapter):
    """Verify timezone offsets (e.g. +05:30) are correctly converted to UTC."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    # 17:28:00 at +05:30 is 11:58:00 UTC (2 min before ref_time)
    raw = {
        "channel_id": 3490289,
        "entry_id": 110,
        "recorded_at": "2026-10-22T17:28:00+05:30",
        "heart_rate": 80.0,
        "spo2": 97.0,
        "ir_ppg": 85000,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.recorded_at.tzinfo == datetime.timezone.utc
    assert res.packet.recorded_at.hour == 11
    assert res.packet.recorded_at.minute == 58
    assert res.packet.quality_state == TelemetryQualityState.VALID


def test_stale_packet_flagged(adapter):
    """Verify observation older than 300 seconds is flagged as STALE."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 120,
        "recorded_at": "2026-10-22T11:50:00Z",  # 10 minutes old (> 300s)
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.quality_state == TelemetryQualityState.STALE
    assert "STALE_TELEMETRY" in res.packet.quality_flags
    assert res.packet.cleaned.physiologically_valid is False


# -----------------------------------------------------------------------------
# 3. Sentinel and Physiological Boundary Tests
# -----------------------------------------------------------------------------
def test_sentinel_zero_and_negative_handling(adapter):
    """Verify sentinel values (0, -999) preserve raw inputs but neutralize clean outputs."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 130,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 0.0,    # Sentinel: lead-off / unmeasured
        "spo2": -999.0,       # Sentinel: sensor error
        "ir_ppg": 5000,       # Low IR: no finger
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    # Raw values preserved exactly
    assert res.packet.raw.raw_heart_rate == 0.0
    assert res.packet.raw.raw_spo2 == -999.0
    assert res.packet.raw.raw_ir_ppg == 5000
    # Clean values mapped to None so SQL check constraints are not violated
    assert res.packet.cleaned.heart_rate is None
    assert res.packet.cleaned.spo2 is None
    assert res.packet.cleaned.optical_contact is False
    assert res.packet.quality_state == TelemetryQualityState.INVALID


def test_optical_lead_off_low_ir_handling(adapter):
    """Verify low IR PPG marks optical lead-off without erasing raw readings."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 140,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 82.0,
        "spo2": 96.0,
        "ir_ppg": 12000,  # Below 50,000 contact threshold
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.raw.raw_heart_rate == 82.0
    assert res.packet.raw.raw_spo2 == 96.0
    assert res.packet.cleaned.optical_contact is False
    assert res.packet.quality_state == TelemetryQualityState.SUSPECT
    assert "OPTICAL_LEAD_OFF" in res.packet.quality_flags


def test_extreme_heart_rate_bounds(adapter):
    """Verify impossible HR bounds (>300 bpm) are flagged INVALID."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 150,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 350.0,  # Impossible hardware reading
        "spo2": 97.0,
        "ir_ppg": 80000,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.cleaned.heart_rate is None
    assert res.packet.quality_state == TelemetryQualityState.INVALID
    assert "HR_IMPOSSIBLE_BOUNDS" in res.packet.quality_flags


# -----------------------------------------------------------------------------
# 4. Migration 003 Text & Structure Verification
# -----------------------------------------------------------------------------
def test_migration_003_sql_structure():
    """Verify migration 003 declares composite key, checks for duplicates, and preserves records."""
    migration_path = os.path.join(
        _PROJECT_ROOT, "queries", "migrations", "003_telemetry_ingestion_safety.sql"
    )
    assert os.path.isfile(migration_path), "Migration 003 file must exist"

    with open(migration_path, "r", encoding="utf-8") as f:
        sql = f.read()

    # Verify composite unique key
    assert "uq_channel_entry" in sql
    assert "(channel_id, entry_id)" in sql
    # Verify pre-check for duplicates before constraint
    assert "GROUP BY channel_id, entry_id" in sql
    assert "HAVING COUNT(*) > 1" in sql
    # Verify no automatic deletion
    assert "DELETE FROM" not in sql
    # Verify addition of ir_ppg column
    assert "ADD COLUMN ir_ppg BIGINT" in sql or "ADD COLUMN IF NOT EXISTS ir_ppg" in sql
    # Verify legacy constraint drop
    assert "DROP CONSTRAINT" in sql
    assert "hardware_telemetry_packets_entry_id_key" in sql


# -----------------------------------------------------------------------------
# 5. Stage 3.8.17 Severity & Bradycardia Classification Tests
# -----------------------------------------------------------------------------
def test_bradycardia_40_bpm_triggers_warning_never_normal(adapter):
    """Verify valid accepted bradycardia reading (HR=40, SpO2=98%) triggers SIRS_WARNING and NEVER NORMAL."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 201,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 40.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "risk_status": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.status == IngestionStatus.ACCEPTED
    assert res.packet.quality_state == TelemetryQualityState.VALID
    assert res.packet.cleaned.physiologically_valid is True
    assert res.packet.cleaned.optical_contact is True
    assert res.packet.cleaned.heart_rate == 40.0
    assert res.packet.cleaned.spo2 == 98.0
    assert res.packet.alert_severity == "SIRS_WARNING"
    assert res.packet.alert_severity != "NORMAL"

    # Schema output validation
    d = res.packet.to_dict()
    assert d["heart_rate"] == 40.0
    assert d["spo2"] == 98.0
    assert d["alert_severity"] == "SIRS_WARNING"
    assert d["quality_state"] == "VALID"
    assert d["physiologically_valid"] is True


def test_normal_vitals_72_bpm_retains_normal_severity(adapter):
    """Verify normal baseline vitals (HR=72, SpO2=98%) evaluate to NORMAL severity."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 202,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 72.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "risk_status": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.quality_state == TelemetryQualityState.VALID
    assert res.packet.cleaned.physiologically_valid is True
    assert res.packet.alert_severity == "NORMAL"


def test_tachycardia_warning_110_bpm_preserved(adapter):
    """Verify clinical tachycardia (HR=110, SpO2=98%) preserves SIRS_WARNING behavior."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 203,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 110.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "risk_status": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.quality_state == TelemetryQualityState.VALID
    assert res.packet.cleaned.physiologically_valid is True
    assert res.packet.alert_severity == "SIRS_WARNING"


def test_critical_combined_vitals_135_bpm_90_spo2_preserved(adapter):
    """Verify combined tachycardia and hypoxemia (HR=135, SpO2=90%) preserves CRITICAL behavior."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 204,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 135.0,
        "spo2": 90.0,
        "ir_ppg": 85000,
        "risk_status": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "CRITICAL"


def test_critical_hypoxemia_80_bpm_84_spo2_preserved(adapter):
    """Verify critical isolated hypoxemia (HR=80, SpO2=84%) evaluates to CRITICAL severity."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 205,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 80.0,
        "spo2": 84.0,
        "ir_ppg": 85000,
        "risk_status": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "CRITICAL"


def test_field4_code_2_forces_critical_severity(adapter):
    """Verify Field 4 = 2 (explicit firmware high risk) forces CRITICAL severity regardless of vitals."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 206,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": 2,  # Direct ThingSpeak field4 name
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "CRITICAL"
    assert res.packet.raw.raw_risk_status == 2


def test_field4_code_0_permits_clinical_evaluation(adapter):
    """Verify Field 4 = 0 allows vitals-based warning/normal evaluation rather than masking risk."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw_brady = {
        "channel_id": 3490289,
        "entry_id": 207,
        "recorded_at": "2026-10-22T11:59:00Z",
        "field1": "45.0",  # ThingSpeak field1
        "field2": "97.0",  # ThingSpeak field2
        "field3": "90000", # ThingSpeak field3
        "field4": "0",     # ThingSpeak field4
    }
    res = adapter.ingest_packet(raw_brady, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "SIRS_WARNING"
    assert res.packet.cleaned.heart_rate == 45.0


def test_invalid_and_sentinel_vitals_never_mislabeled_valid_normal(adapter):
    """Verify missing, malformed, or sentinel vitals are never marked physiologically_valid or VALID quality."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

    # 1. Missing vitals
    res_missing = adapter.ingest_packet({
        "channel_id": 3490289,
        "entry_id": 210,
        "recorded_at": "2026-10-22T11:59:00Z",
        "ir_ppg": 85000,
    }, reference_arrival_time=ref_time)
    assert res_missing.is_success is True
    assert res_missing.packet.quality_state == TelemetryQualityState.INVALID
    assert res_missing.packet.cleaned.physiologically_valid is False
    assert res_missing.packet.cleaned.heart_rate is None
    assert res_missing.packet.cleaned.spo2 is None

    # 2. Impossible out of bounds
    res_bounds = adapter.ingest_packet({
        "channel_id": 3490289,
        "entry_id": 211,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 15.0,  # Below HARDWARE_HR_LIMIT_MIN (20.0)
        "spo2": 98.0,
        "ir_ppg": 85000,
    }, reference_arrival_time=ref_time)
    assert res_bounds.is_success is True
    assert res_bounds.packet.quality_state == TelemetryQualityState.INVALID
    assert res_bounds.packet.cleaned.physiologically_valid is False
    assert res_bounds.packet.cleaned.heart_rate is None

    # 3. Sentinel zero
    res_sentinel = adapter.ingest_packet({
        "channel_id": 3490289,
        "entry_id": 212,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 0.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
    }, reference_arrival_time=ref_time)
    assert res_sentinel.is_success is True
    assert res_sentinel.packet.quality_state == TelemetryQualityState.INVALID
    assert res_sentinel.packet.cleaned.physiologically_valid is False
    assert res_sentinel.packet.cleaned.heart_rate is None

    # 4. Malformed timestamp completely rejected
    res_malformed = adapter.ingest_packet({
        "channel_id": 3490289,
        "entry_id": 213,
        "recorded_at": "unparseable-date",
        "heart_rate": 72.0,
        "spo2": 98.0,
    }, reference_arrival_time=ref_time)
    assert res_malformed.status == IngestionStatus.REJECTED
    assert res_malformed.is_success is False
    assert res_malformed.packet is None


# -----------------------------------------------------------------------------
# 6. Stage 3.8.19 Protocol Hardening & Field 4 Risk-Code Handling Tests
# -----------------------------------------------------------------------------
def test_field4_code_1_forces_warning_severity_never_stable(adapter):
    """Verify Field 4 = 1 triggers SIRS_WARNING even when baseline vitals are normal."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 220,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 75.0,  # Normal HR
        "spo2": 98.0,        # Normal SpO2
        "ir_ppg": 85000,
        "field4": 1,         # Firmware Warning Code 1
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.raw.raw_risk_status == 1
    assert res.packet.alert_severity == "SIRS_WARNING"
    assert res.packet.alert_severity != "NORMAL"


def test_field4_code_1_with_critical_vitals_prioritizes_critical(adapter):
    """Verify independent critical vitals override Code 1 warning to establish CRITICAL severity."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 221,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 135.0, # Critical tachycardia
        "spo2": 90.0,        # Critical hypoxemia
        "ir_ppg": 85000,
        "field4": 1,         # Code 1 must not downgrade critical vitals
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "CRITICAL"


def test_field4_code_3_preserves_legacy_vitals_evaluation(adapter):
    """Verify legacy Field 4 = 3 evaluates vitals independently without false assertions."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

    # 1. Normal vitals under code 3
    raw_norm = {
        "channel_id": 3490289,
        "entry_id": 222,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 72.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": 3,
    }
    res_norm = adapter.ingest_packet(raw_norm, reference_arrival_time=ref_time)
    assert res_norm.is_success is True
    assert res_norm.packet.raw.raw_risk_status == 3
    assert res_norm.packet.alert_severity == "NORMAL"

    # 2. Warning vitals under code 3
    raw_warn = {
        "channel_id": 3490289,
        "entry_id": 223,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 110.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": 3,
    }
    res_warn = adapter.ingest_packet(raw_warn, reference_arrival_time=ref_time)
    assert res_warn.is_success is True
    assert res_warn.packet.alert_severity == "SIRS_WARNING"


def test_field4_missing_or_none_evaluates_vitals_never_coerced_to_zero(adapter):
    """Verify missing/null Field 4 preserves None in raw and derives severity from vitals."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

    raw_none = {
        "channel_id": 3490289,
        "entry_id": 224,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 40.0, # Bradycardia
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": None,     # Explicit None
    }
    res = adapter.ingest_packet(raw_none, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.raw.raw_risk_status is None
    assert res.packet.alert_severity == "SIRS_WARNING"


def test_field4_malformed_string_evaluates_vitals_never_coerced_to_zero(adapter):
    """Verify non-numeric Field 4 preserves None and derives severity from vitals."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

    raw_corrupt = {
        "channel_id": 3490289,
        "entry_id": 225,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": "corrupt_data",
    }
    res = adapter.ingest_packet(raw_corrupt, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.raw.raw_risk_status is None
    assert res.packet.alert_severity == "NORMAL"


def test_field4_unknown_integer_code_5_evaluates_vitals(adapter):
    """Verify custom/unknown integer code 5 preserves 5 in raw and safely derives severity from vitals."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)

    raw_custom = {
        "channel_id": 3490289,
        "entry_id": 226,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 115.0,
        "spo2": 97.0,
        "ir_ppg": 85000,
        "field4": 5,
    }
    res = adapter.ingest_packet(raw_custom, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.raw.raw_risk_status == 5
    assert res.packet.alert_severity == "SIRS_WARNING"
