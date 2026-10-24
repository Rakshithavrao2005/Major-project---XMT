"""
tests/test_field4_backend_contract.py
=============================================================================
Stage 3.8.19: Comprehensive Cross-Layer Field 4 Risk-Code Contract Test Suite.

Validates the backward-compatible Field 4 risk-code contract across:
1. src/sensor/thingspeak_bridge.py
2. src/sensor/live_session.py
3. src/sensor/telemetry_adapter.py
4. src/sensor/alert_engine.py

Guarantees:
- Code 1 is ALWAYS mapped to WARNING across all layers, and NEVER STABLE.
- Missing, NaN, and malformed Field 4 values evaluate strictly to UNKNOWN,
  NEVER silently defaulting to 0 or NORMAL.
- Legacy Code 0 preserves vital-sign evaluation (e.g. bradycardia < 50 bpm -> warning).
- Code 2 preserves CRITICAL precedence.
- Stale and invalid telemetry safeguards remain intact.
=============================================================================
"""

import copy
import datetime
import os
import sys
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.thingspeak_bridge import ThingSpeakSessionBridge
from src.sensor.live_session import get_latest_sensor_telemetry
from src.sensor.telemetry_adapter import (
    TelemetryIngestionAdapter,
    TelemetryQualityState,
    IngestionStatus,
)
from src.sensor.alert_engine import (
    SensorAlertEngine,
    AlertCategory,
    AlertSeverity,
)


@pytest.fixture
def mock_session():
    return {"channel_id": 3490289, "device_id": "ESP32-MAX30102-01"}


@pytest.fixture
def bridge():
    """Dummy bridge instance for pure normalization calls without network."""
    class DummyRepo:
        pass
    class DummyService:
        pass
    return ThingSpeakSessionBridge(
        channel_id=3490289,
        session_repository=DummyRepo(),
        session_service=DummyService(),
    )


@pytest.fixture
def adapter():
    adp = TelemetryIngestionAdapter(freshness_window_seconds=300.0)
    adp.clear_buffer()
    return adp


@pytest.fixture
def alert_engine():
    return SensorAlertEngine()


# =============================================================================
# 1. ThingSpeak Bridge Normalization Tests
# =============================================================================

def test_bridge_code_1_evaluates_to_warning_never_stable(bridge, mock_session):
    """Verify Bridge maps Field 4 = 1 to WARNING and NEVER STABLE."""
    raw_feed = {"entry_id": 100, "field1": "75", "field2": "98", "field3": "85000", "field4": "1"}
    norm = bridge.normalize_feed_packet(raw_feed, mock_session)
    assert norm["risk_status"] == 1
    assert norm["firmware_risk_status_text"] == "WARNING"
    assert norm["firmware_risk_status_text"] != "STABLE"


def test_bridge_missing_and_malformed_field4_evaluates_unknown(bridge, mock_session):
    """Verify Bridge maps missing, None, and non-numeric Field 4 to UNKNOWN, never NORMAL."""
    # Missing
    norm_missing = bridge.normalize_feed_packet({"entry_id": 101, "field1": "75"}, mock_session)
    assert norm_missing["risk_status"] is None
    assert norm_missing["firmware_risk_status_text"] == "UNKNOWN"

    # None
    norm_none = bridge.normalize_feed_packet({"entry_id": 102, "field4": None}, mock_session)
    assert norm_none["risk_status"] is None
    assert norm_none["firmware_risk_status_text"] == "UNKNOWN"

    # String / Corrupt
    norm_corrupt = bridge.normalize_feed_packet({"entry_id": 103, "field4": "corrupt_data"}, mock_session)
    assert norm_corrupt["risk_status"] is None
    assert norm_corrupt["firmware_risk_status_text"] == "UNKNOWN"


def test_bridge_codes_0_2_3_and_custom(bridge, mock_session):
    """Verify Bridge maps codes 0, 2, 3, and custom codes accurately."""
    assert bridge.normalize_feed_packet({"entry_id": 104, "field4": "0"}, mock_session)["firmware_risk_status_text"] == "NORMAL"
    assert bridge.normalize_feed_packet({"entry_id": 105, "field4": "2"}, mock_session)["firmware_risk_status_text"] == "HIGH_RISK"
    assert bridge.normalize_feed_packet({"entry_id": 106, "field4": "3"}, mock_session)["firmware_risk_status_text"] == "CRITICAL_VITALS"
    assert bridge.normalize_feed_packet({"entry_id": 107, "field4": "5"}, mock_session)["firmware_risk_status_text"] == "CUSTOM_RISK_5"


# =============================================================================
# 2. Live Session Status Representation Tests
# =============================================================================

def test_live_session_code_1_displays_warning_never_stable(monkeypatch):
    """Verify get_latest_sensor_telemetry maps code 1 to WARNING and never STABLE."""
    synthetic_df = pd.DataFrame([{
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "entry_id": 201,
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "risk_status": 1,
        "valid_hr": True,
        "valid_spo2": True,
        "finger_detected": True,
        "physiologically_valid": True,
    }])
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: synthetic_df)

    res = get_latest_sensor_telemetry(fetch_remote=False)
    assert res["status"] == "success"
    assert res["risk_status_code"] == 1
    assert "WARNING" in res["firmware_status"]
    assert "STABLE" not in res["firmware_status"]


def test_live_session_missing_field4_displays_unknown_never_normal(monkeypatch):
    """Verify get_latest_sensor_telemetry maps NaN/missing Field 4 to UNKNOWN, never NORMAL (Status 0)."""
    synthetic_df = pd.DataFrame([{
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "entry_id": 202,
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "risk_status": None,  # Missing Field 4
        "valid_hr": True,
        "valid_spo2": True,
        "finger_detected": True,
        "physiologically_valid": True,
    }])
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: synthetic_df)

    res = get_latest_sensor_telemetry(fetch_remote=False)
    assert res["status"] == "success"
    assert res["risk_status_code"] is None
    assert "UNKNOWN" in res["firmware_status"]
    assert "NORMAL" not in res["firmware_status"]


# =============================================================================
# 3. Telemetry Adapter Protocol v1 & v2 Compatibility Tests
# =============================================================================

def test_adapter_code_0_with_bradycardia_yields_warning(adapter):
    """Scenario 2: Code 0 + HR 40, SpO2 98% -> SIRS_WARNING (never NORMAL)."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 301,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 40.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "SIRS_WARNING"
    assert res.packet.alert_severity != "NORMAL"


def test_adapter_code_0_with_tachycardia_yields_warning(adapter):
    """Scenario 3: Code 0 + HR 110, SpO2 97% -> SIRS_WARNING (never NORMAL)."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 302,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 110.0,
        "spo2": 97.0,
        "ir_ppg": 85000,
        "field4": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "SIRS_WARNING"


def test_adapter_code_0_with_normal_vitals_yields_normal(adapter):
    """Scenario 1: Code 0 + HR 75, SpO2 98% -> NORMAL."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 303,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "NORMAL"


def test_adapter_code_0_with_critical_hypoxemia_yields_critical(adapter):
    """Scenario 5: Code 0 + HR 80, SpO2 84% -> CRITICAL."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 304,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 80.0,
        "spo2": 84.0,
        "ir_ppg": 85000,
        "field4": 0,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "CRITICAL"


def test_adapter_code_2_establishes_critical_precedence(adapter):
    """Scenario 4 & Precedence: Code 2 -> CRITICAL regardless of baseline vitals."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 305,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": 2,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "CRITICAL"


def test_adapter_code_1_establishes_warning_never_stable(adapter):
    """Protocol v2: Code 1 -> SIRS_WARNING even with normal baselines."""
    ref_time = datetime.datetime(2026, 10, 22, 12, 0, 0, tzinfo=datetime.timezone.utc)
    raw = {
        "channel_id": 3490289,
        "entry_id": 306,
        "recorded_at": "2026-10-22T11:59:00Z",
        "heart_rate": 75.0,
        "spo2": 98.0,
        "ir_ppg": 85000,
        "field4": 1,
    }
    res = adapter.ingest_packet(raw, reference_arrival_time=ref_time)
    assert res.is_success is True
    assert res.packet.alert_severity == "SIRS_WARNING"
    assert res.packet.alert_severity != "NORMAL"


# =============================================================================
# 4. Sensor Alert Engine Cross-Layer Verification
# =============================================================================

def test_alert_engine_code_1_triggers_warning_status(alert_engine):
    """Verify Alert Engine triggers FIRMWARE_WARNING_STATUS on risk_status = 1."""
    now = datetime.datetime.now(datetime.timezone.utc)
    reading = {
        "heart_rate": 75.0,  # Normal baselines
        "spo2": 98.0,
        "ir_ppg": 85000,
        "risk_status": 1,
    }
    alerts = alert_engine.evaluate_reading(reading, observation_time=now, current_time=now)
    rule_names = [a.rule_name for a in alerts]
    assert "FIRMWARE_WARNING_STATUS" in rule_names
    warning_alert = next(a for a in alerts if a.rule_name == "FIRMWARE_WARNING_STATUS")
    assert warning_alert.severity == AlertSeverity.WARNING
    assert warning_alert.category == AlertCategory.PHYSIOLOGICAL_RISK
