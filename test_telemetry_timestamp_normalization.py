"""
tests/test_telemetry_timestamp_normalization.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive tests for telemetry timestamp normalization, freshness evaluation,
and safe Overview rendering across:
1. Timezone-aware Python datetime.
2. Naive Python datetime.
3. ISO-8601 UTC timestamp string.
4. ISO-8601 timestamp with a timezone offset.
5. Valid dictionary-wrapped timestamp matching actual schema.
6. Missing timestamp (None, empty string, None in dict).
7. Malformed timestamp string.
8. Unsupported dictionary structure.
9. Fresh and stale policy boundaries (300.0s cutoff).
10. Overview page rendering safely when freshness is unavailable.
"""

from __future__ import annotations

import datetime
import os
import sys
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.telemetry_poller import (
    get_telemetry_freshness,
    normalize_telemetry_timestamp,
)
import dashboard.pages.overview as overview_module


class TestTelemetryTimestampNormalization:
    """Tests for normalize_telemetry_timestamp and get_telemetry_freshness."""

    # 1. Timezone-aware Python datetime
    def test_01_timezone_aware_datetime(self):
        dt_aware = datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.timezone.utc)
        dt_norm, status = normalize_telemetry_timestamp(dt_aware)
        assert status == "valid"
        assert dt_norm == dt_aware
        assert dt_norm.tzinfo == datetime.timezone.utc

        f = get_telemetry_freshness(dt_aware, current_time=dt_aware)
        assert f["status"] == "FRESH"
        assert f["is_stale"] is False
        assert f["elapsed_seconds"] == 0.0

    # 2. Naive Python datetime
    def test_02_naive_datetime(self):
        dt_naive = datetime.datetime(2026, 10, 16, 12, 0, 0)
        dt_norm, status = normalize_telemetry_timestamp(dt_naive)
        assert status == "valid"
        assert dt_norm.tzinfo == datetime.timezone.utc
        assert dt_norm.hour == 12

        now_aware = datetime.datetime(2026, 10, 16, 12, 1, 0, tzinfo=datetime.timezone.utc)
        f = get_telemetry_freshness(dt_naive, current_time=now_aware)
        assert f["status"] == "FRESH"
        assert f["is_stale"] is False
        assert f["elapsed_seconds"] == 60.0

    # 3. ISO-8601 UTC timestamp string
    def test_03_iso8601_utc_string(self):
        iso_utc = "2026-10-16T12:00:00Z"
        dt_norm, status = normalize_telemetry_timestamp(iso_utc)
        assert status == "valid"
        assert dt_norm.tzinfo == datetime.timezone.utc
        assert dt_norm.year == 2026
        assert dt_norm.month == 10
        assert dt_norm.day == 16
        assert dt_norm.hour == 12

        now = datetime.datetime(2026, 10, 16, 12, 2, 0, tzinfo=datetime.timezone.utc)
        f = get_telemetry_freshness(iso_utc, current_time=now)
        assert f["status"] == "FRESH"
        assert f["is_stale"] is False
        assert f["elapsed_seconds"] == 120.0

    # 4. ISO-8601 timestamp with a timezone offset
    def test_04_iso8601_with_timezone_offset(self):
        # 17:30 at UTC+05:30 is 12:00:00 UTC
        iso_offset = "2026-10-16T17:30:00+05:30"
        dt_norm, status = normalize_telemetry_timestamp(iso_offset)
        assert status == "valid"
        assert dt_norm.tzinfo == datetime.timezone.utc
        assert dt_norm.hour == 12
        assert dt_norm.minute == 0

        now = datetime.datetime(2026, 10, 16, 12, 1, 30, tzinfo=datetime.timezone.utc)
        f = get_telemetry_freshness(iso_offset, current_time=now)
        assert f["status"] == "FRESH"
        assert f["is_stale"] is False
        assert f["elapsed_seconds"] == 90.0

    # 5. Valid dictionary-wrapped timestamp matching actual schema
    def test_05_valid_dictionary_wrapped_timestamp(self):
        tel_dict = {
            "status": "success",
            "timestamp": "2026-09-30 09:08:43+00:00",
            "entry_id": 290,
            "connected": True,
        }
        dt_norm, status = normalize_telemetry_timestamp(tel_dict)
        assert status == "valid"
        assert dt_norm.year == 2026
        assert dt_norm.month == 9
        assert dt_norm.day == 30
        assert dt_norm.hour == 9
        assert dt_norm.minute == 8

        now = datetime.datetime(2026, 9, 30, 9, 10, 43, tzinfo=datetime.timezone.utc)
        f = get_telemetry_freshness(tel_dict, current_time=now)
        assert f["status"] == "FRESH"
        assert f["is_stale"] is False
        assert f["elapsed_seconds"] == 120.0

        # Also test structured dict e.g. {"year": 2026, "month": 10, "day": 16}
        structured = {"year": 2026, "month": 10, "day": 16, "hour": 8, "minute": 0}
        dt_struct, status_struct = normalize_telemetry_timestamp(structured)
        assert status_struct == "valid"
        assert dt_struct.hour == 8

    # 6. Missing timestamp
    def test_06_missing_timestamp(self):
        f_none = get_telemetry_freshness(None)
        assert f_none["status"] == "MISSING"
        assert f_none["is_stale"] is True
        assert f_none["elapsed_seconds"] is None

        f_empty = get_telemetry_freshness("")
        assert f_empty["status"] == "MISSING"
        assert f_empty["is_stale"] is True

        f_dict_none = get_telemetry_freshness({"timestamp": None})
        assert f_dict_none["status"] == "MISSING"
        assert f_dict_none["is_stale"] is True

    # 7. Malformed timestamp string
    def test_07_malformed_timestamp_string(self):
        f_bad_str = get_telemetry_freshness("invalid-timestamp-string-not-a-date")
        assert f_bad_str["status"] == "UNAVAILABLE"
        assert f_bad_str["is_stale"] is True
        assert f_bad_str["elapsed_seconds"] is None
        assert "Telemetry freshness unavailable" in f_bad_str["description"]

    # 8. Unsupported dictionary structure
    def test_08_unsupported_dictionary_structure(self):
        f_bad_dict = get_telemetry_freshness({"sensor": "MAX30102", "reading_id": 999})
        assert f_bad_dict["status"] == "UNAVAILABLE"
        assert f_bad_dict["is_stale"] is True
        assert f_bad_dict["elapsed_seconds"] is None
        assert "Telemetry freshness unavailable" in f_bad_dict["description"]

    # 9. Fresh and stale policy boundaries (300.0s cutoff)
    def test_09_fresh_and_stale_policy_boundaries(self):
        now = datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.timezone.utc)
        cutoff = 300.0

        # Exactly at boundary: elapsed == 300.0s -> FRESH
        t_boundary = now - datetime.timedelta(seconds=300.0)
        f_boundary = get_telemetry_freshness(t_boundary, current_time=now, stale_threshold_seconds=cutoff)
        assert f_boundary["status"] == "FRESH"
        assert f_boundary["is_stale"] is False
        assert f_boundary["elapsed_seconds"] == 300.0

        # Just past boundary: elapsed == 300.1s -> STALE
        t_stale = now - datetime.timedelta(seconds=300.1)
        f_stale = get_telemetry_freshness(t_stale, current_time=now, stale_threshold_seconds=cutoff)
        assert f_stale["status"] == "STALE"
        assert f_stale["is_stale"] is True
        assert f_stale["elapsed_seconds"] == pytest.approx(300.1, abs=0.01)

    # 10. Overview page rendering safely when freshness is unavailable
    def test_10_overview_page_rendering_safely_when_freshness_unavailable(self, monkeypatch):
        """Overview page must render without crashing even if telemetry is malformed or unavailable."""
        # Mock get_latest_sensor_telemetry to return malformed dictionary
        monkeypatch.setattr(
            overview_module,
            "get_latest_sensor_telemetry",
            lambda fetch_remote=False: {"status": "error", "timestamp": "corrupt-non-date"},
        )

        # Calling render_overview_page must not raise AttributeError or any exception
        try:
            overview_module.render_overview_page()
        except Exception as e:
            pytest.fail(f"render_overview_page raised unexpected exception: {e}")
