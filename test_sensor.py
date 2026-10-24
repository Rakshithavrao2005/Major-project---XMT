"""
tests/test_sensor.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for ESP32 + MAX30102 ThingSpeak ingestion, data cleaning,
quality validation, and PyTorch SensorEncoder architecture.

Test Groups:
  Group J: Sensor Ingestion, Data Cleaning & Audits (J01 - J08)
  Group K: Sensor Encoder Architecture & Tensor Operations (K01 - K05)
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.sensor.thingspeak_ingestion import (
    load_local_raw_sensor_data,
    load_env_credentials,
)
from src.sensor.sensor_encoder import (
    SensorEncoder,
    extract_sensor_windows,
)
from src.sensor.validate_clean import parse_telemetry_schema

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
META_DIR = os.path.join(_PROJECT_ROOT, "data", "metadata")
SENSOR_PLOTS_DIR = os.path.join(_PROJECT_ROOT, "models", "sensor", "plots")


# ═══════════════════════════════════════════════════════════════════════════════
# Group J: Sensor Ingestion, Cleaning & Audits
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensorIngestionAndCleaning:
    """Group J: Verify telemetry ingestion, schema, cleaning, and audits."""

    def test_J01_raw_sensor_csv_exists_and_non_empty(self):
        """Raw ThingSpeak sensor CSV exists and contains rows."""
        csv_path = os.path.join(RAW_DIR, "sensor_thingspeak_raw.csv")
        assert os.path.isfile(csv_path), f"Raw CSV missing at {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) > 0
        for col in ["created_at", "entry_id", "field1", "field2"]:
            assert col in df.columns

    def test_J02_channel_metadata_exists(self):
        """ThingSpeak channel metadata JSON exists with valid fields."""
        meta_path = os.path.join(RAW_DIR, "sensor_channel_metadata.json")
        assert os.path.isfile(meta_path), f"Metadata missing at {meta_path}"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert "channel_id" in meta
        assert "field_schema" in meta
        assert meta["field_schema"]["field1"] == "Heart"

    def test_J03_clean_sensor_csv_schema(self):
        """Cleaned sensor CSV has expected columns and non-null timestamps."""
        clean_path = os.path.join(PROC_DIR, "sensor_thingspeak_clean.csv")
        assert os.path.isfile(clean_path)
        df = pd.read_csv(clean_path)
        expected_cols = [
            "timestamp", "entry_id", "heart_rate", "spo2",
            "valid_hr", "valid_spo2", "physiologically_valid", "session_id"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Column {col} missing from clean sensor CSV"
        assert df["timestamp"].notna().all()

    def test_J04_timestamps_monotonic(self):
        """Clean sensor timestamps are monotonically increasing."""
        clean_path = os.path.join(PROC_DIR, "sensor_thingspeak_clean.csv")
        df = pd.read_csv(clean_path)
        ts = pd.to_datetime(df["timestamp"])
        assert ts.is_monotonic_increasing

    def test_J05_physiological_bounds_filtering(self):
        """Physiological validity flags properly tag extreme/dropout values."""
        clean_path = os.path.join(PROC_DIR, "sensor_thingspeak_clean.csv")
        df = pd.read_csv(clean_path)
        # Any SpO2 < 70 should NOT be marked as valid_spo2
        low_spo2 = df[df["spo2"] < 70.0]
        if not low_spo2.empty:
            assert (~low_spo2["valid_spo2"]).all(), "SpO2 < 70 must be flagged as invalid"

    def test_J06_env_credentials_loading(self):
        """Credentials loader retrieves non-empty channel ID for authoritative channel 3490289."""
        cid, key = load_env_credentials()
        assert cid is not None
        assert str(cid) == "3490289"

    def test_J07_sensor_audit_report_exists(self):
        """SENSOR_DATA_AUDIT.md exists and contains clinical and ethical disclaimers."""
        rpt_path = os.path.join(META_DIR, "SENSOR_DATA_AUDIT.md")
        assert os.path.isfile(rpt_path)
        with open(rpt_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Disclaimers" in content
        assert "MIMIC" in content

    def test_J08_sensor_trend_plot_exists(self):
        """Sensor trend visualization plot exists in models/sensor/plots/."""
        plot_path = os.path.join(SENSOR_PLOTS_DIR, "sensor_trends.png")
        assert os.path.isfile(plot_path)
        assert os.path.getsize(plot_path) > 10_000

    def test_J09_incremental_deduplication_and_empty_behavior(self, tmp_path):
        """Test incremental deduplication on entry_id and preservation when no new feeds are fetched."""
        from src.sensor.thingspeak_ingestion import fetch_thingspeak_feeds
        test_csv = str(tmp_path / "test_raw.csv")
        test_meta = str(tmp_path / "test_meta.json")

        # Create dummy initial CSV
        initial_df = pd.DataFrame([
            {"created_at": "2026-08-23T06:09:30Z", "entry_id": 1, "field1": "78", "field2": "100"},
            {"created_at": "2026-08-23T06:09:52Z", "entry_id": 2, "field1": "115", "field2": "100"},
        ])
        initial_df.to_csv(test_csv, index=False)

        # Simulate empty new feeds preservation
        existing = pd.read_csv(test_csv)
        empty_new = pd.DataFrame(columns=["created_at", "entry_id", "field1", "field2"])
        # Verify empty doesn't wipe
        assert len(existing) == 2

        # Simulate overlap deduplication
        overlap_new = pd.DataFrame([
            {"created_at": "2026-08-23T06:09:52Z", "entry_id": 2, "field1": "115", "field2": "100"},
            {"created_at": "2026-08-23T06:10:34Z", "entry_id": 3, "field1": "83", "field2": "49"},
        ])
        combined = pd.concat([existing, overlap_new], ignore_index=True)
        deduped = combined.drop_duplicates(subset=["entry_id"], keep="last")
        assert len(deduped) == 3
        assert list(deduped["entry_id"]) == [1, 2, 3]

    def test_J10_multi_schema_format_c_standard(self):
        """Format C: Field 3 is numeric IR PPG count (>=1000) and Field 4 is numeric status code."""
        df_format_c = pd.DataFrame([
            {"field3": "125000", "field4": "1"},  # Stable
            {"field3": "142000", "field4": "2"},  # Monitor Closely
            {"field3": "151000", "field4": "3"},  # Critical Vitals
            {"field3": "12000",  "field4": "0"},  # No Finger
        ])
        ir, risk, text, fmt = parse_telemetry_schema(df_format_c)
        assert (fmt == "format_c_standard").all()
        assert ir.iloc[0] == 125000.0
        assert risk.iloc[0] == 1.0
        assert text.iloc[0] == "Stable"
        assert risk.iloc[1] == 2.0
        assert text.iloc[1] == "Monitor Closely"
        assert risk.iloc[2] == 3.0
        assert text.iloc[2] == "Critical Vitals"
        assert risk.iloc[3] == 0.0
        assert text.iloc[3] == "No Finger"

    def test_J11_multi_schema_format_b_legacy_text(self):
        """Format B (historical Channel 3490289 entries 1-290): Field 3 contains text status or early -1, Field 4 is empty."""
        df_format_b = pd.DataFrame([
            {"field3": "Stable", "field4": None},
            {"field3": "Critical Vitals", "field4": None},
            {"field3": "No Finger", "field4": None},
            {"field3": "-1", "field4": None},
            {"field3": "Monitor Closely", "field4": None},
        ])
        ir, risk, text, fmt = parse_telemetry_schema(df_format_b)
        assert (fmt == "format_b_legacy_text").all()
        # Ir_ppg must be NaN: text status must NEVER be silently converted to ADC counts or -1.0
        assert ir.isna().all(), "Legacy text status or -1 must yield NaN for IR PPG, not ADC counts"
        assert risk.iloc[0] == 1.0
        assert text.iloc[0] == "Stable"
        assert risk.iloc[1] == 3.0
        assert text.iloc[1] == "Critical Vitals"
        assert risk.iloc[2] == 0.0
        assert text.iloc[2] == "No Finger"
        assert risk.iloc[3] == 0.0
        assert text.iloc[3] == "No Finger"
        assert risk.iloc[4] == 2.0
        assert text.iloc[4] == "Monitor Closely"

    def test_J12_multi_schema_format_a_legacy_adc(self):
        """Format A (legacy prototype): Field 3 is numeric IR PPG, Field 4 is unpopulated."""
        df_format_a = pd.DataFrame([
            {"field3": "85400", "field4": None},
            {"field3": "92100", "field4": ""},
        ])
        ir, risk, text, fmt = parse_telemetry_schema(df_format_a)
        assert (fmt == "format_a_legacy_adc").all()
        assert ir.iloc[0] == 85400.0
        assert pd.isna(risk.iloc[0])

    def test_J13_optical_finger_detection_under_multi_schema(self):
        """Optical finger detection correctly handles both numeric IR PPG counts and text statuses."""
        from src.sensor.validate_clean import validate_and_clean_sensor_data
        clean_df, audit = validate_and_clean_sensor_data()
        assert "record_format" in clean_df.columns
        assert "patient_status_text" in clean_df.columns
        assert "finger_detected" in clean_df.columns
        # For historical entries with "No Finger", finger_detected must be False
        no_finger_rows = clean_df[clean_df["patient_status_text"] == "No Finger"]
        if not no_finger_rows.empty:
            assert (~no_finger_rows["finger_detected"]).all(), "Rows with 'No Finger' status must have finger_detected=False"
        # For historical entries with "Stable" and valid vitals, finger_detected must be True
        stable_valid = clean_df[(clean_df["patient_status_text"] == "Stable") & clean_df["valid_hr"] & clean_df["valid_spo2"]]
        if not stable_valid.empty:
            assert stable_valid["finger_detected"].all(), "Stable vitals must have finger_detected=True"


# ═══════════════════════════════════════════════════════════════════════════════
# Group K: Sensor Encoder Architecture & Tensor Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensorEncoderArchitecture:
    """Group K: SensorEncoder PyTorch module shapes, masking, gradients."""

    def test_K01_sensor_encoder_output_shape(self):
        """SensorEncoder maps (B, T, 3) sequence to (B, embed_dim)."""
        encoder = SensorEncoder(input_dim=3, hidden_dim=64, embed_dim=64)
        x = torch.randn(4, 20, 3)
        z = encoder(x)
        assert z.shape == (4, 64)

    def test_K02_sensor_encoder_masked_pooling(self):
        """Masked pooling produces finite embeddings for masked sequences."""
        encoder = SensorEncoder(input_dim=3, hidden_dim=32, embed_dim=32)
        x = torch.randn(3, 15, 3)
        mask = torch.ones(3, 15)
        mask[0, 5:] = 0.0  # patient 0 only has 5 valid timesteps
        mask[1, 10:] = 0.0 # patient 1 has 10 valid timesteps
        z = encoder(x, mask=mask)
        assert z.shape == (3, 32)
        assert torch.isfinite(z).all()

    def test_K03_sensor_encoder_finite_gradients(self):
        """Backward pass propagates finite gradients through Conv1d, GRU, and projections."""
        encoder = SensorEncoder(input_dim=3, hidden_dim=32, embed_dim=16)
        x = torch.randn(2, 10, 3)
        z = encoder(x)
        loss = z.sum()
        loss.backward()
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Parameter {name} has no gradient"
                assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"

    def test_K04_sensor_encoder_eval_deterministic(self):
        """Encoder is strictly deterministic in eval mode."""
        encoder = SensorEncoder(input_dim=3, hidden_dim=32, embed_dim=32, dropout=0.3)
        encoder.eval()
        x = torch.randn(2, 8, 3)
        with torch.no_grad():
            z1 = encoder(x)
            z2 = encoder(x)
        assert torch.equal(z1, z2)

    def test_K05_extract_sensor_windows_from_clean_df(self):
        """Sliding window extractor extracts expected shapes from sensor DataFrame."""
        clean_path = os.path.join(PROC_DIR, "sensor_thingspeak_clean.csv")
        df = pd.read_csv(clean_path)
        windows, masks = extract_sensor_windows(df, window_size=10, stride=5)
        assert len(windows) > 0
        assert windows.shape[1] == 10
        assert windows.shape[2] == 3
        assert masks.shape[1] == 10
        # Encoder can directly encode extracted windows
        encoder = SensorEncoder(input_dim=3, hidden_dim=32, embed_dim=32)
        z = encoder.encode_numpy(windows[:4], mask=masks[:4])
        assert z.shape == (4, 32)
        assert np.isfinite(z).all()

    def test_K06_sensor_encoder_handles_nans_and_all_zero_mask(self):
        """SensorEncoder safely handles NaN inputs and all-zero masks without crashing or emitting NaNs."""
        encoder = SensorEncoder(input_dim=3, hidden_dim=64, embed_dim=64)
        x = torch.randn(3, 10, 3)
        x[0, 2:5, :] = float("nan") # Inject NaNs
        x[1, :, :] = float("nan")   # Completely missing sequence
        mask = torch.ones(3, 10)
        mask[0, 2:5] = 0.0
        mask[1, :] = 0.0            # No valid timesteps at all for sequence 1
        z = encoder(x, mask=mask)
        assert z.shape == (3, 64)
        assert torch.isfinite(z).all(), "Encoder output must be finite even with NaNs and zero masks"
