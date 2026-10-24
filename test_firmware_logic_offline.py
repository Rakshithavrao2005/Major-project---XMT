"""
tests/test_firmware_logic_offline.py
=============================================================================
Offline static and algorithmic test suite for ESP32 + MAX30102 firmware logic.

Validates the corrected firmware logic against the requirements of Stage 3.8.6B:
1. Low SpO2 Acceptance: Verifies raw SpO2 readings in [70, 89]% (e.g. 87%) are
   accepted rather than rejected as "unrealistic sensor results".
2. Adaptive Desaturation Tracking: Confirms rapid desaturations (e.g. 98% -> 87%)
   are actively tracked without being blocked by jump filters.
3. Stale Value Elimination: Confirms failed measurement cycles invalidate the
   display values and prevent stale retention.
4. LCD & LED Indication Integrity: Asserts LED colors and LCD states match
   strict safety rules (Green ONLY on valid Normal).
5. Protocol Compatibility: Verifies ThingSpeak Field 1-4 mappings align with
   the backend TelemetryIngestionAdapter.
6. Hardware Pin and Config Invariance: Confirms ESP32 pins, LCD address 0x27,
   and 20-second interval are preserved verbatim in sepsis_risk_monitor.ino.
=============================================================================
"""

import math
import os
import re
import pytest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
INO_PATH = WORKSPACE_ROOT / "hardware" / "sepsis_risk_monitor" / "sepsis_risk_monitor.ino"
if not INO_PATH.exists():
    INO_PATH = WORKSPACE_ROOT / "hardware" / "sepsis_risk_monitor.ino"


class SimulatedFirmwareState:
    """Python simulation of the exact C++ logic implemented in sepsis_risk_monitor.ino."""

    def __init__(self):
        self.HR_MIN_VALID = 35
        self.HR_MAX_VALID = 220
        self.SPO2_MIN_VALID = 70
        self.SPO2_MAX_VALID = 100
        self.IR_FINGER_THRESHOLD = 10000
        self.REQUIRED_STABLE_READINGS = 1

        # Stage 3.8.11 / 3.8.13 HR Spike & Candidate Handling
        self.HR_SPIKE_THRESHOLD = 20.0
        self.REQUIRED_SPIKE_CONFIRMATIONS = 2
        self.lastAcceptedHeartRate = 0.0
        self.candidateHeartRate = 0
        self.storedCandidateHeartRate = 0
        self.consecutiveSpikeCount = 0

        # State
        self.fingerDetected = False
        self.currentMeasurementValid = False
        self.stableReadingCount = 0
        self.filteredHeartRate = 0.0
        self.filteredSpO2 = 0.0
        self.haveFilteredValue = False
        self.displayHeartRate = 0
        self.displaySpO2 = 0
        self.riskStatus = 0
        self.currentState = "STATE_STARTING"

    def step(self, raw_hr: int, raw_spo2: int, valid_hr: int, valid_spo2: int, ppg_ir: int):
        # 1. Optical Check
        if ppg_ir < self.IR_FINGER_THRESHOLD:
            self.fingerDetected = False
            self.currentMeasurementValid = False
            self.stableReadingCount = 0
            self.haveFilteredValue = False
            self.consecutiveSpikeCount = 0
            self.lastAcceptedHeartRate = 0.0
            self.candidateHeartRate = 0
            self.storedCandidateHeartRate = 0
            self.displayHeartRate = 0
            self.displaySpO2 = 0
            self.currentState = "STATE_NO_FINGER"
            self.evaluate_hardware_status()
            return False

        self.fingerDetected = True

        # 2. Calculation validity
        if not valid_hr or not valid_spo2:
            self.currentMeasurementValid = False
            self.stableReadingCount = 0
            self.haveFilteredValue = False
            self.consecutiveSpikeCount = 0
            self.lastAcceptedHeartRate = 0.0
            self.candidateHeartRate = 0
            self.storedCandidateHeartRate = 0
            self.displayHeartRate = 0
            self.displaySpO2 = 0
            self.currentState = "STATE_INVALID_READING"
            self.evaluate_hardware_status()
            return False

        # 3. Boundaries
        if raw_hr < self.HR_MIN_VALID or raw_hr > self.HR_MAX_VALID:
            self.currentMeasurementValid = False
            self.stableReadingCount = 0
            self.haveFilteredValue = False
            self.consecutiveSpikeCount = 0
            self.lastAcceptedHeartRate = 0.0
            self.candidateHeartRate = 0
            self.storedCandidateHeartRate = 0
            self.displayHeartRate = 0
            self.displaySpO2 = 0
            self.currentState = "STATE_INVALID_READING"
            self.evaluate_hardware_status()
            return False

        if raw_spo2 < self.SPO2_MIN_VALID or raw_spo2 > self.SPO2_MAX_VALID:
            self.currentMeasurementValid = False
            self.stableReadingCount = 0
            self.haveFilteredValue = False
            self.consecutiveSpikeCount = 0
            self.lastAcceptedHeartRate = 0.0
            self.candidateHeartRate = 0
            self.storedCandidateHeartRate = 0
            self.displayHeartRate = 0
            self.displaySpO2 = 0
            self.currentState = "STATE_INVALID_READING"
            self.evaluate_hardware_status()
            return False

        # 4. Adaptive Filtering & Spike Handling (Stage 3.8.13)
        if not self.haveFilteredValue:
            self.filteredHeartRate = float(raw_hr)
            self.filteredSpO2 = float(raw_spo2)
            self.lastAcceptedHeartRate = float(raw_hr)
            self.candidateHeartRate = raw_hr
            self.storedCandidateHeartRate = 0
            self.consecutiveSpikeCount = 0
            self.haveFilteredValue = True
        else:
            # Correction A: SpO2 candidate computed in temporary variable (not committed yet)
            candidateFilteredSpO2 = self.filteredSpO2
            spo2_delta = abs(float(raw_spo2) - self.filteredSpO2)
            if spo2_delta > 3.0:
                candidateFilteredSpO2 = (self.filteredSpO2 * 0.40) + (float(raw_spo2) * 0.60)
            else:
                candidateFilteredSpO2 = (self.filteredSpO2 * 0.70) + (float(raw_spo2) * 0.30)

            # Correction B: HR Candidate Consistency Check
            self.candidateHeartRate = raw_hr
            hr_delta = abs(float(raw_hr) - self.lastAcceptedHeartRate)

            if hr_delta > self.HR_SPIKE_THRESHOLD:
                if self.consecutiveSpikeCount == 0:
                    # First unconfirmed candidate
                    self.storedCandidateHeartRate = raw_hr
                    self.consecutiveSpikeCount = 1
                    self.currentMeasurementValid = False
                    self.stableReadingCount = 0
                    self.displayHeartRate = 0
                    self.displaySpO2 = 0
                    self.currentState = "STATE_SAMPLING"
                    self.evaluate_hardware_status()
                    return False
                else:
                    # Candidate 2: check consistency with stored candidate 1
                    candidateConsistency = abs(float(raw_hr) - float(self.storedCandidateHeartRate))
                    if candidateConsistency <= self.HR_SPIKE_THRESHOLD:
                        # Consistent sustained shift confirmed
                        self.consecutiveSpikeCount = 0
                        self.storedCandidateHeartRate = 0
                        self.filteredHeartRate = (self.filteredHeartRate * 0.40) + (float(raw_hr) * 0.60)
                        self.lastAcceptedHeartRate = float(raw_hr)
                        # Commit SpO2 candidate
                        self.filteredSpO2 = candidateFilteredSpO2
                    else:
                        # Inconsistent spikes (e.g. 75 -> 130 -> 40): reject and restart confirmation
                        self.storedCandidateHeartRate = raw_hr
                        self.consecutiveSpikeCount = 1
                        self.currentMeasurementValid = False
                        self.stableReadingCount = 0
                        self.displayHeartRate = 0
                        self.displaySpO2 = 0
                        self.currentState = "STATE_SAMPLING"
                        self.evaluate_hardware_status()
                        return False
            else:
                # Normal variation
                self.consecutiveSpikeCount = 0
                self.storedCandidateHeartRate = 0
                self.filteredHeartRate = (self.filteredHeartRate * 0.70) + (float(raw_hr) * 0.30)
                self.lastAcceptedHeartRate = self.filteredHeartRate
                # Commit SpO2 candidate
                self.filteredSpO2 = candidateFilteredSpO2

        # 5. Stability Gate
        self.stableReadingCount += 1
        if self.stableReadingCount < self.REQUIRED_STABLE_READINGS:
            self.currentState = "STATE_SAMPLING"
            self.currentMeasurementValid = False
            self.evaluate_hardware_status()
            return False

        self.currentMeasurementValid = True
        self.displayHeartRate = int(round(self.filteredHeartRate))
        self.displaySpO2 = int(round(self.filteredSpO2))
        self.evaluate_hardware_status()
        return True

    def evaluate_hardware_status(self):
        if not self.fingerDetected:
            self.currentState = "STATE_NO_FINGER"
            self.riskStatus = 0
            return

        if not self.currentMeasurementValid:
            self.riskStatus = 0
            return

        # High Risk: Critical tachycardia (HR >= 130) OR Hypoxia (SpO2 < 92%)
        if self.displayHeartRate >= 130 or self.displaySpO2 < 92:
            self.currentState = "STATE_HIGH_RISK"
            self.riskStatus = 2
        # Warning (Elevated HR >= 100, Bradycardia < 50, or Suboptimal SpO2 < 95%)
        elif self.displayHeartRate >= 100 or self.displayHeartRate < 50 or self.displaySpO2 < 95:
            self.currentState = "STATE_WARNING"
            self.riskStatus = 1
        # Normal
        else:
            self.currentState = "STATE_NORMAL"
            self.riskStatus = 0

    def get_leds(self):
        red = (self.currentState == "STATE_HIGH_RISK")
        yellow = (self.currentState in ["STATE_STARTING", "STATE_SAMPLING", "STATE_INVALID_READING", "STATE_NO_FINGER", "STATE_WARNING"])
        green = (self.currentState == "STATE_NORMAL")
        return {"RED": red, "YELLOW": yellow, "GREEN": green}

    def get_lcd(self):
        if self.currentState == "STATE_NO_FINGER":
            return {"line0": "CHECK SENSOR    ", "line1": "Place Finger    ", "status_text": "CHECK SENSOR"}
        elif self.currentState == "STATE_SAMPLING":
            return {"line0": "SAMPLING...     ", "line1": "Keep Still      ", "status_text": "SAMPLING..."}
        elif self.currentState == "STATE_INVALID_READING":
            return {"line0": "INVALID READING ", "line1": "Motion / Noise  ", "status_text": "INVALID READING"}
        elif self.currentState == "STATE_HIGH_RISK":
            return {"line0": f"HR:{self.displayHeartRate} S:{self.displaySpO2}%   ", "line1": "Status:HIGH RISK", "status_text": "HIGH RISK"}
        elif self.currentState == "STATE_WARNING":
            return {"line0": f"HR:{self.displayHeartRate} S:{self.displaySpO2}%   ", "line1": "Status: WARNING ", "status_text": "WARNING"}
        elif self.currentState == "STATE_NORMAL":
            return {"line0": f"HR:{self.displayHeartRate} S:{self.displaySpO2}%   ", "line1": "Status: NORMAL  ", "status_text": "NORMAL"}
        return {"line0": "", "line1": "", "status_text": "UNKNOWN"}


# =============================================================================
# 1. Low SpO2 & Desaturation Tracking Tests
# =============================================================================

def test_raw_spo2_87_accepted():
    """Verify that raw SpO2 of 87% is accepted on first stable reading and triggers HIGH RISK (SpO2 < 92%)."""
    fw = SimulatedFirmwareState()
    # Cycle 1: Immediately qualified with REQUIRED_STABLE_READINGS = 1
    ok = fw.step(raw_hr=75, raw_spo2=87, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok
    assert fw.currentMeasurementValid is True
    assert fw.displaySpO2 == 87
    assert fw.displayHeartRate == 75
    # Since SpO2=87% is < 92%, state is HIGH_RISK
    assert fw.currentState == "STATE_HIGH_RISK"
    assert fw.riskStatus == 2
    assert fw.get_leds()["RED"] is True
    assert fw.get_lcd()["status_text"] == "HIGH RISK"


def test_suboptimal_spo2_93_warning():
    """Verify that raw SpO2 of 93% (92 <= SpO2 < 95) triggers STATE_WARNING with Yellow LED and WARNING LCD."""
    fw = SimulatedFirmwareState()
    ok = fw.step(raw_hr=75, raw_spo2=93, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok
    assert fw.currentMeasurementValid is True
    assert fw.displaySpO2 == 93
    assert fw.displayHeartRate == 75
    assert fw.currentState == "STATE_WARNING"
    assert fw.riskStatus == 1
    assert fw.get_leds()["YELLOW"] is True
    assert fw.get_lcd()["status_text"] == "WARNING"


def test_rapid_desaturation_tracks_without_rejection():
    """Verify that a drop from 98% to 87% tracks down within 3 samples and is not rejected."""
    fw = SimulatedFirmwareState()
    # Establish baseline at 98%
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.displaySpO2 == 98
    assert fw.currentState == "STATE_NORMAL"

    # Patient desaturates abruptly to 87%
    fw.step(raw_hr=75, raw_spo2=87, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    # Sample 1: 98 * 0.4 + 87 * 0.6 = 91.4 -> round to 91 (< 92 -> HIGH_RISK)
    assert fw.displaySpO2 <= 92
    assert fw.currentState == "STATE_HIGH_RISK"
    assert fw.riskStatus == 2

    fw.step(raw_hr=75, raw_spo2=87, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    # Sample 2: delta 4.4 > 3 -> 91.4 * 0.40 + 87 * 0.60 = 88.76 -> round to 89
    assert fw.displaySpO2 <= 89

    fw.step(raw_hr=75, raw_spo2=87, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    # Sample 3: delta 1.76 <= 3 -> 88.76 * 0.70 + 87 * 0.30 = 88.23 -> round to 88
    assert fw.displaySpO2 <= 88
    assert fw.displaySpO2 != 98  # Proven NOT locked at 98%


# =============================================================================
# 2. Stale Value Elimination Tests
# =============================================================================

def test_failed_cycle_invalidates_stale_display_values():
    """Verify that a failed calculation cycle immediately invalidates display values."""
    fw = SimulatedFirmwareState()
    # Establish valid baseline
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.displayHeartRate == 75
    assert fw.displaySpO2 == 98
    assert fw.currentMeasurementValid is True

    # Next cycle fails (motion artifact)
    ok = fw.step(raw_hr=0, raw_spo2=0, valid_hr=0, valid_spo2=0, ppg_ir=140000)
    assert not ok
    assert fw.currentMeasurementValid is False
    assert fw.displayHeartRate == 0  # Stale value wiped
    assert fw.displaySpO2 == 0       # Stale value wiped
    assert fw.currentState == "STATE_INVALID_READING"


def test_finger_removal_wipes_display_values():
    """Verify that removing finger wipes display values and switches to CHECK SENSOR."""
    fw = SimulatedFirmwareState()
    # Establish valid baseline
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)

    # Finger removed
    ok = fw.step(raw_hr=0, raw_spo2=0, valid_hr=0, valid_spo2=0, ppg_ir=4500)
    assert not ok
    assert fw.fingerDetected is False
    assert fw.currentMeasurementValid is False
    assert fw.displayHeartRate == 0
    assert fw.displaySpO2 == 0
    assert fw.currentState == "STATE_NO_FINGER"


# =============================================================================
# 3. LED Logic Tests
# =============================================================================

def test_led_and_lcd_alignment_across_states():
    """Verify LED and LCD alignment: Green for LOW risk with LCD 'NORMAL', Yellow for WARNING with LCD 'WARNING', Red for HIGH RISK with LCD 'HIGH RISK'."""
    fw = SimulatedFirmwareState()

    # Starting: Yellow on, Green off, Red off
    fw.currentState = "STATE_STARTING"
    leds = fw.get_leds()
    assert leds["GREEN"] is False and leds["YELLOW"] is True and leds["RED"] is False

    # No finger: Yellow on, Green off, Red off, LCD CHECK SENSOR
    fw.currentState = "STATE_NO_FINGER"
    leds = fw.get_leds()
    lcd = fw.get_lcd()
    assert leds["GREEN"] is False and leds["YELLOW"] is True and leds["RED"] is False
    assert lcd["status_text"] == "CHECK SENSOR"

    # Sampling: Yellow on, Green off, Red off, LCD SAMPLING...
    fw.currentState = "STATE_SAMPLING"
    leds = fw.get_leds()
    lcd = fw.get_lcd()
    assert leds["GREEN"] is False and leds["YELLOW"] is True and leds["RED"] is False
    assert lcd["status_text"] == "SAMPLING..."

    # Invalid: Yellow on, Green off, Red off, LCD INVALID READING
    fw.currentState = "STATE_INVALID_READING"
    leds = fw.get_leds()
    lcd = fw.get_lcd()
    assert leds["GREEN"] is False and leds["YELLOW"] is True and leds["RED"] is False
    assert lcd["status_text"] == "INVALID READING"

    # Low Risk (Normal): Green on, Yellow off, Red off, LCD NORMAL
    fw.currentState = "STATE_NORMAL"
    leds = fw.get_leds()
    lcd = fw.get_lcd()
    assert leds["GREEN"] is True and leds["YELLOW"] is False and leds["RED"] is False
    assert lcd["status_text"] == "NORMAL"

    # Medium Risk (Warning): Yellow on, Green off, Red off, LCD WARNING
    fw.currentState = "STATE_WARNING"
    leds = fw.get_leds()
    lcd = fw.get_lcd()
    assert leds["YELLOW"] is True and leds["GREEN"] is False and leds["RED"] is False
    assert lcd["status_text"] == "WARNING"

    # High Risk: Red on, Green off, Yellow off, LCD HIGH RISK
    fw.currentState = "STATE_HIGH_RISK"
    leds = fw.get_leds()
    lcd = fw.get_lcd()
    assert leds["RED"] is True and leds["GREEN"] is False and leds["YELLOW"] is False
    assert lcd["status_text"] == "HIGH RISK"


# =============================================================================
# 4. Backend Protocol Compatibility Tests
# =============================================================================

def test_protocol_compatibility_with_telemetry_adapter():
    """Verify that firmware telemetry packets are correctly parsed by TelemetryIngestionAdapter."""
    import datetime
    from src.sensor.telemetry_adapter import TelemetryIngestionAdapter, IngestionStatus

    adapter = TelemetryIngestionAdapter(freshness_window_seconds=60.0)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    ts_now = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Normal packet
    res_normal = adapter.ingest_packet({
        "channel_id": 3490289,
        "entry_id": 1001,
        "recorded_at": ts_now,
        "field1": "75",
        "field2": "98",
        "field3": "140000",
        "field4": "0",
    }, reference_arrival_time=now_utc)
    assert res_normal.status == IngestionStatus.ACCEPTED
    assert res_normal.packet.alert_severity == "NORMAL"
    assert res_normal.packet.cleaned.physiologically_valid is True

    # Warning packet (HR 110 triggers SIRS_WARNING)
    res_warn = adapter.ingest_packet({
        "channel_id": 3490289,
        "entry_id": 1002,
        "recorded_at": ts_now,
        "field1": "110",
        "field2": "94",
        "field3": "140000",
        "field4": "0",
    }, reference_arrival_time=now_utc)
    assert res_warn.status == IngestionStatus.ACCEPTED
    assert res_warn.packet.alert_severity == "SIRS_WARNING"

    # High Risk packet (HR 135, SpO2 88, Field 4 = 2)
    res_crit = adapter.ingest_packet({
        "channel_id": 3490289,
        "entry_id": 1003,
        "recorded_at": ts_now,
        "field1": "135",
        "field2": "88",
        "field3": "140000",
        "field4": "2",
    }, reference_arrival_time=now_utc)
    assert res_crit.status == IngestionStatus.ACCEPTED
    assert res_crit.packet.alert_severity == "CRITICAL"


# =============================================================================
# 5. Static Firmware Source Code Invariance Tests
# =============================================================================

def test_firmware_pins_and_hardware_configuration_preserved():
    """Verify that hardware pins, LCD address, and interval in the .ino file are intact."""
    assert INO_PATH.exists(), f"Firmware file not found at {INO_PATH}"
    with open(INO_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Pins
    assert "#define RED_LED     4" in content
    assert "#define YELLOW_LED  16" in content
    assert "#define GREEN_LED   17" in content
    assert "Wire.begin(21, 22);" in content

    # LCD
    assert "LiquidCrystal_I2C lcd(0x27, 16, 2);" in content

    # Interval & Stability
    assert "uploadInterval = 20000;" in content
    assert "REQUIRED_STABLE_READINGS = 1;" in content

    # Credentials configuration variables present
    assert "WIFI_SSID" in content
    assert "WIFI_PASSWORD" in content
    assert "CHANNEL_ID = 3490289;" in content
    assert "WRITE_API_KEY" in content

    # Clinical disclaimer present
    assert "NOT CLINICAL SEPSIS DIAGNOSIS" in content
    assert "NOT CONSTITUTE A MEDICAL DIAGNOSTIC DEVICE" in content.upper()

    # Stage 3.8.11 / 3.8.13 Spike Handling Constants
    assert "HR_SPIKE_THRESHOLD = 20.0;" in content
    assert "REQUIRED_SPIKE_CONFIRMATIONS = 2;" in content
    assert "lastAcceptedHeartRate" in content
    assert "candidateHeartRate" in content
    assert "storedCandidateHeartRate" in content


# =============================================================================
# 6. Stage 3.8.13 HR Spike, Candidate Consistency, & SpO2 Deferral Tests
# =============================================================================

def test_hr_rejected_cycle_does_not_mutate_accepted_spo2_filter():
    """Correction A: Verify that an HR-rejected cycle does not mutate the accepted SpO2 filter."""
    fw = SimulatedFirmwareState()
    # Cycle 1: Establish clean baseline at 75 BPM, 98% SpO2
    ok1 = fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok1 is True
    assert fw.filteredSpO2 == 98.0
    assert fw.displaySpO2 == 98

    # Cycle 2: Motion artifact causes HR spike (130 BPM, delta = 55) and distorted SpO2 (88%)
    ok2 = fw.step(raw_hr=130, raw_spo2=88, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok2 is False
    assert fw.currentMeasurementValid is False
    assert fw.displaySpO2 == 0
    # Crucial check: accepted filter state MUST NOT be mutated by the rejected cycle
    assert fw.filteredSpO2 == 98.0

    # Cycle 3: Clean measurement arrives (75 BPM, 98% SpO2)
    ok3 = fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok3 is True
    assert fw.currentMeasurementValid is True
    assert fw.filteredSpO2 == 98.0
    assert fw.displaySpO2 == 98


def test_inconsistent_spike_sequence_does_not_confirm_shift():
    """Correction B: Verify that 75 -> 130 -> 40 does NOT confirm a sustained shift."""
    fw = SimulatedFirmwareState()
    # Baseline at 75 BPM
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.lastAcceptedHeartRate == 75.0

    # Cycle 1: Spikes high to 130 BPM (delta = 55 > 20) -> Candidate 1 stored
    ok_c1 = fw.step(raw_hr=130, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok_c1 is False
    assert fw.consecutiveSpikeCount == 1
    assert fw.storedCandidateHeartRate == 130

    # Cycle 2: Plunges low to 40 BPM (|40 - 130| = 90 > 20) -> Inconsistent!
    ok_c2 = fw.step(raw_hr=40, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok_c2 is False
    assert fw.currentMeasurementValid is False
    # Shift was NOT confirmed; baseline remains at 75
    assert fw.lastAcceptedHeartRate == 75.0
    # Candidate was reset to 40
    assert fw.storedCandidateHeartRate == 40
    assert fw.consecutiveSpikeCount == 1

    # Cycle 3: Return to normal baseline 75 BPM
    ok_c3 = fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok_c3 is True
    assert fw.currentMeasurementValid is True
    assert fw.displayHeartRate == 75
    assert fw.consecutiveSpikeCount == 0
    assert fw.storedCandidateHeartRate == 0


def test_consistent_spike_sequence_confirms_shift():
    """Correction B: Verify that 75 -> 130 -> 135 confirms a genuine sustained shift."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.lastAcceptedHeartRate == 75.0

    # Cycle 1: 130 BPM (delta = 55 > 20) -> Candidate 1 stored
    ok_c1 = fw.step(raw_hr=130, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok_c1 is False
    assert fw.consecutiveSpikeCount == 1
    assert fw.storedCandidateHeartRate == 130

    # Cycle 2: 135 BPM (|135 - 130| = 5 <= 20) -> Consistent! Shift confirmed!
    ok_c2 = fw.step(raw_hr=135, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok_c2 is True
    assert fw.currentMeasurementValid is True
    # 75 * 0.40 + 135 * 0.60 = 111.0
    assert fw.displayHeartRate == 111
    assert fw.lastAcceptedHeartRate == 135.0
    assert fw.consecutiveSpikeCount == 0
    assert fw.storedCandidateHeartRate == 0


def test_confirmed_high_hr_reaches_risk_state_logic():
    """Verify that a confirmed high HR appropriately reaches warning and high-risk logic."""
    fw = SimulatedFirmwareState()
    # Baseline normal
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_NORMAL"
    assert fw.riskStatus == 0

    # Confirmed tachycardia shift to 140 BPM with normal SpO2 (98%)
    fw.step(raw_hr=140, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    fw.step(raw_hr=140, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    # Display HR = 75*0.4 + 140*0.6 = 114 -> Warning state (HR >= 100)
    assert fw.currentState == "STATE_WARNING"
    assert fw.riskStatus == 1

    # Now patient develops hypoxia (SpO2 = 88%) with sustained HR=140
    fw.step(raw_hr=140, raw_spo2=88, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    fw.step(raw_hr=140, raw_spo2=88, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    fw.step(raw_hr=140, raw_spo2=88, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    # HR is >= 130 and SpO2 < 92 -> HIGH RISK (Risk Status = 2)
    assert fw.displayHeartRate >= 130
    assert fw.displaySpO2 < 92
    assert fw.currentState == "STATE_HIGH_RISK"
    assert fw.riskStatus == 2


def test_rejected_readings_cannot_trigger_green_led_or_be_uploaded_or_labeled_normal():
    """Verify that rejected readings cannot trigger green LED, be uploaded, or be labeled normal."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=72, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentMeasurementValid is True

    # Spurious jump to 125 BPM
    ok = fw.step(raw_hr=125, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok is False
    assert fw.currentMeasurementValid is False
    assert fw.displayHeartRate == 0  # Stale value not retained as fresh
    assert fw.displaySpO2 == 0
    assert fw.currentState == "STATE_SAMPLING"

    # LED check
    leds = fw.get_leds()
    assert leds["GREEN"] is False
    assert leds["YELLOW"] is True
    assert leds["RED"] is False

    # Telemetry gate check
    assert fw.currentMeasurementValid is False  # Upload blocked


def test_invalid_sample_resets_spike_confirmation_counter():
    """Verify that a calculation failure resets unconfirmed candidate confirmation."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=72, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)

    # First cycle of suspicious jump
    fw.step(raw_hr=120, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.consecutiveSpikeCount == 1
    assert fw.storedCandidateHeartRate == 120

    # Next cycle fails (motion noise)
    fw.step(raw_hr=0, raw_spo2=0, valid_hr=0, valid_spo2=0, ppg_ir=140000)
    assert fw.consecutiveSpikeCount == 0
    assert fw.storedCandidateHeartRate == 0
    assert fw.haveFilteredValue is False
    assert fw.displayHeartRate == 0
    assert fw.currentState == "STATE_INVALID_READING"


# =============================================================================
# 7. Stage 3.8.15 Bradycardia Safety & 4-Step Sequence Tests
# =============================================================================

def test_bradycardia_40_bpm_triggers_warning_state_with_yellow_warning_display():
    """Verify that confirmed bradycardia (HR=40 BPM, SpO2=98%) triggers STATE_WARNING with Yellow LED and WARNING LCD."""
    fw = SimulatedFirmwareState()
    # Baseline normal reading
    fw.step(raw_hr=72, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_NORMAL"
    leds_normal = fw.get_leds()
    assert leds_normal["GREEN"] is True
    assert leds_normal["YELLOW"] is False

    # Patient develops confirmed sustained bradycardia (40 BPM)
    # Step 1: Unconfirmed jump -> Yellow sampling, Green off
    fw.step(raw_hr=40, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_SAMPLING"
    assert fw.get_leds()["YELLOW"] is True
    assert fw.get_leds()["GREEN"] is False

    # Step 2: Confirmed shift (40*0.6 + 72*0.4 = 52.8)
    fw.step(raw_hr=40, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    # Step 3: Sustained low HR (52.8*0.7 + 40*0.3 = 48.96 -> round to 49 (< 50)
    fw.step(raw_hr=40, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentMeasurementValid is True
    assert fw.displayHeartRate < 50
    assert fw.displaySpO2 == 98

    # State is STATE_WARNING (medium risk), LED is YELLOW, LCD displays WARNING
    assert fw.currentState == "STATE_WARNING"
    assert fw.riskStatus == 1
    leds_brady = fw.get_leds()
    assert leds_brady["YELLOW"] is True
    assert leds_brady["GREEN"] is False
    assert leds_brady["RED"] is False
    assert fw.get_lcd()["status_text"] == "WARNING"


def test_four_step_complex_sequence_75_130_40_42():
    """Verify that 75 -> 130 -> 40 -> 42 rejects anomalies and confirms consistent shift on step 4."""
    fw = SimulatedFirmwareState()
    # Baseline at 75 BPM, 98% SpO2
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.lastAcceptedHeartRate == 75.0
    assert fw.currentState == "STATE_NORMAL"

    # Step 1: 130 BPM (delta = 55 > 20) -> Stored candidate 1 = 130, rejected
    ok1 = fw.step(raw_hr=130, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok1 is False
    assert fw.consecutiveSpikeCount == 1
    assert fw.storedCandidateHeartRate == 130
    assert fw.currentMeasurementValid is False
    assert fw.get_leds()["GREEN"] is False

    # Step 2: 40 BPM (|40 - 130| = 90 > 20) -> Inconsistent! Rejected, stored candidate reset to 40
    ok2 = fw.step(raw_hr=40, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok2 is False
    assert fw.consecutiveSpikeCount == 1
    assert fw.storedCandidateHeartRate == 40
    assert fw.lastAcceptedHeartRate == 75.0  # Baseline preserved
    assert fw.currentMeasurementValid is False
    assert fw.get_leds()["GREEN"] is False

    # Step 3: 42 BPM (|42 - 40| = 2 <= 20) -> Consistent! Shift confirmed to low HR
    ok3 = fw.step(raw_hr=42, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok3 is True
    assert fw.currentMeasurementValid is True
    assert fw.lastAcceptedHeartRate == 42.0
    assert fw.consecutiveSpikeCount == 0
    assert fw.storedCandidateHeartRate == 0

    # Step 4: Next sustained reading at 42 BPM continues cleanly
    ok4 = fw.step(raw_hr=42, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok4 is True
    assert fw.currentMeasurementValid is True


# =============================================================================
# 8. Stage 3.8.20 Comprehensive LED & Risk-State Alignment Test Suite
# =============================================================================

def test_startup_yellow_led():
    """1. Startup -> YELLOW LED on, GREEN off, RED off."""
    fw = SimulatedFirmwareState()
    assert fw.currentState == "STATE_STARTING"
    leds = fw.get_leds()
    assert leds["YELLOW"] is True
    assert leds["GREEN"] is False
    assert leds["RED"] is False


def test_no_finger_yellow_led():
    """2. No finger -> YELLOW on, GREEN off, RED off, LCD CHECK SENSOR."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=5000) # Below 10,000 counts
    assert fw.currentState == "STATE_NO_FINGER"
    leds = fw.get_leds()
    assert leds["YELLOW"] is True
    assert leds["GREEN"] is False
    assert leds["RED"] is False
    assert fw.get_lcd()["status_text"] == "CHECK SENSOR"


def test_insufficient_samples_yellow_led():
    """3. Insufficient samples -> YELLOW on, GREEN off, RED off."""
    fw = SimulatedFirmwareState()
    fw.REQUIRED_STABLE_READINGS = 3
    # First sample
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_SAMPLING"
    leds = fw.get_leds()
    assert leds["YELLOW"] is True
    assert leds["GREEN"] is False
    assert leds["RED"] is False
    assert fw.get_lcd()["status_text"] == "SAMPLING..."


def test_pending_hr_spike_confirmation_yellow_led():
    """4. Pending HR spike confirmation -> YELLOW on, GREEN off, RED off."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000) # Baseline
    # Sudden jump from 75 to 110 (delta = 35 > 20)
    ok = fw.step(raw_hr=110, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok is False
    assert fw.currentState == "STATE_SAMPLING"
    assert fw.consecutiveSpikeCount == 1
    leds = fw.get_leds()
    assert leds["YELLOW"] is True
    assert leds["GREEN"] is False
    assert leds["RED"] is False


def test_valid_low_risk_green_led_normal_lcd():
    """5. Valid LOW risk -> GREEN on, LCD NORMAL, actual normal risk classification."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=72, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_NORMAL"
    leds = fw.get_leds()
    assert leds["GREEN"] is True
    assert leds["YELLOW"] is False
    assert leds["RED"] is False
    assert fw.get_lcd()["status_text"] == "NORMAL"
    assert fw.riskStatus == 0


def test_valid_medium_risk_yellow_led_warning_lcd_retains_warning():
    """6. Valid MEDIUM risk -> YELLOW on, LCD WARNING, actual warning risk classification retained."""
    fw = SimulatedFirmwareState()
    # Baseline at 110 bpm (tachycardia warning)
    fw.step(raw_hr=110, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_WARNING"
    leds = fw.get_leds()
    assert leds["YELLOW"] is True
    assert leds["GREEN"] is False
    assert leds["RED"] is False
    assert fw.get_lcd()["status_text"] == "WARNING"
    assert fw.displayHeartRate == 110
    assert fw.riskStatus == 1 # Explicit Field 4 = 1 for warning state


def test_configured_high_risk_red_led_high_risk_lcd():
    """7. Existing HIGH_RISK condition -> RED on, LCD HIGH RISK."""
    fw = SimulatedFirmwareState()
    # HR=135, SpO2=90% -> High Sepsis Risk
    fw.step(raw_hr=135, raw_spo2=90, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_HIGH_RISK"
    leds = fw.get_leds()
    assert leds["RED"] is True
    assert leds["GREEN"] is False
    assert leds["YELLOW"] is False
    assert fw.get_lcd()["status_text"] == "HIGH RISK"
    assert fw.riskStatus == 2


def test_invalid_hr_or_spo2_never_green():
    """8. Invalid HR or SpO2 -> never GREEN, always YELLOW."""
    fw = SimulatedFirmwareState()
    # Invalid HR
    fw.step(raw_hr=10, raw_spo2=98, valid_hr=0, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_INVALID_READING"
    assert fw.get_leds()["GREEN"] is False
    assert fw.get_leds()["YELLOW"] is True

    # Invalid SpO2
    fw.step(raw_hr=75, raw_spo2=50, valid_hr=1, valid_spo2=0, ppg_ir=140000)
    assert fw.currentState == "STATE_INVALID_READING"
    assert fw.get_leds()["GREEN"] is False
    assert fw.get_leds()["YELLOW"] is True


def test_sensor_removal_after_valid_reading_safe_sampling_state():
    """9. Sensor removal after a previously valid reading -> transitions to safe no-finger state."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.currentState == "STATE_NORMAL"
    assert fw.get_leds()["GREEN"] is True

    # Finger removed
    fw.step(raw_hr=0, raw_spo2=0, valid_hr=0, valid_spo2=0, ppg_ir=3000)
    assert fw.currentState == "STATE_NO_FINGER"
    assert fw.currentMeasurementValid is False
    assert fw.get_leds()["GREEN"] is False
    assert fw.get_leds()["YELLOW"] is True
    assert fw.get_lcd()["status_text"] == "CHECK SENSOR"


def test_rejected_spike_does_not_contaminate_accepted_readings():
    """10. Rejected spike does not contaminate accepted readings or incorrectly trigger GREEN/RED."""
    fw = SimulatedFirmwareState()
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.lastAcceptedHeartRate == 75.0

    # Rejected spike to 135 (delta = 60 > 20)
    ok = fw.step(raw_hr=135, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert ok is False
    assert fw.currentMeasurementValid is False
    assert fw.lastAcceptedHeartRate == 75.0
    # While rejected/sampling, Yellow is on, Green and Red are off
    assert fw.get_leds()["YELLOW"] is True
    assert fw.get_leds()["GREEN"] is False
    assert fw.get_leds()["RED"] is False


def test_confirmed_high_risk_transition_red():
    """11. Confirmed high-risk transition -> RED on, LCD HIGH RISK."""
    fw = SimulatedFirmwareState()
    # Baseline at 75 bpm, 98%
    fw.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    # Severe hypoxia to 75% (98*0.4 + 75*0.6 = 84.2 -> round to 84 < 85)
    fw.step(raw_hr=75, raw_spo2=75, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw.displaySpO2 < 85
    assert fw.currentState == "STATE_HIGH_RISK"
    assert fw.get_leds()["RED"] is True
    assert fw.get_leds()["GREEN"] is False
    assert fw.get_leds()["YELLOW"] is False


def test_field4_compatibility_preserved():
    """12. Field 4 remains compatible with backend contract: 2 for HIGH_RISK, 1 for WARNING, 0 for NORMAL."""
    # Test 1: Normal reading -> riskStatus = 0
    fw1 = SimulatedFirmwareState()
    fw1.step(raw_hr=75, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw1.riskStatus == 0

    # Test 2: Warning reading -> riskStatus = 1
    fw2 = SimulatedFirmwareState()
    fw2.step(raw_hr=110, raw_spo2=98, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw2.riskStatus == 1

    # Test 3: High Risk reading -> riskStatus = 2
    fw3 = SimulatedFirmwareState()
    fw3.step(raw_hr=135, raw_spo2=90, valid_hr=1, valid_spo2=1, ppg_ir=140000)
    assert fw3.riskStatus == 2


def test_every_state_leaves_led_outputs_mutually_consistent():
    """13. Every state transition leaves LED outputs mutually consistent (exactly 1 LED ON)."""
    fw = SimulatedFirmwareState()
    for state in ["STATE_STARTING", "STATE_NO_FINGER", "STATE_SAMPLING", "STATE_INVALID_READING", "STATE_NORMAL", "STATE_WARNING", "STATE_HIGH_RISK"]:
        fw.currentState = state
        leds = fw.get_leds()
        # Exactly one LED must be active at any given time
        active_count = sum(1 for val in leds.values() if val is True)
        assert active_count == 1, f"State {state} produced inconsistent active LEDs: {leds}"


def test_invalid_readings_remain_blocked_from_telemetry_upload():
    """14. Invalid readings remain blocked from telemetry upload."""
    fw = SimulatedFirmwareState()
    # Cycle with invalid calculations
    fw.step(raw_hr=0, raw_spo2=0, valid_hr=0, valid_spo2=0, ppg_ir=140000)
    assert fw.currentMeasurementValid is False
    # If simulated upload function checks currentMeasurementValid, it must block
    upload_allowed = fw.currentMeasurementValid
    assert upload_allowed is False
