"""
tests/test_hourly_feature_extractor.py
─────────────────────────────────────────────────────────────────────────────
Unit Test Suite for Sliding-Window Feature Extractor.

SAFETY & PROVENANCE NOTICE:
=============================================================================
ALL DATA AND OBSERVATIONS IN THIS TEST FILE ARE STRICTLY SYNTHETIC FIXTURES
CONSTRUCTED EXCLUSIVELY FOR UNIT TESTING OF SOFTWARE MECHANICS AND ANTI-LEAKAGE
GUARDRAILS.
THEY ARE NOT CLINICAL PATIENT DATA, NOT REAL EHR OBSERVATIONS, AND MUST NEVER
BE USED FOR CLINICAL TRAINING, VALIDATION, OR BENCHMARK EVALUATION.
=============================================================================
"""

import datetime
import numpy as np
import pandas as pd
import pytest

from src.hourly.feature_extractor import (
    HourlySlidingWindowFeatureExtractor,
    PHYSIOLOGICAL_BOUNDS,
)


@pytest.fixture
def synthetic_patient_timeline():
    """
    Creates a controlled synthetic timeline of timestamped vitals and labs.
    NOTICE: PURELY SYNTHETIC FIXTURE FOR TESTING LOGIC ONLY.
    """
    base_time = pd.Timestamp("2026-10-20 00:00:00")

    # Synthetic vitals across 10 hours
    vitals_data = [
        # Hour 0 (00:00)
        {"stay_id": 99901, "charttime": base_time, "itemid": 220045, "valuenum": 72.0},   # HR = 72
        {"stay_id": 99901, "charttime": base_time, "itemid": 220277, "valuenum": 98.0},   # SpO2 = 98
        {"stay_id": 99901, "charttime": base_time, "itemid": 220052, "valuenum": 85.0},   # MAP = 85
        {"stay_id": 99901, "charttime": base_time, "itemid": 223761, "valuenum": 98.6},   # Temp F = 98.6
        {"stay_id": 99901, "charttime": base_time, "itemid": 220739, "valuenum": 4.0},    # GCS Eye = 4
        {"stay_id": 99901, "charttime": base_time, "itemid": 223900, "valuenum": 5.0},    # GCS Verbal = 5
        {"stay_id": 99901, "charttime": base_time, "itemid": 223901, "valuenum": 6.0},    # GCS Motor = 6
        # Hour 2 (02:00)
        {"stay_id": 99901, "charttime": base_time + pd.Timedelta(hours=2), "itemid": 220045, "valuenum": 80.0},
        {"stay_id": 99901, "charttime": base_time + pd.Timedelta(hours=2), "itemid": 220277, "valuenum": 97.0},
        # Hour 4 (04:00) - Tachycardia event
        {"stay_id": 99901, "charttime": base_time + pd.Timedelta(hours=4), "itemid": 220045, "valuenum": 110.0},
        {"stay_id": 99901, "charttime": base_time + pd.Timedelta(hours=4), "itemid": 220052, "valuenum": 70.0},
        # Hour 5 (05:00) - Artifact: Impossible HR = 450 bpm (Must be rejected)
        {"stay_id": 99901, "charttime": base_time + pd.Timedelta(hours=5), "itemid": 220045, "valuenum": 450.0},
        # Hour 7 (07:00) - FUTURE Observation relative to t=06:00
        {"stay_id": 99901, "charttime": base_time + pd.Timedelta(hours=7), "itemid": 220045, "valuenum": 130.0},
    ]

    # Synthetic labs with specimen draw (charttime) and result availability (storetime)
    labs_data = [
        # Lab 1: Drawn at 01:00, stored at 02:00 -> Available at t >= 02:00
        {
            "stay_id": 99901,
            "charttime": base_time + pd.Timedelta(hours=1),
            "storetime": base_time + pd.Timedelta(hours=2),
            "itemid": 50813,  # Lactate
            "valuenum": 1.4,
        },
        # Lab 2: Drawn at 05:00, stored at 07:00 -> In transit at t = 06:00! MUST BE EXCLUDED at t=06:00
        {
            "stay_id": 99901,
            "charttime": base_time + pd.Timedelta(hours=5),
            "storetime": base_time + pd.Timedelta(hours=7),
            "itemid": 50813,  # Lactate = 4.2 (severe sepsis warning)
            "valuenum": 4.2,
        },
        # Lab 3: Drawn at 03:00, stored at 03:30 -> Available at t >= 03:30
        {
            "stay_id": 99901,
            "charttime": base_time + pd.Timedelta(hours=3),
            "storetime": base_time + pd.Timedelta(minutes=210),
            "itemid": 50912,  # Creatinine
            "valuenum": 1.2,
        },
    ]

    return base_time, pd.DataFrame(vitals_data), pd.DataFrame(labs_data)


def test_strict_temporal_cutoff_vitals(synthetic_patient_timeline):
    """Verifies observations with charttime > t are strictly excluded."""
    base_time, vitals_df, labs_df = synthetic_patient_timeline
    extractor = HourlySlidingWindowFeatureExtractor(vitals_window_hours=6.0)

    # Prediction time at Hour 6 (06:00)
    t = base_time + pd.Timedelta(hours=6)
    feat = extractor.extract_features(prediction_time=t, vitals_df=vitals_df, labs_df=labs_df)

    # Hour 7 HR = 130 bpm must NOT be seen
    # Maximum valid HR before Hour 6 was 110 bpm at Hour 4
    assert feat["hr_max"] == 110.0
    assert feat["hr_last"] == 110.0
    assert feat["hr_n"] == 3  # Hour 0 (72), Hour 2 (80), Hour 4 (110). Hour 5 was rejected as artifact.


def test_strict_storetime_availability_labs(synthetic_patient_timeline):
    """
    CRITICAL ANTI-LEAKAGE:
    A lab drawn at 05:00 but stored/verified at 07:00 must be EXCLUDED at t = 06:00.
    At t = 07:30, it must be INCLUDED.
    """
    base_time, vitals_df, labs_df = synthetic_patient_timeline
    extractor = HourlySlidingWindowFeatureExtractor(labs_window_hours=24.0)

    # 1. Prediction at t = 06:00: Lab 2 is in transit (storetime = 07:00)
    t_early = base_time + pd.Timedelta(hours=6)
    feat_early = extractor.extract_features(prediction_time=t_early, vitals_df=vitals_df, labs_df=labs_df)

    # Only Lab 1 (1.4 mmol/L) was verified before t=06:00
    assert feat_early["lactate_last"] == 1.4
    assert feat_early["lactate_n"] == 1

    # 2. Prediction at t = 07:30: Lab 2 has now verified (storetime = 07:00 <= 07:30)
    t_late = base_time + pd.Timedelta(hours=7, minutes=30)
    feat_late = extractor.extract_features(prediction_time=t_late, vitals_df=vitals_df, labs_df=labs_df)

    assert feat_late["lactate_last"] == 4.2
    assert feat_late["lactate_n"] == 2


def test_physiological_bound_rejection(synthetic_patient_timeline):
    """Impossible values (e.g. HR = 450 bpm) must be discarded by physiological boundary screening."""
    base_time, vitals_df, labs_df = synthetic_patient_timeline
    extractor = HourlySlidingWindowFeatureExtractor()

    t = base_time + pd.Timedelta(hours=6)
    feat = extractor.extract_features(prediction_time=t, vitals_df=vitals_df, labs_df=labs_df)

    # 450 bpm is above 300 bpm bound and must not be hr_max
    assert feat["hr_max"] <= 300.0
    assert feat["hr_max"] == 110.0


def test_temperature_fahrenheit_conversion(synthetic_patient_timeline):
    """Verifies that Fahrenheit (item 223761) is converted to Celsius accurately."""
    base_time, vitals_df, labs_df = synthetic_patient_timeline
    extractor = HourlySlidingWindowFeatureExtractor()

    t = base_time + pd.Timedelta(hours=1)
    feat = extractor.extract_features(prediction_time=t, vitals_df=vitals_df, labs_df=labs_df)

    # 98.6 F converts to exactly 37.0 C
    assert "temp_c_last" in feat
    assert abs(feat["temp_c_last"] - 37.0) < 0.1


def test_sliding_window_aggregation(synthetic_patient_timeline):
    """Observations older than the lookback window are excluded from windowed statistics."""
    base_time, vitals_df, labs_df = synthetic_patient_timeline
    # Set small 3-hour window
    extractor = HourlySlidingWindowFeatureExtractor(vitals_window_hours=3.0)

    # t = 06:00. Lookback window is [03:00, 06:00]
    t = base_time + pd.Timedelta(hours=6)
    feat = extractor.extract_features(prediction_time=t, vitals_df=vitals_df, labs_df=labs_df)

    # Hour 0 (72) and Hour 2 (80) are outside the 3h window.
    # Only Hour 4 (110) is inside [03:00, 06:00].
    assert feat["hr_n"] == 1
    assert feat["hr_mean"] == 110.0
    assert feat["hr_min"] == 110.0
    assert feat["hr_max"] == 110.0


def test_staleness_detection(synthetic_patient_timeline):
    """Checks that measurements older than vitals_stale_hours are flagged as stale."""
    base_time, vitals_df, labs_df = synthetic_patient_timeline
    # Staleness threshold of 2.0 hours
    extractor = HourlySlidingWindowFeatureExtractor(vitals_stale_hours=2.0)

    # Map was measured at 04:00. At t = 07:00, it is 3 hours old (stale)
    t = base_time + pd.Timedelta(hours=7)
    feat = extractor.extract_features(prediction_time=t, vitals_df=vitals_df, labs_df=labs_df)

    assert feat["map_hours_since_last"] == 3.0
    assert feat["map_is_stale"] == 1


def test_missingness_preservation_and_imputation():
    """Verifies that missingness flags are preserved even after continuous imputation."""
    features = {
        "hr_mean": np.nan,
        "hr_measured": 0,
        "hr_n": 0,
        "lactate_mean": np.nan,
        "lactate_measured": 0,
        "lactate_n": 0,
        "temp_c_mean": 37.2,
        "temp_c_measured": 1,
        "temp_c_n": 2,
    }
    train_medians = {
        "hr_mean": 82.0,
        "lactate_mean": 1.6,
        "temp_c_mean": 36.8,
    }

    imputed = HourlySlidingWindowFeatureExtractor.impute_features(features, train_medians)

    # Continuous values imputed
    assert imputed["hr_mean"] == 82.0
    assert imputed["lactate_mean"] == 1.6
    # Original measured value untouched
    assert imputed["temp_c_mean"] == 37.2
    # Missingness indicator flags strictly preserved
    assert imputed["hr_measured"] == 0
    assert imputed["hr_n"] == 0
    assert imputed["lactate_measured"] == 0


def test_gcs_total_calculation(synthetic_patient_timeline):
    """GCS eye, verbal, motor sum to total GCS."""
    base_time, vitals_df, labs_df = synthetic_patient_timeline
    extractor = HourlySlidingWindowFeatureExtractor()

    t = base_time + pd.Timedelta(hours=1)
    feat = extractor.extract_features(prediction_time=t, vitals_df=vitals_df, labs_df=labs_df)

    assert feat["gcs_eye_last"] == 4.0
    assert feat["gcs_verbal_last"] == 5.0
    assert feat["gcs_motor_last"] == 6.0
    assert feat["gcs_total_last"] == 15.0
