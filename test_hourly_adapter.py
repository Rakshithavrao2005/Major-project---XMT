"""
tests/test_hourly_adapter.py
─────────────────────────────────────────────────────────────────────────────
Stage 1 Isolated Adapter Smoke Test Suite.

PROVENANCE & INTEGRITY NOTICE:
All records in this test suite are 100% SYNTHETIC FIXTURES (Stay IDs 90001-90005).
Zero real patient records from MIMIC-IV are used.
Tests verify:
1. Wide-to-long transformation for vitals.
2. Long-lab filtering, anti-leakage storetime filtering, and timestamp anomaly audits.
3. Target alignment and active surveillance masking.
4. Feature extractor compatibility without fabricated GCS / FiO2.
5. PyTorch Temporal Transformer forward-pass and finite gradients.
"""

import datetime
import numpy as np
import pandas as pd
import pytest
import torch

from src.hourly.adapter import (
    melt_wide_vitals_chunk,
    process_labs_chunk,
    process_labels_chunk,
    WIDE_VITAL_TO_ITEMID,
)
from src.hourly.feature_extractor import HourlySlidingWindowFeatureExtractor
from src.hourly.temporal_model import HourlyTemporalTransformer
from src.hourly.train_pipeline import MaskedBCEWithLogitsLoss


@pytest.fixture
def synthetic_five_stays_fixture():
    """
    Creates an isolated 5-stay synthetic test fixture:
    - Stay 90001: True negative control (stays 10h, no sepsis).
    - Stay 90002: Incident sepsis at Hour 6 (Positive at hr 5, post-onset hr 6+).
    - Stay 90003: Discharge censored (discharged at hour 5).
    - Stay 90004: Lab latency test (storetime before, equal to, and after prediction time).
    - Stay 90005: Clinical edge cases (duplicate events, missing vitals, reverse timestamps).
    """
    base_time = pd.Timestamp("2026-10-21 00:00:00")

    # 1. Wide Vitals (Simulating longitudinal_vitals.zip schema)
    vitals_rows = []
    # Stay 90001: Normal trajectory
    for hr in range(10):
        t = base_time + pd.Timedelta(hours=hr)
        vitals_rows.append({
            "subject_id": 1001, "hadm_id": 2001, "stay_id": 90001,
            "charttime": str(t), "heart_rate": 72.0 + hr, "sbp": 120.0, "dbp": 80.0,
            "mbp": 93.3, "sbp_ni": np.nan, "dbp_ni": np.nan, "mbp_ni": np.nan,
            "resp_rate": 16.0, "temperature": 37.0, "temperature_site": "Oral",
            "spo2": 98.0, "glucose": 105.0
        })

    # Stay 90004: Hour 0 to 4 vitals
    for hr in range(5):
        t = base_time + pd.Timedelta(hours=hr)
        vitals_rows.append({
            "subject_id": 1004, "hadm_id": 2004, "stay_id": 90004,
            "charttime": str(t), "heart_rate": 85.0, "sbp": 110.0, "dbp": 70.0,
            "mbp": 83.3, "sbp_ni": np.nan, "dbp_ni": np.nan, "mbp_ni": np.nan,
            "resp_rate": 18.0, "temperature": 36.8, "temperature_site": "Oral",
            "spo2": 97.0, "glucose": 110.0
        })

    # Stay 90005: Duplicates & Missing vitals
    t0 = base_time
    # Normal reading
    vitals_rows.append({
        "subject_id": 1005, "hadm_id": 2005, "stay_id": 90005,
        "charttime": str(t0), "heart_rate": 80.0, "sbp": 120.0, "dbp": 80.0,
        "mbp": 93.0, "sbp_ni": np.nan, "dbp_ni": np.nan, "mbp_ni": np.nan,
        "resp_rate": 16.0, "temperature": 37.0, "temperature_site": "Oral",
        "spo2": 98.0, "glucose": 100.0
    })
    # Duplicate timestamp with slightly different heart_rate
    vitals_rows.append({
        "subject_id": 1005, "hadm_id": 2005, "stay_id": 90005,
        "charttime": str(t0), "heart_rate": 82.0, "sbp": 120.0, "dbp": 80.0,
        "mbp": 93.0, "sbp_ni": np.nan, "dbp_ni": np.nan, "mbp_ni": np.nan,
        "resp_rate": 16.0, "temperature": 37.0, "temperature_site": "Oral",
        "spo2": 98.0, "glucose": 100.0
    })
    # Reading with missing BP and temp
    vitals_rows.append({
        "subject_id": 1005, "hadm_id": 2005, "stay_id": 90005,
        "charttime": str(t0 + pd.Timedelta(hours=1)), "heart_rate": 84.0,
        "sbp": np.nan, "dbp": np.nan, "mbp": np.nan, "sbp_ni": np.nan,
        "dbp_ni": np.nan, "mbp_ni": np.nan, "resp_rate": 18.0,
        "temperature": np.nan, "temperature_site": None, "spo2": 99.0, "glucose": np.nan
    })

    # 2. Long Labs (Simulating bq-results-...172200.zip schema)
    labs_rows = [
        # Stay 90001: Baseline labs at Hour 0 (stored at Hour 1)
        {
            "subject_id": 1001, "hadm_id": 2001, "stay_id": 90001,
            "charttime": str(base_time), "storetime": str(base_time + pd.Timedelta(hours=1)),
            "itemid": 50813, "item_label": "Lactate", "valuenum": 1.2
        },
        {
            "subject_id": 1001, "hadm_id": 2001, "stay_id": 90001,
            "charttime": str(base_time), "storetime": str(base_time + pd.Timedelta(hours=1)),
            "itemid": 51265, "item_label": "Platelet Count", "valuenum": 250.0
        },
        # Stay 90004: Timing tests relative to Prediction Time t = 03:00
        # Draw 1: Drawn at 01:00, stored at 02:00 -> BEFORE t (must be INCLUDED)
        {
            "subject_id": 1004, "hadm_id": 2004, "stay_id": 90004,
            "charttime": str(base_time + pd.Timedelta(hours=1)),
            "storetime": str(base_time + pd.Timedelta(hours=2)),
            "itemid": 50813, "item_label": "Lactate", "valuenum": 1.5
        },
        # Draw 2: Drawn at 02:00, stored at 03:00 -> EQUAL to t (must be INCLUDED)
        {
            "subject_id": 1004, "hadm_id": 2004, "stay_id": 90004,
            "charttime": str(base_time + pd.Timedelta(hours=2)),
            "storetime": str(base_time + pd.Timedelta(hours=3)),
            "itemid": 50813, "item_label": "Lactate", "valuenum": 2.1
        },
        # Draw 3: Drawn at 02:30, stored at 04:00 -> AFTER t (IN TRANSIT! MUST BE EXCLUDED at t=03:00)
        {
            "subject_id": 1004, "hadm_id": 2004, "stay_id": 90004,
            "charttime": str(base_time + pd.Timedelta(minutes=150)),
            "storetime": str(base_time + pd.Timedelta(hours=4)),
            "itemid": 50813, "item_label": "Lactate", "valuenum": 4.5
        },
        # Stay 90005: Anomalies
        # Reverse timestamp: storetime earlier than charttime (data entry lag error)
        {
            "subject_id": 1005, "hadm_id": 2005, "stay_id": 90005,
            "charttime": str(base_time + pd.Timedelta(hours=2)),
            "storetime": str(base_time + pd.Timedelta(hours=1)),
            "itemid": 50912, "item_label": "Creatinine", "valuenum": 1.1
        },
        # Null storetime
        {
            "subject_id": 1005, "hadm_id": 2005, "stay_id": 90005,
            "charttime": str(base_time + pd.Timedelta(hours=1)),
            "storetime": None,
            "itemid": 50912, "item_label": "Creatinine", "valuenum": 1.0
        },
        # Duplicate measurement at exact same timestamp
        {
            "subject_id": 1005, "hadm_id": 2005, "stay_id": 90005,
            "charttime": str(base_time + pd.Timedelta(hours=3)),
            "storetime": str(base_time + pd.Timedelta(hours=3, minutes=30)),
            "itemid": 51265, "item_label": "Platelet Count", "valuenum": 180.0
        },
        {
            "subject_id": 1005, "hadm_id": 2005, "stay_id": 90005,
            "charttime": str(base_time + pd.Timedelta(hours=3)),
            "storetime": str(base_time + pd.Timedelta(hours=3, minutes=30)),
            "itemid": 51265, "item_label": "Platelet Count", "valuenum": 182.0
        },
    ]

    # 3. Hourly Labels (Simulating hourly_sepsis_label.csv schema)
    labels_rows = []
    # Stay 90001: All negative hours
    for hr in range(10):
        t_start = base_time + pd.Timedelta(hours=hr)
        t_end = t_start + pd.Timedelta(hours=1)
        labels_rows.append({
            "subject_id": 1001, "hadm_id": 2001, "stay_id": 90001,
            "hour_index": hr, "starttime": str(t_start), "endtime": str(t_end),
            "onset_time": "", "label_status": "NEGATIVE"
        })

    # Stay 90002: Incident sepsis at Hour 6
    onset_t = base_time + pd.Timedelta(hours=6)
    for hr in range(10):
        t_start = base_time + pd.Timedelta(hours=hr)
        t_end = t_start + pd.Timedelta(hours=1)
        if hr < 5:
            st = "NEGATIVE"
        elif hr == 5:
            st = "POSITIVE"  # Sepsis occurs in (t_end, t_end + 1h] -> [05:00, 06:00] predicting 06:00
        else:
            st = "POST_ONSET"
        labels_rows.append({
            "subject_id": 1002, "hadm_id": 2002, "stay_id": 90002,
            "hour_index": hr, "starttime": str(t_start), "endtime": str(t_end),
            "onset_time": str(onset_t), "label_status": st
        })

    # Stay 90003: Discharge censored at Hour 4
    for hr in range(5):
        t_start = base_time + pd.Timedelta(hours=hr)
        t_end = t_start + pd.Timedelta(hours=1)
        st = "DISCHARGE_CENSORED" if hr >= 3 else "NEGATIVE"
        labels_rows.append({
            "subject_id": 1003, "hadm_id": 2003, "stay_id": 90003,
            "hour_index": hr, "starttime": str(t_start), "endtime": str(t_end),
            "onset_time": "", "label_status": st
        })

    vitals_df = pd.DataFrame(vitals_rows)
    labs_df = pd.DataFrame(labs_rows)
    labels_df = pd.DataFrame(labels_rows)

    return {
        "base_time": base_time,
        "valid_stays": {90001, 90002, 90003, 90004, 90005},
        "vitals_df": vitals_df,
        "labs_df": labs_df,
        "labels_df": labels_df,
    }


def test_vitals_adapter_melting_and_filtering(synthetic_five_stays_fixture):
    """Verifies wide vitals are melted to canonical long format without fabricating GCS/FiO2."""
    raw_vitals = synthetic_five_stays_fixture["vitals_df"]
    valid_stays = synthetic_five_stays_fixture["valid_stays"]

    melted, metrics = melt_wide_vitals_chunk(raw_vitals, valid_stay_ids=valid_stays)

    assert set(melted.columns) == {"stay_id", "charttime", "itemid", "valuenum"}
    assert metrics["retained_stays"] == 3  # Stays 90001, 90004, 90005 have vitals
    assert metrics["melted_events"] > 0
    assert not melted.isna().any().any()

    # Confirm canonical item IDs are present
    assert 220045 in melted["itemid"].values  # HR
    assert 220277 in melted["itemid"].values  # SpO2
    assert 223762 in melted["itemid"].values  # Temp Celsius

    # Confirm GCS and FiO2 item IDs are strictly absent (NOT fabricated)
    assert 220739 not in melted["itemid"].values  # GCS Eye opening
    assert 223835 not in melted["itemid"].values  # Bedside FiO2


def test_labs_adapter_storetime_availability_and_anomaly_audits(synthetic_five_stays_fixture):
    """
    Verifies:
    1. Null storetime rows are filtered out.
    2. Chronological anomalies (storetime < charttime) are audited.
    3. Duplicate measurements are detected and counted.
    """
    raw_labs = synthetic_five_stays_fixture["labs_df"]
    valid_stays = synthetic_five_stays_fixture["valid_stays"]

    processed_labs, metrics = process_labs_chunk(raw_labs, valid_stay_ids=valid_stays, allow_null_storetime=False)

    assert set(processed_labs.columns) == {"stay_id", "charttime", "storetime", "itemid", "valuenum"}
    # One lab had null storetime -> must be dropped
    assert metrics["dropped_null_storetime"] == 1
    # One lab had reverse timestamps (storetime < charttime) -> must be flagged
    assert metrics["anomalous_reverse_timestamps"] == 1
    # Duplicate measurement at 03:00 for stay 90005 -> must be detected
    assert metrics["duplicate_measurements"] == 2


def test_temporal_feature_extractor_storetime_gate(synthetic_five_stays_fixture):
    """
    Tests that at prediction time t = 03:00 for Stay 90004:
    - Lab stored at 02:00 is visible.
    - Lab stored at 03:00 is visible (storetime == t).
    - Lab drawn at 02:30 but stored at 04:00 is strictly INVISIBLE (storetime > t).
    """
    base_time = synthetic_five_stays_fixture["base_time"]
    raw_vitals = synthetic_five_stays_fixture["vitals_df"]
    raw_labs = synthetic_five_stays_fixture["labs_df"]

    melted_vitals, _ = melt_wide_vitals_chunk(raw_vitals)
    processed_labs, _ = process_labs_chunk(raw_labs)

    # Filter to stay 90004
    v4 = melted_vitals[melted_vitals["stay_id"] == 90004]
    l4 = processed_labs[processed_labs["stay_id"] == 90004]

    extractor = HourlySlidingWindowFeatureExtractor(vitals_window_hours=6.0, labs_window_hours=24.0)

    # Prediction time at Hour 3 (03:00)
    t = base_time + pd.Timedelta(hours=3)
    feats = extractor.extract_features(prediction_time=t, vitals_df=v4, labs_df=l4)

    # In Stay 90004:
    # Lactate at 01:00 (stored 02:00) = 1.5
    # Lactate at 02:00 (stored 03:00) = 2.1
    # Lactate at 02:30 (stored 04:00) = 4.5 -> MUST BE EXCLUDED!
    assert feats["lactate_last"] == 2.1
    assert feats["lactate_max"] == 2.1
    assert feats["lactate_n"] == 2  # Only the 2 verified labs are counted
    assert feats["lactate_measured"] == 1

    # Unmeasured features must be NaN, NOT fabricated
    assert np.isnan(feats["gcs_total_last"])
    assert np.isnan(feats["pao2_fio2_ratio"])
    assert feats["gcs_eye_measured"] == 0
    assert feats["pao2_measured"] == 0


def test_labels_adapter_active_surveillance_and_censoring(synthetic_five_stays_fixture):
    """
    Verifies that:
    - POSITIVE label produces is_septic_next_hour = 1, active_surveillance_mask = 1.
    - NEGATIVE label produces is_septic_next_hour = 0, active_surveillance_mask = 1.
    - POST_ONSET produces is_septic_next_hour = NaN, active_surveillance_mask = 0.
    - DISCHARGE_CENSORED produces is_septic_next_hour = NaN, active_surveillance_mask = 0.
    """
    raw_labels = synthetic_five_stays_fixture["labels_df"]
    valid_stays = synthetic_five_stays_fixture["valid_stays"]

    processed_labels, metrics = process_labels_chunk(raw_labels, valid_stay_ids=valid_stays)

    assert metrics["positive_hours"] == 1
    assert metrics["post_onset_censored"] == 4
    assert metrics["discharge_censored"] == 2

    # Check Stay 90002 (Incident sepsis at Hour 6)
    s2 = processed_labels[processed_labels["stay_id"] == 90002].sort_values("hour_index")
    # Hour 5: Pre-onset prediction window -> active, target = 1
    h5 = s2[s2["hour_index"] == 5].iloc[0]
    assert h5["is_septic_next_hour"] == 1.0
    assert h5["active_surveillance_mask"] == 1

    # Hour 6: Post-onset -> censored, mask = 0, target = NaN
    h6 = s2[s2["hour_index"] == 6].iloc[0]
    assert pd.isna(h6["is_septic_next_hour"])
    assert h6["active_surveillance_mask"] == 0


def test_model_forward_pass_and_finite_gradients_on_synthetic_tensors():
    """
    Constructs a small synthetic batch of feature sequences and confirms:
    - HourlyTemporalTransformer forward pass produces logits of shape (B, T, 1).
    - MaskedBCEWithLogitsLoss computes scalar loss ignoring censored steps.
    - Backward pass computes finite, non-NaN gradients across all model parameters.
    """
    batch_size = 3
    seq_len = 8
    num_features = 25

    # Synthetic features: (B, T, D)
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, num_features)

    # Targets: next-hour labels, with NaNs at censored positions
    targets = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, np.nan, np.nan, np.nan, np.nan],  # Septic patient (censored after onset)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan],        # Discharge censored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],              # Negative control
    ], dtype=torch.float32)

    # Active surveillance masks: 1 = surveillance, 0 = censored
    masks = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=torch.float32)

    model = HourlyTemporalTransformer(
        num_features=num_features,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
    )
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward pass
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 1)

    # Loss computation
    loss = criterion(logits, targets, masks)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Confirm all parameter gradients are finite and non-NaN
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


# ── Stage 2.1 Targeted Duplicate-Handling Test Suite ─────────────────────────

def test_exact_duplicate_rows_collapsed():
    """Confirms identical rows (same stay, charttime, storetime, item, val) collapse to 1 row."""
    base_t = "2026-10-21 10:00:00"
    store_t = "2026-10-21 11:00:00"
    raw_df = pd.DataFrame([
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t, "itemid": 50813, "valuenum": 2.1},
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t, "itemid": 50813, "valuenum": 2.1},
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t, "itemid": 50813, "valuenum": 2.1},
    ])
    res, metrics = process_labs_chunk(raw_df)

    assert len(res) == 1
    assert res.iloc[0]["valuenum"] == 2.1
    assert metrics["exact_duplicates_collapsed"] == 2
    assert metrics["duplicate_measurements"] == 3


def test_conflicting_values_same_timestamp_preserved_without_averaging():
    """
    Confirms simultaneous conflicting measurements (same chart & store, differing val)
    are BOTH preserved deterministically. Zero synthetic averaging.
    """
    base_t = "2026-10-21 10:00:00"
    store_t = "2026-10-21 11:00:00"
    raw_df = pd.DataFrame([
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t, "itemid": 50813, "valuenum": 2.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t, "itemid": 50813, "valuenum": 4.0},
    ])
    res, metrics = process_labs_chunk(raw_df)

    assert len(res) == 2
    assert res.iloc[0]["valuenum"] == 2.0
    assert res.iloc[1]["valuenum"] == 4.0
    assert metrics["conflicting_values_detected"] == 2
    assert metrics["duplicate_measurements"] == 2


def test_sequential_revisions_distinct_storetimes_preserved():
    """
    Confirms same charttime with distinct storetimes (e.g. preliminary vs final verification)
    are BOTH preserved in staging to prevent false data blackout or lookahead leakage.
    """
    base_t = "2026-10-21 10:00:00"
    store_t1 = "2026-10-21 11:00:00"
    store_t2 = "2026-10-21 12:30:00"
    raw_df = pd.DataFrame([
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t1, "itemid": 50813, "valuenum": 2.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t2, "itemid": 50813, "valuenum": 2.8},
    ])
    res, metrics = process_labs_chunk(raw_df)

    assert len(res) == 2
    assert metrics["sequential_revisions_retained"] == 2
    assert metrics["exact_duplicates_collapsed"] == 0
    assert metrics["conflicting_values_detected"] == 0

    # Ensure chronologically sorted by storetime
    assert res.iloc[0]["storetime"] == pd.Timestamp(store_t1)
    assert res.iloc[0]["valuenum"] == 2.0
    assert res.iloc[1]["storetime"] == pd.Timestamp(store_t2)
    assert res.iloc[1]["valuenum"] == 2.8


def test_duplicate_handling_deterministic_order():
    """Confirms shuffling input rows produces identical deterministic output DataFrame."""
    base_t = "2026-10-21 10:00:00"
    rows = [
        {"stay_id": 90001, "charttime": base_t, "storetime": "2026-10-21 11:00:00", "itemid": 50813, "valuenum": 2.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": "2026-10-21 11:00:00", "itemid": 50813, "valuenum": 4.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": "2026-10-21 12:30:00", "itemid": 50813, "valuenum": 3.5},
        {"stay_id": 90002, "charttime": base_t, "storetime": "2026-10-21 11:15:00", "itemid": 50912, "valuenum": 1.1},
    ]
    df_order_a = pd.DataFrame(rows)
    df_order_b = pd.DataFrame(list(reversed(rows)))

    res_a, _ = process_labs_chunk(df_order_a)
    res_b, _ = process_labs_chunk(df_order_b)

    pd.testing.assert_frame_equal(res_a, res_b)


def test_conservative_observation_policy_sequential_distinct_storetimes():
    """
    Verifies the Conservative Laboratory Observation Policy:
    - Two distinct values with the same lab name and charttime but different storetime values:
      1. Before later storetime (t = 11:30): Future result at 12:30 is strictly excluded.
         Only the 11:00 result (2.0) is available (lactate_last=2.0, lactate_n=1, lactate_mean=2.0).
      2. After later storetime (t = 13:00): Both distinct observations are retained.
         Neither observation is discarded.
         Both contribute to summary stats:
         lactate_n = 2, lactate_min = 2.0, lactate_max = 2.8, lactate_mean = 2.4, lactate_last = 2.8.
    """
    base_t = "2026-10-21 10:00:00"
    store_t1 = "2026-10-21 11:00:00"
    store_t2 = "2026-10-21 12:30:00"
    raw_df = pd.DataFrame([
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t1, "itemid": 50813, "valuenum": 2.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t2, "itemid": 50813, "valuenum": 2.8},
    ])
    res_labs, _ = process_labs_chunk(raw_df)

    extractor = HourlySlidingWindowFeatureExtractor()

    # Evaluation at t = 11:30: only first result is available
    t_early = pd.Timestamp("2026-10-21 11:30:00")
    feats_early = extractor.extract_features(t_early, labs_df=res_labs)
    assert feats_early["lactate_last"] == 2.0
    assert feats_early["lactate_n"] == 1
    assert feats_early["lactate_mean"] == 2.0
    assert feats_early["lactate_min"] == 2.0
    assert feats_early["lactate_max"] == 2.0

    # Evaluation at t = 13:00: both results are available and retained
    t_late = pd.Timestamp("2026-10-21 13:00:00")
    feats_late = extractor.extract_features(t_late, labs_df=res_labs)
    assert feats_late["lactate_last"] == 2.8
    assert feats_late["lactate_n"] == 2
    assert feats_late["lactate_min"] == 2.0
    assert feats_late["lactate_max"] == 2.8
    assert feats_late["lactate_mean"] == 2.4


def test_simultaneous_distinct_measurements_preserved_in_feature_extractor():
    """
    Verifies that distinct simultaneous measurements (same charttime & storetime, differing valuenum)
    are BOTH preserved and contribute accurately to window stats without synthetic averaging.
    """
    base_t = "2026-10-21 10:00:00"
    store_t = "2026-10-21 11:00:00"
    raw_df = pd.DataFrame([
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t, "itemid": 51301, "valuenum": 7.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": store_t, "itemid": 51301, "valuenum": 18.0},
    ])
    res_labs, _ = process_labs_chunk(raw_df)

    extractor = HourlySlidingWindowFeatureExtractor()
    t_pred = pd.Timestamp("2026-10-21 12:00:00")
    feats = extractor.extract_features(t_pred, labs_df=res_labs)

    assert feats["wbc_n"] == 2
    assert feats["wbc_min"] == 7.0
    assert feats["wbc_max"] == 18.0
    assert feats["wbc_mean"] == 12.5


def test_shuffling_input_rows_does_not_change_extracted_features():
    """
    Verifies that shuffling input laboratory and vitals rows does not alter
    the resulting extracted feature dictionary, guaranteeing full determinism.
    """
    base_t = "2026-10-21 10:00:00"
    labs_rows = [
        {"stay_id": 90001, "charttime": base_t, "storetime": "2026-10-21 11:00:00", "itemid": 50813, "valuenum": 2.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": "2026-10-21 12:30:00", "itemid": 50813, "valuenum": 2.8},
        {"stay_id": 90001, "charttime": base_t, "storetime": "2026-10-21 11:00:00", "itemid": 51301, "valuenum": 7.0},
        {"stay_id": 90001, "charttime": base_t, "storetime": "2026-10-21 11:00:00", "itemid": 51301, "valuenum": 18.0},
        {"stay_id": 90001, "charttime": "2026-10-21 09:00:00", "storetime": "2026-10-21 09:30:00", "itemid": 50912, "valuenum": 1.2},
    ]
    vitals_rows = [
        {"stay_id": 90001, "charttime": base_t, "itemid": 220045, "valuenum": 80.0},
        {"stay_id": 90001, "charttime": base_t, "itemid": 220045, "valuenum": 84.0},
        {"stay_id": 90001, "charttime": "2026-10-21 09:00:00", "itemid": 220052, "valuenum": 75.0},
    ]

    extractor = HourlySlidingWindowFeatureExtractor()
    t_eval = pd.Timestamp("2026-10-21 13:00:00")

    # Order 1: original
    df_l1, _ = process_labs_chunk(pd.DataFrame(labs_rows))
    df_v1 = pd.DataFrame(vitals_rows)
    f1 = extractor.extract_features(t_eval, vitals_df=df_v1, labs_df=df_l1)

    # Order 2: reversed / permuted
    df_l2, _ = process_labs_chunk(pd.DataFrame(list(reversed(labs_rows))))
    df_v2 = pd.DataFrame(list(reversed(vitals_rows)))
    f2 = extractor.extract_features(t_eval, vitals_df=df_v2, labs_df=df_l2)

    for k in f1:
        if isinstance(f1[k], float) and np.isnan(f1[k]):
            assert np.isnan(f2[k]), f"Mismatch on {k}: {f1[k]} vs {f2[k]}"
        else:
            assert f1[k] == f2[k], f"Mismatch on {k}: {f1[k]} vs {f2[k]}"


def test_biological_timestamp_alignment_newer_specimen_not_overridden():
    """
    Stage 2.7 Biological Alignment Test:
    Verifies that an older specimen with a later storetime does NOT override
    a newer specimen with an earlier storetime.
    - Specimen A: drawn 06:00, stored 09:30, lactate = 1.2
    - Specimen B: drawn 08:30, stored 08:45, lactate = 3.8
    At t = 10:00:
    - lactate_last must be 3.8 (from 08:30 specimen, NOT 1.2 from 06:00 specimen).
    - lactate_delta must be 3.8 - 1.2 = +2.6 (in collection-time order).
    - lactate_hours_since_last must be 1.5h (10:00 - 08:30 charttime, NOT 0.5h from storetime).
    """
    raw_df = pd.DataFrame([
        {
            "stay_id": 90001,
            "charttime": "2026-10-21 06:00:00",
            "storetime": "2026-10-21 09:30:00",
            "itemid": 50813,
            "valuenum": 1.2,
        },
        {
            "stay_id": 90001,
            "charttime": "2026-10-21 08:30:00",
            "storetime": "2026-10-21 08:45:00",
            "itemid": 50813,
            "valuenum": 3.8,
        },
    ])
    res_labs, _ = process_labs_chunk(raw_df)

    extractor = HourlySlidingWindowFeatureExtractor()
    t_eval = pd.Timestamp("2026-10-21 10:00:00")
    feats = extractor.extract_features(t_eval, labs_df=res_labs)

    assert feats["lactate_last"] == 3.8
    assert feats["lactate_delta"] == pytest.approx(2.6)
    assert feats["lactate_hours_since_last"] == 1.5
    assert feats["lactate_n"] == 2
    assert feats["lactate_min"] == 1.2
    assert feats["lactate_max"] == 3.8
    assert feats["lactate_mean"] == 2.5


def test_temporal_gating_future_storetime_and_charttime_excluded():
    """
    Verifies that:
    1. A lab with storetime > prediction_time is strictly excluded.
    2. A lab with charttime > prediction_time is strictly excluded.
    """
    raw_df = pd.DataFrame([
        # Valid historical lab
        {
            "stay_id": 90001,
            "charttime": "2026-10-21 08:00:00",
            "storetime": "2026-10-21 08:45:00",
            "itemid": 50813,
            "valuenum": 1.5,
        },
        # Future storetime (in transit)
        {
            "stay_id": 90001,
            "charttime": "2026-10-21 09:30:00",
            "storetime": "2026-10-21 10:30:00",
            "itemid": 50813,
            "valuenum": 4.5,
        },
        # Future charttime (data recording anomaly)
        {
            "stay_id": 90001,
            "charttime": "2026-10-21 10:15:00",
            "storetime": "2026-10-21 10:00:00",
            "itemid": 50813,
            "valuenum": 5.0,
        },
    ])
    res_labs, _ = process_labs_chunk(raw_df)

    extractor = HourlySlidingWindowFeatureExtractor()
    t_eval = pd.Timestamp("2026-10-21 10:00:00")
    feats = extractor.extract_features(t_eval, labs_df=res_labs)

    assert feats["lactate_n"] == 1
    assert feats["lactate_last"] == 1.5
    assert feats["lactate_max"] == 1.5


def test_lookback_window_based_on_charttime():
    """
    Verifies that the 24-hour lookback window is determined strictly by charttime.
    - Draw 1: charttime = 25h ago, storetime = 23h ago -> Out of window!
    - Draw 2: charttime = 23h ago, storetime = 22h ago -> In window!
    """
    t_eval = pd.Timestamp("2026-10-21 12:00:00")
    raw_df_old = pd.DataFrame([
        {
            "stay_id": 90001,
            "charttime": "2026-10-20 11:00:00",  # 25 hours ago relative to t_eval
            "storetime": "2026-10-20 13:00:00",  # 23 hours ago
            "itemid": 50813,
            "valuenum": 1.5,
        }
    ])
    raw_df_in = pd.DataFrame([
        {
            "stay_id": 90001,
            "charttime": "2026-10-20 13:00:00",  # 23 hours ago relative to t_eval
            "storetime": "2026-10-20 14:00:00",  # 22 hours ago
            "itemid": 50813,
            "valuenum": 2.5,
        }
    ])
    extractor = HourlySlidingWindowFeatureExtractor(labs_window_hours=24.0, labs_stale_hours=24.0)

    # 1. 25h old charttime: out of window (n=0, measured=0, stale=1, mean=NaN, last=1.5)
    res_old, _ = process_labs_chunk(raw_df_old)
    feats_old = extractor.extract_features(t_eval, labs_df=res_old)
    assert feats_old["lactate_n"] == 0
    assert np.isnan(feats_old["lactate_mean"])
    assert feats_old["lactate_measured"] == 0
    assert feats_old["lactate_last"] == 1.5
    assert feats_old["lactate_hours_since_last"] == 25.0
    assert feats_old["lactate_is_stale"] == 1

    # 2. 23h old charttime: in window (n=1, measured=1, stale=0, mean=2.5, last=2.5)
    res_in, _ = process_labs_chunk(raw_df_in)
    feats_in = extractor.extract_features(t_eval, labs_df=res_in)
    assert feats_in["lactate_n"] == 1
    assert feats_in["lactate_mean"] == 2.5
    assert feats_in["lactate_measured"] == 1
    assert feats_in["lactate_last"] == 2.5
    assert feats_in["lactate_hours_since_last"] == 23.0
    assert feats_in["lactate_is_stale"] == 0
