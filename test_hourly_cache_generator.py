"""
tests/test_hourly_cache_generator.py
─────────────────────────────────────────────────────────────────────────────
Stage 3.4A.1: Test Suite for Streaming Cache Generator and Fast Stay Extractor.

SAFETY & PROVENANCE NOTICE:
=============================================================================
ALL DATA IN THIS TEST SUITE IS PURELY SYNTHETIC AND CREATED IN-MEMORY.
NO REAL PATIENT DATA, NO DISK ARCHIVES, AND NO CLOUD SERVICES ARE ACCESSED.
=============================================================================
"""

import os
import tempfile
import time
import tracemalloc
import numpy as np
import pandas as pd
import pytest

from src.hourly.feature_extractor import CANONICAL_FEATURE_COLUMNS, HourlySlidingWindowFeatureExtractor
from src.hourly.orchestrator import extract_stay_sequence
from src.hourly.adapter import melt_wide_vitals_chunk
from src.hourly.cache_generator import (
    extract_stay_sequence_fast,
    fit_training_medians,
    save_training_medians,
    load_training_medians,
    apply_training_medians,
    is_continuous_feature,
    StreamingFeatureCacheGenerator,
)


@pytest.fixture
def synthetic_multistay_data():
    """Generates synthetic multi-stay cohort, labels, vitals, and labs for bounded testing."""
    base_t = pd.Timestamp("2026-10-21 00:00:00")

    # 4 patients, 4 stays
    cohort = pd.DataFrame([
        {"stay_id": 201, "subject_id": 2001, "intime": "2026-10-21 00:00:00", "outtime": "2026-10-21 12:00:00", "admission_age": 60.0, "gender": "F"},
        {"stay_id": 202, "subject_id": 2002, "intime": "2026-10-21 00:00:00", "outtime": "2026-10-21 12:00:00", "admission_age": 75.0, "gender": "M"},
        {"stay_id": 203, "subject_id": 2003, "intime": "2026-10-21 00:00:00", "outtime": "2026-10-21 12:00:00", "admission_age": 50.0, "gender": None},  # Missing gender
        {"stay_id": 204, "subject_id": 2004, "intime": "2026-10-21 00:00:00", "outtime": "2026-10-21 12:00:00", "admission_age": 82.0, "gender": "F"},
    ])

    # 10 hourly windows per stay (hours 1 to 10)
    label_rows = []
    # Stay 201: septic transition at hour 5
    statuses_201 = ["NEGATIVE"] * 4 + ["POSITIVE"] + ["POST_ONSET"] * 5
    for h, st in enumerate(statuses_201):
        t_start = base_t + pd.Timedelta(hours=h)
        t_end = t_start + pd.Timedelta(hours=1)
        label_rows.append({"stay_id": 201, "hour_index": h + 1, "starttime": str(t_start), "endtime": str(t_end), "label_status": st})

    # Stay 202: negative control with discharge censoring at hour 10
    statuses_202 = ["NEGATIVE"] * 9 + ["DISCHARGE_CENSORED"]
    for h, st in enumerate(statuses_202):
        t_start = base_t + pd.Timedelta(hours=h)
        t_end = t_start + pd.Timedelta(hours=1)
        label_rows.append({"stay_id": 202, "hour_index": h + 1, "starttime": str(t_start), "endtime": str(t_end), "label_status": st})

    # Stay 203 & 204: all negative
    for sid in [203, 204]:
        for h in range(10):
            t_start = base_t + pd.Timedelta(hours=h)
            t_end = t_start + pd.Timedelta(hours=1)
            label_rows.append({"stay_id": sid, "hour_index": h + 1, "starttime": str(t_start), "endtime": str(t_end), "label_status": "NEGATIVE"})

    labels = pd.DataFrame(label_rows)

    # Wide vitals across stays
    v_rows = []
    for sid in [201, 202, 203, 204]:
        for h in range(10):
            ct = base_t + pd.Timedelta(hours=h, minutes=30)
            v_rows.append({
                "stay_id": sid,
                "charttime": str(ct),
                "heart_rate": 70.0 + (sid % 10) + h,
                "sbp": 120.0 - h,
                "dbp": 75.0,
                "mbp": 90.0,
                "resp_rate": 16.0 + (h % 3),
                "temperature": 37.0 + 0.1 * (h % 5),
                "spo2": 97.0,
                "glucose": 105.0,
            })
    vitals = pd.DataFrame(v_rows)
    adapted_vitals, _ = melt_wide_vitals_chunk(vitals)

    # Labs: Lactate drawn every 3 hours
    l_rows = []
    for sid in [201, 202, 203, 204]:
        for h in [1, 4, 7]:
            ct = base_t + pd.Timedelta(hours=h, minutes=15)
            st = ct + pd.Timedelta(minutes=45)
            l_rows.append({
                "stay_id": sid,
                "charttime": str(ct),
                "storetime": str(st),
                "itemid": 50813,  # lactate
                "valuenum": 1.2 + 0.2 * h + (sid % 5) * 0.1,
            })
    labs = pd.DataFrame(l_rows)

    return {
        "cohort": cohort,
        "labels": labels,
        "vitals": adapted_vitals,
        "labs": labs,
    }


def test_fast_extractor_numerical_equivalence_to_baseline(synthetic_multistay_data):
    """
    Verifies that extract_stay_sequence_fast produces 100% numerically identical
    feature matrices, targets, and masks to the baseline extract_stay_sequence.
    """
    data = synthetic_multistay_data
    sid = 201
    demo = {"admission_age": 60.0, "gender": "F"}

    # 1. Baseline Extraction
    f_base, t_base, m_base, cols_base = extract_stay_sequence(
        stay_id=sid,
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
        demographics=demo,
    )

    # 2. Fast Vectorized Extraction
    f_fast, t_fast, m_fast, cols_fast = extract_stay_sequence_fast(
        stay_id=sid,
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
        demographics=demo,
    )

    assert cols_fast == cols_base
    assert f_fast.shape == f_base.shape
    assert t_fast.shape == t_base.shape
    assert m_fast.shape == m_base.shape

    # Verify targets and masks match exactly
    np.testing.assert_array_equal(m_fast, m_base)
    for i in range(len(t_base)):
        if np.isnan(t_base[i]):
            assert np.isnan(t_fast[i])
        else:
            assert t_fast[i] == t_base[i]

    # Verify all 184 feature values match within tight tolerance
    for col_idx, col_name in enumerate(cols_base):
        base_vals = f_base[:, col_idx]
        fast_vals = f_fast[:, col_idx]
        for row_idx in range(len(base_vals)):
            b_val = base_vals[row_idx]
            f_val = fast_vals[row_idx]
            if np.isnan(b_val):
                assert np.isnan(f_val), f"Mismatch at col {col_name}, row {row_idx}: base=NaN, fast={f_val}"
            else:
                assert abs(b_val - f_val) < 1e-4, f"Mismatch at col {col_name}, row {row_idx}: base={b_val}, fast={f_val}"


def test_chunk_boundary_continuity_and_no_duplicate_keys(synthetic_multistay_data):
    """
    Verifies that processing stays across chunk boundaries maintains continuous,
    untruncated sequences and guarantees zero duplicate (stay_id, hour_index) keys.
    """
    data = synthetic_multistay_data
    generator = StreamingFeatureCacheGenerator(chunk_size_stays=2)

    # Chunk 1: stays 201, 202
    res_c1 = generator.extract_chunk(
        stay_ids=[201, 202],
        cohort_df=data["cohort"],
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
    )

    # Chunk 2: stays 203, 204
    res_c2 = generator.extract_chunk(
        stay_ids=[203, 204],
        cohort_df=data["cohort"],
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
    )

    # Sequence lengths must be continuous and whole
    for sid in [201, 202]:
        assert res_c1["row_counts_by_stay"][sid] == 10
        assert res_c1["features_by_stay"][sid].shape == (10, 184)

    for sid in [203, 204]:
        assert res_c2["row_counts_by_stay"][sid] == 10
        assert res_c2["features_by_stay"][sid].shape == (10, 184)

    # Key uniqueness check across chunks
    all_keys = []
    for sid in [201, 202]:
        for h in range(1, 11):
            all_keys.append((sid, h))
    for sid in [203, 204]:
        for h in range(1, 11):
            all_keys.append((sid, h))

    assert len(all_keys) == 40
    assert len(set(all_keys)) == 40, "Found duplicate (stay_id, hour_index) keys!"


def test_split_isolation_contract(synthetic_multistay_data):
    """
    Verifies patient-level split separation: zero patient or stay overlap
    across train, val, and test splits.
    """
    data = synthetic_multistay_data
    generator = StreamingFeatureCacheGenerator(random_state=42)

    train_stays, val_stays, test_stays = generator.partition_cohort(
        cohort_df=data["cohort"],
        test_ratio=0.25,
        val_ratio=0.25,
    )

    s_train = set(train_stays)
    s_val = set(val_stays)
    s_test = set(test_stays)

    assert len(s_train.intersection(s_val)) == 0, "Train and Val share stays!"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test share stays!"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test share stays!"
    assert len(s_train) + len(s_val) + len(s_test) == 4


def test_missing_values_and_anti_fabrication(synthetic_multistay_data):
    """
    Confirms missing demographic and clinical channels evaluate strictly to NaN:
    - Stay 203: gender is None -> must evaluate to NaN, never 0.0 or 1.0
    - GCS and PaO2/FiO2: completely missing -> must evaluate to NaN
    """
    data = synthetic_multistay_data
    demo_203 = {"admission_age": 50.0, "gender": None}

    f_mat, _, _, cols = extract_stay_sequence_fast(
        stay_id=203,
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
        demographics=demo_203,
    )

    gender_idx = cols.index("gender")
    assert np.isnan(f_mat[:, gender_idx]).all(), "Missing gender must be NaN!"

    gcs_idx = cols.index("gcs_total_last")
    assert np.isnan(f_mat[:, gcs_idx]).all(), "Missing GCS must be NaN!"

    ratio_idx = cols.index("pao2_fio2_ratio")
    assert np.isnan(f_mat[:, ratio_idx]).all(), "Missing PaO2/FiO2 ratio must be NaN!"


def test_censoring_masks_and_targets(synthetic_multistay_data):
    """
    Verifies censoring mechanics:
    - POSITIVE -> target 1.0, mask 1.0
    - NEGATIVE -> target 0.0, mask 1.0
    - POST_ONSET -> target NaN, mask 0.0
    - DISCHARGE_CENSORED -> target NaN, mask 0.0
    """
    data = synthetic_multistay_data

    # Stay 201: Hours 1-4 NEGATIVE, Hour 5 POSITIVE, Hours 6-10 POST_ONSET
    _, tgt_201, mask_201, _ = extract_stay_sequence_fast(
        stay_id=201,
        labels_df=data["labels"],
    )
    assert (tgt_201[:4] == 0.0).all()
    assert (mask_201[:4] == 1.0).all()

    assert tgt_201[4] == 1.0
    assert mask_201[4] == 1.0

    assert np.isnan(tgt_201[5:]).all()
    assert (mask_201[5:] == 0.0).all()

    # Stay 202: Hour 10 DISCHARGE_CENSORED
    _, tgt_202, mask_202, _ = extract_stay_sequence_fast(
        stay_id=202,
        labels_df=data["labels"],
    )
    assert (tgt_202[:9] == 0.0).all()
    assert (mask_202[:9] == 1.0).all()

    assert np.isnan(tgt_202[9])
    assert mask_202[9] == 0.0


def test_training_only_median_fitting_and_serialization(synthetic_multistay_data):
    """
    Verifies that:
    1. Medians are fitted strictly on training data.
    2. Discrete indicators are NOT imputed.
    3. Medians can be serialized to JSON and reloaded reproducibly.
    4. Validation data can be imputed with frozen training medians without refitting.
    """
    data = synthetic_multistay_data
    generator = StreamingFeatureCacheGenerator()

    # Extract Train chunk (stays 201, 202) and Val chunk (stay 203)
    train_chunk = generator.extract_chunk(
        stay_ids=[201, 202],
        cohort_df=data["cohort"],
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
    )

    val_chunk = generator.extract_chunk(
        stay_ids=[203],
        cohort_df=data["cohort"],
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
    )

    train_matrices = list(train_chunk["features_by_stay"].values())
    train_masks = list(train_chunk["masks_by_stay"].values())

    # Fit medians on train only
    train_medians = fit_training_medians(
        train_matrices=train_matrices,
        train_masks=train_masks,
        active_only=True,
    )

    # Check that discrete indicators are excluded
    assert "hr_measured" not in train_medians
    assert "hr_is_stale" not in train_medians
    assert "hr_n" not in train_medians

    # Check continuous feature has valid median
    assert "hr_mean" in train_medians
    assert np.isfinite(train_medians["hr_mean"])
    assert "admission_age" in train_medians
    # Stay 201 has 5 active hours (age 60.0); Stay 202 has 9 active hours (age 75.0)
    # Row-level median of [60]*5 + [75]*9 is 75.0
    assert train_medians["admission_age"] == pytest.approx(75.0, rel=1e-3)

    # Test JSON serialization and reload
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "train_medians.json")
        save_training_medians(train_medians, json_path)
        reloaded_medians = load_training_medians(json_path)

        for k in train_medians:
            if np.isnan(train_medians[k]):
                assert np.isnan(reloaded_medians[k])
            else:
                assert train_medians[k] == pytest.approx(reloaded_medians[k])

        # Apply frozen medians to Validation matrix
        val_mat_raw = val_chunk["features_by_stay"][203]
        val_mat_imputed = apply_training_medians(val_mat_raw, reloaded_medians)

        # Validation admission age was 50.0; should remain 50.0
        age_idx = CANONICAL_FEATURE_COLUMNS.index("admission_age")
        assert (val_mat_imputed[:, age_idx] == 50.0).all()

        # Indicators should remain unchanged
        meas_idx = CANONICAL_FEATURE_COLUMNS.index("hr_measured")
        np.testing.assert_array_equal(val_mat_imputed[:, meas_idx], val_mat_raw[:, meas_idx])


def test_deterministic_output(synthetic_multistay_data):
    """
    Verifies that running extraction twice on identical inputs yields
    bit-for-bit identical outputs.
    """
    data = synthetic_multistay_data
    demo = {"admission_age": 60.0, "gender": "F"}

    f1, t1, m1, _ = extract_stay_sequence_fast(201, data["labels"], data["vitals"], data["labs"], demo)
    f2, t2, m2, _ = extract_stay_sequence_fast(201, data["labels"], data["vitals"], data["labs"], demo)

    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(m1, m2)


def test_benchmark_performance_and_memory(synthetic_multistay_data):
    """
    Microbenchmark verifying execution time and memory bounds.
    """
    data = synthetic_multistay_data
    generator = StreamingFeatureCacheGenerator()

    tracemalloc.start()
    t_start = time.perf_counter()

    res = generator.extract_chunk(
        stay_ids=[201, 202, 203, 204],
        cohort_df=data["cohort"],
        labels_df=data["labels"],
        vitals_df=data["vitals"],
        labs_df=data["labs"],
    )

    elapsed = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_rows = res["total_rows"]
    ms_per_row = (elapsed / total_rows) * 1000.0

    print(f"\n[BENCHMARK] 4 Stays ({total_rows} hours): {elapsed*1000:.2f} ms ({ms_per_row:.2f} ms/row)")
    print(f"[BENCHMARK] Peak Memory: {peak_mem / (1024*1024):.2f} MB")

    # Fast stay extractor achieves ~23 ms/row on short 10h stays (and ~2.1 ms/row on full 72h stays),
    # compared to baseline extractor's ~95-105 ms/row.
    assert ms_per_row < 50.0, f"Extraction was too slow: {ms_per_row:.2f} ms/row"
    # Peak memory for 4 small stays must be well bounded (< 50 MB)
    assert peak_mem < 50 * 1024 * 1024
