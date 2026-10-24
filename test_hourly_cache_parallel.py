"""
tests/test_hourly_cache_parallel.py
─────────────────────────────────────────────────────────────────────────────
Stage 3.4A: Focused Synthetic Verification of Parallel Cache Generator.

SAFETY & PROVENANCE NOTICE:
=============================================================================
ALL DATA IN THIS TEST SUITE IS PURELY SYNTHETIC AND CREATED IN-MEMORY.
NO REAL PATIENT DATA, NO DISK ARCHIVES, AND NO CLOUD SERVICES ARE ACCESSED.
=============================================================================
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.hourly.feature_extractor import CANONICAL_FEATURE_COLUMNS
from src.hourly.adapter import melt_wide_vitals_chunk
from src.hourly.cache_generator import fit_training_medians, extract_stay_sequence_fast
from src.hourly.run_cache_parallel import (
    ParallelFeatureCacheGenerator,
    _worker_scan_stay_batch,
    _worker_stream_stay_batch,
    build_pyarrow_schema,
)


@pytest.fixture
def synthetic_parallel_data():
    """Generates synthetic multi-stay cohort, labels, vitals, and labs for bounded parallel testing."""
    base_t = pd.Timestamp("2026-10-22 00:00:00")

    # 6 stays across 6 subjects
    cohort = pd.DataFrame([
        {"stay_id": 301, "subject_id": 3001, "intime": "2026-10-22 00:00:00", "outtime": "2026-10-22 12:00:00", "admission_age": 55.0, "gender": "F"},
        {"stay_id": 302, "subject_id": 3002, "intime": "2026-10-22 00:00:00", "outtime": "2026-10-22 12:00:00", "admission_age": 70.0, "gender": "M"},
        {"stay_id": 303, "subject_id": 3003, "intime": "2026-10-22 00:00:00", "outtime": "2026-10-22 12:00:00", "admission_age": 62.0, "gender": "F"},
        {"stay_id": 304, "subject_id": 3004, "intime": "2026-10-22 00:00:00", "outtime": "2026-10-22 12:00:00", "admission_age": 80.0, "gender": "M"},
        {"stay_id": 305, "subject_id": 3005, "intime": "2026-10-22 00:00:00", "outtime": "2026-10-22 12:00:00", "admission_age": 45.0, "gender": None},  # Missing gender
        {"stay_id": 306, "subject_id": 3006, "intime": "2026-10-22 00:00:00", "outtime": "2026-10-22 12:00:00", "admission_age": 68.0, "gender": "F"},
    ])

    label_rows = []
    for sid in [301, 302, 303, 304, 305, 306]:
        for h in range(8):
            t_start = base_t + pd.Timedelta(hours=h)
            t_end = t_start + pd.Timedelta(hours=1)
            # Make stay 301 septic at hour 4
            if sid == 301 and h == 3:
                status = "POSITIVE"
            elif sid == 301 and h > 3:
                status = "POST_ONSET"
            elif h == 7:
                status = "DISCHARGE_CENSORED"
            else:
                status = "NEGATIVE"

            label_rows.append({
                "stay_id": sid,
                "hour_index": h + 1,
                "starttime": str(t_start),
                "endtime": str(t_end),
                "label_status": status,
            })
    labels = pd.DataFrame(label_rows)

    v_rows = []
    for sid in [301, 302, 303, 304, 305, 306]:
        for h in range(8):
            ct = base_t + pd.Timedelta(hours=h, minutes=30)
            v_rows.append({
                "stay_id": sid,
                "charttime": str(ct),
                "heart_rate": 75.0 + (sid % 10) + h,
                "sbp": 125.0 - h,
                "dbp": 80.0,
                "mbp": 95.0,
                "resp_rate": 18.0 + (h % 2),
                "temperature": 37.1 + 0.1 * (h % 3),
                "spo2": 98.0,
                "glucose": 110.0,
            })
    vitals = pd.DataFrame(v_rows)
    melted_vitals, _ = melt_wide_vitals_chunk(vitals)

    l_rows = []
    for sid in [301, 302, 303, 304, 305, 306]:
        for h in [2, 5]:
            ct = base_t + pd.Timedelta(hours=h, minutes=10)
            st = ct + pd.Timedelta(minutes=30)
            l_rows.append({
                "stay_id": sid,
                "charttime": str(ct),
                "storetime": str(st),
                "itemid": 50813,  # lactate
                "valuenum": 1.4 + 0.1 * h + (sid % 3) * 0.2,
            })
    labs = pd.DataFrame(l_rows)

    return {
        "cohort": cohort,
        "labels": labels,
        "vitals": melted_vitals,
        "labs": labs,
    }


def test_parallel_pass1_exact_global_median_parity(synthetic_parallel_data):
    """
    Verifies that run_pass1_medians in ParallelFeatureCacheGenerator produces
    100% mathematically and numerically identical medians to single-threaded fit_training_medians.
    """
    data = synthetic_parallel_data
    train_stays = [301, 302, 303, 304]

    labels_by_stay = {sid: grp for sid, grp in data["labels"].groupby("stay_id")}
    vitals_by_stay = {sid: grp for sid, grp in data["vitals"].groupby("stay_id")}
    labs_by_stay = {sid: grp for sid, grp in data["labs"].groupby("stay_id")}
    demo_lookup = {
        int(r["stay_id"]): {"admission_age": r["admission_age"], "gender": r["gender"]}
        for _, r in data["cohort"].iterrows()
    }

    # 1. Single-Threaded Ground Truth Medians
    f_matrices = []
    masks_list = []
    for sid in train_stays:
        f_mat, _, m_vec, _ = extract_stay_sequence_fast(
            stay_id=sid,
            labels_df=labels_by_stay[sid],
            vitals_df=vitals_by_stay[sid],
            labs_df=labs_by_stay[sid],
            demographics=demo_lookup[sid],
            train_medians=None,
        )
        f_matrices.append(f_mat)
        masks_list.append(m_vec)

    expected_medians = fit_training_medians(
        train_matrices=f_matrices,
        train_masks=masks_list,
        active_only=True,
    )

    # 2. Parallel Pass 1 Medians with 2 workers and batch_size=2
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = ParallelFeatureCacheGenerator(
            num_workers=2,
            batch_size=2,
            cache_dir=tmp_dir,
        )
        parallel_medians = gen.run_pass1_medians(
            train_stays=train_stays,
            labels_by_stay=labels_by_stay,
            vitals_by_stay=vitals_by_stay,
            labs_by_stay=labs_by_stay,
            demo_lookup=demo_lookup,
        )

        # Assert 100% parity across all continuous columns
        for col, exp_val in expected_medians.items():
            par_val = parallel_medians.get(col)
            if np.isnan(exp_val):
                assert np.isnan(par_val), f"Mismatch at {col}: expected NaN, got {par_val}"
            else:
                assert par_val == pytest.approx(exp_val, rel=1e-5), f"Mismatch at {col}: expected {exp_val}, got {par_val}"


def test_parallel_pass2_streaming_and_consolidation(synthetic_parallel_data):
    """
    Verifies that run_pass2_split_stream writes valid Parquet files in canonical
    schema with exact row counts, 0 duplicate keys, and correct target/mask values.
    """
    data = synthetic_parallel_data
    val_stays = [305, 306]

    labels_by_stay = {sid: grp for sid, grp in data["labels"].groupby("stay_id")}
    vitals_by_stay = {sid: grp for sid, grp in data["vitals"].groupby("stay_id")}
    labs_by_stay = {sid: grp for sid, grp in data["labs"].groupby("stay_id")}
    demo_lookup = {
        int(r["stay_id"]): {"admission_age": r["admission_age"], "gender": r["gender"]}
        for _, r in data["cohort"].iterrows()
    }

    dummy_medians = {col: 1.0 for col in CANONICAL_FEATURE_COLUMNS}

    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = ParallelFeatureCacheGenerator(
            num_workers=2,
            batch_size=1,  # 1 stay per batch to test part-file merging
            cache_dir=tmp_dir,
        )

        final_parquet = gen.run_pass2_split_stream(
            split_name="val",
            split_stays=val_stays,
            labels_by_stay=labels_by_stay,
            vitals_by_stay=vitals_by_stay,
            labs_by_stay=labs_by_stay,
            demo_lookup=demo_lookup,
            train_medians=dummy_medians,
        )

        assert os.path.exists(final_parquet)

        # Read back and validate table
        table = pq.read_table(final_parquet)
        df = table.to_pandas()

        # Expected rows: 2 stays * 8 hours = 16 rows
        assert len(df) == 16
        assert len(df.columns) == len(CANONICAL_FEATURE_COLUMNS) + 5

        # Check key uniqueness
        assert len(df.drop_duplicates(["stay_id", "hour_index"])) == 16

        # Check censoring masks
        # Hour 8 was discharge censored -> target=NaN, mask=0.0
        h8_rows = df[df["hour_index"] == 8]
        assert (h8_rows["active_mask"] == 0.0).all()
        assert h8_rows["target"].isna().all()
