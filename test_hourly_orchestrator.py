"""
tests/test_hourly_orchestrator.py
─────────────────────────────────────────────────────────────────────────────
Stage 3.1: Synthetic Integration Test Suite for Hourly Pipeline Orchestrator.

SAFETY & PROVENANCE NOTICE:
=============================================================================
ALL DATA IN THIS TEST SUITE IS PURELY SYNTHETIC AND CREATED IN-MEMORY.
NO REAL PATIENT DATA, NO DISK ARCHIVES, AND NO CLOUD SERVICES ARE ACCESSED.
=============================================================================
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.hourly.adapter import melt_wide_vitals_chunk, process_labs_chunk
from src.hourly.feature_extractor import CANONICAL_FEATURE_COLUMNS, HourlySlidingWindowFeatureExtractor
from src.hourly.orchestrator import (
    HourlyPipelineOrchestrator,
    validate_dataframe_schema,
    map_label_status_to_target,
    extract_stay_sequence,
)
from src.hourly.temporal_model import HourlyTemporalTransformer
from src.hourly.train_pipeline import MaskedBCEWithLogitsLoss


@pytest.fixture
def synthetic_orchestrator_data():
    """Builds a tiny in-memory synthetic 2-stay dataset for integration testing."""
    base_t = pd.Timestamp("2026-10-21 00:00:00")

    cohort = pd.DataFrame([
        {
            "stay_id": 101,
            "subject_id": 1001,
            "intime": "2026-10-21 00:00:00",
            "outtime": "2026-10-21 08:00:00",
            "admission_age": 65.0,
            "gender": "F",
        },
        {
            "stay_id": 102,
            "subject_id": 1002,
            "intime": "2026-10-21 00:00:00",
            "outtime": "2026-10-21 08:00:00",
            "admission_age": 72.0,
            "gender": None,  # Explicitly missing gender
        },
    ])

    # Stay 101: 5 hours (Hour 0 to 4) - septic transition at Hour 3
    # Stay 102: 4 hours (Hour 0 to 3) - negative control with discharge censoring at Hour 3
    labels_rows = []
    # Stay 101: Hour 0-1 NEGATIVE, Hour 2 POSITIVE, Hour 3 POST_ONSET, Hour 4 POST_ONSET
    statuses_101 = ["NEGATIVE", "NEGATIVE", "POSITIVE", "POST_ONSET", "POST_ONSET"]
    for h, st in enumerate(statuses_101):
        t_start = base_t + pd.Timedelta(hours=h)
        t_end = t_start + pd.Timedelta(hours=1)
        labels_rows.append({
            "stay_id": 101,
            "hour_index": h,
            "starttime": str(t_start),
            "endtime": str(t_end),
            "label_status": st,
        })

    # Stay 102: Hour 0-2 NEGATIVE, Hour 3 DISCHARGE_CENSORED
    statuses_102 = ["NEGATIVE", "NEGATIVE", "NEGATIVE", "DISCHARGE_CENSORED"]
    for h, st in enumerate(statuses_102):
        t_start = base_t + pd.Timedelta(hours=h)
        t_end = t_start + pd.Timedelta(hours=1)
        labels_rows.append({
            "stay_id": 102,
            "hour_index": h,
            "starttime": str(t_start),
            "endtime": str(t_end),
            "label_status": st,
        })

    labels = pd.DataFrame(labels_rows)

    # Wide vitals
    vitals_raw = pd.DataFrame([
        {
            "stay_id": 101,
            "charttime": "2026-10-21 00:30:00",
            "heart_rate": 80.0,
            "sbp": 120.0,
            "dbp": 80.0,
            "mbp": 93.3,
            "resp_rate": 16.0,
            "temperature": 37.0,
            "spo2": 98.0,
            "glucose": 100.0,
        },
        {
            "stay_id": 101,
            "charttime": "2026-10-21 01:30:00",
            "heart_rate": 95.0,
            "sbp": 110.0,
            "dbp": 70.0,
            "mbp": 83.3,
            "resp_rate": 20.0,
            "temperature": 38.2,
            "spo2": 96.0,
            "glucose": 110.0,
        },
        {
            "stay_id": 102,
            "charttime": "2026-10-21 01:00:00",
            "heart_rate": 72.0,
            "sbp": 125.0,
            "dbp": 75.0,
            "mbp": 91.0,
            "resp_rate": 14.0,
            "temperature": 36.8,
            "spo2": 99.0,
            "glucose": 95.0,
        },
    ])

    # Long labs
    labs_raw = pd.DataFrame([
        # Stay 101: Lactate drawn at 01:00, stored at 01:45
        {
            "stay_id": 101,
            "charttime": "2026-10-21 01:00:00",
            "storetime": "2026-10-21 01:45:00",
            "itemid": 50813,  # Lactate
            "valuenum": 2.2,
        },
        # Stay 101: Later lactate drawn at 02:00, stored at 02:30
        {
            "stay_id": 101,
            "charttime": "2026-10-21 02:00:00",
            "storetime": "2026-10-21 02:30:00",
            "itemid": 50813,
            "valuenum": 4.1,
        },
        # Stay 102: WBC drawn at 01:00, stored at 01:30
        {
            "stay_id": 102,
            "charttime": "2026-10-21 01:00:00",
            "storetime": "2026-10-21 01:30:00",
            "itemid": 51301,  # WBC
            "valuenum": 8.5,
        },
    ])

    return {
        "cohort": cohort,
        "labels": labels,
        "vitals_raw": vitals_raw,
        "labs_raw": labs_raw,
    }


def test_schema_validation_success_and_failures():
    """Confirms schema validator raises clear ValueErrors on missing columns."""
    valid_cohort = pd.DataFrame(columns=["stay_id", "subject_id", "intime", "outtime"])
    validate_dataframe_schema(valid_cohort, "cohort")

    # Missing column
    invalid_cohort = pd.DataFrame(columns=["stay_id", "subject_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe_schema(invalid_cohort, "cohort")

    # Invalid type
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        validate_dataframe_schema([1, 2, 3], "cohort")  # type: ignore


def test_demographic_missingness_handling(synthetic_orchestrator_data):
    """
    Confirms missing or unknown demographics are represented strictly as NaN,
    and NEVER silently converted to a real category.
    """
    cohort = synthetic_orchestrator_data["cohort"]
    labels = synthetic_orchestrator_data["labels"]

    orchestrator = HourlyPipelineOrchestrator()
    seq_dict = orchestrator.build_stay_sequences(
        cohort_df=cohort,
        labels_df=labels,
        vitals_df=None,
        labs_df=None,
    )

    cols = orchestrator.feature_columns
    age_idx = cols.index("admission_age")
    gender_idx = cols.index("gender")

    # Stay 101: female -> gender = 1.0, age = 65.0
    f101 = seq_dict["features_by_stay"][101]
    assert f101[0, age_idx] == 65.0
    assert f101[0, gender_idx] == 1.0

    # Stay 102: None gender -> gender MUST BE NaN, NEVER 0.0 or 1.0!
    f102 = seq_dict["features_by_stay"][102]
    assert f102[0, age_idx] == 72.0
    assert np.isnan(f102[0, gender_idx]), "Missing gender must evaluate to NaN, not a default category"


def test_label_status_mapping_semantics():
    """
    Verifies label mapping:
    - POSITIVE -> 1.0, active mask 1.0
    - NEGATIVE -> 0.0, active mask 1.0
    - POST_ONSET -> NaN, active mask 0.0
    - DISCHARGE_CENSORED -> NaN, active mask 0.0
    - Unknown -> NaN, active mask 0.0
    """
    assert map_label_status_to_target("POSITIVE") == (1.0, 1.0)
    assert map_label_status_to_target("negative") == (0.0, 1.0)
    assert map_label_status_to_target("POST_ONSET")[1] == 0.0
    assert np.isnan(map_label_status_to_target("POST_ONSET")[0])
    assert map_label_status_to_target("DISCHARGE_CENSORED")[1] == 0.0
    assert np.isnan(map_label_status_to_target("DISCHARGE_CENSORED")[0])
    assert map_label_status_to_target("UNKNOWN_STATUS")[1] == 0.0
    assert np.isnan(map_label_status_to_target(None)[0])


def test_end_to_end_orchestrator_dataset_creation(synthetic_orchestrator_data):
    """
    Tests end-to-end flow:
    raw vitals -> melt_wide_vitals_chunk -> adapter
    raw labs -> process_labs_chunk -> adapter
    orchestrator -> HourlySequenceDataset
    """
    cohort = synthetic_orchestrator_data["cohort"]
    labels = synthetic_orchestrator_data["labels"]
    vitals_raw = synthetic_orchestrator_data["vitals_raw"]
    labs_raw = synthetic_orchestrator_data["labs_raw"]

    # 1. Adapt vitals and labs
    adapted_vitals, _ = melt_wide_vitals_chunk(vitals_raw)
    adapted_labs, _ = process_labs_chunk(labs_raw)

    # 2. Build dataset via orchestrator
    orchestrator = HourlyPipelineOrchestrator()
    dataset = orchestrator.create_sequence_dataset(
        cohort_df=cohort,
        labels_df=labels,
        vitals_df=adapted_vitals,
        labs_df=adapted_labs,
    )

    assert len(dataset) == 2

    # Check Stay 101 sequence
    seq101, tgt101, mask101, sid101 = dataset[0]
    assert sid101 == 101
    assert seq101.shape == (5, len(CANONICAL_FEATURE_COLUMNS))
    assert tgt101.shape == (5,)
    assert mask101.shape == (5,)

    # Targets: [0.0, 0.0, 1.0, NaN, NaN]
    assert tgt101[0].item() == 0.0
    assert tgt101[1].item() == 0.0
    assert tgt101[2].item() == 1.0
    assert torch.isnan(tgt101[3])
    assert torch.isnan(tgt101[4])

    # Masks: [1.0, 1.0, 1.0, 0.0, 0.0]
    assert list(mask101.numpy()) == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_synthetic_model_forward_and_gradient_verification(synthetic_orchestrator_data):
    """
    Verifies that the assembled dataset passes cleanly through:
    HourlyTemporalTransformer -> MaskedBCEWithLogitsLoss
    with matching tensor shapes, finite loss, and finite gradients.
    """
    cohort = synthetic_orchestrator_data["cohort"]
    labels = synthetic_orchestrator_data["labels"]
    vitals_raw = synthetic_orchestrator_data["vitals_raw"]
    labs_raw = synthetic_orchestrator_data["labs_raw"]

    adapted_vitals, _ = melt_wide_vitals_chunk(vitals_raw)
    adapted_labs, _ = process_labs_chunk(labs_raw)

    # Use training medians to impute features for Transformer consumption
    train_medians = {col: 0.0 for col in CANONICAL_FEATURE_COLUMNS}
    orchestrator = HourlyPipelineOrchestrator(train_medians=train_medians)

    dataset = orchestrator.create_sequence_dataset(
        cohort_df=cohort,
        labels_df=labels,
        vitals_df=adapted_vitals,
        labs_df=adapted_labs,
    )

    model = HourlyTemporalTransformer(
        num_features=len(CANONICAL_FEATURE_COLUMNS),
        embed_dim=32,
        num_heads=2,
        num_layers=2,
    )

    result = HourlyPipelineOrchestrator.verify_forward_step(model, dataset)

    assert result["seq_len"] == 5
    assert result["num_features"] == len(CANONICAL_FEATURE_COLUMNS)
    assert result["logits_shape"] == [1, 5, 1]
    assert result["loss"] > 0.0
    assert result["finite_gradients"] is True
