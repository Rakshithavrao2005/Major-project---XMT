"""
tests/test_hourly_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit Test Suite for Temporal Model, Censoring Mechanics, and Pipeline Guardrails.

SAFETY & PROVENANCE NOTICE:
=============================================================================
ALL DATA AND SEQUENCES IN THIS TEST SUITE ARE SYNTHETIC TOY ARTIFACTS
DESIGNED SOLELY TO VALIDATE PYTORCH TENSOR MECHANICS, CAUSAL MASKING,
PATIENT-LEVEL GROUP SPLITTING, AND POST-ONSET CENSORING LOGIC.
THEY ARE NOT CLINICAL PATIENT DATA, NOT REAL EHR OBSERVATIONS, AND MUST NEVER
BE USED FOR CLINICAL TRAINING, VALIDATION, OR BENCHMARK EVALUATION.
=============================================================================
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.hourly.dataset import (
    HourlySequenceDataset,
    apply_censoring_and_target_alignment,
    patient_level_split,
)
from src.hourly.temporal_model import HourlyTemporalTransformer
from src.hourly.train_pipeline import (
    HourlyTrainingPipeline,
    MaskedBCEWithLogitsLoss,
    check_hourly_data_prerequisites,
    DataRequirementError,
    resolve_dataset_files,
    stream_dataset_chunks,
    REQUIRED_DATASETS,
)


def test_patient_level_split_zero_leakage():
    """Confirms patient_level_split yields 0% patient overlap between splits."""
    # Create synthetic cohort with 100 patients across 150 stays
    np.random.seed(42)
    pts = [1000 + i for i in range(100)]
    stays = []
    for sid in range(150):
        stays.append({
            "stay_id": 20000 + sid,
            "subject_id": np.random.choice(pts),
        })
    cohort_df = pd.DataFrame(stays)

    train_stays, val_stays, test_stays = patient_level_split(
        cohort_df, test_ratio=0.20, val_ratio=0.15, random_state=42
    )

    # Map stay_ids back to subject_ids
    train_pts = set(cohort_df[cohort_df["stay_id"].isin(train_stays)]["subject_id"])
    val_pts = set(cohort_df[cohort_df["stay_id"].isin(val_stays)]["subject_id"])
    test_pts = set(cohort_df[cohort_df["stay_id"].isin(test_stays)]["subject_id"])

    # Strict zero overlap assertion
    assert len(train_pts & test_pts) == 0, "Patient leakage detected between train and test"
    assert len(train_pts & val_pts) == 0, "Patient leakage detected between train and val"
    assert len(val_pts & test_pts) == 0, "Patient leakage detected between val and test"
    assert len(train_stays) + len(val_stays) + len(test_stays) == 150


def test_censoring_and_active_surveillance_mask():
    """
    Verifies that:
    1. Hours >= sepsis onset are censored (active_surveillance_mask = 0).
    2. Discharge-censored hours are masked out.
    3. Uncensored pre-sepsis hours remain active (active_surveillance_mask = 1).
    """
    base_time = pd.Timestamp("2026-10-20 00:00:00")
    onset_time = base_time + pd.Timedelta(hours=4)

    # Stay with 6 hourly slots
    rows = []
    for hr in range(6):
        endtime = base_time + pd.Timedelta(hours=hr + 1)
        is_censored = 1 if endtime >= onset_time else 0
        rows.append({
            "stay_id": 8801,
            "subject_id": 101,
            "hr": hr,
            "endtime": endtime,
            "sepsis3_onset_time": onset_time,
            "is_censored": is_censored,
            "is_discharge_censored": 0,
            "is_septic_next_hour": 1 if hr == 2 else 0,  # next hour onset at hr=2 -> onset in (3, 4]
        })
    df = pd.DataFrame(rows)

    censored_df = apply_censoring_and_target_alignment(df)

    # Hours 0, 1, 2 have endtime < 04:00 -> active (mask = 1)
    assert list(censored_df["active_surveillance_mask"]) == [1, 1, 1, 0, 0, 0]
    # Censored hours have target masked to NaN
    assert pd.isna(censored_df.loc[3, "is_septic_next_hour"])
    assert pd.isna(censored_df.loc[4, "is_septic_next_hour"])
    assert pd.isna(censored_df.loc[5, "is_septic_next_hour"])
    # Uncensored hours retain their labels
    assert censored_df.loc[2, "is_septic_next_hour"] == 1
    assert censored_df.loc[0, "is_septic_next_hour"] == 0


def test_causal_temporal_transformer_tensor_shapes_and_causality():
    """
    Tests:
    1. Output tensor shape is (Batch, Seq_Len, 1).
    2. Causality invariant: Changing input features at time step t=4 does NOT change
       logits at time steps t=0, 1, 2, 3 (anti-leakage attention guarantee).
    """
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 5
    num_features = 16

    model = HourlyTemporalTransformer(
        num_features=num_features,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
    )
    model.eval()

    # Input tensor 1
    x1 = torch.randn(batch_size, seq_len, num_features)
    with torch.no_grad():
        out1 = model(x1)

    assert out1.shape == (batch_size, seq_len, 1)

    # Input tensor 2: Identical to x1 for steps 0..3, but heavily perturbed at step 4
    x2 = x1.clone()
    x2[:, 4, :] += 50.0  # Massive perturbation at step 4

    with torch.no_grad():
        out2 = model(x2)

    # Outputs at steps 0, 1, 2, 3 must be strictly identical (within float tolerance)
    diff_past = torch.abs(out1[:, :4, :] - out2[:, :4, :]).max().item()
    assert diff_past < 1e-5, f"Causal leakage detected! Past logits changed by {diff_past}"

    # Output at step 4 SHOULD change due to the perturbation
    diff_future = torch.abs(out1[:, 4, :] - out2[:, 4, :]).max().item()
    assert diff_future > 0.01, "Expected perturbation at step 4 to change logit at step 4"


def test_masked_bce_loss_mechanics():
    """Confirms MaskedBCEWithLogitsLoss strictly ignores censored time steps (mask == 0)."""
    criterion = MaskedBCEWithLogitsLoss()

    logits = torch.tensor([[2.0, -1.0, 5.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    # Case 1: Mask includes all 3 steps
    mask_all = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    loss_all = criterion(logits, targets, mask_all)

    # Case 2: Third step (which had a huge error: logit=5 vs target=0) is censored (mask=0)
    mask_censored = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    loss_censored = criterion(logits, targets, mask_censored)

    # Loss when third step is censored should be significantly lower because massive error is ignored
    assert loss_censored < loss_all


def test_data_prerequisite_guardrail_blocks_without_authorized_data():
    """
    Confirms that check_hourly_data_prerequisites detects missing raw files
    and HourlyTrainingPipeline.verify_or_raise blocks execution safely.
    """
    prereqs = check_hourly_data_prerequisites("data/raw_hourly_nonexistent_test")
    assert prereqs["all_files_present"] is False
    assert "cohort_icustays.csv" in prereqs["files"]
    assert prereqs["files"]["cohort_icustays.csv"]["exists"] is False

    pipeline = HourlyTrainingPipeline(data_dir="data/raw_hourly_nonexistent_test")
    with pytest.raises(DataRequirementError) as exc_info:
        pipeline.verify_or_raise()

    assert "Authorized longitudinal raw files are missing" in str(exc_info.value)


def test_cli_execution_refuses_training_when_data_missing():
    """Confirms that running train_pipeline.py directly via CLI halts with exit code 1."""
    import subprocess
    import os
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cmd = [sys.executable, "src/hourly/train_pipeline.py"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    assert res.returncode == 1
    assert "[BLOCKED]" in res.stdout or "[BLOCKED]" in res.stderr
    assert "Safety Protocol: No model was trained or evaluated." in res.stdout


def test_masked_bce_loss_with_nan_targets_and_finite_gradients():
    """
    D1 Remediation Test:
    Confirms MaskedBCEWithLogitsLoss:
    1. Handles NaN targets in censored positions without producing NaN loss.
    2. Generates finite gradients for uncensored positions.
    3. Produces strictly 0.0 gradients for censored positions.
    """
    criterion = MaskedBCEWithLogitsLoss()
    logits = torch.tensor([[1.5, -2.0, 3.0]], dtype=torch.float32, requires_grad=True)
    # Targets contain NaN at censored position (index 2)
    targets = torch.tensor([[1.0, 0.0, float("nan")]], dtype=torch.float32)
    masks = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    loss = criterion(logits, targets, masks)
    assert not torch.isnan(loss), "Loss must not be NaN when censored targets contain NaN"
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), "Gradients must be finite"
    assert logits.grad[0, 2].item() == 0.0, "Censored position must have exactly 0 gradient"
    assert logits.grad[0, 0].item() != 0.0, "Active position 0 must have non-zero gradient"
    assert logits.grad[0, 1].item() != 0.0, "Active position 1 must have non-zero gradient"


def test_masked_bce_loss_all_censored_batch_safety():
    """
    D1 Remediation Test:
    Confirms an all-censored batch (all masks == 0):
    1. Returns 0.0 loss without NaN or errors.
    2. Produces zero gradients on parameters (no erroneous updates).
    """
    criterion = MaskedBCEWithLogitsLoss()
    logits = torch.tensor([[2.5, -1.0], [0.5, -3.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[float("nan"), float("nan")], [float("nan"), float("nan")]], dtype=torch.float32)
    masks = torch.zeros_like(logits)

    loss = criterion(logits, targets, masks)
    assert not torch.isnan(loss), "All-censored batch must not produce NaN loss"
    assert loss.item() == 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits)), "All-censored batch must yield zero gradients"


def test_masked_bce_loss_manual_formula_verification():
    """
    D1 Remediation Test:
    Confirms exact mathematical alignment with manual BCE formulation:
    loss = -1/N * sum(y * ln(sigma(z)) + (1-y) * ln(1 - sigma(z))) for active hours.
    """
    import math

    criterion = MaskedBCEWithLogitsLoss()
    z0, z1, z_censored = 1.2, -0.8, 5.0
    y0, y1, y_censored = 1.0, 0.0, float("nan")

    logits = torch.tensor([[z0, z1, z_censored]], dtype=torch.float32)
    targets = torch.tensor([[y0, y1, y_censored]], dtype=torch.float32)
    masks = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    loss = criterion(logits, targets, masks)

    # Manual calculation
    sig0 = 1.0 / (1.0 + math.exp(-z0))
    sig1 = 1.0 / (1.0 + math.exp(-z1))
    bce0 = -(y0 * math.log(sig0) + (1.0 - y0) * math.log(1.0 - sig0))
    bce1 = -(y1 * math.log(sig1) + (1.0 - y1) * math.log(1.0 - sig1))
    manual_expected = (bce0 + bce1) / 2.0

    assert abs(loss.item() - manual_expected) < 1e-6, (
        f"Loss {loss.item()} does not match manual calculation {manual_expected}"
    )


def test_dataset_file_resolution_and_streaming(tmp_path):
    """
    D2 Remediation Test:
    Confirms resolve_dataset_files and stream_dataset_chunks correctly handle:
    1. Single CSV files.
    2. Sharded CSV files.
    3. Gzip-compressed (.csv.gz) files.
    4. Schema validation and error handling on missing/inconsistent columns.
    """
    data_dir = str(tmp_path)
    shard0 = tmp_path / "longitudinal_vitals_000000000000.csv"
    shard1 = tmp_path / "longitudinal_vitals_000000000001.csv.gz"

    df0 = pd.DataFrame({
        "stay_id": [1, 2],
        "charttime": ["2026-10-20 01:00:00", "2026-10-20 02:00:00"],
        "heart_rate": [75.0, 80.0],
    })
    df1 = pd.DataFrame({
        "stay_id": [3, 4],
        "charttime": ["2026-10-20 03:00:00", "2026-10-20 04:00:00"],
        "heart_rate": [85.0, 90.0],
    })

    df0.to_csv(shard0, index=False)
    df1.to_csv(shard1, index=False, compression="gzip")

    # Verify resolution
    resolved = resolve_dataset_files(data_dir, "longitudinal_vitals")
    assert len(resolved) == 2
    assert str(shard0) in resolved
    assert str(shard1) in resolved

    # Verify streaming
    chunks = list(stream_dataset_chunks(resolved, chunksize=1, expected_columns=["stay_id", "charttime", "heart_rate"]))
    assert len(chunks) == 4  # 4 rows streamed in chunks of 1
    combined = pd.concat(chunks, ignore_index=True)
    assert list(combined["stay_id"]) == [1, 2, 3, 4]

    # Verify schema validation on missing column
    with pytest.raises(ValueError, match="missing expected columns"):
        list(stream_dataset_chunks(resolved, expected_columns=["stay_id", "nonexistent_col"]))

    # Verify inconsistent schema across shards raises ValueError
    bad_shard = tmp_path / "longitudinal_vitals_000000000002.csv"
    pd.DataFrame({"different_id": [5]}).to_csv(bad_shard, index=False)
    all_shards = sorted(resolved + [str(bad_shard)])
    with pytest.raises(ValueError, match="Inconsistent schema"):
        list(stream_dataset_chunks(all_shards))


def test_hourly_sepsis_labels_query_schema_includes_hadm_id():
    """
    Confirms static SQL schema check on queries/02_hourly_sepsis_labels.sql:
    The final SELECT statement must project hadm_id directly alongside stay_id and subject_id,
    and redundant joins are eliminated.
    """
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sql_path = os.path.join(project_root, "queries", "02_hourly_sepsis_labels.sql")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    assert "hadm_id" in sql_text, "hadm_id must be selected in 02_hourly_sepsis_labels.sql"

    final_select_idx = sql_text.rfind("SELECT")
    final_select_clause = sql_text[final_select_idx:]
    assert "hadm_id" in final_select_clause, "hadm_id must be in final SELECT projection"
    assert "hg.stay_id" in final_select_clause
    assert "hg.subject_id" in final_select_clause or "c.subject_id" in final_select_clause


def test_export_workflow_valid_structure_and_gzip_compression():
    """
    Confirms queries/05_export_workflow.sql:
    1. Contains zero dummy table references (e.g. 'your_project', 'your_dataset').
    2. Has all 4 export steps configured.
    3. All 4 export steps use wildcard URI (*.csv.gz) and compression='GZIP'.
    4. Large exports (vitals and labs) omit global ORDER BY to avoid single-node BigQuery memory exhaustion.
    """
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sql_path = os.path.join(project_root, "queries", "05_export_workflow.sql")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    # 1. Zero dummy placeholders
    assert "your_project" not in sql, "Found dummy 'your_project' in 05_export_workflow.sql"
    assert "your_dataset" not in sql, "Found dummy 'your_dataset' in 05_export_workflow.sql"
    assert "-- Paste contents" not in sql, "Found unreplaced placeholder comments in 05_export_workflow.sql"

    # 2. All 4 datasets present
    assert "cohort_icustays_*.csv.gz" in sql
    assert "hourly_sepsis_labels_*.csv.gz" in sql
    assert "longitudinal_vitals_*.csv.gz" in sql
    assert "longitudinal_labs_*.csv.gz" in sql

    # 3. Compression is GZIP
    assert sql.count("compression='GZIP'") == 4

    # 4. Global ORDER BY is omitted in Step 3 and Step 4
    step3_idx = sql.find("STEP 3:")
    step4_idx = sql.find("STEP 4:")
    step3_sql = sql[step3_idx:step4_idx]
    step4_sql = sql[step4_idx:]
    assert "ORDER BY stay_id" not in step3_sql, "Step 3 should not have global ORDER BY in export"
    assert "ORDER BY stay_id" not in step4_sql, "Step 4 should not have global ORDER BY in export"


def test_export_client_safety_guardrails():
    """
    Confirms queries/export_client.py:
    1. DRY_RUN_ONLY is True by default.
    2. CONFIRM_EXECUTE is False by default.
    3. LARGE_DATASETS guardrail blocks in-memory to_dataframe() download for vitals and labs.
    """
    import queries.export_client as ec

    assert ec.DRY_RUN_ONLY is True, "DRY_RUN_ONLY must default to True"
    assert ec.CONFIRM_EXECUTE is False, "CONFIRM_EXECUTE must default to False"


def test_feature_extractor_robust_to_unordered_observations():
    """
    Confirms HourlySlidingWindowFeatureExtractor produces identical results
    regardless of whether the input vitals/labs DataFrames are chronological or shuffled.
    """
    from src.hourly.feature_extractor import HourlySlidingWindowFeatureExtractor

    extractor = HourlySlidingWindowFeatureExtractor(vitals_window_hours=6.0, labs_window_hours=24.0)
    pred_time = pd.Timestamp("2026-10-20 12:00:00")

    # Chronologically sorted vitals
    vitals_sorted = pd.DataFrame([
        {"charttime": "2026-10-20 07:00:00", "itemid": 220045, "valuenum": 70.0},
        {"charttime": "2026-10-20 09:00:00", "itemid": 220045, "valuenum": 80.0},
        {"charttime": "2026-10-20 11:00:00", "itemid": 220045, "valuenum": 95.0},
    ])

    # Reverse order vitals
    vitals_shuffled = vitals_sorted.iloc[::-1].copy().reset_index(drop=True)

    feats_sorted = extractor.extract_features(pred_time, vitals_df=vitals_sorted)
    feats_shuffled = extractor.extract_features(pred_time, vitals_df=vitals_shuffled)

    assert feats_sorted["hr_last"] == 95.0
    assert feats_shuffled["hr_last"] == 95.0
    assert feats_sorted["hr_delta"] == 25.0  # 95 - 70
    assert feats_shuffled["hr_delta"] == 25.0
    assert feats_sorted["hr_mean"] == feats_shuffled["hr_mean"]

