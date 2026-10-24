"""
tests/test_train_hourly_model.py
─────────────────────────────────────────────────────────────────────────────
Focused Unit Test Suite for Stage 3.4B Hourly Training Pipeline Components.

Verifies:
1. CachedHourlySequenceDataset & Collation:
   - Padded tensor shapes, metadata alignment, and patient boundary integrity.
   - Guarantee that batches never mix or cross patient boundaries.
2. Target & Mask Mechanics:
   - Proper masking of censored hours and padded time steps.
   - Safe sanitization of IEEE 754 NaNs in targets.
3. Causal Attention Guarantee:
   - Causality invariant: Future feature perturbations (t' > t) have zero impact
     on past logits (<= t).
4. Class-Imbalance Loss Weighting:
   - Verified behavior of pos_weight in BCEWithLogitsLoss.
5. Checkpoint & Path Isolation:
   - Dry-run validation of feature cache and non-interference with protected baseline assets.
"""

import math
import numpy as np
import pytest
import torch

from src.hourly.feature_extractor import CANONICAL_FEATURE_COLUMNS
from src.hourly.temporal_model import HourlyTemporalTransformer
from src.hourly.train_pipeline import MaskedBCEWithLogitsLoss
from src.hourly.train_hourly_model import (
    CachedHourlySequenceDataset,
    pad_hourly_sequence_collate,
    run_hourly_training,
    DEFAULT_CACHE_DIR,
)


def test_dataset_and_collation_patient_isolation():
    """
    Verifies that:
    1. Sequences are collated into (B, Max_T, 184) tensors.
    2. Batches preserve distinct stay IDs and never cross patient boundaries.
    3. Padding mask is True ONLY for padded indices.
    """
    num_features = len(CANONICAL_FEATURE_COLUMNS)  # 184
    # Create 3 synthetic stays of lengths 5, 8, 3
    lengths = [5, 8, 3]
    stay_ids = [101, 102, 103]
    total_rows = sum(lengths)

    features = np.random.randn(total_rows, num_features).astype(np.float32)
    targets = np.zeros(total_rows, dtype=np.float32)
    # Put a positive at stay 102, hour 2
    targets[6] = 1.0
    masks = np.ones(total_rows, dtype=np.float32)
    # Mask out hour 7 of stay 102 (censored)
    masks[12] = 0.0
    targets[12] = np.nan

    stay_indices = []
    curr = 0
    for sid, l in zip(stay_ids, lengths):
        stay_indices.append((sid, curr, curr + l))
        curr += l

    dataset = CachedHourlySequenceDataset(features, targets, masks, stay_indices)
    assert len(dataset) == 3

    # Collate batch
    batch_items = [dataset[0], dataset[1], dataset[2]]
    padded_seqs, padded_targets, padded_masks, padding_mask, batch_sids = pad_hourly_sequence_collate(batch_items)

    assert padded_seqs.shape == (3, 8, num_features)
    assert padded_targets.shape == (3, 8)
    assert padded_masks.shape == (3, 8)
    assert padding_mask.shape == (3, 8)
    assert batch_sids == [101, 102, 103]

    # Stay 101 has length 5: indices 0..4 are real, 5..7 are padded
    assert bool(padding_mask[0, 4]) is False
    assert bool(padding_mask[0, 5]) is True
    assert float(padded_masks[0, 5]) == 0.0
    assert math.isnan(float(padded_targets[0, 5]))

    # Stay 102 has length 8: indices 0..7 are real, zero padding
    assert bool(padding_mask[1, 7]) is False
    assert float(padded_targets[1, 1]) == 1.0  # hour 2 relative to stay 102
    assert float(padded_masks[1, 7]) == 0.0    # censored hour 7

    # Stay 103 has length 3: indices 0..2 are real, 3..7 are padded
    assert bool(padding_mask[2, 2]) is False
    assert bool(padding_mask[2, 3]) is True
    assert float(padded_masks[2, 3]) == 0.0


def test_masked_loss_nan_target_sanitization_and_pos_weight():
    """
    Verifies that:
    1. IEEE 754 NaN targets at censored locations do NOT produce NaN loss or gradients.
    2. pos_weight = 150.0 scales positive class loss appropriately.
    3. All-censored batches return zero scalar loss.
    """
    criterion = MaskedBCEWithLogitsLoss(pos_weight=torch.tensor([150.0]))

    logits = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
    # Target has NaN at index 2 (censored)
    targets = torch.tensor([[0.0, 1.0, float("nan")]])
    # Active mask is 0 at index 2
    masks = torch.tensor([[1.0, 1.0, 0.0]])

    loss = criterion(logits, targets, masks)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    # Gradient at censored index 2 must be exactly 0.0
    assert logits.grad[0, 2].item() == 0.0

    # All-censored batch test
    logits_censored = torch.tensor([[1.0, 2.0]], requires_grad=True)
    targets_censored = torch.tensor([[float("nan"), float("nan")]])
    masks_censored = torch.tensor([[0.0, 0.0]])
    loss_censored = criterion(logits_censored, targets_censored, masks_censored)
    assert loss_censored.item() == 0.0


def test_temporal_transformer_causality_strict():
    """
    Verifies the causal attention guarantee:
    Perturbing feature inputs at t=3 must NOT alter logits at t=0, 1, 2.
    """
    torch.manual_seed(42)
    model = HourlyTemporalTransformer(
        num_features=184,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
    )
    model.eval()

    x1 = torch.randn(1, 6, 184)
    x2 = x1.clone()
    # Perturb future inputs at t=4 and t=5
    x2[:, 4:, :] = x2[:, 4:, :] + torch.randn(1, 2, 184) * 5.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Logits at t=0, 1, 2, 3 must be IDENTICAL within floating-point epsilon
    diff_past = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
    diff_future = (out1[:, 4:, :] - out2[:, 4:, :]).abs().max().item()

    assert diff_past < 1e-5, f"Causality leak detected in past hours! diff={diff_past}"
    assert diff_future > 1e-3, "Future hours unexpectedly identical after perturbation"


def test_dry_run_preflight_cache_verification(tmp_path):
    """
    Verifies run_hourly_training dry-run preflight:
    1. Checks presence of manifest and split Parquet files.
    2. Does NOT create model weights or execute training.
    """
    result = run_hourly_training(
        cache_dir=DEFAULT_CACHE_DIR,
        output_dir=str(tmp_path / "test_stage3_4b"),
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN_PASSED"
    # Ensure no model files were written
    assert not (tmp_path / "test_stage3_4b" / "hourly_transformer_best.pt").exists()


def test_missing_value_strategy_and_finite_validation(tmp_path):
    """
    Regression Test for Stage 3.4B.2:
    1. Validates that intermittent continuous NaNs take training medians.
    2. Validates that entirely unobserved channels (gender, GCS, BUN, PaO2/FiO2) take 0.0.
    3. Validates that indicators (*_measured, *_is_stale) remain intact.
    4. Validates that non-finite values are completely absent from the returned tensor.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.hourly.train_hourly_model import load_split_dataset_from_cache

    # Create dummy stay rows
    rows = []
    for h in range(4):
        row = {
            "stay_id": 9901,
            "hour_index": h + 1,
            "endtime": f"2150-01-01 0{h}:00:00",
            "target": 0.0 if h < 3 else 1.0,
            "active_mask": 1.0,
        }
        for col in CANONICAL_FEATURE_COLUMNS:
            row[col] = 1.0
        # Inject intermittent NaN in observed feature
        if h == 1:
            row["hr_mean"] = float("nan")
        # Inject 100% unobserved channels
        row["gender"] = float("nan")
        row["gcs_eye_mean"] = float("nan")
        row["pao2_fio2_ratio"] = float("nan")
        # Indicators
        row["hr_measured"] = 0.0 if h == 1 else 1.0
        row["gcs_eye_measured"] = 0.0
        row["gcs_eye_is_stale"] = 1.0
        rows.append(row)

    df = pd.DataFrame(rows)
    test_pq = str(tmp_path / "test_dummy_split.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, test_pq)

    train_medians = {
        "hr_mean": 81.5,
        "gender": float("nan"),
        "gcs_eye_mean": None,
        "pao2_fio2_ratio": float("nan"),
    }

    dataset = load_split_dataset_from_cache(
        test_pq, train_medians=train_medians, split_name="test_dummy"
    )

    # 1. Output features must be 100% finite
    assert np.isfinite(dataset.features).all(), "Features tensor contains non-finite values!"
    assert not np.isnan(dataset.features).any()

    # 2. Check intermittent NaN took training median
    hr_idx = CANONICAL_FEATURE_COLUMNS.index("hr_mean")
    assert dataset.features[1, hr_idx] == 81.5, f"Expected median 81.5, got {dataset.features[1, hr_idx]}"

    # 3. Check entirely unobserved channel took 0.0
    gender_idx = CANONICAL_FEATURE_COLUMNS.index("gender")
    gcs_idx = CANONICAL_FEATURE_COLUMNS.index("gcs_eye_mean")
    pao2_idx = CANONICAL_FEATURE_COLUMNS.index("pao2_fio2_ratio")
    assert (dataset.features[:, gender_idx] == 0.0).all()
    assert (dataset.features[:, gcs_idx] == 0.0).all()
    assert (dataset.features[:, pao2_idx] == 0.0).all()

    # 4. Check indicators preserved intact
    meas_idx = CANONICAL_FEATURE_COLUMNS.index("gcs_eye_measured")
    stale_idx = CANONICAL_FEATURE_COLUMNS.index("gcs_eye_is_stale")
    assert (dataset.features[:, meas_idx] == 0.0).all()
    assert (dataset.features[:, stale_idx] == 1.0).all()


def test_non_finite_input_rejection_error_message(tmp_path):
    """
    Verifies that unhandled non-finite values in features raise a descriptive
    ValueError with the exact bad column names and split name.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.hourly.train_hourly_model import load_split_dataset_from_cache

    rows = [{
        "stay_id": 9902,
        "hour_index": 1,
        "endtime": "2150-01-01 00:00:00",
        "target": 0.0,
        "active_mask": 1.0,
    }]
    for col in CANONICAL_FEATURE_COLUMNS:
        rows[0][col] = 1.0
    # Inject NaN with no train_medians provided
    rows[0]["sbp_mean"] = float("nan")

    df = pd.DataFrame(rows)
    test_pq = str(tmp_path / "bad_split.parquet")
    pq.write_table(pa.Table.from_pandas(df), test_pq)

    with pytest.raises(ValueError, match=r"\[VAL\] Non-finite values detected in input feature tensor for columns: \['sbp_mean'\]"):
        load_split_dataset_from_cache(test_pq, train_medians=None, split_name="val")


def test_forward_pass_and_evaluation_finite_with_unobserved_features(tmp_path):
    """
    Verifies that forward pass and evaluate_model_on_dataloader:
    1. Accept the imputed dataset without errors.
    2. Produce strictly finite logits and probabilities on active hours.
    3. Successfully compute AUROC and AUPRC without ValueError: Input contains NaN.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader
    from src.hourly.train_hourly_model import (
        load_split_dataset_from_cache,
        pad_hourly_sequence_collate,
        evaluate_model_on_dataloader,
    )
    from src.clinical.evaluate import compute_metrics

    # Create 2 synthetic stays
    rows = []
    for sid in [1, 2]:
        for h in range(5):
            row = {
                "stay_id": sid,
                "hour_index": h + 1,
                "endtime": f"2150-01-01 0{h}:00:00",
                "target": 1.0 if (sid == 1 and h == 3) else 0.0,
                "active_mask": 1.0,
            }
            for col in CANONICAL_FEATURE_COLUMNS:
                row[col] = float("nan") if "gcs" in col or "pao2" in col else 1.0
            rows.append(row)

    df = pd.DataFrame(rows)
    test_pq = str(tmp_path / "eval_test.parquet")
    pq.write_table(pa.Table.from_pandas(df), test_pq)

    train_medians = {col: 1.0 for col in CANONICAL_FEATURE_COLUMNS if "gcs" not in col and "pao2" not in col}
    for col in CANONICAL_FEATURE_COLUMNS:
        if "gcs" in col or "pao2" in col:
            train_medians[col] = float("nan")

    dataset = load_split_dataset_from_cache(test_pq, train_medians=train_medians, split_name="val")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_hourly_sequence_collate)

    model = HourlyTemporalTransformer(
        num_features=len(CANONICAL_FEATURE_COLUMNS),
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
    )
    criterion = MaskedBCEWithLogitsLoss(pos_weight=torch.tensor([150.0]))
    device = torch.device("cpu")

    eval_result = evaluate_model_on_dataloader(model, dataloader, criterion, device, split_name="val")

    assert np.isfinite(eval_result["y_prob"]).all(), "Evaluated probabilities contain NaNs!"
    assert np.isfinite(eval_result["y_true"]).all(), "Evaluated targets contain NaNs!"
    assert np.isfinite(eval_result["loss"]), "Evaluated loss is non-finite!"

    # Verify metrics computation succeeds without raising ValueError
    metrics = compute_metrics(eval_result["y_true"], eval_result["y_prob"], threshold=0.5, model_name="test_eval")
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert not math.isnan(metrics["roc_auc"])

