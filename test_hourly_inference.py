"""
tests/test_hourly_inference.py
─────────────────────────────────────────────────────────────────────────────
Focused Unit Test Suite for Stage 3.5.1 Hourly Sepsis Inference Pipeline.

Validates:
1. Checkpoint loading, architecture compatibility, and parameter freezing.
2. Canonical 184-feature ordering and training-derived missing-value imputation.
3. Timestamp sorting and strict causal sequence invariance.
4. Explicit calibrator selection (temperature, platt, none) and probability bounds.
5. Descriptive input validation errors (empty, malformed, non-chronological, missing cols).
6. Deterministic inference across repeated runs.
7. Guarantee that inference forward passes never update model weights.
8. Research disclaimer enforcement.
"""

import math
import numpy as np
import pandas as pd
import pytest
import torch

from src.hourly.feature_extractor import CANONICAL_FEATURE_COLUMNS
from src.hourly.inference import (
    HourlySepsisPredictor,
    RESEARCH_DISCLAIMER,
    DEFAULT_CHECKPOINT_PATH,
)


def _create_synthetic_sequence(num_hours: int = 5) -> pd.DataFrame:
    """Helper creating a valid synthetic DataFrame with canonical features."""
    rows = []
    base_time = pd.Timestamp("2150-01-01 00:00:00")
    for h in range(num_hours):
        row = {
            "stay_id": 12345,
            "hour_index": h + 1,
            "endtime": str(base_time + pd.Timedelta(hours=h + 1)),
        }
        for col in CANONICAL_FEATURE_COLUMNS:
            row[col] = 1.0
        # Realistic indicators
        row["hr_mean"] = 82.0 + h * 0.5
        row["hr_measured"] = 1.0
        row["hr_is_stale"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_checkpoint_loading_and_architecture():
    """Verifies that HourlySepsisPredictor loads weights and reconstructs exact architecture."""
    predictor = HourlySepsisPredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH)

    total_params = sum(p.numel() for p in predictor.model.parameters())
    assert total_params == 81153, f"Expected 81,153 params, got {total_params}"
    assert len(predictor.feature_columns) == 184
    assert predictor.feature_columns == list(CANONICAL_FEATURE_COLUMNS)
    assert predictor.val_auprc == 0.0704
    assert predictor.checkpoint_epoch == 4

    # Verify all parameters are frozen (no gradients)
    for p in predictor.model.parameters():
        assert p.requires_grad is False


def test_feature_ordering_and_missing_value_handling():
    """
    Verifies that:
    1. Intermittent NaNs in observed channels take training-derived medians.
    2. Entirely unobserved channels take neutral 0.0 embeddings.
    3. Indicators remain intact.
    4. Output feature matrix is 100% finite.
    """
    predictor = HourlySepsisPredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH)
    df = _create_synthetic_sequence(num_hours=4)

    # Inject intermittent NaN in observed continuous feature
    df.loc[1, "hr_mean"] = float("nan")
    # Inject 100% unobserved channels
    df["gender"] = float("nan")
    df["gcs_eye_mean"] = float("nan")
    df["pao2_fio2_ratio"] = float("nan")
    df["gcs_eye_measured"] = 0.0
    df["gcs_eye_is_stale"] = 1.0

    features = predictor.preprocess_sequence(df)

    assert features.shape == (4, 184)
    assert np.isfinite(features).all()
    assert not np.isnan(features).any()

    # Intermittent NaN took training median
    hr_idx = predictor.feature_columns.index("hr_mean")
    expected_hr_median = predictor.train_medians["hr_mean"]
    assert features[1, hr_idx] == expected_hr_median

    # Unobserved channels took 0.0
    gender_idx = predictor.feature_columns.index("gender")
    gcs_idx = predictor.feature_columns.index("gcs_eye_mean")
    pao2_idx = predictor.feature_columns.index("pao2_fio2_ratio")
    assert (features[:, gender_idx] == 0.0).all()
    assert (features[:, gcs_idx] == 0.0).all()
    assert (features[:, pao2_idx] == 0.0).all()

    # Indicators intact
    meas_idx = predictor.feature_columns.index("gcs_eye_measured")
    stale_idx = predictor.feature_columns.index("gcs_eye_is_stale")
    assert (features[:, meas_idx] == 0.0).all()
    assert (features[:, stale_idx] == 1.0).all()


def test_timestamp_ordering_and_causal_sequence_behavior():
    """
    Verifies that:
    1. Input sequence is executed chronologically.
    2. Causal invariance: Perturbing features at t=5 does NOT alter predictions at t=1..4.
    """
    predictor = HourlySepsisPredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH)
    df1 = _create_synthetic_sequence(num_hours=5)

    res1 = predictor.predict_sequence(df1)
    assert res1.sequence_length == 5
    assert len(res1.hourly_risk_trajectory) == 5

    # Create df2 with massive change only at hour 5
    df2 = df1.copy()
    for col in CANONICAL_FEATURE_COLUMNS:
        df2.loc[4, col] = 999.0

    res2 = predictor.predict_sequence(df2)

    # Hours 1..4 trajectories and raw logits must be IDENTICAL within floating-point epsilon
    for t in range(4):
        diff_prob = abs(res1.hourly_risk_trajectory[t] - res2.hourly_risk_trajectory[t])
        diff_logit = abs(res1.raw_logits[t] - res2.raw_logits[t])
        assert diff_prob < 1e-5, f"Causality violation at hour {t+1}: prob diff={diff_prob}"
        assert diff_logit < 1e-5, f"Causality violation at hour {t+1}: logit diff={diff_logit}"

    # Hour 5 prediction must change
    assert abs(res1.risk_score - res2.risk_score) > 1e-3


def test_calibrator_selection_and_probability_bounds():
    """Verifies unmixed calibrator selection and probability bounds in [0.0, 1.0]."""
    df = _create_synthetic_sequence(num_hours=3)

    # 1. Temperature calibrator (default, threshold=0.90)
    pred_temp = HourlySepsisPredictor(calibrator_type="temperature")
    res_temp = pred_temp.predict_sequence(df)
    assert res_temp.calibrator_type == "temperature"
    assert res_temp.decision_threshold == 0.90
    assert 0.0 <= res_temp.risk_score <= 1.0
    for p in res_temp.hourly_risk_trajectory:
        assert 0.0 <= p <= 1.0

    # 2. Platt calibrator (optional, threshold=0.05)
    pred_platt = HourlySepsisPredictor(calibrator_type="platt")
    res_platt = pred_platt.predict_sequence(df)
    assert res_platt.calibrator_type == "platt"
    assert res_platt.decision_threshold == 0.05
    assert 0.0 <= res_platt.risk_score <= 1.0

    # 3. None calibrator
    pred_none = HourlySepsisPredictor(calibrator_type="none")
    res_none = pred_none.predict_sequence(df)
    assert res_none.calibrator_type == "none"
    assert 0.0 <= res_none.risk_score <= 1.0

    # 4. Invalid calibrator
    with pytest.raises(ValueError, match="Unknown calibrator_type"):
        HourlySepsisPredictor(calibrator_type="invalid_cal")


def test_invalid_inputs_descriptive_errors():
    """Verifies that invalid inputs produce clear, descriptive validation errors."""
    predictor = HourlySepsisPredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH)

    # 1. Non-DataFrame
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        predictor.predict_sequence({"not": "a dataframe"})  # type: ignore

    # 2. Empty DataFrame
    with pytest.raises(ValueError, match="Input sequence DataFrame is empty"):
        predictor.predict_sequence(pd.DataFrame())

    # 3. Missing canonical feature column
    df = _create_synthetic_sequence(num_hours=2)
    df_missing = df.drop(columns=["heart_rate" if "heart_rate" in df.columns else "hr_mean"])
    with pytest.raises(ValueError, match="missing .* required canonical features"):
        predictor.predict_sequence(df_missing)

    # 4. Non-chronological hour index
    df_non_chrono = df.copy()
    df_non_chrono["hour_index"] = [2, 1]
    with pytest.raises(ValueError, match="Non-chronological sequence"):
        predictor.predict_sequence(df_non_chrono)

    # 5. Malformed endtime timestamp
    df_bad_time = df.copy()
    df_bad_time = df_bad_time.drop(columns=["hour_index"])
    df_bad_time["endtime"] = ["not_a_time", "also_bad"]
    with pytest.raises(ValueError, match="Malformed timestamps"):
        predictor.predict_sequence(df_bad_time)

    # 6. Exceeds max sequence length (120 hours)
    df_long = _create_synthetic_sequence(num_hours=125)
    with pytest.raises(ValueError, match="exceeds maximum supported positional encoding length"):
        predictor.predict_sequence(df_long)


def test_deterministic_inference():
    """Verifies that repeated inference on identical inputs produces identical bit-for-bit results."""
    predictor = HourlySepsisPredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH)
    df = _create_synthetic_sequence(num_hours=6)

    res1 = predictor.predict_sequence(df)
    res2 = predictor.predict_sequence(df)

    assert res1.risk_score == res2.risk_score
    assert res1.hourly_risk_trajectory == res2.hourly_risk_trajectory
    assert res1.raw_logits == res2.raw_logits
    assert res1.flagged_for_surveillance == res2.flagged_for_surveillance


def test_model_weights_not_updated_during_inference():
    """Verifies that repeated forward passes never alter trained model weights."""
    predictor = HourlySepsisPredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH)
    df = _create_synthetic_sequence(num_hours=4)

    # Snapshot weights
    initial_weights = [p.clone() for p in predictor.model.parameters()]

    # Run multiple inference predictions
    for _ in range(5):
        _ = predictor.predict_sequence(df)

    # Assert weights are bit-for-bit identical
    for w_init, p_curr in zip(initial_weights, predictor.model.parameters()):
        assert torch.equal(w_init, p_curr), "Model weights altered during inference!"


def test_research_disclaimer_present():
    """Verifies that the research disclaimer is attached to every prediction output."""
    predictor = HourlySepsisPredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH)
    df = _create_synthetic_sequence(num_hours=2)

    res = predictor.predict_sequence(df)
    assert res.disclaimer == RESEARCH_DISCLAIMER
    assert "RESEARCH PROTOTYPE PREDICTION ONLY" in res.disclaimer
    d = res.to_dict()
    assert "disclaimer" in d
    assert "risk_score" in d
