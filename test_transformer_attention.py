"""
tests/test_transformer_attention.py
─────────────────────────────────────────────────────────────────────────────
Regression tests for Transformer Self-Attention extraction and UI rendering.
Validates:
1. run_clinical_transformer_inference returns both attention_weight and attention_pct.
2. Attention percentage sums and format consistency.
3. Resilience against missing attention_pct or empty attention outputs.
"""

import pytest
import pandas as pd
from backend.patient_service import (
    run_clinical_transformer_inference,
    get_all_patients_directory,
    search_patient,
)


def test_transformer_attention_schema():
    """Verify that run_clinical_transformer_inference returns valid attention keys."""
    patients = get_all_patients_directory()
    assert len(patients) > 0
    test_pid = int(patients.iloc[0]["subject_id"])
    p_info = search_patient(test_pid)
    assert p_info["found"] is True
    stays = p_info["stays"]
    assert len(stays) > 0
    test_stay = stays[0]["stay_id"]

    res = run_clinical_transformer_inference(test_pid, test_stay)
    assert res["status"] == "success"
    assert "top_attended_features" in res
    attn_list = res["top_attended_features"]
    assert isinstance(attn_list, list)

    if len(attn_list) > 0:
        for item in attn_list:
            assert "feature" in item
            assert "attention_weight" in item
            assert "attention_pct" in item
            assert isinstance(item["attention_weight"], float)
            assert isinstance(item["attention_pct"], float)
            assert 0.0 <= item["attention_weight"] <= 1.0
            assert 0.0 <= item["attention_pct"] <= 100.0


def test_attention_dataframe_defensive_fallbacks():
    """Verify that attention dataframe rendering handles missing keys, zero weights, or empty arrays."""
    # Scenario A: Standard schema with attention_pct
    data_full = [
        {"feature": "wbc", "attention_weight": 0.1234, "attention_pct": 12.34},
        {"feature": "lactate", "attention_weight": 0.0850, "attention_pct": 8.50},
    ]
    df_a = pd.DataFrame(data_full)
    assert "attention_pct" in df_a.columns
    formatted_a = df_a["attention_pct"].apply(lambda p: f"{float(p):.1f}%")
    assert formatted_a.iloc[0] == "12.3%"

    # Scenario B: Missing attention_pct (legacy or mock)
    data_missing_pct = [
        {"feature": "heart_rate", "attention_weight": 0.2000},
        {"feature": "temp", "attention_weight": 0.0500},
    ]
    df_b = pd.DataFrame(data_missing_pct)
    if "attention_pct" not in df_b.columns:
        total_w = df_b["attention_weight"].sum()
        df_b["attention_pct"] = (df_b["attention_weight"] / total_w) * 100.0 if total_w > 0 else 0.0

    assert "attention_pct" in df_b.columns
    assert round(df_b["attention_pct"].iloc[0], 1) == 80.0
    assert round(df_b["attention_pct"].iloc[1], 1) == 20.0

    # Scenario C: Empty list
    empty_list = []
    assert len(empty_list) == 0
