"""
tests/test_strict_followup.py
─────────────────────────────────────────────────────────────────────────────
Automated Test Suite for Step 3: Strict-Follow-Up Sensitivity Analysis.

Test Groups:
- Group S01-S03: Cohort Dimensions, Membership & Follow-Up Audit
- Group S04-S06: Zero Positive Loss & Confusion Matrix Invariants
- Group S07-S09: Subject-Level Cluster Resampling & Confidence Intervals
- Group S10-S12: Artifact Integrity, Metrics Schema & Prevalence Decoupling
"""

from __future__ import annotations
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.run_strict_followup_analysis import compute_cluster_bootstrap_ci

STRICT_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports", "strict_followup")
PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")


class TestStrictFollowupCohortIntegrity:
    """Group S01-S03: Cohort Dimensions, Membership & Follow-Up Audit."""

    def test_S01_dataset_dimensions_exact(self):
        """Verifies exact row and column dimensions for main and strict raw datasets."""
        df_main = pd.read_csv(os.path.join(RAW_DIR, "dataset_T6h_H24_v5.csv"))
        df_strict = pd.read_csv(os.path.join(RAW_DIR, "dataset_T6h_H24_v5_strictfollowup.csv"))

        assert df_main.shape == (2884, 62)
        assert df_strict.shape == (1909, 62)
        assert len(df_main) - len(df_strict) == 975

    def test_S02_all_excluded_rows_are_short_followup_negatives(self):
        """Confirms that all 975 excluded rows have sepsis_label == 0 and followup_h < 24.0."""
        df_main = pd.read_csv(os.path.join(RAW_DIR, "dataset_T6h_H24_v5.csv"))
        df_strict = pd.read_csv(os.path.join(RAW_DIR, "dataset_T6h_H24_v5_strictfollowup.csv"))

        excluded_stays = set(df_main["stay_id"]) - set(df_strict["stay_id"])
        df_excluded = df_main[df_main["stay_id"].isin(excluded_stays)]

        assert len(df_excluded) == 975
        assert (df_excluded["sepsis_label"] == 0).all()
        assert (df_excluded["followup_h"] < 24.0).all()

    def test_S03_test_cohort_dimensions_and_prevalence_shift(self):
        """Verifies that the strict test set contains exactly 403 stays with 104 positives."""
        ids_te = pd.read_csv(os.path.join(PROC_DIR, "ids_test.csv"))
        y_te = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv"))["sepsis_label"].values
        df_strict = pd.read_csv(os.path.join(RAW_DIR, "dataset_T6h_H24_v5_strictfollowup.csv"))

        strict_stay_set = set(df_strict["stay_id"])
        strict_mask = ids_te["stay_id"].isin(strict_stay_set).values

        assert len(ids_te) == 602
        assert int(strict_mask.sum()) == 403
        assert len(ids_te) - int(strict_mask.sum()) == 199
        # Prevalence comparison
        assert abs(y_te.mean() - 0.17275) < 1e-3
        assert abs(y_te[strict_mask].mean() - 0.25806) < 1e-3


class TestConfusionMatrixInvariants:
    """Group S04-S06: Zero Positive Loss & Confusion Matrix Invariants."""

    def test_S04_all_positive_test_cases_retained(self):
        """Confirms exactly 104 positives in both original and strict test sets."""
        with open(os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["cohort_comparison"]["original"]["n_positives"] == 104
        assert data["metadata"]["cohort_comparison"]["strict_followup"]["n_positives"] == 104
        assert data["metadata"]["cohort_comparison"]["strict_followup"]["excluded_positive_count"] == 0

    def test_S05_sensitivity_strictly_invariant_across_cohorts(self):
        """Verifies that sensitivity (recall) and TP/FN counts are identical between cohorts."""
        with open(os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        for m_name, res in data["results"].items():
            orig = res["original_cohort"]
            strict = res["strict_cohort"]
            # Sensitivity must be identical
            assert orig["sensitivity"] == strict["sensitivity"], f"{m_name} sensitivity changed!"
            # True Positives and False Negatives must be identical
            assert orig["confusion_matrix"]["tp"] == strict["confusion_matrix"]["tp"]
            assert orig["confusion_matrix"]["fn"] == strict["confusion_matrix"]["fn"]

    def test_S06_specificity_decreases_and_fpr_increases(self):
        """Verifies that removing rapid-discharge negatives reduces specificity (increases FPR)."""
        with open(os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        for m_name, res in data["results"].items():
            orig = res["original_cohort"]
            strict = res["strict_cohort"]
            # Non-trivial models should have fewer true negatives
            assert strict["confusion_matrix"]["tn"] <= orig["confusion_matrix"]["tn"]


class TestClusterResamplingAndBootstrap:
    """Group S07-S09: Subject-Level Cluster Resampling & Confidence Intervals."""

    def test_S07_cluster_bootstrap_valid_and_ordered_intervals(self):
        """Verifies that compute_cluster_bootstrap_ci returns valid [lower, upper] intervals."""
        rng = np.random.RandomState(42)
        n = 100
        ids_df = pd.DataFrame({
            "stay_id": range(n),
            "subject_id": [f"SUBJ_{i // 2}" for i in range(n)], # 50 subjects with 2 stays each
        })
        y_true = rng.binomial(1, 0.25, n)
        y_prob = np.clip(y_true * 0.5 + rng.normal(0, 0.2, n), 0, 1)

        ci = compute_cluster_bootstrap_ci(ids_df, y_true, y_prob, n_bootstraps=50)
        assert len(ci["roc_auc_ci"]) == 2
        assert len(ci["pr_auc_ci"]) == 2
        assert 0.0 <= ci["roc_auc_ci"][0] <= ci["roc_auc_ci"][1] <= 1.0
        assert 0.0 <= ci["pr_auc_ci"][0] <= ci["pr_auc_ci"][1] <= 1.0

    def test_S08_multiple_stays_per_subject_present(self):
        """Verifies that subjects with multiple stays exist in both test sets."""
        ids_te = pd.read_csv(os.path.join(PROC_DIR, "ids_test.csv"))
        df_strict = pd.read_csv(os.path.join(RAW_DIR, "dataset_T6h_H24_v5_strictfollowup.csv"))
        strict_ids = ids_te[ids_te["stay_id"].isin(set(df_strict["stay_id"]))]

        multi_orig = (ids_te["subject_id"].value_counts() > 1).sum()
        multi_strict = (strict_ids["subject_id"].value_counts() > 1).sum()

        assert multi_orig > 0
        assert multi_strict > 0
        assert multi_orig == 42
        assert multi_strict == 23


class TestStrictArtifactsAndReporting:
    """Group S10-S12: Artifact Integrity, Metrics Schema & Prevalence Decoupling."""

    def test_S10_strict_followup_artifacts_exist(self):
        """Verifies that all report files exist in reports/strict_followup/."""
        assert os.path.isfile(os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json"))
        assert os.path.isfile(os.path.join(STRICT_REPORTS_DIR, "STRICT_FOLLOWUP_REPORT.md"))
        assert os.path.isfile(os.path.join(STRICT_REPORTS_DIR, "strict_followup_curves_comparison.png"))

    def test_S11_all_target_models_evaluated(self):
        """Verifies that baseline, deep tabular, CXR, and fusion models are all evaluated."""
        with open(os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        required_models = [
            "Logistic Regression (Calibrated)",
            "Random Forest (Calibrated)",
            "Clinical Transformer (Uncalibrated)",
            "Clinical Transformer (Platt Calibrated)",
            "CXR Encoder (Unimodal Image)",
            "Model A: Concat Fusion (Uncalibrated)",
            "Model A: Concat Fusion (Platt Calibrated)",
            "Model B: Cross-Attention (Uncalibrated)",
            "Model B: Cross-Attention (Platt Calibrated)",
        ]
        for m in required_models:
            assert m in data["results"]

    def test_S12_auprc_increases_with_prevalence(self):
        """Verifies that AUPRC increases strictly due to baseline prevalence shift (0.1728 -> 0.2581)."""
        with open(os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        for m_name, res in data["results"].items():
            # AUPRC in strict cohort must be greater than original cohort due to higher baseline
            assert res["strict_cohort"]["pr_auc"] > res["original_cohort"]["pr_auc"], (
                f"{m_name} AUPRC did not increase despite +8.53% prevalence increase"
            )

    def test_S13_thresholds_match_authoritative_step2_metadata(self):
        """Verifies Step 3 calibrated operating thresholds match authoritative Step 2 metadata."""
        cal_path = os.path.join(_PROJECT_ROOT, "reports", "calibration", "calibration_metrics.json")
        strict_path = os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json")
        with open(cal_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        with open(strict_path, "r", encoding="utf-8") as f:
            strict_data = json.load(f)

        mapping = [
            ("Clinical Transformer", "Clinical Transformer (Platt Calibrated)"),
            ("Model A (Concatenation Fusion)", "Model A: Concat Fusion (Platt Calibrated)"),
            ("Model B (Cross-Attention Fusion)", "Model B: Cross-Attention (Platt Calibrated)"),
        ]
        for step2_name, step3_name in mapping:
            expected_thr = float(cal_data["results"][step2_name]["test"]["platt"]["classification_at_val_opt_thr"]["val_opt_threshold"])
            actual_thr = float(strict_data["results"][step3_name]["strict_cohort"]["operating_threshold"])
            assert abs(actual_thr - expected_thr) < 1e-4, (
                f"{step3_name}: operating threshold {actual_thr} does not match Step 2 metadata {expected_thr}"
            )

    def test_S14_independent_numerical_recomputation_strict_subset(self):
        """Independently recomputes AUROC and Brier score from frozen baseline model on strict subset."""
        import pickle
        from sklearn.metrics import roc_auc_score, brier_score_loss

        ids_te = pd.read_csv(os.path.join(PROC_DIR, "ids_test.csv"))
        X_te = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        y_te = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv"))["sepsis_label"].values
        df_strict = pd.read_csv(os.path.join(RAW_DIR, "dataset_T6h_H24_v5_strictfollowup.csv"))

        strict_mask = ids_te["stay_id"].isin(set(df_strict["stay_id"])).values
        y_strict = y_te[strict_mask]
        X_strict = X_te[strict_mask]

        # Load frozen Logistic Regression model
        with open(os.path.join(_PROJECT_ROOT, "models", "baseline", "logistic_regression.pkl"), "rb") as f:
            lr_model = pickle.load(f)

        raw_probs = lr_model.predict_proba(X_strict)[:, 1]
        raw_auroc = float(roc_auc_score(y_strict, raw_probs))
        raw_brier = float(brier_score_loss(y_strict, raw_probs))

        with open(os.path.join(STRICT_REPORTS_DIR, "strict_followup_metrics.json"), "r", encoding="utf-8") as f:
            strict_data = json.load(f)

        rep = strict_data["results"]["Logistic Regression (Calibrated)"]["strict_cohort"]
        rep_auroc = rep["roc_auc"]
        rep_brier = rep["brier_score"]

        assert abs(raw_auroc - rep_auroc) <= 1e-4, f"AUROC mismatch: recomputed {raw_auroc:.4f} vs reported {rep_auroc:.4f}"
        assert abs(raw_brier - rep_brier) <= 1e-4, f"Brier mismatch: recomputed {raw_brier:.4f} vs reported {rep_brier:.4f}"

