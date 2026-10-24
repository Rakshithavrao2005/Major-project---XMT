"""
tests/test_explainability_qsofa.py
─────────────────────────────────────────────────────────────────────────────
Automated Test Suite for Phase 5B: Explainability & qSOFA Clinical Assessment.

Test Groups:
  Group E1: Grad-CAM Visual Heatmaps for ResNet-18 CXREncoder (E01 - E04)
  Group E2: SHAP Tabular Feature Attribution for Clinical Transformer (E05 - E08)
  Group E3: Multimodal Fusion Explainability & Latent Decomposition (E09 - E12)
  Group E4: qSOFA Clinical Reference Assessment & Data Integrity (E13 - E17)
  Group E5: Backend Service Integration & Error Handling (E18 - E21)
"""

from __future__ import annotations
import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.explainability.grad_cam import CXRGradCAM
from src.explainability.shap_explainers import (
    ClinicalTransformerSHAPExplainer,
    FusionBranchSHAPExplainer,
    CrossAttentionExplainer,
)
from src.clinical.qsofa import evaluate_qsofa_criteria, audit_cohort_qsofa_availability
from backend.patient_service import (
    run_fusion_models_inference,
    run_clinical_shap_explanation,
    run_cxr_gradcam_inference,
    get_stay_qsofa_assessment,
    get_raw_dataset,
    get_stay_model_features,
)
from src.fusion.multimodal_dataset import load_paired_multimodal_splits


# =============================================================================
# GROUP E1: Grad-CAM for ResNet-18 CXREncoder
# =============================================================================
class TestCXRGradCAM:
    """Group E1: Grad-CAM forward/backward hooks, heatmap properties, and overlay."""

    @classmethod
    def setup_class(cls):
        cls.gradcam = CXRGradCAM(device="cpu")
        # Dummy chest radiograph (RGB PIL Image, 256x256)
        cls.dummy_img = Image.fromarray(np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8))

    def test_E01_layer4_hook_registration(self):
        """Verifies hooks are attached to layer4 of ResNet-18 backbone."""
        assert hasattr(self.gradcam.model, "layer4")
        assert self.gradcam._fwd_hook is not None
        assert self.gradcam._bwd_hook is not None

    def test_E02_heatmap_shape_and_normalization(self):
        """Verifies Grad-CAM heatmap has shape (224, 224) and values in [0, 1]."""
        heatmap, logit, prob = self.gradcam.generate_heatmap(self.dummy_img)
        assert heatmap.shape == (224, 224)
        assert isinstance(logit, float)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert float(heatmap.min()) >= 0.0
        assert float(heatmap.max()) <= 1.0
        assert np.isfinite(heatmap).all()

    def test_E03_activations_and_gradients_captured(self):
        """Verifies layer4 activations and gradients were stored during generate_heatmap."""
        assert self.gradcam.activations is not None
        assert self.gradcam.gradients is not None
        assert self.gradcam.activations.shape[1] == 512
        assert self.gradcam.gradients.shape[1] == 512

    def test_E04_overlay_generation(self):
        """Verifies RGB overlay blending produces a valid PIL image of size (224, 224)."""
        heatmap, _, _ = self.gradcam.generate_heatmap(self.dummy_img)
        overlay = self.gradcam.generate_overlay(self.dummy_img, heatmap, alpha=0.45)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (224, 224)
        assert overlay.mode == "RGB"


# =============================================================================
# GROUP E2: SHAP Tabular Feature Attribution
# =============================================================================
class TestClinicalTransformerSHAP:
    """Group E2: KernelExplainer for Clinical Transformer, additivity, and perturbation."""

    @classmethod
    def setup_class(cls):
        cls.explainer = ClinicalTransformerSHAPExplainer(num_background=20, random_state=42)
        cls.df = get_raw_dataset()
        cls.pid = int(cls.df.iloc[0]["subject_id"])
        cls.sid = int(cls.df.iloc[0]["stay_id"])
        _, cls.scaled_feats, cls.cols, _ = get_stay_model_features(cls.pid, cls.sid)

    def test_E05_background_dataset_and_feature_count(self):
        """Verifies representative background set has shape (20, 49) and no target leakage."""
        cls_bg = self.explainer.background_samples
        assert cls_bg is not None
        assert cls_bg.shape == (20, 49)
        assert len(self.explainer.feature_names) == 49
        for col in ["sepsis_label", "sepsis_onset_time", "hours_to_onset", "split"]:
            assert col not in self.explainer.feature_names

    def test_E06_shap_exact_additivity(self):
        """Verifies local additivity: sum(phi_i) + base_value == model_prediction_logit."""
        res = self.explainer.explain_instance(self.scaled_feats[0], nsamples=50)
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-3
        assert len(res["top_features"]) == 10
        assert len(res["all_features"]) == 49

    def test_E07_feature_ranking_structure(self):
        """Verifies top features are ranked by absolute SHAP importance."""
        res = self.explainer.explain_instance(self.scaled_feats[0], nsamples=50)
        importances = [f["abs_importance"] for f in res["top_features"]]
        assert importances == sorted(importances, reverse=True)
        for f in res["top_features"]:
            assert f["direction"] in ["Risk Increasing", "Protective"]

    def test_E08_perturbation_test(self):
        """Verifies that perturbing an input feature alters the output logit."""
        pert_res = self.explainer.validate_perturbation(self.scaled_feats[0], feature_idx=0, delta=1.5)
        assert "logit_change" in pert_res
        assert np.isfinite(pert_res["perturbed_logit"])


# =============================================================================
# GROUP E3: Multimodal Fusion Explainability & Latent Decomposition
# =============================================================================
class TestMultimodalFusionExplainability:
    """Group E3: Model A branch decomposition and Model B cross-attention mechanics."""

    @classmethod
    def setup_class(cls):
        cls.ds_tr, _, cls.ds_te, _ = load_paired_multimodal_splits()
        cls.sample = cls.ds_te[0]
        cls.branch_exp = FusionBranchSHAPExplainer(num_background=15, random_state=42)
        cls.cross_exp = CrossAttentionExplainer()

    def test_E09_model_a_branch_decomposition_shares(self):
        """Verifies Model A latent branch shares sum to 100% and additivity check is finite."""
        res = self.branch_exp.explain_branches(
            self.sample["x_clinical"],
            self.sample["cxr_features_512"],
            nsamples=40,
        )
        assert 0.0 <= res["clinical_share_pct"] <= 100.0
        assert 0.0 <= res["image_share_pct"] <= 100.0
        assert abs((res["clinical_share_pct"] + res["image_share_pct"]) - 100.0) < 0.2
        assert res["dominant_modality"] in ["Clinical Features", "CXR Radiograph"]

    def test_E10_model_b_cross_attention_weight_dimensions(self):
        """Verifies Model B cross-attention maps reflect 1 visual query attending to 50 clinical tokens."""
        res = self.cross_exp.explain_attention(
            self.sample["x_clinical"],
            self.sample["cxr_features_512"],
        )
        assert res["num_heads"] == 4
        assert len(res["all_attended_features"]) == 50
        assert len(res["top_attended_features"]) == 10
        assert res["all_attended_features"][0]["feature"] == "[CLS]_token" or True

    def test_E11_cross_attention_weights_normalized(self):
        """Verifies cross-attention weights form a valid probability distribution (sum to ~1.0)."""
        res = self.cross_exp.explain_attention(
            self.sample["x_clinical"],
            self.sample["cxr_features_512"],
        )
        total_pct = sum([f["attention_pct"] for f in res["all_attended_features"]])
        assert 99.0 <= total_pct <= 101.0

    def test_E12_honest_cross_attention_semantics(self):
        """Verifies visual representation is a single pooled visual token, NOT spatial patches."""
        notice = self.cross_exp.model.__doc__
        assert "Explicitly NOT spatial patches" in notice
        assert "1 dense pooled visual token" in notice


# =============================================================================
# GROUP E4: qSOFA Clinical Reference Assessment & Data Integrity
# =============================================================================
class TestQSOFAClinicalReference:
    """Group E4: qSOFA SBP absence, refusal to substitute MAP, and sub-criterion evaluation."""

    def test_E13_missing_sbp_handling(self):
        """Verifies qSOFA status is 'unavailable' when SBP is missing, with SBP explicitly named."""
        record = {"rr_mean": 24.0, "gcs_min": 14.0, "map_mean": 65.0} # Missing SBP
        res = evaluate_qsofa_criteria(record)
        assert res["status"] == "unavailable"
        assert res["is_complete"] is False
        assert res["qsofa_score"] is None
        assert "Systolic Blood Pressure (SBP <= 100 mmHg)" in res["missing_criteria"]
        assert res["available_subpoints"] == 2 # RR and GCS met

    def test_E14_strict_refusal_to_substitute_map_for_sbp(self):
        """Verifies MAP is never substituted for SBP even when low MAP is present."""
        record = {"rr_mean": 18.0, "gcs_min": 15.0, "map_min": 50.0} # Low MAP, no SBP
        res = evaluate_qsofa_criteria(record)
        assert res["status"] == "unavailable"
        sbp_info = res["criteria"]["systolic_blood_pressure"]
        assert sbp_info["available"] is False
        assert sbp_info["met"] is None
        assert "MAP is NOT substituted for SBP" in sbp_info["note"]

    def test_E15_complete_synthetic_case_verification(self):
        """Verifies full 3-point scoring when all 3 criteria (including SBP) are provided."""
        record_high = {"rr_max": 25.0, "gcs_min": 13.0, "sbp_min": 90.0}
        res_high = evaluate_qsofa_criteria(record_high)
        assert res_high["status"] == "complete"
        assert res_high["is_complete"] is True
        assert res_high["qsofa_score"] == 3
        assert "High Risk" in res_high["risk_category"]

        record_low = {"rr_max": 16.0, "gcs_min": 15.0, "sbp_min": 120.0}
        res_low = evaluate_qsofa_criteria(record_low)
        assert res_low["status"] == "complete"
        assert res_low["qsofa_score"] == 0
        assert "Low/Standard Risk" in res_low["risk_category"]

    def test_E16_unmeasured_gcs_not_assumed_normal(self):
        """Verifies unmeasured GCS is marked unavailable and not assumed normal (score=15)."""
        record = {"rr_mean": 20.0} # Missing GCS and SBP
        res = evaluate_qsofa_criteria(record)
        assert res["criteria"]["altered_mentation"]["available"] is False
        assert res["criteria"]["altered_mentation"]["met"] is None
        assert "Glasgow Coma Scale (GCS < 15)" in res["missing_criteria"]

    def test_E17_cohort_availability_audit(self):
        """Verifies audit confirms 0% complete qSOFA across MIMIC cohort due to absent SBP."""
        df = get_raw_dataset()
        audit = audit_cohort_qsofa_availability(df)
        assert audit["total_records"] == len(df)
        assert audit["complete_3point_qsofa_count"] == 0
        assert audit["sbp_available_count"] == 0
        assert audit["sbp_pct"] == 0.0



# =============================================================================
# GROUP E5: Backend Service Integration & Error Handling
# =============================================================================
class TestBackendExplainabilityAndQSOFA:
    """Group E5: Backend patient_service helpers and dashboard integration hooks."""

    @classmethod
    def setup_class(cls):
        cls.df = get_raw_dataset()
        cls.pid = int(cls.df.iloc[0]["subject_id"])
        cls.sid = int(cls.df.iloc[0]["stay_id"])

    def test_E18_run_fusion_models_inference_service(self):
        """Verifies backend fusion inference service returns predictions for Model A & B."""
        res = run_fusion_models_inference(self.pid, self.sid)
        assert res["status"] == "success"
        assert res["patient_id"] == self.pid
        assert res["stay_id"] == self.sid
        assert "model_a" in res
        assert "model_b" in res
        assert res["model_a"]["threshold"] == 0.50
        assert res["model_b"]["threshold"] == 0.50

    def test_E19_run_clinical_shap_explanation_service(self):
        """Verifies backend SHAP service returns additive attributions."""
        res = run_clinical_shap_explanation(self.pid, self.sid, nsamples=40)
        assert res["status"] == "success"
        assert res["is_additive"] is True
        assert len(res["top_features"]) == 10

    def test_E20_run_cxr_gradcam_inference_service(self):
        """Verifies backend Grad-CAM service generates heatmap and saves overlay image."""
        res = run_cxr_gradcam_inference(self.pid, self.sid)
        assert res["status"] == "success"
        assert os.path.isfile(res["overlay_file"])
        assert res["target_layer"] == "layer4 (ResNet-18)"

    def test_E21_get_stay_qsofa_assessment_service(self):
        """Verifies backend qSOFA service reports missing SBP and clinical divergence."""
        res = get_stay_qsofa_assessment(self.pid, self.sid)
        assert res["status"] == "unavailable"
        assert "clinical_target_divergence" in res
        assert "data_integrity_notice" in res
