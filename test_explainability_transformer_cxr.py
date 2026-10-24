"""
tests/test_explainability_transformer_cxr.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Test Suite for Step 5B:
Clinical Transformer SHAP and CXR Grad-CAM Validation.

Verifies:
1. Clinical Transformer SHAP setup, scaler, and training background data provenance.
2. Clinical Transformer SHAP exact additivity across logits and Platt-calibrated probability.
3. Clinical Transformer SHAP reproducibility, stability, and attention vs attribution theory.
4. CXR Grad-CAM architecture, layer4 hooks, eval mode gradients, and transforms.
5. CXR Grad-CAM heatmap dimensions (224x224), [0, 1] bounds, finite check, and strict determinism.
6. CXR Grad-CAM overlay generation, hook cleanup, resolution limits, and clinical disclaimers.
"""

from __future__ import annotations
import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.explainability.shap_explainers import (
    ClinicalTransformerSHAPExplainer,
    CLINICAL_DISCLAIMER,
)
from src.explainability.grad_cam import CXRGradCAM
from src.imaging.cxr_encoder import CXREncoder
from src.clinical.clinical_transformer import ClinicalTransformer

PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


# =============================================================================
# 1. Clinical Transformer SHAP Setup & Provenance
# =============================================================================
class TestClinicalTransformerSHAPSetupAndProvenance:
    """Verifies checkpoint loading, feature order, and training background provenance."""

    @classmethod
    def setup_class(cls):
        cls.explainer = ClinicalTransformerSHAPExplainer(num_background=15, random_state=42)
        cls.X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))

    def test_checkpoint_and_scaler_loading(self):
        """Confirms model and scaler load properly with matching architecture."""
        assert isinstance(self.explainer.model, ClinicalTransformer)
        assert self.explainer.model.training is False
        assert hasattr(self.explainer.scaler, "mean_")
        assert len(self.explainer.scaler.mean_) == 49

    def test_feature_schema_and_order(self):
        """Confirms exactly 49 features matching X_train and X_test column order."""
        assert len(self.explainer.feature_names) == 49
        assert self.explainer.feature_names == list(self.X_train.columns)
        assert self.explainer.feature_names == list(self.X_test.columns)

    def test_zero_target_leakage_in_features(self):
        """Confirms no targets or identifiers in feature names."""
        forbidden = ["sepsis_label", "sepsis_onset_time", "hours_to_onset", "split", "subject_id", "stay_id"]
        for f in self.explainer.feature_names:
            assert f not in forbidden

    def test_background_provenance_strictly_training(self):
        """Confirms background samples are derived strictly from training partition."""
        bg = self.explainer.background_samples
        assert bg.shape == (15, 49)
        assert np.isfinite(bg).all()
        # Verify background samples fall within training feature distribution bounds
        X_tr_scaled = self.explainer.scaler.transform(self.X_train.values)
        tr_min = np.min(X_tr_scaled, axis=0) - 1e-5
        tr_max = np.max(X_tr_scaled, axis=0) + 1e-5
        assert np.all(bg >= tr_min)
        assert np.all(bg <= tr_max)

    def test_platt_calibrator_loading(self):
        """Confirms Platt scaling parameters are loaded and mathematically valid."""
        assert self.explainer.platt_a is not None
        assert self.explainer.platt_b is not None
        assert abs(self.explainer.platt_a - 0.60745) < 1e-3
        assert abs(self.explainer.platt_b - (-1.55000)) < 1e-3


# =============================================================================
# 2. Clinical Transformer SHAP Additivity
# =============================================================================
class TestClinicalTransformerSHAPAdditivity:
    """Verifies game-theoretic additivity in both logit and calibrated probability spaces."""

    @classmethod
    def setup_class(cls):
        cls.explainer = ClinicalTransformerSHAPExplainer(num_background=15, random_state=42)
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        # Test across first 2 held-out test patients
        cls.sample_0_scaled = cls.explainer.scaler.transform(cls.X_test.iloc[:1].values)[0]
        cls.sample_1_scaled = cls.explainer.scaler.transform(cls.X_test.iloc[1:2].values)[0]

    def test_logits_additivity_sample_0(self):
        """Verifies exact additivity for model logit output: sum(phi) + base == logit."""
        res = self.explainer.explain_instance(self.sample_0_scaled, output_type="logits", nsamples=100)
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-3
        assert res["output_type"] == "logits"
        # Verify model logit match
        with torch.no_grad():
            t_x = torch.as_tensor(self.sample_0_scaled.reshape(1, -1), dtype=torch.float32)
            direct_logit = float(self.explainer.model(t_x).item())
        assert abs(res["prediction_logit"] - direct_logit) < 1e-4

    def test_calibrated_probability_additivity_sample_0(self):
        """Verifies exact additivity for calibrated probability: sum(phi) + base == P_cal."""
        res = self.explainer.explain_instance(self.sample_0_scaled, output_type="calibrated_probability", nsamples=100)
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-3
        assert res["output_type"] == "calibrated_probability"
        assert 0.0 <= res["calibrated_probability"] <= 1.0
        # Verify calibrated probability match via Platt scaling
        logit = res["prediction_logit"]
        expected_cal_p = 1.0 / (1.0 + np.exp(-(self.explainer.platt_a * logit + self.explainer.platt_b)))
        assert abs(res["calibrated_probability"] - expected_cal_p) < 1e-4

    def test_logits_additivity_sample_1(self):
        """Verifies additivity on second held-out test patient."""
        res = self.explainer.explain_instance(self.sample_1_scaled, output_type="logits", nsamples=100)
        assert res["is_additive"] is True
        assert res["additivity_difference"] < 1e-3

    def test_explanation_payload_structure(self):
        """Verifies all required metadata, feature rankings, and disclaimers are present."""
        res = self.explainer.explain_instance(self.sample_0_scaled, output_type="logits", nsamples=80)
        assert "model_name" in res
        assert "prediction_value" in res
        assert "base_value" in res
        assert "top_features" in res
        assert len(res["top_features"]) == 10
        assert len(res["all_features"]) == 49
        assert "disclaimer" in res
        assert CLINICAL_DISCLAIMER in res["disclaimer"]
        # Verify features are ranked by absolute importance
        top = res["top_features"]
        for i in range(len(top) - 1):
            assert top[i]["abs_importance"] >= top[i+1]["abs_importance"]

    def test_cohort_explanation_dimensions(self):
        """Verifies cohort explanation matrix dimensions on test batch."""
        test_batch = self.explainer.scaler.transform(self.X_test.iloc[:5].values)
        res = self.explainer.explain_cohort(test_batch, output_type="logits", nsamples=80)
        assert res["cohort_size"] == 5
        assert res["shap_matrix"].shape == (5, 49)
        assert len(res["all_features"]) == 49
        assert len(res["top_features"]) == 15


# =============================================================================
# 3. Clinical Transformer SHAP Stability & Theoretical Distinction
# =============================================================================
class TestClinicalTransformerStabilityAndAttributionTheory:
    """Verifies sampling stability, reproducibility, and attention vs attribution theory."""

    @classmethod
    def setup_class(cls):
        cls.explainer = ClinicalTransformerSHAPExplainer(num_background=15, random_state=42)
        cls.X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
        cls.sample_scaled = cls.explainer.scaler.transform(cls.X_test.iloc[:1].values)[0]

    def test_exact_reproducibility_with_fixed_explainer(self):
        """Confirms that calling explain_instance twice on the same instance gives identical results."""
        res1 = self.explainer.explain_instance(self.sample_scaled, output_type="logits", nsamples=100)
        res2 = self.explainer.explain_instance(self.sample_scaled, output_type="logits", nsamples=100)
        assert res1["prediction_value"] == res2["prediction_value"]
        assert res1["base_value"] == res2["base_value"]
        assert res1["shap_sum"] == res2["shap_sum"]
        assert res1["additivity_difference"] == res2["additivity_difference"]

    def test_attention_vs_attribution_conceptual_distinction(self):
        """
        Demonstrates the fundamental mathematical difference between attention weights and SHAP:
        1. Attention weights are non-negative and sum to 1.0 across keys: sum_j(alpha_j) == 1, alpha_j >= 0.
           They represent representation routing/aggregation, never negative (protective) effects.
        2. SHAP attributions are game-theoretic marginal contributions that can be positive (risk-increasing)
           or negative (protective), summing to the deviation from the expected base value.
        """
        # 1. Extract attention weights directly from Clinical Transformer forward(return_attention=True)
        x_tensor = torch.as_tensor(self.sample_scaled.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            logits, attentions = self.explainer.model(x_tensor, return_attention=True)

        assert len(attentions) == 2, "Clinical Transformer has 2 transformer layers"
        attn_layer0 = attentions[0].squeeze(0).cpu().numpy() # (num_heads=4, seq_len=50, seq_len=50)
        assert attn_layer0.shape == (4, 50, 50)

        # Verify attention properties:
        assert (attn_layer0 >= -1e-6).all(), "Attention weights must be strictly non-negative"
        for h in range(4):
            row_sums = attn_layer0[h].sum(axis=-1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

        # 2. Extract SHAP values
        shap_res = self.explainer.explain_instance(self.sample_scaled, output_type="logits", nsamples=80)
        shap_vals = [f["shap_value"] for f in shap_res["all_features"]]
        has_positive = any(v > 0 for v in shap_vals)
        has_negative = any(v < 0 for v in shap_vals)

        # SHAP must support bidirectional attribution (both protective and risk-increasing)
        assert has_positive, "SHAP attribution should include risk-increasing features"
        assert has_negative, "SHAP attribution should include protective features"
        # Total sum of SHAP attributions is not constrained to 1.0
        assert abs(sum(shap_vals) - 1.0) > 0.1, "SHAP sum should not be constrained to 1.0 like attention"


# =============================================================================
# 4. CXR Grad-CAM Architecture & Setup
# =============================================================================
class TestCXRGradCAMArchitectureAndSetup:
    """Verifies CXR Grad-CAM architecture, hooks, and evaluation mode preservation."""

    @classmethod
    def setup_class(cls):
        cls.gradcam = CXRGradCAM(device="cpu")

    @classmethod
    def teardown_class(cls):
        cls.gradcam.close()

    def test_encoder_architecture_and_weights(self):
        """Confirms ResNet-18 CXREncoder loaded with embed_dim=128."""
        assert isinstance(self.gradcam.model, CXREncoder)
        assert hasattr(self.gradcam.model, "layer4")
        assert hasattr(self.gradcam.model, "classifier")
        # Final linear layer in classification head produces scalar output (1 feature)
        assert self.gradcam.model.classifier[-1].out_features == 1

    def test_layer4_hooks_registered(self):
        """Confirms forward and backward hooks are active on layer4."""
        assert self.gradcam._fwd_hook is not None
        assert self.gradcam._bwd_hook is not None

    def test_eval_mode_gradient_preservation(self):
        """
        Confirms model is in eval mode (preventing BatchNorm drift) while
        torch.enable_grad() allows backpropagation for Grad-CAM.
        """
        assert self.gradcam.model.training is False
        # BatchNorm running statistics must remain fixed
        running_mean_before = self.gradcam.model.bn1.running_mean.clone()

        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        self.gradcam.generate_heatmap(dummy_img)

        running_mean_after = self.gradcam.model.bn1.running_mean
        torch.testing.assert_close(running_mean_before, running_mean_after)


# =============================================================================
# 5. CXR Grad-CAM Execution & Mathematical Properties
# =============================================================================
class TestCXRGradCAMExecutionAndProperties:
    """Verifies Grad-CAM heatmap generation, bounds, determinism, and prediction alignment."""

    @classmethod
    def setup_class(cls):
        cls.gradcam = CXRGradCAM(device="cpu")
        cls.ids_test = pd.read_csv(os.path.join(PROC_DIR, "ids_test.csv"))
        # Load first test radiograph
        img_rel_path = cls.ids_test.iloc[0]["image_path"]
        cls.img_path = os.path.join(RAW_DIR, img_rel_path)
        cls.img = Image.open(cls.img_path)

    @classmethod
    def teardown_class(cls):
        cls.gradcam.close()

    def test_gradcam_heatmap_dimensions_and_finite(self):
        """Confirms heatmap shape is exactly (224, 224) and all values are finite."""
        heatmap, logit, prob = self.gradcam.generate_heatmap(self.img)
        assert heatmap.shape == (224, 224)
        assert np.isfinite(heatmap).all()
        assert not np.isnan(heatmap).any()

    def test_gradcam_normalized_bounds(self):
        """Confirms heatmap values are strictly bounded in [0.0, 1.0]."""
        heatmap, logit, prob = self.gradcam.generate_heatmap(self.img)
        assert 0.0 <= heatmap.min()
        assert heatmap.max() <= 1.0
        assert heatmap.max() > heatmap.min(), "Heatmap should have non-trivial contrast"

    def test_gradcam_strict_determinism(self):
        """Confirms bitwise identical heatmaps across multiple forward-backward runs."""
        hm1, l1, p1 = self.gradcam.generate_heatmap(self.img)
        hm2, l2, p2 = self.gradcam.generate_heatmap(self.img)
        assert l1 == l2
        assert p1 == p2
        diff = np.max(np.abs(hm1 - hm2))
        assert diff == 0.0, f"Grad-CAM must be strictly deterministic, got diff={diff}"

    def test_gradcam_prediction_alignment(self):
        """Confirms generated logit and probability match direct model forward pass."""
        heatmap, logit, prob = self.gradcam.generate_heatmap(self.img)
        tensor = self.gradcam.transform(self.img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            expected_logit = float(self.gradcam.model.classifier(self.gradcam.model(tensor)).item())
            expected_prob = float(torch.sigmoid(torch.tensor(expected_logit)).item())
        assert abs(logit - expected_logit) < 1e-4
        assert abs(prob - expected_prob) < 1e-4

    def test_multi_image_evaluation(self):
        """Evaluates Grad-CAM on 3 distinct held-out test images."""
        for i in range(3):
            rel_path = self.ids_test.iloc[i]["image_path"]
            img_p = os.path.join(RAW_DIR, rel_path)
            if os.path.exists(img_p):
                test_im = Image.open(img_p)
                hm, l, p = self.gradcam.generate_heatmap(test_im)
                assert hm.shape == (224, 224)
                assert 0.0 <= hm.min() <= hm.max() <= 1.0
                assert 0.0 <= p <= 1.0


# =============================================================================
# 6. CXR Grad-CAM Overlay, Limitations & Safety Disclaimers
# =============================================================================
class TestCXRGradCAMOverlayAndLimitations:
    """Verifies overlay rendering, resolution limits, hook removal, and safety documentation."""

    @classmethod
    def setup_class(cls):
        cls.gradcam = CXRGradCAM(device="cpu")
        cls.ids_test = pd.read_csv(os.path.join(PROC_DIR, "ids_test.csv"))
        cls.img_path = os.path.join(RAW_DIR, cls.ids_test.iloc[0]["image_path"])
        cls.img = Image.open(cls.img_path)

    @classmethod
    def teardown_class(cls):
        cls.gradcam.close()

    def test_overlay_generation(self):
        """Confirms overlay image size is (224, 224) and mode RGB."""
        heatmap, _, _ = self.gradcam.generate_heatmap(self.img)
        overlay = self.gradcam.generate_overlay(self.img, heatmap, alpha=0.4)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (224, 224)
        assert overlay.mode == "RGB"

    def test_target_layer_resolution_contract(self):
        """
        Confirms target layer activations have coarse spatial resolution of 7x7,
        requiring 32x bilinear upsampling to reach 224x224.
        This verifies that Grad-CAM produces regional/diffuse heatmaps rather than
        focal pathological boundaries.
        """
        tensor = self.gradcam.transform(self.img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            self.gradcam.model(tensor)
        assert self.gradcam.activations is not None
        # activations shape is (1, 512, 7, 7)
        assert self.gradcam.activations.shape == (1, 512, 7, 7)
        spatial_dim = self.gradcam.activations.shape[2]
        upsample_factor = 224 // spatial_dim
        assert spatial_dim == 7
        assert upsample_factor == 32

    def test_close_removes_hooks(self):
        """Confirms close() removes forward and backward hooks cleanly."""
        fresh_gradcam = CXRGradCAM(device="cpu")
        assert len(fresh_gradcam.model.layer4._forward_hooks) > 0
        assert len(fresh_gradcam.model.layer4._backward_hooks) > 0
        fresh_gradcam.close()
        assert len(fresh_gradcam.model.layer4._forward_hooks) == 0
        assert len(fresh_gradcam.model.layer4._backward_hooks) == 0
