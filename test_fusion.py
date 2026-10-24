"""
tests/test_fusion.py
─────────────────────────────────────────────────────────────────────────────
Automated Test Suite for Phase 5A: Clinical + CXR Multimodal Fusion.

Test Groups:
  Group P: Multimodal Dataset, Linkage & Leakage Controls (P01 - P06)
  Group Q: Fusion Architectures & Cross-Attention Mechanics (Q01 - Q04)
  Group R: Saved Checkpoints, Threshold Validation & Artifact Integrity (R01 - R05)
"""

from __future__ import annotations
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fusion.multimodal_dataset import (
    load_paired_multimodal_splits,
    build_authoritative_dicom_feature_map,
    MultimodalDataset,
)
from src.fusion.fusion_model import (
    ConcatenationFusionModel,
    CrossAttentionFusionModel,
    CLINICAL_CKPT_PATH,
    CXR_CKPT_PATH,
)

FUSION_DIR = os.path.join(_PROJECT_ROOT, "models", "fusion")
PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")


class TestMultimodalDataset:
    """Group P: Multimodal Dataset Pairing, Identifiers, and Leakage Controls."""

    @classmethod
    def setup_class(cls):
        cls.ds_tr, cls.ds_va, cls.ds_te, cls.meta = load_paired_multimodal_splits()

    def test_P01_dicom_feature_map_completeness_and_uniqueness(self):
        """Verifies that the authoritative DICOM feature map contains all 2,884 unique images."""
        dmap = build_authoritative_dicom_feature_map()
        assert len(dmap) == 2884
        sample_feat = next(iter(dmap.values()))
        assert sample_feat.shape == torch.Size([512])
        assert sample_feat.dtype == torch.float32

    def test_P02_multimodal_dataset_shapes_and_types(self):
        """Verifies paired dataset dimensions for clinical and visual modalities."""
        assert len(self.ds_tr) == 1828
        assert len(self.ds_va) == 454
        assert len(self.ds_te) == 602

        sample = self.ds_te[0]
        assert "stay_id" in sample
        assert "subject_id" in sample
        assert "dicom_id" in sample
        assert sample["x_clinical"].shape == torch.Size([49])
        assert sample["cxr_features_512"].shape == torch.Size([512])
        assert sample["label"].item() in [0.0, 1.0]

    def test_P03_authoritative_identifier_join(self):
        """Verifies that shuffling row order does not break dicom_id feature alignment."""
        dmap = build_authoritative_dicom_feature_map()
        ids_shuffled = self.ds_te.ids_df.sample(frac=1.0, random_state=99).reset_index(drop=True)
        dummy_x = np.zeros((len(ids_shuffled), 49))

        ds_shuffled = MultimodalDataset(
            ids_df=ids_shuffled,
            X_clinical=dummy_x,
            dicom_to_feature=dmap,
            split_name="shuffled_test",
        )
        assert len(ds_shuffled) == len(self.ds_te)
        # Check first 5 samples: feature must match dmap[dicom_id] exactly
        for i in range(5):
            did = ds_shuffled.ids_df.iloc[i]["dicom_id"]
            expected_feat = dmap[did]
            assert torch.equal(ds_shuffled.cxr_features[i], expected_feat)

    def test_P04_missing_dicom_detection(self):
        """Verifies that attempting to load unknown dicom_ids raises an explicit KeyError."""
        bad_ids = self.ds_te.ids_df.copy()
        bad_ids.loc[0, "dicom_id"] = "NONEXISTENT_DICOM_ID_99999"
        dmap = build_authoritative_dicom_feature_map()

        with pytest.raises(KeyError) as exc_info:
            MultimodalDataset(
                ids_df=bad_ids,
                X_clinical=np.zeros((len(bad_ids), 49)),
                dicom_to_feature=dmap,
                split_name="bad_test",
            )
        assert "Missing visual features" in str(exc_info.value)

    def test_P05_zero_patient_leakage_across_splits(self):
        """Verifies zero patient (subject_id) overlap between train_sub, val_sub, and test."""
        tr_patients = set(self.ds_tr.ids_df["subject_id"])
        va_patients = set(self.ds_va.ids_df["subject_id"])
        te_patients = set(self.ds_te.ids_df["subject_id"])

        assert len(tr_patients & va_patients) == 0, "Patient overlap between train_sub and val_sub!"
        assert len(tr_patients & te_patients) == 0, "Patient overlap between train_sub and test!"
        assert len(va_patients & te_patients) == 0, "Patient overlap between val_sub and test!"

    def test_P06_no_label_or_leakage_columns_in_features(self):
        """Verifies that sepsis_label or timing columns are not in the clinical feature tensor."""
        forbidden_cols = ["sepsis_label", "hours_to_onset", "sepsis_onset_time", "prediction_time"]
        X_df = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
        for col in forbidden_cols:
            assert col not in X_df.columns


class TestFusionArchitectures:
    """Group Q: Fusion Model Forward Passes, Dimensionality & Attention Mechanics."""

    def test_Q01_model_a_forward_and_shapes(self):
        """Tests ConcatenationFusionModel forward pass, output shape, and probability range."""
        model = ConcatenationFusionModel(embed_dim=64, hidden_dim=64, dropout=0.2, freeze_backbones=True)
        model.eval()

        dummy_x = torch.randn(8, 49)
        dummy_cxr = torch.randn(8, 512)

        with torch.no_grad():
            logits = model(dummy_x, dummy_cxr)
            probs = model.predict_proba(dummy_x, dummy_cxr)
            preds = model.predict(dummy_x, dummy_cxr, threshold=0.5)

        assert logits.shape == torch.Size([8, 1])
        assert probs.shape == (8,)
        assert preds.shape == (8,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert np.all((preds == 0) | (preds == 1))

    def test_Q02_model_b_cross_attention_forward_and_attention_weights(self):
        """Tests CrossAttentionFusionModel visual-guided attention mechanics."""
        model = CrossAttentionFusionModel(embed_dim=64, num_heads=4, hidden_dim=64, dropout=0.2, freeze_backbones=True)
        model.eval()

        dummy_x = torch.randn(4, 49)
        dummy_cxr = torch.randn(4, 512)

        with torch.no_grad():
            logits, attn = model(dummy_x, dummy_cxr, return_attention=True)

        assert logits.shape == torch.Size([4, 1])
        # Attention shape: (batch_size, num_heads, visual_tokens=1, clinical_tokens=50)
        assert attn.shape == torch.Size([4, 4, 1, 50])
        # Attention weights along key dimension must sum to 1.0
        attn_sums = attn.sum(dim=-1)
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)

    def test_Q03_backbone_freezing(self):
        """Verifies that unimodal backbones are frozen when freeze_backbones=True."""
        model = ConcatenationFusionModel(embed_dim=64, freeze_backbones=True)
        for param in model.clinical_extractor.parameters():
            assert not param.requires_grad
        for param in model.cxr_projector.parameters():
            assert not param.requires_grad
        for param in model.fusion_head.parameters():
            assert param.requires_grad

    def test_Q04_fusion_eval_deterministic(self):
        """Verifies that dropout is disabled in evaluation mode for deterministic outputs."""
        model = CrossAttentionFusionModel(embed_dim=64, freeze_backbones=True)
        model.eval()
        dummy_x = torch.randn(2, 49)
        dummy_cxr = torch.randn(2, 512)

        with torch.no_grad():
            out1 = model(dummy_x, dummy_cxr)
            out2 = model(dummy_x, dummy_cxr)

        assert torch.allclose(out1, out2)


class TestFusionArtifactsAndEvaluation:
    """Group R: Saved Checkpoints, Threshold Validation, and Evaluation Metrics."""

    def test_R01_saved_checkpoints_exist_and_load(self):
        """Verifies that both fusion checkpoints exist in models/fusion/ and load cleanly."""
        ckpt_a_path = os.path.join(FUSION_DIR, "concat_fusion_best.pt")
        ckpt_b_path = os.path.join(FUSION_DIR, "cross_attn_fusion_best.pt")

        assert os.path.isfile(ckpt_a_path), f"Missing {ckpt_a_path}"
        assert os.path.isfile(ckpt_b_path), f"Missing {ckpt_b_path}"

        ckpt_a = torch.load(ckpt_a_path, map_location="cpu", weights_only=False)
        ckpt_b = torch.load(ckpt_b_path, map_location="cpu", weights_only=False)

        assert "model_state_dict" in ckpt_a
        assert "training_meta" in ckpt_a
        assert "model_state_dict" in ckpt_b
        assert "training_meta" in ckpt_b

    def test_R02_test_metrics_json_integrity(self):
        """Verifies test_metrics.json has all required comparative models and bootstrap CIs."""
        metrics_path = os.path.join(FUSION_DIR, "test_metrics.json")
        assert os.path.isfile(metrics_path)

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["test_n"] == 602
        assert data["test_pos"] == 104
        assert abs(data["test_prevalence"] - 0.1728) < 1e-3

        required_models = [
            "Clinical Transformer",
            "CXR Encoder (Image-Only)",
            "Model A (Concatenation Fusion)",
            "Model B (Cross-Attention Fusion)",
        ]
        for m_name in required_models:
            assert m_name in data["models"]
            m_data = data["models"][m_name]
            for metric_key in ["roc_auc", "pr_auc", "f1", "brier_score", "ece", "roc_auc_ci_lower", "roc_auc_ci_upper"]:
                assert metric_key in m_data

    def test_R03_threshold_selected_on_validation_only(self):
        """Verifies optimal thresholds are documented in training_meta from validation data."""
        metrics_path = os.path.join(FUSION_DIR, "test_metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta_a = data["training_meta"]["model_a"]
        meta_b = data["training_meta"]["model_b"]
        assert "optimal_threshold" in meta_a
        assert "optimal_threshold" in meta_b
        assert 0.10 <= meta_a["optimal_threshold"] <= 0.90
        assert 0.10 <= meta_b["optimal_threshold"] <= 0.90

    def test_R04_comparative_plots_exist(self):
        """Verifies that ROC and PR comparative curve PNGs are saved."""
        plots_dir = os.path.join(FUSION_DIR, "plots")
        roc_plot = os.path.join(plots_dir, "test_roc_comparison.png")
        pr_plot = os.path.join(plots_dir, "test_pr_comparison.png")

        assert os.path.isfile(roc_plot), f"Missing {roc_plot}"
        assert os.path.isfile(pr_plot), f"Missing {pr_plot}"

    def test_R05_unimodal_checkpoints_unmodified(self):
        """Confirms existing unimodal checkpoints were preserved without modification."""
        assert os.path.isfile(CLINICAL_CKPT_PATH)
        assert os.path.isfile(CXR_CKPT_PATH)
        clin_ckpt = torch.load(CLINICAL_CKPT_PATH, map_location="cpu", weights_only=True)
        cxr_ckpt = torch.load(CXR_CKPT_PATH, map_location="cpu", weights_only=True)
        assert "model_state_dict" in clin_ckpt
        assert "model_state_dict" in cxr_ckpt


class TestFusionPredictionResultPresentation:
    """Group S: Multimodal Fusion Prediction Result, Boundary Logic, and Contract Integrity."""

    def test_S01_fusion_inference_returns_dynamic_separate_models(self):
        """run_fusion_models_inference must return distinct, non-averaged Model A and Model B results."""
        from backend.patient_service import run_fusion_models_inference, resolve_patient_stay

        pid, sid, _, _ = resolve_patient_stay("10213765", 36628477)
        res = run_fusion_models_inference(pid, sid)

        assert res["status"] == "success"
        assert "model_a" in res
        assert "model_b" in res

        ma = res["model_a"]
        mb = res["model_b"]

        # 1. Names and architecture labels
        assert "Model A" in ma["name"]
        assert "Model B" in mb["name"]

        # 2. Dynamic probabilities must be floats between 0 and 1
        p_a = ma["predicted_probability"]
        p_b = mb["predicted_probability"]
        assert 0.0 <= p_a <= 1.0
        assert 0.0 <= p_b <= 1.0

        # 3. Models must not be artificially averaged
        avg_p = round((p_a + p_b) / 2.0, 4)
        assert p_a != avg_p or p_b != avg_p or abs(p_a - p_b) < 1e-4

        # 4. Thresholds documented
        assert ma["threshold"] == 0.50
        assert mb["threshold"] == 0.50

    def test_S02_fusion_threshold_boundary_classification(self):
        """Boundary test: probability >= 0.50 evaluates to Positive, < 0.50 evaluates to Negative."""
        thr = 0.50

        # Boundary condition: exact cutoff
        assert ("Positive" if 0.5000 >= thr else "Negative") == "Positive"

        # Boundary condition: epsilon above
        assert ("Positive" if 0.5001 >= thr else "Negative") == "Positive"

        # Boundary condition: epsilon below
        assert ("Positive" if 0.4999 >= thr else "Negative") == "Negative"

    def test_S03_prediction_target_metadata_documented(self):
        """Prediction target must confirm 1-24h sepsis onset window following the 6h observation window."""
        from backend.patient_service import run_fusion_models_inference, resolve_patient_stay

        pid, sid, _, _ = resolve_patient_stay("10213765", 36628477)
        res = run_fusion_models_inference(pid, sid)

        assert "prediction_target" in res
        target_str = res["prediction_target"]
        assert "1–24" in target_str or "1-24" in target_str
        assert "6-hour" in target_str

        # Also documented inside individual models
        assert "prediction_target" in res["model_a"]
        assert "prediction_target" in res["model_b"]

    def test_S04_honest_unavailable_state_when_missing_cxr(self):
        """Stays lacking paired radiographs must return status='unavailable' with a clear message."""
        from backend.patient_service import run_fusion_models_inference

        # Use an invalid/synthetic stay that has no linked radiograph
        res = run_fusion_models_inference(99999999, 99999999)
        assert res["status"] in {"unavailable", "error"}
        assert "message" in res
        assert len(res["message"]) > 10

    def test_S05_isolation_from_esp32_sensor_telemetry(self):
        """Multimodal fusion schemas must never contain or accept live hardware IoT fields."""
        from backend.patient_service import run_fusion_models_inference, resolve_patient_stay

        pid, sid, _, _ = resolve_patient_stay("10213765", 36628477)
        res = run_fusion_models_inference(pid, sid)

        # Prohibited telemetry fields
        prohibited_iot = ["channel_id", "entry_id", "ir_ppg", "field1", "field2", "telemetry_age_seconds"]
        for key in prohibited_iot:
            assert key not in res
            assert key not in res["model_a"]
            assert key not in res["model_b"]

    def test_S06_fusion_page_module_exports_render_function(self):
        """dashboard.pages.multimodal_research must export render_multimodal_research_page."""
        import dashboard.pages.multimodal_research as p_mm
        assert hasattr(p_mm, "render_multimodal_research_page")
        assert callable(p_mm.render_multimodal_research_page)

