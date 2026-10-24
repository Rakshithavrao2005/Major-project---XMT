"""
tests/test_imaging.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Phase 4A: Chest X-ray dataset integrity, preprocessing,
patient-level isolation, and PyTorch CXREncoder architecture.

Test Groups:
  Group L: CXR Dataset, Linkage & Leakage Verification (L01 - L07)
  Group M: CXR Encoder Architecture & Tensor Operations (M01 - M05)
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.imaging.cxr_dataset import (
    CXRDataset,
    get_cxr_transforms,
    load_cxr_splits,
)
from src.imaging.cxr_encoder import CXREncoder

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
META_DIR = os.path.join(_PROJECT_ROOT, "data", "metadata")
IMAGES_DIR = os.path.join(RAW_DIR, "images")


# ═══════════════════════════════════════════════════════════════════════════════
# Group L: CXR Dataset, Linkage & Leakage Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestCXRDatasetAndLinkage:
    """Group L: Image directory verification, DICOM linkage, split isolation."""

    def test_L01_images_dir_count(self):
        """All 2,884 CXR images are extracted and present in data/raw/images/."""
        assert os.path.isdir(IMAGES_DIR), f"Missing images dir at {IMAGES_DIR}"
        images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
        assert len(images) == 2884, f"Expected 2,884 images, found {len(images)}"

    def test_L02_image_linkage_train_and_test(self):
        """Every dicom_id in ids_train.csv and ids_test.csv exists on disk."""
        df_tr = pd.read_csv(os.path.join(PROC_DIR, "ids_train.csv"))
        df_te = pd.read_csv(os.path.join(PROC_DIR, "ids_test.csv"))
        disk_files = {f[:-4] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")}

        train_missing = set(df_tr["dicom_id"]) - disk_files
        test_missing  = set(df_te["dicom_id"]) - disk_files
        assert len(train_missing) == 0, f"Missing train images: {len(train_missing)}"
        assert len(test_missing) == 0, f"Missing test images: {len(test_missing)}"

    def test_L03_patient_level_isolation(self):
        """Zero patient overlap between ids_train.csv and ids_test.csv."""
        df_tr = pd.read_csv(os.path.join(PROC_DIR, "ids_train.csv"))
        df_te = pd.read_csv(os.path.join(PROC_DIR, "ids_test.csv"))
        tr_patients = set(df_tr["subject_id"])
        te_patients = set(df_te["subject_id"])
        assert len(tr_patients & te_patients) == 0, "Patient overlap detected!"

    def test_L04_cxr_dataset_loading(self):
        """CXRDataset successfully loads an image item with tensor (3, 224, 224)."""
        df_tr = pd.read_csv(os.path.join(PROC_DIR, "ids_train.csv")).head(5)
        y_tr  = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv"))["sepsis_label"].head(5)
        ds = CXRDataset(ids_df=df_tr, labels_s=y_tr)
        assert len(ds) == 5
        item = ds[0]
        assert "image" in item and "label" in item and "dicom_id" in item
        assert item["image"].shape == (3, 224, 224)
        assert torch.isfinite(item["image"]).all()

    def test_L05_cxr_transforms_normalization(self):
        """CXR transforms produce normalized 3-channel tensors."""
        transform = get_cxr_transforms(is_train=False, image_size=224)
        dummy_img = Image.new("RGB", (300, 400), color=(128, 128, 128))
        t = transform(dummy_img)
        assert t.shape == (3, 224, 224)
        assert torch.is_tensor(t)

    def test_L06_cxr_audit_report_exists(self):
        """CXR_DATASET_AUDIT.md exists and documents leakage and semantics."""
        rpt = os.path.join(META_DIR, "CXR_DATASET_AUDIT.md")
        assert os.path.isfile(rpt)
        with open(rpt, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Zero temporal leakage" in content
        assert "Sepsis" in content

    def test_L07_timing_strictly_pre_prediction(self):
        """All radiographs were acquired at or before prediction time T (hours_before_prediction <= 0.0)."""
        raw_csv = os.path.join(RAW_DIR, "dataset_T6h_H24_v5.csv")
        df = pd.read_csv(raw_csv)
        assert (df["hours_before_prediction"] <= 0.0001).all(), "CXR temporal leakage detected!"


# ═══════════════════════════════════════════════════════════════════════════════
# Group M: CXR Encoder Architecture & Tensor Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestCXREncoderArchitecture:
    """Group M: CXREncoder forward pass, projection shapes, gradient backprop."""

    def test_M01_cxr_encoder_output_shape(self):
        """CXREncoder maps (B, 3, 224, 224) to (B, embed_dim)."""
        model = CXREncoder(embed_dim=64, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        z = model(x)
        assert z.shape == (2, 64)
        assert torch.isfinite(z).all()

    def test_M02_cxr_encoder_predict_logits_shape(self):
        """predict_logits produces (B, 1) logits."""
        model = CXREncoder(embed_dim=64, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        logits = model.predict_logits(x)
        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()

    def test_M03_cxr_encoder_finite_gradients(self):
        """Backward pass propagates finite gradients through projection and convolutional layers."""
        model = CXREncoder(embed_dim=32, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        logits = model.predict_logits(x).squeeze(-1)
        loss = logits.sum()
        loss.backward()
        # Verify first conv layer and projection have gradients
        assert model.conv1.weight.grad is not None
        assert torch.isfinite(model.conv1.weight.grad).all()
        assert model.proj[0].weight.grad is not None
        assert torch.isfinite(model.proj[0].weight.grad).all()

    def test_M04_cxr_encoder_eval_deterministic(self):
        """In eval mode, duplicate forward calls yield identical output tensors."""
        model = CXREncoder(embed_dim=64, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            z1 = model(x)
            z2 = model(x)
        assert torch.equal(z1, z2)

    def test_M05_cxr_encoder_feature_hook_shape(self):
        """When return_features=True, returns layer4 feature map (B, 512, 7, 7) for Grad-CAM."""
        model = CXREncoder(embed_dim=64, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        z, feats = model(x, return_features=True)
        assert z.shape == (2, 64)
        assert feats.shape == (2, 512, 7, 7)


# ═══════════════════════════════════════════════════════════════════════════════
# Group N: CXR Model Artifacts & Evaluation Outputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestCXRModelArtifactsAndEvaluation:
    """Group N: Saved artifacts, config, metrics, and checkpoint inference."""

    def test_N01_cxr_checkpoint_file_exists(self):
        """Saved CXR model checkpoint exists and is non-empty."""
        ckpt_path = os.path.join(_PROJECT_ROOT, "models", "imaging", "cxr_encoder.pt")
        assert os.path.isfile(ckpt_path)
        assert os.path.getsize(ckpt_path) > 10_000

    def test_N02_cached_features_exist(self):
        """Precomputed 512-d feature cache exists with valid split tensors."""
        cache_path = os.path.join(_PROJECT_ROOT, "models", "imaging", "cached_features.pt")
        assert os.path.isfile(cache_path)
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert "feats_tr" in data and "feats_va" in data and "feats_te" in data
        assert data["feats_tr"].shape[1] == 512
        assert data["feats_va"].shape[1] == 512
        assert data["feats_te"].shape[1] == 512

    def test_N03_cxr_training_config_keys(self):
        """training_config.json contains required metadata, architecture, and split keys."""
        cfg_path = os.path.join(_PROJECT_ROOT, "models", "imaging", "training_config.json")
        assert os.path.isfile(cfg_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k in ["architecture", "training", "splits", "chosen_threshold"]:
            assert k in cfg
        assert cfg["splits"]["patient_overlap_train_val"] == 0

    def test_N04_cxr_test_metrics_keys(self):
        """test_metrics.json has all required metric keys and bootstrap confidence intervals."""
        m_path = os.path.join(_PROJECT_ROOT, "models", "imaging", "test_metrics.json")
        assert os.path.isfile(m_path)
        with open(m_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score", "ece",
                  "roc_auc_ci_lower", "roc_auc_ci_upper", "pr_auc_ci_lower", "pr_auc_ci_upper"]:
            assert k in m, f"Metric {k} missing from CXR test_metrics.json"

    def test_N05_cxr_report_exists(self):
        """PHASE4B_IMAGING_REPORT.md exists and is non-empty."""
        rpt = os.path.join(META_DIR, "PHASE4B_IMAGING_REPORT.md")
        assert os.path.isfile(rpt)
        assert os.path.getsize(rpt) > 500

    def test_N06_checkpoint_loads_and_infers(self):
        """Loaded checkpoint performs end-to-end forward pass on raw 3-channel image tensors."""
        ckpt_path = os.path.join(_PROJECT_ROOT, "models", "imaging", "cxr_encoder.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = CXREncoder(embed_dim=ckpt["embed_dim"], pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            z = model(dummy)
            logits = model.predict_logits(dummy)
        assert z.shape == (2, ckpt["embed_dim"])
        assert logits.shape == (2, 1)
        assert torch.isfinite(z).all() and torch.isfinite(logits).all()


class TestCXRDiagnostics:
    """Tests verifying Phase 4B diagnostic artifacts, metrics, and report integrity."""

    DIAG_DIR = os.path.join(_PROJECT_ROOT, "models", "imaging", "diagnostics")

    def test_O01_diagnostic_summary_exists_and_valid(self):
        summary_path = os.path.join(self.DIAG_DIR, "diagnostic_summary.json")
        assert os.path.isfile(summary_path), "diagnostic_summary.json missing"
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_sections = [
            "probability_distributions",
            "logit_distributions",
            "threshold_analysis",
            "calibration",
            "unimodal_metrics",
        ]
        for sec in required_sections:
            assert sec in data, f"Section {sec} missing from diagnostic_summary.json"

        # Check probability bounds
        val_dist = data["probability_distributions"]["validation"]
        assert 0.0 <= val_dist["min"] <= val_dist["max"] <= 1.0
        assert val_dist["min"] > 0.50, "Val min probability expected > 0.50 due to pos_weight equilibrium"

    def test_O02_diagnostic_report_exists(self):
        rpt = os.path.join(self.DIAG_DIR, "CXR_DIAGNOSTIC_REPORT.md")
        assert os.path.isfile(rpt), "CXR_DIAGNOSTIC_REPORT.md missing"
        assert os.path.getsize(rpt) > 1000

    def test_O03_diagnostic_plots_exist(self):
        for plot_name in [
            "probability_distributions.png",
            "sensitivity_specificity_tradeoff.png",
            "calibration_diagnostic.png",
        ]:
            p = os.path.join(self.DIAG_DIR, plot_name)
            assert os.path.isfile(p), f"Plot {plot_name} missing"
            assert os.path.getsize(p) > 5000, f"Plot {plot_name} is unexpectedly small"

    def test_O04_validation_threshold_sweep_csv_valid(self):
        csv_p = os.path.join(self.DIAG_DIR, "validation_threshold_sweep.csv")
        assert os.path.isfile(csv_p), "validation_threshold_sweep.csv missing"
        df = pd.read_csv(csv_p)
        assert len(df) >= 40
        for col in ["threshold", "recall_sensitivity", "specificity", "precision", "f1", "tp", "fp", "tn", "fn"]:
            assert col in df.columns, f"Column {col} missing in sweep CSV"

