"""
tests/test_transformer.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Phase 3A: Clinical Transformer architecture, training, and artifacts.

Test Groups:
  Group H: Transformer Architecture & Tensor Operations (H01 - H09)
  Group I: Transformer Artifacts & Pipeline Integration (I01 - I08)
"""

import os
import json
import pickle
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.clinical.clinical_transformer import (
    TabularFeatureTokenizer,
    TransformerEncoderBlock,
    ClinicalTransformer,
)
from src.clinical.train_transformer import (
    load_data,
    create_patient_validation_split,
)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
TRANSFORMER_DIR = os.path.join(_PROJECT_ROOT, "models", "transformer")
META_DIR = os.path.join(_PROJECT_ROOT, "data", "metadata")


# ═══════════════════════════════════════════════════════════════════════════════
# Group H: Transformer Architecture & Tensor Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformerArchitecture:
    """Group H: Core module tensor shapes, finite gradients, and outputs."""

    def test_H01_tokenizer_shape(self):
        """FeatureTokenizer projects (B, num_features) to (B, num_features, embed_dim)."""
        tokenizer = TabularFeatureTokenizer(num_features=49, embed_dim=64)
        x = torch.randn(4, 49)
        tokens = tokenizer(x)
        assert tokens.shape == (4, 49, 64), f"Expected (4, 49, 64), got {tokens.shape}"

    def test_H02_encoder_block_shape(self):
        """TransformerEncoderBlock preserves sequence shape (B, seq_len, embed_dim)."""
        block = TransformerEncoderBlock(embed_dim=64, num_heads=4, dim_feedforward=128)
        x = torch.randn(4, 50, 64)
        out, attn = block(x, return_attention=False)
        assert out.shape == (4, 50, 64)
        assert attn is None

    def test_H03_encoder_block_attention_weights(self):
        """When return_attention=True, weights shape is (B, heads, seq, seq) and normalized."""
        block = TransformerEncoderBlock(embed_dim=64, num_heads=4, dim_feedforward=128)
        block.eval()
        x = torch.randn(2, 50, 64)
        out, attn = block(x, return_attention=True)
        assert attn is not None
        assert attn.shape == (2, 4, 50, 50)
        # Check softmax sum along key dimension ~ 1.0
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_H04_forward_logits_shape(self):
        """ClinicalTransformer forward pass yields (B, 1) logits."""
        model = ClinicalTransformer(num_features=49, embed_dim=64, num_heads=4, num_layers=2)
        x = torch.randn(8, 49)
        logits = model(x)
        assert logits.shape == (8, 1)

    def test_H05_logits_finite(self):
        """Forward pass outputs are strictly finite (no NaN or Inf)."""
        model = ClinicalTransformer(num_features=49, embed_dim=64, num_heads=4, num_layers=2)
        x = torch.randn(16, 49)
        logits = model(x)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"

    def test_H06_loss_backward(self):
        """Loss computation and backward pass update all trainable parameters with finite grads."""
        model = ClinicalTransformer(num_features=49, embed_dim=32, num_heads=2, num_layers=1)
        criterion = nn.BCEWithLogitsLoss()
        x = torch.randn(4, 49)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"

    def test_H07_predict_proba_range(self):
        """predict_proba yields probabilities strictly within [0.0, 1.0]."""
        model = ClinicalTransformer(num_features=49, embed_dim=32, num_heads=2, num_layers=1)
        x = np.random.randn(10, 49).astype(np.float32)
        probs = model.predict_proba(x)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (10,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_H08_predict_binary(self):
        """predict returns binary decisions {0, 1}."""
        model = ClinicalTransformer(num_features=49, embed_dim=32, num_heads=2, num_layers=1)
        x = np.random.randn(10, 49).astype(np.float32)
        preds = model.predict(x, threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_H09_dropout_eval_deterministic(self):
        """In eval mode, duplicate forward calls on the same input yield identical outputs."""
        model = ClinicalTransformer(num_features=49, embed_dim=64, num_heads=4, num_layers=2)
        model.eval()
        x = torch.randn(4, 49)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2), "Evaluation mode must be strictly deterministic"


# ═══════════════════════════════════════════════════════════════════════════════
# Group I: Transformer Artifacts & Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformerArtifacts:
    """Group I: Saved model artifacts, metadata, and patient isolation."""

    def test_I01_checkpoint_file_exists(self):
        """Saved PyTorch checkpoint file exists and has non-zero size."""
        ckpt_path = os.path.join(TRANSFORMER_DIR, "clinical_transformer.pt")
        assert os.path.isfile(ckpt_path), f"Checkpoint missing at {ckpt_path}"
        assert os.path.getsize(ckpt_path) > 10_000

    def test_I02_checkpoint_loads_and_infers(self):
        """Saved checkpoint can be loaded and successfully run inference."""
        ckpt_path = os.path.join(TRANSFORMER_DIR, "clinical_transformer.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in ckpt
        assert "val_pr_auc" in ckpt

        model = ClinicalTransformer(
            num_features=ckpt["num_features"],
            embed_dim=ckpt["embed_dim"],
            num_heads=ckpt["num_heads"],
            num_layers=ckpt["num_layers"],
            dim_feedforward=ckpt["dim_feedforward"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy = torch.randn(2, 49)
        with torch.no_grad():
            logits = model(dummy)
        assert logits.shape == (2, 1)

    def test_I03_scaler_artifact_exists(self):
        """StandardScaler exists and transforms 49 features."""
        scaler_path = os.path.join(TRANSFORMER_DIR, "scaler.pkl")
        assert os.path.isfile(scaler_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        assert scaler.n_features_in_ == 49

    def test_I04_config_json_keys(self):
        """training_config.json contains required metadata, architecture, and split keys."""
        config_path = os.path.join(TRANSFORMER_DIR, "training_config.json")
        assert os.path.isfile(config_path)
        with open(config_path, "r") as f:
            cfg = json.load(f)
        for key in ["num_features", "architecture", "training", "splits", "chosen_threshold"]:
            assert key in cfg, f"Key {key} missing from training_config.json"
        assert cfg["splits"]["patient_overlap_train_val"] == 0

    def test_I05_test_metrics_json_keys(self):
        """test_metrics.json has all required metric keys and above-chance performance."""
        metrics_path = os.path.join(TRANSFORMER_DIR, "test_metrics.json")
        assert os.path.isfile(metrics_path)
        with open(metrics_path, "r") as f:
            m = json.load(f)
        for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score", "ece"]:
            assert k in m, f"Metric {k} missing from test_metrics.json"
        assert m["roc_auc"] > 0.50, f"ROC-AUC {m['roc_auc']} not above chance"
        assert m["pr_auc"] > m["positive_rate"], f"PR-AUC {m['pr_auc']} not above prevalence baseline"

    def test_I06_history_json_epochs(self):
        """training_history.json exists and tracks training and validation loss."""
        hist_path = os.path.join(TRANSFORMER_DIR, "training_history.json")
        assert os.path.isfile(hist_path)
        with open(hist_path, "r") as f:
            hist = json.load(f)
        assert len(hist) > 0
        assert "train_loss" in hist[0] and "val_pr_auc" in hist[0]

    def test_I07_phase3a_report_exists(self):
        """PHASE3A_TRANSFORMER_REPORT.md exists and is non-empty."""
        rpt_path = os.path.join(META_DIR, "PHASE3A_TRANSFORMER_REPORT.md")
        assert os.path.isfile(rpt_path)
        assert os.path.getsize(rpt_path) > 500

    def test_I08_zero_patient_leakage_in_splits(self):
        """Re-verify that patient-level split function yields strictly 0 overlapping subject_ids."""
        X_tr, y_tr, ids_tr, _, _, _ = load_data()
        tr_idx, va_idx = create_patient_validation_split(ids_tr, y_tr, n_splits=5, random_state=42)
        tr_patients = set(ids_tr.iloc[tr_idx]["subject_id"])
        va_patients = set(ids_tr.iloc[va_idx]["subject_id"])
        assert len(tr_patients & va_patients) == 0
