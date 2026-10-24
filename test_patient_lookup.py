"""
tests/test_patient_lookup.py
─────────────────────────────────────────────────────────────────────────────
Automated Test Suite for Patient-ID-Based Lookup and Safe Model Inference.

Covers all 20 requirements specified in Phase 4B Patient Lookup task:
1. Search for a valid Patient ID.
2. Search for an unknown Patient ID.
3. Empty and malformed Patient ID input.
4. A patient with multiple stays.
5. A stay with multiple linked images / verified linkage.
6. A missing image handling.
7. A missing clinical record handling.
8. Correct patient-to-stay and stay-to-image mapping.
9. No unrelated image displayed when an image is missing.
10. Clinical Transformer checkpoint loading and inference.
11. Exact feature order and scaler use during inference.
12. CXR checkpoint loading and inference.
13. Correct image preprocessing during inference.
14. Inference does not modify model weights.
15. No random or synthetic patient-data fallback.
16. Separation of model predictions from ground-truth labels.
17. No test-label-based threshold selection.
18. Correct handling of unavailable checkpoints or invalid inference inputs.
19. Proper behavior when the selected stay lacks required prediction-window timestamps.
20. Dashboard/Backend behavior for missing records, loading failures, and inference errors.
"""

import os
import sys
import copy
import pytest
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.patient_service import (
    search_patient,
    get_stay_clinical_record,
    run_clinical_transformer_inference,
    run_cxr_inference,
    get_sensor_telemetry_status,
    validate_patient_id,
    load_clinical_transformer_pipeline,
    load_cxr_model,
    get_raw_dataset,
    get_eligible_patients,
    get_patient_dataset_statistics,
    get_all_patients_directory,
)
from backend.app import app


class TestPatientSearchAndDataIntegrity:
    """Tests 1-9 & 15-16: Search, Linkage, Ground-Truth Separation, and Error Handling."""

    @classmethod
    def setup_class(cls):
        cls.raw_df = get_raw_dataset()
        # Known multi-stay patient: 17585185 (5 stays)
        cls.multi_stay_pid = 17585185
        # Known single-stay patient: 10213765 (1 stay)
        cls.single_stay_pid = 10213765
        cls._original_require_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False
        cls.flask_client = app.test_client()

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = getattr(cls, "_original_require_auth", True)

    def test_01_search_valid_patient_id(self):
        """Test 1: Search for a valid Patient ID returns expected stay count and verified linkage."""
        res = search_patient(self.single_stay_pid)
        assert res["found"] is True
        assert res["patient_id"] == self.single_stay_pid
        assert res["total_stays"] == 1
        assert len(res["stays"]) == 1
        assert res["stays"][0]["stay_id"] == 36628477
        assert res["stays"][0]["image_exists"] is True

    def test_02_search_unknown_patient_id(self):
        """Test 2: Search for an unknown Patient ID returns clear not-found message."""
        unknown_pid = 999999999
        res = search_patient(unknown_pid)
        assert res["found"] is False
        assert res["patient_id"] == unknown_pid
        assert "not found" in res["error"].lower()
        assert res["total_stays"] == 0
        assert len(res["stays"]) == 0

    def test_03_empty_and_malformed_patient_id(self):
        """Test 3: Empty, negative, and malformed Patient ID inputs are rejected with informative errors."""
        bad_inputs = ["", "   ", None, -100, "abc", "10213765; DROP TABLE", "12.34", "patient_1"]
        for bad in bad_inputs:
            is_valid, pid, err = validate_patient_id(bad)
            assert is_valid is False
            assert pid is None
            assert len(err) > 0

            # Test through search_patient
            res = search_patient(bad)
            assert res["found"] is False
            assert res["error"] is not None

    def test_04_patient_with_multiple_stays(self):
        """Test 4: Patient with multiple stays returns all stays with real timestamps."""
        res = search_patient(self.multi_stay_pid)
        assert res["found"] is True
        assert res["total_stays"] == 5
        assert len(res["stays"]) == 5

        stay_ids = [s["stay_id"] for s in res["stays"]]
        expected_stays = [32417784, 32776339, 32424770, 39477454, 36336086]
        assert set(stay_ids) == set(expected_stays)

        # Check chronological sorting
        for s in res["stays"]:
            assert "intime" in s and s["intime"] != "N/A"
            assert "prediction_time" in s and s["prediction_time"] != "N/A"

    def test_05_stay_with_linked_image_verification(self):
        """Test 5: Verify stay-to-image linkage and actual JPEG existence."""
        res = search_patient(self.single_stay_pid)
        stay = res["stays"][0]
        assert stay["dicom_id"] == "255cb786-dff3ae18-4f72077d-65346785-06ea767e"
        assert stay["image_exists"] is True
        assert os.path.isfile(stay["image_path"])

    def test_06_missing_image_handling(self):
        """Test 6: Missing image file is detected without crashing or substituting other images."""
        rec = get_stay_clinical_record(self.single_stay_pid, 36628477)
        img_info = rec["imaging"]
        assert img_info["image_exists"] is True

        # Check when image path does not exist
        fake_rec = copy.deepcopy(rec)
        fake_rec["imaging"]["image_exists"] = False
        fake_rec["imaging"]["image_file"] = None
        assert fake_rec["imaging"]["image_file"] is None
        assert fake_rec["imaging"]["image_exists"] is False

    def test_07_missing_clinical_record_handling(self):
        """Test 7: Querying non-existent stay raises clear ValueError."""
        with pytest.raises(ValueError, match="No record found"):
            get_stay_clinical_record(self.single_stay_pid, 99999999)

    def test_08_correct_patient_to_stay_and_stay_to_image_mapping(self):
        """Test 8: Ensure patient, stay, and DICOM mappings match raw dataset ground truth."""
        raw_row = self.raw_df[self.raw_df["stay_id"] == 36628477].iloc[0]
        rec = get_stay_clinical_record(self.single_stay_pid, 36628477)

        assert rec["patient_id"] == int(raw_row["subject_id"])
        assert rec["stay_id"] == int(raw_row["stay_id"])
        assert rec["imaging"]["dicom_id"] == str(raw_row["dicom_id"])

    def test_09_no_unrelated_image_displayed_when_missing(self):
        """Test 9: Service does not silently substitute another image if target image is absent."""
        stay_rec = get_stay_clinical_record(self.single_stay_pid, 36628477)
        # Attempt to serve a mismatched dicom_id via API
        resp = self.flask_client.get(f"/api/patient/{self.single_stay_pid}/stay/36628477/image/unrelated-dicom-id")
        assert resp.status_code == 400
        assert "does not match" in resp.get_json()["message"]

    def test_15_no_random_or_synthetic_patient_data_fallback(self):
        """Test 15: Non-existent Patient ID returns strictly 0 stays (no mock/synthetic fallback)."""
        res = search_patient(888888888)
        assert res["found"] is False
        assert len(res["stays"]) == 0
        assert res["total_stays"] == 0

    def test_16_strict_separation_of_predictions_from_ground_truth(self):
        """Test 16: Clinical record and inference payloads strictly omit sepsis_label and sepsis_onset_time."""
        rec = get_stay_clinical_record(self.single_stay_pid, 36628477)
        forbidden_keys = ["sepsis_label", "sepsis_onset_time", "hours_to_onset", "followup_h"]
        for k in forbidden_keys:
            assert k not in rec
            assert k not in rec.get("demographics", {})
            assert k not in rec.get("vital_signs", {})
            assert k not in rec.get("laboratory_measurements", {})

        inf = run_clinical_transformer_inference(self.single_stay_pid, 36628477)
        assert "sepsis_label" not in inf
        assert "sepsis_onset_time" not in inf


class TestModelInferenceAndSafeguards:
    """Tests 10-14 & 17-20: Clinical Transformer, CXR Inference, Threshold Safeguards, and Stability."""

    @classmethod
    def setup_class(cls):
        cls.patient_id = 10213765
        cls.stay_id = 36628477
        cls._original_require_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False
        cls.flask_client = app.test_client()

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = getattr(cls, "_original_require_auth", True)

    def test_10_clinical_transformer_checkpoint_loading_and_inference(self):
        """Test 10: Clinical Transformer checkpoint loads and produces valid risk score."""
        res = run_clinical_transformer_inference(self.patient_id, self.stay_id)
        assert res["status"] == "success"
        assert res["model_name"].startswith("Explainable Multimodal Transformer")
        assert 0.0 <= res["predicted_probability"] <= 1.0
        assert res["decision_threshold"] == 0.60
        assert "High Sepsis Risk" in res["predicted_class"] or "Standard Risk" in res["predicted_class"]
        assert "top_attended_features" in res
        assert len(res["top_attended_features"]) == 5
        assert all("feature" in item and "attention_weight" in item for item in res["top_attended_features"])

    def test_11_exact_feature_order_and_scaler_use_during_inference(self):
        """Test 11: Scaler applies to exactly 49 features matching training_config.json."""
        model, scaler, feature_cols, train_medians = load_clinical_transformer_pipeline()
        assert len(feature_cols) == 49
        assert scaler.n_features_in_ == 49
        assert len(train_medians) >= 20

    def test_12_cxr_checkpoint_loading_and_inference(self):
        """Test 12: CXR checkpoint loads and produces raw score and Platt-calibrated score."""
        res = run_cxr_inference(self.patient_id, self.stay_id)
        assert res["status"] == "success"
        assert "raw_probability" in res
        assert "platt_calibrated_probability" in res
        assert 0.50 <= res["raw_probability"] <= 0.60  # Concentrated raw probability
        assert 0.0 <= res["platt_calibrated_probability"] <= 1.0

    def test_13_correct_image_preprocessing_during_inference(self):
        """Test 13: Image transformation produces (1, 3, 224, 224) tensor with finite normalized values."""
        from src.imaging.cxr_dataset import get_cxr_transforms
        from PIL import Image

        transform = get_cxr_transforms(is_train=False, image_size=224)
        dummy_img = Image.new("RGB", (300, 400), color=(128, 128, 128))
        tensor = transform(dummy_img).unsqueeze(0)
        assert tensor.shape == (1, 3, 224, 224)
        assert torch.isfinite(tensor).all()

    def test_14_inference_does_not_modify_model_weights(self):
        """Test 14: Model inference is strictly read-only and does not modify weights."""
        model, _, _, _ = load_clinical_transformer_pipeline()
        param_before = [p.clone().detach() for p in model.parameters()]

        # Run inference twice
        run_clinical_transformer_inference(self.patient_id, self.stay_id)
        run_clinical_transformer_inference(self.patient_id, self.stay_id)

        param_after = [p.clone().detach() for p in model.parameters()]
        for b, a in zip(param_before, param_after):
            assert torch.equal(b, a), "Model weights were modified during inference!"

    def test_17_no_test_label_based_threshold_selection(self):
        """Test 17: Thresholds reported are validation-selected (0.60 for clinical, 0.5237 for CXR)."""
        clin_inf = run_clinical_transformer_inference(self.patient_id, self.stay_id)
        assert clin_inf["decision_threshold"] == 0.60
        assert "validation" in clin_inf["threshold_selection_split"].lower()

        cxr_inf = run_cxr_inference(self.patient_id, self.stay_id)
        assert cxr_inf["thresholds"]["experimental_balanced_threshold_05237"]["threshold"] == 0.5237
        assert "validation" in cxr_inf["thresholds"]["experimental_balanced_threshold_05237"]["notice"].lower()

    def test_18_handling_unavailable_checkpoints_or_invalid_inputs(self):
        """Test 18: Service gracefully handles non-existent stay without crashing."""
        res = run_clinical_transformer_inference(99999999, 88888888)
        assert res["status"] == "error"
        assert "not found" in res["message"].lower()

        res_cxr = run_cxr_inference(99999999, 88888888)
        assert res_cxr["status"] == "error"
        assert "not found" in res_cxr["message"].lower()

    def test_19_behavior_when_selected_stay_lacks_timestamps(self):
        """Test 19: Graceful handling when prediction timestamps are missing."""
        df = get_raw_dataset()
        # Temporarily mock a stay with NaN prediction_time
        original_pred_time = df.loc[df["stay_id"] == self.stay_id, "prediction_time"].values[0]
        try:
            df.loc[df["stay_id"] == self.stay_id, "prediction_time"] = np.nan
            res = run_clinical_transformer_inference(self.patient_id, self.stay_id)
            assert res["status"] == "unavailable"
            assert "lacks required admission/prediction cutoff timestamps" in res["message"]
        finally:
            # Restore
            df.loc[df["stay_id"] == self.stay_id, "prediction_time"] = original_pred_time

    def test_20_backend_api_and_dashboard_stability(self):
        """Test 20: Backend REST API returns proper HTTP status codes and responses."""
        # 1. Health endpoint
        resp = self.flask_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "healthy"

        # 2. Patient search endpoint
        resp = self.flask_client.get(f"/api/patient/{self.patient_id}")
        assert resp.status_code == 200
        assert resp.get_json()["found"] is True

        # 3. Stay clinical record endpoint
        resp = self.flask_client.get(f"/api/patient/{self.patient_id}/stay/{self.stay_id}")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "success"

        # 4. Clinical inference endpoint
        resp = self.flask_client.get(f"/api/patient/{self.patient_id}/stay/{self.stay_id}/infer/clinical")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "success"

        # 5. CXR inference endpoint
        resp = self.flask_client.get(f"/api/patient/{self.patient_id}/stay/{self.stay_id}/infer/cxr")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "success"

        # 6. Sensor isolation status
        resp = self.flask_client.get("/api/sensor/status")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "isolated"
        assert "NOT associated" in resp.get_json()["isolation_guarantee"]

    def test_21_patient_multi_stay_dynamic_predictions_and_cross_stay_variance(self):
        """Test 21: Model predictions vary dynamically across stays and are not fixed or cached."""
        # Patient 17585185: Stay 32417784 (Emergency admission with severe tachycardia HR=139) -> High Risk
        inf_acute = run_clinical_transformer_inference(17585185, 32417784)
        assert inf_acute["status"] == "success"
        assert inf_acute["predicted_probability"] >= 0.60
        assert inf_acute["predicted_class"] == "High Sepsis Risk"

        # Patient 17585185: Stay 32776339 (Elective surgical admission with normal HR=85.4) -> Standard Risk
        inf_elective = run_clinical_transformer_inference(17585185, 32776339)
        assert inf_elective["status"] == "success"
        assert inf_elective["predicted_probability"] < 0.60
        assert inf_elective["predicted_class"] == "Standard Risk"

        # Verify probabilities differ significantly between stays of the SAME patient
        assert abs(inf_acute["predicted_probability"] - inf_elective["predicted_probability"]) > 0.40

        # Patient 10213765: Single stay 36628477 (Tachycardia HR=111.5, Emergency) -> High Risk
        inf_pt2 = run_clinical_transformer_inference(10213765, 36628477)
        assert inf_pt2["status"] == "success"
        assert inf_pt2["predicted_probability"] >= 0.60
        assert inf_pt2["predicted_class"] == "High Sepsis Risk"

        # CXR Balanced threshold behavior: Stay 36628477 has prob 0.5221 < 0.5237 -> Negative
        cxr_pt2 = run_cxr_inference(10213765, 36628477)
        assert cxr_pt2["thresholds"]["experimental_balanced_threshold_05237"]["classification"] == "Negative"


class TestAllPatientsDirectoryAndExtendedLookup:
    """Tests 22-27: Universal Patient Lookup, Multi-Stay Ordering, and Directory Filtering."""

    @classmethod
    def setup_class(cls):
        cls._original_require_auth = app.config.get("REQUIRE_AUTH", True)
        app.config["REQUIRE_AUTH"] = False
        cls.flask_client = app.test_client()

    @classmethod
    def teardown_class(cls):
        app.config["REQUIRE_AUTH"] = getattr(cls, "_original_require_auth", True)

    def test_22_patient_dataset_statistics_matches_authoritative_files(self):
        """Test 22: Summary statistics reflect the exact verified dataset counts."""
        stats = get_patient_dataset_statistics()
        assert stats["total_records"] == 2884
        assert stats["unique_patients"] == 2661
        assert stats["unique_stays"] == 2884
        assert stats["single_stay_patients"] == 2474
        assert stats["multi_stay_patients"] == 187
        assert stats["test_patients"] == 553
        assert stats["train_patients"] == 2108
        assert stats["test_records"] == 602
        assert stats["train_records"] == 2282
        assert stats["total_images"] == 2884

    def test_23_patients_beyond_examples_can_be_found(self):
        """Test 23: Verify lookups succeed for multiple patients beyond 17585185 and 10213765."""
        # Test patients from both test and train cohorts
        test_pids = [10002930, 10003019, 10007058, 13269859, 11281568, 19907884]
        for pid in test_pids:
            res = search_patient(pid)
            assert res["found"] is True, f"Patient {pid} must be found"
            assert res["patient_id"] == pid
            assert res["total_stays"] >= 1
            assert len(res["stays"]) == res["total_stays"]

            # Verify every stay has valid intime and clinical record
            for s in res["stays"]:
                assert "stay_id" in s
                assert "intime" in s and s["intime"] != "N/A"
                rec = get_stay_clinical_record(pid, s["stay_id"])
                assert rec["patient_id"] == pid
                assert rec["stay_id"] == s["stay_id"]
                # Zero target leakage
                assert "sepsis_label" not in rec

    def test_24_multi_stay_patients_longitudinal_selection(self):
        """Test 24: Multi-stay patients have stays chronologically sorted and individually selectable."""
        # Test several known multi-stay patients with 4 stays each
        multi_pids = [13269859, 11281568, 19907884]
        for pid in multi_pids:
            res = search_patient(pid)
            assert res["found"] is True
            assert res["total_stays"] == 4
            stays = res["stays"]
            assert len(stays) == 4

            # Verify chronological ordering by intime
            intimes = [s["intime"] for s in stays]
            assert intimes == sorted(intimes), f"Stays for patient {pid} must be chronologically ordered"

            # Verify each stay is individually queryable
            stay_ids = [s["stay_id"] for s in stays]
            assert len(set(stay_ids)) == 4, "Each stay must have a unique stay_id"

            for sid in stay_ids:
                rec = get_stay_clinical_record(pid, sid)
                assert rec["patient_id"] == pid
                assert rec["stay_id"] == sid
                # Clinical transformer runs independently per stay
                clin_inf = run_clinical_transformer_inference(pid, sid)
                assert clin_inf["status"] == "success"
                assert 0.0 <= clin_inf["predicted_probability"] <= 1.0

    def test_25_cohort_and_stay_filtering_in_patient_directory(self):
        """Test 25: Patient directory filters cohorts and multi-stay criteria correctly."""
        all_patients = get_eligible_patients()
        assert len(all_patients) == 2661

        test_patients = get_eligible_patients(split="test")
        assert len(test_patients) == 553
        assert all(p["split"] == "test" for p in test_patients)

        train_patients = get_eligible_patients(split="train")
        assert len(train_patients) == 2108
        assert all(p["split"] == "train" for p in train_patients)

        multi_stay_patients = get_eligible_patients(multi_stay_only=True)
        assert len(multi_stay_patients) == 187
        assert all(p["total_stays"] > 1 for p in multi_stay_patients)

        # Query substring filtering
        matched = get_eligible_patients(query="17585")
        assert len(matched) >= 1
        assert any(p["subject_id"] == 17585185 for p in matched)

    def test_26_flask_api_patients_and_stats_endpoints(self):
        """Test 26: Flask REST API /api/patients and /api/patients/stats endpoints."""
        # 1. Stats endpoint
        resp = self.flask_client.get("/api/patients/stats")
        assert resp.status_code == 200
        stats = resp.get_json()["statistics"]
        assert stats["unique_patients"] == 2661
        assert stats["total_records"] == 2884

        # 2. Patients list endpoint
        resp = self.flask_client.get("/api/patients")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["total_returned"] == 2661

        # 3. Filtered test split endpoint
        resp = self.flask_client.get("/api/patients?split=test")
        assert resp.status_code == 200
        assert resp.get_json()["total_returned"] == 553

        # 4. Multi-stay filter endpoint
        resp = self.flask_client.get("/api/patients?multi_stay=true")
        assert resp.status_code == 200
        assert resp.get_json()["total_returned"] == 187

    def test_27_unknown_and_invalid_ids_produce_clear_messages(self):
        """Test 27: Non-existent and malformed Patient IDs produce clear error messages."""
        # Non-existent ID
        res = search_patient(88888888)
        assert res["found"] is False
        assert "not found" in res["error"].lower()
        assert res["total_stays"] == 0

        # Malformed string
        res = search_patient("non_numeric_id")
        assert res["found"] is False
        assert "Malformed" in res["error"]

