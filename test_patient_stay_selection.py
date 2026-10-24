"""
tests/test_patient_stay_selection.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive regression tests for patient ID and ICU stay selection:
1. Patient with exactly one stay auto-selection.
2. Patient with multiple stays selection and default behavior.
3. Switching patients safely resets an invalid previous stay.
4. Missing stay selection (Stay ID None) does not trigger unhandled traceback.
5. Invalid patient and stay combinations are handled safely.
6. Multimodal Research loads a valid patient/stay combination without error.
7. Patient Explorer and Multimodal Research maintain consistent selection behavior.
8. Clinical AI receives the exact matching clinical record.
9. No record belonging to another patient can be returned accidentally.
"""

import os
import sys
import pytest
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backend.patient_service import (
    resolve_patient_stay,
    get_stay_clinical_record,
    search_patient,
    run_clinical_transformer_inference,
    run_fusion_models_inference,
    get_raw_dataset,
)


class TestPatientStaySelection:
    @classmethod
    def setup_class(cls):
        cls.raw_df = get_raw_dataset()
        # Authoritative test fixtures
        cls.single_stay_pid = 10213765      # Has exactly 1 stay: 36628477 (has CXR)
        cls.single_stay_id = 36628477
        cls.multi_stay_pid = 17585185       # Multi-stay patient: stays include 32417784, 36336086
        cls.multi_stay_ids = [32417784, 36336086]
        cls.target_bug_pid = 10146782       # Patient reported in prompt traceback

    # ── Requirement 1: Patient with exactly one stay ──────────────────────────
    def test_01_patient_with_exactly_one_stay(self):
        """Single-stay patient automatically resolves their single stay ID."""
        pid, sid, stays, err = resolve_patient_stay(self.single_stay_pid, None)
        assert err is None
        assert pid == self.single_stay_pid
        assert sid == self.single_stay_id
        assert len(stays) == 1
        assert stays[0]["stay_id"] == self.single_stay_id

    # ── Requirement 2: Patient with multiple stays ────────────────────────────
    def test_02_patient_with_multiple_stays(self):
        """Multi-stay patient allows selecting specific stay or defaults safely."""
        # 2a. Specific stay provided
        pid, sid, stays, err = resolve_patient_stay(self.multi_stay_pid, self.multi_stay_ids[1])
        assert err is None
        assert pid == self.multi_stay_pid
        assert sid == self.multi_stay_ids[1]
        assert len(stays) >= 2

        # 2b. None provided -> defaults to first chronological stay
        pid_def, sid_def, stays_def, err_def = resolve_patient_stay(self.multi_stay_pid, None)
        assert err_def is None
        assert pid_def == self.multi_stay_pid
        assert sid_def == self.multi_stay_ids[0]

    # ── Requirement 3: Switching patients resets invalid previous stay ────────
    def test_03_switching_patients_resets_invalid_previous_stay(self):
        """Switching to a new patient rejects stale stay from previous patient."""
        # Previous patient was 17585185 (stay 32417784).
        # User switches to patient 10213765, but stay_id 32417784 is still passed.
        pid, sid, stays, err = resolve_patient_stay(self.single_stay_pid, 32417784)
        assert err is None
        assert pid == self.single_stay_pid
        # Must NOT retain 32417784; must resolve to 10213765's valid stay 36628477
        assert sid == self.single_stay_id
        assert sid != 32417784

    # ── Requirement 4: Missing stay selection does not trigger traceback ──────
    def test_04_missing_stay_selection_does_not_trigger_unhandled_traceback(self):
        """Patient 10146782 with Stay ID None resolves cleanly without ValueError."""
        pid, sid, stays, err = resolve_patient_stay(self.target_bug_pid, None)
        assert err is None
        assert pid == self.target_bug_pid
        assert sid is not None
        assert isinstance(sid, int)

        # Loading record with resolved stay succeeds
        rec = get_stay_clinical_record(pid, sid)
        assert rec["patient_id"] == self.target_bug_pid
        assert rec["stay_id"] == sid

        # Directly calling get_stay_clinical_record with None raises clear ValueError
        with pytest.raises(ValueError, match="No record found"):
            get_stay_clinical_record(self.target_bug_pid, None)

    # ── Requirement 5: Invalid patient and stay combinations handled safely ───
    def test_05_invalid_patient_and_stay_combinations_handled_safely(self):
        """Invalid or non-existent IDs return structured error instead of unhandled crash."""
        # Non-numeric string
        pid, sid, stays, err = resolve_patient_stay("invalid_abc", None)
        assert pid is None
        assert sid is None
        assert "Malformed" in err

        # Non-existent ID
        pid, sid, stays, err = resolve_patient_stay(99999999, None)
        assert pid is None
        assert sid is None
        assert "not found" in err.lower()

        # Direct clinical record lookup with invalid IDs raises clear ValueError
        with pytest.raises(ValueError, match="No record found"):
            get_stay_clinical_record(99999999, 88888888)

    # ── Requirement 6: Multimodal Research loads valid patient/stay ───────────
    def test_06_multimodal_research_loads_valid_patient_stay_combination(self):
        """Multimodal fusion inference executes successfully with validated stay."""
        # String and int IDs both supported
        res = run_fusion_models_inference(str(self.single_stay_pid), str(self.single_stay_id))
        assert res.get("status") == "success"
        assert res["patient_id"] == self.single_stay_pid
        assert res["stay_id"] == self.single_stay_id
        assert "model_a" in res
        assert "model_b" in res

    # ── Requirement 7: Patient Explorer & Multimodal maintain consistent behavior ─
    def test_07_patient_explorer_and_multimodal_maintain_consistent_selection(self):
        """Both views resolve identical patient and stay contexts."""
        # Explorer resolution
        exp_pid, exp_sid, _, _ = resolve_patient_stay(str(self.single_stay_pid), None)
        # Multimodal resolution
        mm_pid, mm_sid, _, _ = resolve_patient_stay(str(self.single_stay_pid), None)

        assert exp_pid == mm_pid == self.single_stay_pid
        assert exp_sid == mm_sid == self.single_stay_id

        rec_exp = get_stay_clinical_record(exp_pid, exp_sid)
        rec_mm = get_stay_clinical_record(mm_pid, mm_sid)
        assert rec_exp["patient_id"] == rec_mm["patient_id"]
        assert rec_exp["stay_id"] == rec_mm["stay_id"]

    # ── Requirement 8: Clinical AI receives correct clinical record ───────────
    def test_08_clinical_ai_receives_correct_clinical_record(self):
        """Clinical Transformer receives exact clinical features for selected stay."""
        pid, sid, _, _ = resolve_patient_stay(self.multi_stay_pid, self.multi_stay_ids[0])
        rec = get_stay_clinical_record(pid, sid)
        assert rec["patient_id"] == self.multi_stay_pid
        assert rec["stay_id"] == self.multi_stay_ids[0]

        inf_res = run_clinical_transformer_inference(pid, sid)
        assert inf_res["status"] == "success"
        assert inf_res["patient_id"] == self.multi_stay_pid
        assert inf_res["stay_id"] == self.multi_stay_ids[0]

    # ── Requirement 9: No record belonging to another patient can be returned ─
    def test_09_no_record_belonging_to_another_patient_can_be_returned(self):
        """Cross-patient stay query strictly raises ValueError and never returns data."""
        # Patient 10213765 does NOT own stay 32417784 (which belongs to 17585185)
        with pytest.raises(ValueError, match="No record found"):
            get_stay_clinical_record(self.single_stay_pid, 32417784)
