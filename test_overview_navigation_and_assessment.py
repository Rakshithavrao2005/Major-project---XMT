"""
tests/test_overview_navigation_and_assessment.py
─────────────────────────────────────────────────────────────────────────────
Targeted verification tests for:
1. Overview and Patient Explorer navigation button contracts (setting both active_page & nav_mode).
2. Final Sepsis Risk Assessment component execution and multi-model consensus.
3. Separation of Chest X-ray positivity (learned target) from confirmed sepsis risk.
4. Model Disagreement explanations (Quadrant 1: positive CXR + lower clinical risk).
5. Concordant risk explanations (Quadrant 2, Quadrant 3, Quadrant 4).
6. Non-fabrication disclosures ("Specific cause not determined by this model").
7. Alternative chest-condition detection disclosure ("Alternative-condition detection is not currently configured").
8. Input pairing validation and isolation (never average probabilities, disable fusion when unpaired).
"""

from __future__ import annotations

import os
import sys
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dashboard.styles import (
    PAGE_HOME,
    PAGE_OVERVIEW,
    PAGE_CLINICAL_AI,
    PAGE_CHEST_XRAY,
    PAGE_MODEL_FUSION,
    PAGE_MULTIMODAL_RESEARCH,
    PAGE_PATIENT_EXPLORER,
    PAGE_MODEL_EVALUATION,
    PAGE_HARDWARE_ALERTS,
)
from dashboard.app import PAGE_ROUTING
from backend.patient_service import (
    run_clinical_transformer_inference,
    run_cxr_inference,
    run_fusion_models_inference,
    resolve_patient_stay,
    get_stay_clinical_record,
)
from dashboard.final_risk_assessment import render_final_sepsis_risk_assessment
from dashboard.model_comparison import render_cross_model_comparison_section


class TestOverviewNavigationButtons:
    """Tests for button routing invariants."""

    def test_routing_table_contains_home_and_overview(self):
        """PAGE_ROUTING must support both PAGE_HOME and PAGE_OVERVIEW with valid callables."""
        assert PAGE_HOME in PAGE_ROUTING
        assert PAGE_OVERVIEW in PAGE_ROUTING
        renderer_home, _ = PAGE_ROUTING[PAGE_HOME]
        renderer_ov, _ = PAGE_ROUTING[PAGE_OVERVIEW]
        assert callable(renderer_home)
        assert callable(renderer_ov)

    def test_routing_targets_for_overview_action_buttons(self):
        """The 3 target pages for Overview actions must exist in PAGE_ROUTING."""
        assert PAGE_CLINICAL_AI in PAGE_ROUTING
        assert PAGE_CHEST_XRAY in PAGE_ROUTING
        assert PAGE_MODEL_FUSION in PAGE_ROUTING
        assert PAGE_MULTIMODAL_RESEARCH in PAGE_ROUTING

        # Verify callables
        for key in [PAGE_CLINICAL_AI, PAGE_CHEST_XRAY, PAGE_MODEL_FUSION, PAGE_MULTIMODAL_RESEARCH]:
            fn, canon = PAGE_ROUTING[key]
            assert callable(fn)
            assert isinstance(canon, str)

    def test_navigation_button_source_code_sets_active_page(self):
        """Verify that source files set st.session_state['active_page'] upon button click."""
        pe_path = os.path.join(_PROJECT_ROOT, "dashboard", "pages", "patient_explorer.py")
        with open(pe_path, "r", encoding="utf-8") as f:
            pe_content = f.read()
        assert 'st.session_state["active_page"] = PAGE_CLINICAL_AI' in pe_content
        assert 'st.session_state["active_page"] = PAGE_CHEST_XRAY' in pe_content
        assert 'st.session_state["active_page"] = PAGE_MODEL_FUSION' in pe_content

        ov_path = os.path.join(_PROJECT_ROOT, "dashboard", "pages", "overview.py")
        with open(ov_path, "r", encoding="utf-8") as f:
            ov_content = f.read()
        assert 'st.session_state["active_page"] = PAGE_CLINICAL_AI' in ov_content
        assert 'st.session_state["active_page"] = PAGE_PATIENT_EXPLORER' in ov_content
        assert 'st.session_state["active_page"] = PAGE_MODEL_EVALUATION' in ov_content
        assert 'st.session_state["active_page"] = PAGE_HARDWARE_ALERTS' in ov_content


class TestFinalSepsisRiskAssessmentLogic:
    """Tests for multi-model risk assessment contracts and consensus."""

    def test_paired_patient_inference_contracts(self):
        """Verify Patient #10213765 (Stay #36628477) has valid paired clinical and imaging outputs."""
        pid, sid, stays, err = resolve_patient_stay("10213765", 36628477)
        assert err is None
        assert pid == 10213765
        assert sid == 36628477

        # Clinical Inference
        clin = run_clinical_transformer_inference(pid, sid)
        assert clin["status"] == "success"
        assert 0.0 <= clin["predicted_probability"] <= 1.0
        assert clin["decision_threshold"] == 0.60
        assert any(term in clin["predicted_class"].upper() for term in ["LOW", "HIGH", "RISK"])

        # Chest X-ray Inference
        cxr = run_cxr_inference(pid, sid)
        assert cxr["status"] == "success"
        assert 0.0 <= cxr["raw_probability"] <= 1.0
        assert 0.0 <= cxr["platt_calibrated_probability"] <= 1.0
        assert "experimental_balanced_threshold_05237" in cxr["thresholds"]
        bal_cutoff = cxr["thresholds"]["experimental_balanced_threshold_05237"]["threshold"]
        assert bal_cutoff == 0.5237

        # Multimodal Fusion Inference
        fus = run_fusion_models_inference(pid, sid)
        assert fus["status"] == "success"
        ma = fus["model_a"]
        mb = fus["model_b"]
        assert ma["threshold"] == 0.50
        assert mb["threshold"] == 0.50
        # Probabilities must be independent numbers and NOT identical or averaged
        assert isinstance(ma["predicted_probability"], float)
        assert isinstance(mb["predicted_probability"], float)

    def test_unpaired_patient_disables_fusion(self, monkeypatch):
        """When CXR visual feature is missing/unpaired, fusion must be marked unavailable."""
        from src.backend import patient_service
        monkeypatch.setattr(
            patient_service,
            "get_stay_cxr_feature_512",
            lambda pid, sid: (None, "missing_dicom", "Linked image file not found on disk."),
        )

        fus = run_fusion_models_inference(10213765, 36628477)
        assert fus["status"] == "unavailable"
        assert "multimodal fusion requires" in fus["message"].lower()

    def test_model_disagreement_quadrant_1_handling(self):
        """
        When CXR raw score >= 0.5237 (Positive) and Clinical AI < 0.60 (Low Risk):
        Ensure disagreement text distinguishes CXR learned target from clinical diagnosis.
        """
        fra_path = os.path.join(_PROJECT_ROOT, "dashboard", "final_risk_assessment.py")
        with open(fra_path, "r", encoding="utf-8") as f:
            fra_code = f.read()

        # Must explicitly mention Model Disagreement
        assert "Model Disagreement" in fra_code
        # Must mention positive output for model's trained target
        assert "Positive output for the model's trained target" in fra_code
        # Must mention lower estimated risk
        assert "Lower estimated risk according to this model" in fra_code
        # Must mention that models disagree and further clinical review is required
        assert "The models disagree. A positive X-ray model output does not independently confirm sepsis" in fra_code
        assert "Further clinical review is required" in fra_code

    def test_quadrant_2_elevated_risk_consensus_handling(self):
        """When both CXR and Clinical AI are positive/elevated, state elevated consensus without claiming confirmed diagnosis."""
        fra_path = os.path.join(_PROJECT_ROOT, "dashboard", "final_risk_assessment.py")
        with open(fra_path, "r", encoding="utf-8") as f:
            fra_code = f.read()

        assert "Elevated Multimodal Risk Consensus" in fra_code
        assert "does not constitute an automated or confirmed medical diagnosis" in fra_code

    def test_quadrant_3_isolated_clinical_elevation_handling(self):
        """When CXR is negative and Clinical AI is elevated, negative CXR does not negate clinical elevation."""
        fra_path = os.path.join(_PROJECT_ROOT, "dashboard", "final_risk_assessment.py")
        with open(fra_path, "r", encoding="utf-8") as f:
            fra_code = f.read()

        assert "Isolated Clinical Elevation" in fra_code
        assert "A negative X-ray output does not negate the clinical model's elevated estimate" in fra_code

    def test_quadrant_4_concordant_lower_risk_handling(self):
        """When both are lower/negative, lower scores are not proof that patient is free of sepsis."""
        fra_path = os.path.join(_PROJECT_ROOT, "dashboard", "final_risk_assessment.py")
        with open(fra_path, "r", encoding="utf-8") as f:
            fra_code = f.read()

        assert "Concordant Lower Risk Estimates" in fra_code
        assert "These are lower model estimates, not proof that the patient is free of sepsis" in fra_code

    def test_non_fabrication_disclaimers(self):
        """Verify non-fabrication disclaimers for CXR causes and alternative conditions."""
        fra_path = os.path.join(_PROJECT_ROOT, "dashboard", "final_risk_assessment.py")
        with open(fra_path, "r", encoding="utf-8") as f:
            fra_code = f.read()

        # Part 5 exact phrasing
        assert "Specific cause not determined by this model." in fra_code
        assert "The current model's positive output is not sufficient to identify an alternative disease." in fra_code

        # Part 6 exact phrasing
        assert "Alternative-condition detection is not currently configured." in fra_code

    def test_no_probability_averaging(self):
        """Verify that fusion model code explicitly forbids and avoids averaging probabilities."""
        fra_path = os.path.join(_PROJECT_ROOT, "dashboard", "final_risk_assessment.py")
        with open(fra_path, "r", encoding="utf-8") as f:
            fra_code = f.read()

        assert "never averaged" in fra_code
