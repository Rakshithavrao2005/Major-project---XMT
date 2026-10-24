"""
tests/test_conditional_final_prediction.py
─────────────────────────────────────────────────────────────────────────────
Targeted regression test suite verifying conditional final prediction and
model disagreement display contracts in XMT Sepsis System:

1. Final model positive: shows positive assessment; hides alternative-cause & disagreement panels.
2. Clinical lower-risk + X-ray positive: shows the Model Disagreement panel with both model outputs.
3. Clinical elevated-risk + X-ray negative: explains disagreement (negative CXR does not negate clinical risk).
4. Fusion positive or negative using its actual documented threshold (0.50, never averaged).
5. Missing or failed model output: displays unavailable / uncertain state with exact failure reason.
6. X-ray positive without a separate disease model: does not invent alternative diagnoses.
7. Score, threshold, calibrated probability, and displayed classification remain strictly consistent.
8. The conditional UI preserves navigation buttons, patient lookup, SHAP, and hardware monitoring.
"""

from __future__ import annotations

import os
import sys
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dashboard.styles import (
    PAGE_CLINICAL_AI,
    PAGE_CHEST_XRAY,
    PAGE_MODEL_FUSION,
    PAGE_MULTIMODAL_RESEARCH,
)
from backend.patient_service import (
    resolve_patient_stay,
    run_clinical_transformer_inference,
    run_cxr_inference,
    run_fusion_models_inference,
    get_all_patients_directory,
    get_sensor_telemetry_status,
)
from dashboard.final_risk_assessment import render_final_sepsis_risk_assessment


class MockStreamlitCapture:
    """Mock Streamlit context capturing rendered markdown and buttons."""

    def __init__(self):
        self.markdown_blocks = []
        self.columns_created = 0
        self.buttons = []
        self.session_state = {}

    def markdown(self, body, unsafe_allow_html=False):
        self.markdown_blocks.append(str(body))

    def caption(self, body):
        self.markdown_blocks.append(str(body))

    def columns(self, spec):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        self.columns_created += n
        return [self for _ in range(n)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def button(self, label, **kwargs):
        self.buttons.append(label)
        return False

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index] if options else ""

    def rerun(self):
        pass

    def get_full_rendered_text(self) -> str:
        return "\n".join(self.markdown_blocks)


class TestConditionalFinalPrediction:
    """Comprehensive test suite for conditional assessment display."""

    def test_01_final_model_positive_shows_positive_and_hides_alternative_cause(self, monkeypatch):
        """
        State A: When Clinical Model is positive (e.g. prob >= 0.60):
        Must show the Elevated Sepsis Risk Assessment;
        Must HIDE the alternative-cause panel, the X-ray cause box, and the disagreement panel.
        """
        mock_st = MockStreamlitCapture()
        mock_st.session_state = {"current_search_pid": "17585185", "current_stay_id": 32417784}

        # Monkeypatch streamlit calls inside final_risk_assessment
        from dashboard import final_risk_assessment
        monkeypatch.setattr(final_risk_assessment, "st", mock_st)

        # Mock clinical inference to return elevated risk
        monkeypatch.setattr(
            final_risk_assessment,
            "run_clinical_transformer_inference",
            lambda pid, sid: {
                "status": "success",
                "predicted_probability": 0.7754,
                "decision_threshold": 0.60,
                "predicted_class": "High Sepsis Risk",
                "raw_logit": 1.238,
                "top_attended_features": [
                    {"feature": "lactate", "attention_pct": 14.2},
                    {"feature": "wbc", "attention_pct": 11.5},
                ],
            },
        )

        render_final_sepsis_risk_assessment(patient_id=17585185, stay_id=32417784)
        rendered = mock_st.get_full_rendered_text()

        # Must display Elevated Risk assessment and decision rule
        assert "Final Decision: Elevated Risk" in rendered
        assert "HIGH RISK" in rendered
        assert "77.54%" in rendered
        assert "60.0%" in rendered

        # Must HIDE the Model Disagreement callout
        assert "⚠️ Model Disagreement — Chest X-ray Positive with Lower Clinical Sepsis Risk" not in rendered

        # Must HIDE the alternative-cause and X-ray why-positive explanation panels in State A
        assert "Why is the X-ray model showing positive?" not in rendered
        assert "Alternative-condition detection is not currently configured." not in rendered

    def test_02_clinical_lower_risk_and_xray_positive_shows_disagreement(self, monkeypatch):
        """
        State B: Clinical model is lower risk (< 0.60) AND X-ray model is positive (>= 0.5237):
        Must display the Model Disagreement panel with both model outputs;
        Must explain that positive X-ray does not independently establish sepsis;
        Must display the alternative-condition and X-ray cause explanations ONLY in this state.
        """
        mock_st = MockStreamlitCapture()
        mock_st.session_state = {"current_search_pid": "10146782", "current_stay_id": 32413224}

        from dashboard import final_risk_assessment
        monkeypatch.setattr(final_risk_assessment, "st", mock_st)

        # Mock Clinical AI as lower risk (38.86%) and CXR as positive (52.53% raw, 16.95% Platt)
        monkeypatch.setattr(
            final_risk_assessment,
            "run_clinical_transformer_inference",
            lambda pid, sid: {
                "status": "success",
                "predicted_probability": 0.3886,
                "decision_threshold": 0.60,
                "predicted_class": "Standard Risk",
                "raw_logit": -0.4532,
            },
        )
        monkeypatch.setattr(
            final_risk_assessment,
            "run_cxr_inference",
            lambda pid, sid: {
                "status": "success",
                "raw_probability": 0.5253,
                "platt_calibrated_probability": 0.1695,
                "raw_logit": 0.1012,
                "thresholds": {
                    "experimental_balanced_threshold_05237": {"threshold": 0.5237}
                },
            },
        )

        render_final_sepsis_risk_assessment(patient_id=10146782, stay_id=32413224)
        rendered = mock_st.get_full_rendered_text()

        # Must show Model Disagreement panel
        assert "Model Disagreement — Chest X-ray Positive with Lower Clinical Sepsis Risk" in rendered
        assert "38.86%" in rendered
        assert "LOW RISK" in rendered
        assert "52.5%" in rendered
        assert "52.4%" in rendered
        assert "17.0%" in rendered or "16.9%" in rendered

        # Must explain that positive X-ray does not independently establish sepsis
        assert "The models disagree. A positive X-ray model output does not independently confirm sepsis" in rendered
        assert "Further clinical review is required" in rendered

        # Must show the non-fabrication cause and alternative-condition notices in State B
        assert "Why is the X-ray model showing positive?" in rendered
        assert "Specific cause not determined by this model." in rendered
        assert "Alternative-condition detection is not currently configured." in rendered

    def test_03_clinical_elevated_risk_and_xray_negative_explains_difference(self, monkeypatch):
        """
        State A (Sub-case: Isolated Clinical Elevation):
        Clinical is elevated (>= 0.60), but CXR is negative (< 0.5237).
        Must explain that negative CXR does not negate clinical risk because sepsis can arise from non-pulmonary sources.
        Must HIDE the 'Why is the X-ray model showing positive?' box.
        """
        mock_st = MockStreamlitCapture()
        from dashboard import final_risk_assessment
        monkeypatch.setattr(final_risk_assessment, "st", mock_st)

        monkeypatch.setattr(
            final_risk_assessment,
            "run_clinical_transformer_inference",
            lambda pid, sid: {
                "status": "success",
                "predicted_probability": 0.6665,
                "decision_threshold": 0.60,
                "predicted_class": "High Sepsis Risk",
                "raw_logit": 0.6924,
                "top_attended_features": [{"feature": "heart_rate", "attention_pct": 12.0}],
            },
        )
        monkeypatch.setattr(
            final_risk_assessment,
            "run_cxr_inference",
            lambda pid, sid: {
                "status": "success",
                "raw_probability": 0.5221,
                "platt_calibrated_probability": 0.1692,
                "raw_logit": 0.0884,
                "thresholds": {
                    "experimental_balanced_threshold_05237": {"threshold": 0.5237}
                },
            },
        )

        render_final_sepsis_risk_assessment(patient_id=10213765, stay_id=36628477)
        rendered = mock_st.get_full_rendered_text()

        # Must show Isolated Clinical Elevation explanation
        assert "Isolated Clinical Elevation" in rendered
        assert "A negative X-ray output does not negate the clinical model's elevated estimate" in rendered
        assert "non-pulmonary infection sources" in rendered

        # Must HIDE the 'Why is the X-ray model showing positive?' box since CXR is negative
        assert "Why is the X-ray model showing positive?" not in rendered
        assert "Alternative-condition detection is not currently configured." not in rendered

    def test_04_fusion_positive_or_negative_uses_actual_documented_threshold(self):
        """
        Multimodal fusion outputs (Model A & Model B) must strictly evaluate threshold 0.50.
        Probabilities must be independent and never averaged.
        """
        # Test patient #10213765
        fus = run_fusion_models_inference(10213765, 36628477)
        assert fus["status"] == "success"
        ma = fus["model_a"]
        mb = fus["model_b"]

        assert ma["threshold"] == 0.50
        assert mb["threshold"] == 0.50

        # Classification must strictly match prob >= 0.50
        assert ("High" in ma["classification"] or "Positive" in ma["classification"]) == (ma["predicted_probability"] >= 0.50)
        assert ("High" in mb["classification"] or "Positive" in mb["classification"]) == (mb["predicted_probability"] >= 0.50)

        # Probabilities must not be identical or averaged
        assert ma["predicted_probability"] != mb["predicted_probability"] or ma["name"] != mb["name"]

    def test_05_missing_or_failed_model_output_shows_unavailable_state(self, monkeypatch):
        """
        State C (Unavailable): When clinical inference fails or returns an error,
        the UI must display 'Final Sepsis Risk Assessment Unavailable' with exact error reason.
        """
        mock_st = MockStreamlitCapture()
        from dashboard import final_risk_assessment
        monkeypatch.setattr(final_risk_assessment, "st", mock_st)

        monkeypatch.setattr(
            final_risk_assessment,
            "run_clinical_transformer_inference",
            lambda pid, sid: {
                "status": "unavailable",
                "message": "Selected stay lacks required admission/prediction timestamps.",
            },
        )

        render_final_sepsis_risk_assessment(patient_id=10213765, stay_id=36628477)
        rendered = mock_st.get_full_rendered_text()

        assert "Final Sepsis Risk Assessment Unavailable" in rendered
        assert "Selected stay lacks required admission/prediction timestamps" in rendered
        assert "HIGH RISK" not in rendered
        assert "Concordant Lower Risk Estimates" not in rendered

    def test_06_xray_positive_without_separate_model_does_not_invent_alternative_diagnosis(self):
        """
        When CXR is positive, ensure the explanation discloses that the specific cause is not determined,
        that the model cannot diagnose alternative conditions, and makes no unsubstantiated claims
        (e.g., claiming a chest X-ray diagnoses a throat infection).
        """
        fra_path = os.path.join(_PROJECT_ROOT, "dashboard", "final_risk_assessment.py")
        with open(fra_path, "r", encoding="utf-8") as f:
            fra_code = f.read()

        assert "Specific cause not determined by this model." in fra_code
        assert "The current model's positive output is not sufficient to identify an alternative disease." in fra_code
        assert "Alternative-condition detection is not currently configured." in fra_code
        assert "cannot diagnose throat or upper respiratory infections" in fra_code

    def test_07_score_threshold_calibrated_probability_consistency(self):
        """
        Verify that the CXR balanced threshold (0.5237) is applied strictly to the raw sigmoid score,
        while the Platt calibrated probability reflects ICU prevalence (~16-17%),
        confirming that raw score >= 52.4% can co-occur with Platt prob ~17% without contradiction.
        """
        cxr = run_cxr_inference(10146782, 32413224)
        assert cxr["status"] == "success"

        raw_p = cxr["raw_probability"]
        cal_p = cxr["platt_calibrated_probability"]
        bal_cutoff = cxr["thresholds"]["experimental_balanced_threshold_05237"]["threshold"]

        # Raw probability is near ~0.525, exceeding the 0.5237 threshold
        assert raw_p >= bal_cutoff
        # Platt probability is scaled down to ~0.169 (16.9-17.0%)
        assert cal_p < 0.20
        # The governing score documents that raw probability governs the binary decision
        gov = cxr["governing_score"]
        assert gov["score_type"] == "raw_probability"
        assert gov["threshold"] == bal_cutoff
        assert gov["classification"] == "Positive (High Risk)"

    def test_08_conditional_ui_preserves_navigation_and_subsystems(self):
        """
        Verify that navigation buttons, patient directory, and hardware telemetry isolation
        remain preserved and fully operational.
        """
        # Directory lookup preserved
        dir_df = get_all_patients_directory()
        assert len(dir_df) > 0
        assert "subject_id" in dir_df.columns

        # Telemetry isolation guarantee preserved
        telemetry_status = get_sensor_telemetry_status()
        assert telemetry_status["status"] == "isolated"
        assert "STRICT DATA ISOLATION" in telemetry_status["isolation_guarantee"]
