# XMT-Sepsis-System: Final Academic Submission Package

**Course / Degree:** B.Tech / M.Tech in Computer Science & Engineering (Data Science)  
**Project Title:** Explainable Multimodal Transformer for Sepsis Disease Prediction in a Proactive Telemedicine Framework (XMT-Sepsis)  
**Academic Year:** 2026  
**Package Location:** `submission/`  

---

## 1. Submission Package Overview

This directory contains the complete academic submission documentation, presentation outlines, report blueprint, live demonstration scripts, and benchmark results tables for the **XMT-Sepsis-System**.

All deliverables have been synthesized strictly from verified local implementation artifacts without modifying any underlying source code, checkpoints, datasets, or configurations.

---

## 2. Package Deliverables & File Index

| Deliverable | File Path | Purpose & Content |
| :--- | :--- | :--- |
| **1. Presentation Outline** | [submission/PROJECT_PRESENTATION_OUTLINE.md](file:///C:/Users/Varshini/.gemini/antigravity/scratch/XMT-Sepsis-System/submission/PROJECT_PRESENTATION_OUTLINE.md) | Slide-by-slide 12-slide presentation structure covering motivation, architecture, data engineering, models, hardware, explainability, results, and limitations. |
| **2. Project Report Outline** | [submission/PROJECT_REPORT_OUTLINE.md](file:///C:/Users/Varshini/.gemini/antigravity/scratch/XMT-Sepsis-System/submission/PROJECT_REPORT_OUTLINE.md) | Comprehensive 9-chapter formal project report blueprint covering literature review, methodology, architecture, explainability, testing, ethics, and future directions. |
| **3. Live Demonstration Script** | [submission/LIVE_DEMO_SCRIPT.md](file:///C:/Users/Varshini/.gemini/antigravity/scratch/XMT-Sepsis-System/submission/LIVE_DEMO_SCRIPT.md) | Timed 5–7 minute walkthrough using the live Streamlit dashboard (`http://127.0.0.1:8502`) with exact spoken narratives, button clicks, and panel Q&A cheat sheet. |
| **4. Academic Results Tables** | [submission/ACADEMIC_RESULTS_TABLE.md](file:///C:/Users/Varshini/.gemini/antigravity/scratch/XMT-Sepsis-System/submission/ACADEMIC_RESULTS_TABLE.md) | Quantitative benchmark matrix on the held-out test cohort ($N=602$) comparing majority baseline, qSOFA, Logistic Regression, Random Forest, FT-Transformer, and Fusion Models A & B, with clear separation from the hourly task. |
| **5. Package Guide (This File)**| [submission/README.md](file:///C:/Users/Varshini/.gemini/antigravity/scratch/XMT-Sepsis-System/submission/README.md) | Executive summary, verification status, critical disclosures, and system preservation confirmation. |

---

## 3. Authoritative System Verification Summary

- **Software Daemons:**
  - Flask REST API on `http://127.0.0.1:5000` (PID 28760, healthy, HTTP 200).
  - Streamlit Presentation Dashboard on `http://127.0.0.1:8502` (PID 9092, healthy, HTTP 200).
- **Core Models Verified:**
  - `clinical_transformer.pt`: Deep Tabular FT-Transformer on 49 ICU biomarkers.
  - `cxr_encoder.pt`: ResNet-18 convolutional backbone on admission chest radiographs.
  - `concat_fusion_best.pt` (Model A): Multimodal early concatenation MLP.
  - `cross_attn_fusion_best.pt` (Model B): Dense visual cross-attention across clinical feature tokens.
- **Explainability Engines Verified:**
  - Local Kernel SHAP computing top-5 biomarker attributions on any selected patient.
  - Layer-4 Grad-CAM generating pulmonary activation heatmaps overlaid on radiographs.
- **Hardware Status:**
  - Microcontroller: ESP32 Dev Module + MAX30102 pulse oximeter on I²C.
  - Firmware on COM4 compiled and flashed with exit code 0.
  - Bedside state machine: optical lead-off ($IR < 10,000$ counts), 20 BPM HR spike rejection, and Tri-color LED indicators.

---

## 4. Mandatory Disclosures & Unverified Items

1. **Live Sensor Streaming Status:**
   - The hardware firmware, USB serial flashing, and backend ingestion gate are 100% code-verified.
   - However, during offline evaluation, the physical sensor was not actively transmitting live Wi-Fi packets to ThingSpeak; the dashboard transparently flags `SILENCE_WARNING` and reports `telemetry freshness: None` rather than generating synthetic live readings.
2. **Hourly Longitudinal Prediction Status:**
   - The temporal sequence model (`hourly_transformer_best.pt`) is fully developed and passes all 31 unit tests.
   - The longitudinal full-cohort ingestion pipeline in `data/raw_hourly/` remains **preparatory and gated** awaiting authorized institutional BigQuery export.
3. **Evaluation Cohorts & qSOFA Subscale:**
   - Machine learning benchmark models were evaluated on the complete held-out test cohort of $N = 602$ ICU stays (104 positive, $17.28\%$ prevalence).
   - Consensus 3-point qSOFA was unavailable due to unextracted SBP; the audited 2-point subscale (RR, GCS) was evaluated on $N = 595$ stays because 7 stays lacked GCS observations in the first 6 hours and were safely excluded rather than imputed.
4. **Accuracy Trade-Off & Sensitivity Bounds:**
   - In an imbalanced cohort ($17.28\%$ sepsis prevalence, 104/602), an all-negative trivial model achieves $82.72\%$ baseline accuracy with $0\%$ sensitivity.
   - Reaching $\ge 87\%$ accuracy mathematically requires sensitivity of at least $24.77\%$ ($0.1728 \times \text{Sens} + 0.8272 \ge 0.87$) even under hypothetical perfect $100\%$ specificity.
   - In empirical validation threshold sweeps, maximizing accuracy pushed cutoffs high ($\tau \ge 0.75 - 0.84$), achieving only $83.5\% - 84.1\%$ peak accuracy while collapsing sensitivity down to $5.2\% - 9.1\%$ (missing $>90\%$ of septic cases).
   - Clinical decision support deliberately prioritizes high screening sensitivity ($\ge 65\%$) and well-calibrated probabilities ($\text{Brier} \approx 0.13$) at prevalence-aligned thresholds ($\tau \approx 0.14 - 0.21$) over artificial accuracy chasing.
5. **Research Disclaimer:**
   - This platform is an **experimental academic research prototype** evaluated on retrospective de-identified MIMIC-IV data. It is **NOT** a certified medical diagnostic tool and must not be used for patient management. Neither validation F1-maximizing thresholds nor benchmark screening thresholds have been clinically validated in prospective interventional trials.

---

## 5. Confirmation of System Preservation

- **Python Source Files Modified:** 0
- **Model Checkpoints Overwritten:** 0
- **Calibration Parameters Altered:** 0
- **Production Thresholds Changed:** 0
- **Running Services Restarted or Interrupted:** 0
- **Locked Test Set Contaminated:** 0
