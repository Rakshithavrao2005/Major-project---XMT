# XMT-Sepsis-System: Final Project Presentation Outline

**Degree / Discipline:** B.Tech / M.Tech in Computer Science & Engineering (Data Science)  
**Project Title:** Explainable Multimodal Transformer for Sepsis Disease Prediction in a Proactive Telemedicine Framework (XMT-Sepsis)  
**Academic Status:** Final Capstone / Major Project Presentation  
**Project Root:** `C:\Users\Varshini\.gemini\antigravity\scratch\XMT-Sepsis-System`  

---

## Slide 1: Title & Overview
- **Title:** Explainable Multimodal Transformer for Sepsis Disease Prediction in a Proactive Telemedicine Framework
- **Subtitle:** Joint Modeling of Tabular ICU Biomarkers, Chest Radiographs, and Bedside IoT Telemetry with Explainable AI
- **Domain:** Healthcare Informatics • Deep Learning • Multimodal Fusion • Embedded Edge IoT • Explainable AI (XAI)
- **Key Message:** Early sepsis detection within the first hours of hospital admission using transparent, multimodal deep learning paired with an edge IoT vital signs acquisition prototype.
- **Academic Disclaimer:** *Research Prototype Only. Evaluated on retrospective de-identified MIMIC-IV data. Not certified for diagnostic or independent clinical use.*

---

## Slide 2: Problem Statement & Clinical Motivation
- **Clinical Challenge of Sepsis:**
  - Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection (Sepsis-3 definition).
  - Every hour of delayed antibiotic intervention increases mortality by 7.6% (Kumar et al., *Crit Care Med*).
- **Limitations of Current Approaches:**
  - *Heuristic Scores (qSOFA, SIRS, MEWS):* High specificity but poor sensitivity (~10% to 50%), missing subtle deteriorating trajectories.
  - *Black-Box Clinical AI:* Complex ensembles often lack transparency, causing clinician distrust and alert fatigue.
  - *Unimodal Isolation:* Hospital records, pulmonary radiographs, and wearable bedside telemetry operate in separate, unintegrated silos.
- **Research Goal:** Build a patient-centric, explainable multimodal deep learning framework that detects sepsis risk within 1–24 hours post-admission, supported by local attribution explanations (SHAP & Grad-CAM) and bedside edge telemetry.

---

## Slide 3: Objectives & Key Research Contributions
1. **Multimodal Architecture Design:**
   - Unimodal Feature Tokenizer Transformer (FT-Transformer) evaluating 49 continuous and categorical laboratory biomarkers.
   - Deep Convolutional Encoder (ResNet-18) extracting 512-dimensional representations from admission chest radiographs (CXR).
   - Comparative Multimodal Fusion: Feature Concatenation MLP (Model A) vs. Cross-Attention Fusion (Model B).
2. **Strict Leakage-Safe Data Engineering:**
   - Temporal observation window fixed strictly at $T = \text{intime} + 6\text{h}$; prediction horizon defined as $H = 24\text{h}$ (1–24 hours following $T$).
   - Patient-level split isolation guaranteeing 0% subject overlap across training ($N=2,282$), validation ($N=454$), and locked test ($N=602$) sets.
3. **Rigorous Post-Hoc Probability Calibration:**
   - Implementation of Platt scaling to realign uncalibrated logits with empirical sepsis prevalence ($\sim 17.28\%$), reducing Brier score from 0.23 to 0.13.
4. **Transparent Explainability Framework:**
   - Local Kernel SHAP computing individualized biomarker attribution scores.
   - Layer-4 Grad-CAM activation maps localizing pulmonary abnormalities on chest radiographs.
5. **Bedside Edge IoT Integration:**
   - ESP32 microcontroller with MAX30102 pulse oximeter acquiring HR, $\text{SpO}_2$, and optical PPG counts with bedside LCD/LED alerts and bounded cloud telemetry ingestion.

---

## Slide 4: System Architecture & Data Flow
```
┌────────────────────────────────────────────────────────────────────────┐
│                   HOSPITAL CLINICAL DATA PIPELINE                      │
│                                                                        │
│  [MIMIC-IV Cohort: 2,661 Patients / 2,884 Stays]                       │
│        │                                                               │
│        ├─► [49 Tabular Biomarkers: Vitals, CBC, ABG, Chemistries]     │
│        │         │                                                     │
│        │         ▼                                                     │
│        │   [FT-Transformer] ──► Clinical Latent z_clin in R^64         │
│        │                               │                               │
│        └─► [Admission Chest X-ray]     │ (Model A Concat or            │
│                  │                     │  Model B Cross-Attention)     │
│                  ▼                     ▼                               │
│            [ResNet-18 CXR]  ──► Visual Latent z_img in R^64            │
│                                        │                               │
│                                        ▼                               │
│                             [Joint Multimodal Head]                    │
│                                        │                               │
│                                        ▼                               │
│                             [Platt Scaling Calibrator]                 │
│                                        │                               │
│                                        ▼                               │
│                      [Calibrated Probability & Risk Tier]              │
│                                        │                               │
│                   ┌────────────────────┴───────────────────┐           │
│                   ▼                                        ▼           │
│       [Local Kernel SHAP (Top-5)]            [Layer-4 Grad-CAM Heatmap]│
└───────────────────┼────────────────────────────────────────┼───────────┘
                    │                                        │
                    ▼                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   BEDSIDE EDGE IOT MONITOR (ESP32)                     │
│                                                                        │
│   MAX30102 PPG Sensor ──► ESP32 Microcontroller ──► Bedside LCD / LEDs │
│                                   │                                    │
│                                   ▼                                    │
│                  Session-Bounded Telemetry Poller                      │
│                  (Optical Lead-Off & Silence Surveillance)             │
└───────────────────────────────────┼────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                STREAMLIT & FLASK TELEMEDICINE INTERFACE                │
│                                                                        │
│   • Patient Explorer   • Clinical AI (SHAP)   • CXR & Grad-CAM         │
│   • Model Comparison   • Model Evaluation     • Live Hardware Vitals   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Slide 5: Dataset & Preprocessing Pipeline
- **Primary Source:** De-identified MIMIC-IV Clinical Database (v2.2) and MIMIC-CXR.
- **Cohort Composition:**
  - Total Stays: 2,884 across 2,661 unique patients.
  - Overall Sepsis Prevalence: 16.50% (476 septic admissions).
- **Partitioning Strategy (Patient-Grouped Stratified K-Fold):**
  - **Training Set (`ids_train.csv`):** 2,282 stays, 2,108 unique subjects (372 septic, 16.30%).
  - **Validation Sub-Split (`val_sub`):** 454 stays, 422 unique subjects (77 septic, 16.96%).
  - **Locked Test Set (`ids_test.csv`):** 602 stays, 553 unique subjects (104 septic, 17.28%).
  - **Zero Leakage:** Strictly 0 shared patients between train, validation, and test splits.
- **Preprocessing Safeguards:**
  - Continuous missing values imputed with training-set medians exclusively for model tensors.
  - Missingness indicator flags (`_measured`) recorded as informative features.
  - Continuous features normalized via `StandardScaler` fitted strictly on training data (`scaler.pkl`).

---

## Slide 6: Deep Learning Architectures
1. **Clinical Feature Tokenizer Transformer (FT-Transformer):**
   - Maps 49 continuous/categorical features into 64-dimensional feature tokens.
   - Prepends a learnable `[CLS]` token $\to$ passed through 2 Transformer encoder layers (4 attention heads, $d_{\text{ff}} = 128$, Pre-LN, GELU, Dropout 0.2).
   - Generates contextualized clinical representation $z_{\text{clin}} \in \mathbb{R}^{64}$.
2. **CXR Visual Representation (ResNet-18):**
   - Input: $224 \times 224$ normalized pulmonary radiograph.
   - Backbone: Pretrained ResNet-18 extracting 512-dimensional pooled convolutional feature vector $f_{512}$.
   - Projection Head: Linear projection + LayerNorm mapping $f_{512} \to z_{\text{img}} \in \mathbb{R}^{64}$.
3. **Multimodal Fusion Architectures:**
   - **Model A (Concatenation Fusion):** Concatenates $[z_{\text{clin}} \,\|\, z_{\text{img}}] \in \mathbb{R}^{128} \to \text{Linear}(128, 64) \to \text{LayerNorm} \to \text{Dropout} \to \text{Linear}(64, 1)$.
   - **Model B (Cross-Attention Fusion):** Dense visual token attends across all 50 clinical feature tokens via Multi-Head Cross-Attention, extracting cross-modal interactions.

---

## Slide 7: Probability Calibration & Operating Thresholds
- **The Problem of Confidence Misalignment:**
  - Raw neural network outputs produce compressed or overconfident probabilities.
  - An uncalibrated cutoff ($\tau = 0.50$ or $0.60$) causes severe under-calling (sensitivity $< 2\%$) because the base sepsis rate is only $\sim 17\%$.
- **Platt Scaling:**
  - Logistic transformation fitted on uncalibrated validation logits: $P_{\text{cal}} = \sigma(a \cdot z + b)$.
  - Fitted parameters on `val_sub`:
    - Clinical Transformer: $a = 0.6075, b = -1.5500$.
    - Model A Concat: $a = 0.6263, b = -1.5294$.
- **Operating Policy:**
  - Shifts decision threshold to prevalence-aligned operating points ($\tau_{\text{Trans}} = 0.14 - 0.21, \tau_{\text{ModA}} = 0.18$).
  - Results in Brier score reduction from $0.23$ to $0.13$ and recovers sensitivity to $66\% - 79\%$.

---

## Slide 8: Edge IoT Hardware Architecture (ESP32 + MAX30102)
- **Hardware Stack:**
  - Microcontroller: ESP32 Dev Module (Tensilica Xtensa Dual-Core 32-bit, 240 MHz).
  - Sensor: MAX30102 High-Sensitivity Pulse Oximeter & Heart Rate IC.
  - Peripherals: 16x2 I²C LCD (Address `0x27`), Tri-color Alert LEDs (Red GPIO 4, Yellow GPIO 16, Green GPIO 17).
- **Embedded Safeguards (Firmware):**
  - *Optical Lead-Off Gate:* $ppgIR < 10,000$ counts triggers `STATE_NO_FINGER`, clears stale vitals, and lights Yellow LED.
  - *Physiological Bound Filtering:* Validates $HR \in [35, 220]\,\text{BPM}$ and $SpO_2 \in [70, 100]\%$.
  - *HR Spike Rejection:* Sudden jumps $> 20\,\text{BPM}$ require 2-cycle candidate confirmation before baseline adoption.
  - *Adaptive Smoothing:* 70/30 smoothing with 40/60 dynamic tracking during acute desaturations.
- **Architectural Isolation Invariant:**
  - Ambulatory hardware telemetry streams into dedicated prospective monitoring sessions.
  - It is **strictly segregated** from historical hospital ICU EHR databases and cannot falsely trigger the 49-biomarker hospital model.

---

## Slide 9: Explainability Framework (SHAP & Grad-CAM)
1. **Local Kernel SHAP (Tabular Biomarkers):**
   - Approximates Shapley values $\phi_i$ across all 49 clinical biomarkers for any selected patient.
   - Quantifies exact positive (risk-elevating) and negative (protective) contributions to the log-odds of sepsis.
   - Displays top-5 influential biomarkers with directional force bars on the dashboard.
2. **Layer-4 Grad-CAM (Chest Radiographs):**
   - Computes gradients of the target class score with respect to ResNet-18 layer-4 convolutional feature maps.
   - Generates spatial activation heatmaps identifying pulmonary opacities, infiltrates, or consolidations linked to sepsis etiology.
3. **qSOFA Clinical Subscale:**
   - Evaluates consensus 2-point audited criteria (Respiratory Rate $\ge 22$, GCS $< 15$).
   - Explicitly refuses to substitute MAP for unmeasured SBP, maintaining clinical integrity.

---

## 10. Slide 10: Quantitative Experimental Results

### Held-Out Test Set Performance ($N = 602$, 104 Positives, Prevalence = 17.28%)

| Model Architecture | Input Modalities | AUROC | AUPRC | Brier Score | Calibrated Threshold ($\tau$) | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | None (*Always Neg*) | 0.5000 | 0.1728 | 0.1728 | — | 0.00% | 100.00% | 0.00% | 0.0000 |
| **qSOFA (Audited 2-Point)\*** | RR, GCS | 0.4501 | 0.1603 | 0.3374 | Score $\ge 2$ | 9.71% | 83.54% | 10.99% | 0.1031 |
| **Logistic Regression (Cal.)**| 49 Labs & Vitals | **0.7162** | **0.3348** | **0.1316** | 0.16 | **72.12%** | **61.45%** | **28.09%** | **0.4043** |
| **Random Forest (Cal.)** | 49 Labs & Vitals | 0.6887 | 0.3314 | 0.1467 | 0.40 | 32.69% | 88.76% | 37.78% | 0.3505 |
| **Clinical Transformer (Platt)**| 49 Labs & Vitals | **0.7069** | **0.3281** | **0.1326** | 0.14 | **78.85%** | **53.41%** | **26.11%** | **0.3923** |
| **Model A Concat Fusion (Platt)**| 49 Labs + CXR Image| **0.7114** | **0.3335** | **0.1325** | 0.18 | **66.35%** | **67.87%** | **30.13%** | **0.4144** |
| **Model B Cross-Attn (Platt)** | 49 Labs + CXR Image| **0.7056** | **0.3245** | **0.1334** | 0.12 | **82.69%** | **44.98%** | **23.89%** | **0.3707** |

*\*Note on Cohorts and Thresholds:*
1. **qSOFA Cohort ($N = 595$):** Full 3-point qSOFA was unavailable due to unextracted SBP. The audited 2-point subscale was evaluated on 595 stays because 7 stays lacked GCS observations in the first 6 hours and were safely excluded rather than imputed. Machine learning models were evaluated on the complete $N = 602$ test cohort.
2. **Threshold Provenance:** Reported ML thresholds ($\tau_{\text{Trans}}=0.14, \tau_{\text{ModA}}=0.18$) are fixed benchmark cutoffs selected on validation data to achieve $\ge 65\%$ screening sensitivity. Live production serving uses validation F1-maximizing thresholds ($\tau_{\text{Trans}}=0.21, \tau_{\text{ModA}}=0.18$); neither are clinically validated.

*Key Result:* Deep multimodal models dramatically outperform consensus clinical scoring (qSOFA AUROC 0.45 $\to$ Multimodal AUROC 0.71) while providing clinically actionable screening sensitivity ($66\% - 83\%$).

---

## Slide 11: Limitations & Critical Disclosures
1. **The 87% Accuracy Constraint:**
   - Under class imbalance ($17.28\%$ sepsis prevalence, 104/602), an all-negative trivial model achieves $82.72\%$ baseline accuracy with $0\%$ sensitivity.
   - Reaching $\ge 87\%$ accuracy mathematically requires sensitivity of at least $24.77\%$ ($0.1728 \times \text{Sens} + 0.8272 \ge 0.87$) even under hypothetical perfect $100\%$ specificity (zero false positives).
   - In empirical validation threshold sweeps, maximizing accuracy pushed decision thresholds up ($\tau \ge 0.75 - 0.84$), reaching only $83.5\% - 84.1\%$ peak accuracy while collapsing sensitivity down to $5.2\% - 9.1\%$ (missing $>90\%$ of septic patients).
   - In clinical screening, prioritizing high sensitivity ($\ge 65\%$) and well-calibrated risk probabilities (Brier score $\approx 0.13$) at $\tau \in [0.14, 0.18]$ provides clinically actionable surveillance, whereas uncalibrated accuracy chasing generates unsafe false negatives.
2. **Current Implementation Disclosures:**
   - **Hardware Telemetry Streaming:** Code-integrated, firmware flashed, and ingestion-gated. Live streaming requires physical Wi-Fi transmission during the demo; when idle, it displays safe silence surveillance.
   - **Hourly Longitudinal Prediction:** Phase 7 preparatory; safely guarded with `DataRequirementError` awaiting authorized full-cohort BigQuery longitudinal export.
   - **Retrospective Data Scope:** Evaluated on retrospective MIMIC-IV data; not validated across external healthcare institutions.

---

## Slide 12: Future Work & Conclusion
- **Future Enhancements:**
  - External multi-center clinical validation across diverse hospital cohorts.
  - Ingestion of full longitudinal hourly Parquet streams upon completion of institutional cloud authorization.
  - Low-power Bluetooth Low Energy (BLE) integration directly from ESP32 to mobile clinical tablets.
- **Summary Conclusion:**
  - XMT-Sepsis delivers an end-to-end, explainable, multimodal decision-support platform for early sepsis surveillance.
  - Successfully bridges raw ICU laboratory data, pulmonary chest imaging, edge IoT hardware acquisition, and local feature attribution into a unified telemedicine system.
