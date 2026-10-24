# XMT-Sepsis-System: Final Academic Project Report Outline

**Degree:** Bachelor of Technology / Master of Technology in Computer Science & Engineering (Data Science)  
**Document Type:** Formal Project Report Structure & Detailed Chapter Blueprint  
**System Name:** Explainable Multimodal Transformer for Sepsis Disease Prediction in a Proactive Telemedicine Framework (XMT-Sepsis)  
**Location:** `submission/PROJECT_REPORT_OUTLINE.md`  

---

## Chapter 1: Introduction
### 1.1 Background & Clinical Motivation
- Pathophysiology of sepsis and septic shock (Sepsis-3 consensus definitions).
- Global disease burden: 48.9 million cases and 11 million sepsis-related deaths annually.
- Importance of the "Golden Hour" in critical care medicine.
- Why clinical scoring tools (qSOFA, SIRS, NEWS) exhibit poor sensitivity in early hospital admission windows.

### 1.2 Problem Statement & Scope
- High dimensional, irregularly sampled electronic health record (EHR) data.
- Isolated diagnostic silos (laboratory blood tests, radiological imaging, ambulatory wearable vitals).
- The opacity of complex black-box machine learning models in high-stakes healthcare settings.
- Prediction Task Definition: Binary classification of sepsis onset within 1–24 hours post-observation ($T = \text{intime} + 6\text{h}$, $H = 24\text{h}$).

### 1.3 Project Objectives & Contributions
- End-to-end multimodal deep learning pipeline incorporating tabular EHR and chest radiographs.
- Rigorous leakage-safe patient-level split partitioning.
- Integration of post-hoc Platt probability calibration.
- Local explainability framework using Kernel SHAP and Layer-4 Grad-CAM.
- Physical bedside edge IoT telemetry prototype with safety guardrails.

---

## Chapter 2: Literature Review & Related Work
### 2.1 Clinical Risk Scoring Systems
- Sequential Organ Failure Assessment (SOFA) and quick SOFA (qSOFA).
- Systemic Inflammatory Response Syndrome (SIRS) criteria.
- Modified Early Warning Score (MEWS) and National Early Warning Score (NEWS).

### 2.2 Machine Learning in Sepsis Prediction
- Classical machine learning approaches: Logistic Regression, Random Forests, Support Vector Machines, XGBoost.
- Recurrent and temporal deep learning models: LSTMs, GRUs, and Temporal Convolutional Networks.
- Limitations of unimodal models relying solely on vital sign trends.

### 2.3 Tabular Deep Learning & Transformers
- Emergence of Tabular Transformers (FT-Transformer, TabNet, SAINT).
- Self-attention mechanisms over numerical and categorical biomarker embeddings.

### 2.4 Multimodal Medical Fusion
- Early (Feature-level) Concatenation vs. Intermediate (Cross-modal attention) Fusion vs. Late (Decision-level) Fusion.
- Vision-Language and Vision-Tabular architectures in critical care informatics.

### 2.5 Explainable Artificial Intelligence (XAI) in Medicine
- Cooperative game theory and Shapley values (SHAP).
- Gradient-weighted Class Activation Mapping (Grad-CAM) for radiological interpretability.
- Regulatory requirements for algorithmic transparency in clinical decision support (EU AI Act, FDA SaMD).

---

## Chapter 3: Dataset & Data Engineering Methodology
### 3.1 Data Sources & Cohort Selection
- MIMIC-IV Clinical Database (v2.2) and MIMIC-CXR.
- Cohort definition criteria: adult ICU admissions ($\ge 18$ years), observation window $[0, 6\text{h}]$, minimum 6h ICU stay.
- Linked admission chest radiographs paired via DICOM metadata.

### 3.2 Feature Specification (49 Tabular Variables)
- **Demographics:** Age, Gender, Admission Type (one-hot encoded).
- **Vital Signs (Summary Statistics):** Heart Rate (mean, min, max, count, last 1h mean), $\text{SpO}_2$ (mean, min, max, count, last 1h mean), Respiratory Rate (mean, max), Mean Arterial Pressure (mean, min), Temperature (max).
- **Neurological Assessment:** Glasgow Coma Scale (min, motor min, verbal min, eye min).
- **Laboratory Panels:** Creatinine (mean, max, count), Bilirubin (max, count), Platelets (min), White Blood Cell count (mean, max, count), Lactate (max, mean, count), $\text{PaO}_2$ (min).
- **Informative Missingness Flags:** 7 binary indicators signaling whether diagnostic panels were drawn.

### 3.3 Leakage-Safe Partitioning Strategy
- Patient-grouped stratified partitioning on `subject_id`:
  - Training Partition (`ids_train.csv`): 2,282 stays (2,108 unique patients).
  - Validation Split (`val_sub`): 454 stays (422 unique patients, Stratified Group K-Fold, $K=5$, seed=42).
  - Locked Test Partition (`ids_test.csv`): 602 stays (553 unique patients).
- Mathematical proof of 0% patient overlap across all three splits.

### 3.4 Preprocessing & Normalization
- Train-only median imputation: medians computed strictly on training data; missingness preserved via binary flags.
- Continuous feature standardization: `StandardScaler` fitted strictly on `train_sub` ($N = 1,828$).

---

## Chapter 4: System Architecture & Implementation
### 4.1 Unimodal Deep Models
- **Clinical Transformer (FT-Transformer):**
  - Linear feature tokenization mapping 49 features $\to$ tokens $\in \mathbb{R}^{64}$.
  - Prepending `[CLS]` token $\to$ 2 Transformer layers (4 attention heads, $d_{\text{ff}} = 128$).
  - Contextualized clinical embedding $z_{\text{clin}} \in \mathbb{R}^{64}$.
- **CXR Encoder (ResNet-18):**
  - Image preprocessing: $224 \times 224 \times 3$, ImageNet normalization.
  - Convolutional backbone extracting pooled 512-d feature vector $f_{512}$.
  - Projection head mapping $f_{512} \to z_{\text{img}} \in \mathbb{R}^{64}$.

### 4.2 Multimodal Fusion Models
- **Model A (Concatenation Fusion):**
  - Concatenates $[z_{\text{clin}} \,\|\, z_{\text{img}}] \in \mathbb{R}^{128}$.
  - Fusion MLP: $\text{Linear}(128, 64) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.2) \to \text{Linear}(64, 1)$.
- **Model B (Cross-Attention Fusion):**
  - Visual token attends across 50 clinical feature tokens via Multi-Head Cross-Attention.
  - Residual connection and joint classification head.

### 4.3 Probability Calibration Pipeline
- Post-hoc Platt Scaling: $P_{\text{cal}} = \sigma(a \cdot z + b)$.
- Fitting via maximum likelihood on uncalibrated validation logits (`val_sub`).
- Prevalence realignment from uncalibrated thresholds down to $\tau \in [0.12, 0.21]$.

### 4.4 Edge IoT Hardware & Firmware Implementation
- Hardware: ESP32 Dev Module + MAX30102 pulse oximeter + 16x2 I²C LCD + Tri-color LEDs.
- Firmware State Machine:
  - `STATE_NO_FINGER`: $ppgIR < 10,000$ counts (optical lead-off, yellow LED).
  - `STATE_SAMPLING` / `STATE_NORMAL` / `STATE_WARNING` / `STATE_HIGH_RISK`.
- Noise rejection: 20 BPM HR spike verification filter, adaptive smoothing, and bounded ThingSpeak upload (20s rate limit).

### 4.5 Full-Stack Telemedicine Software Architecture
- Flask REST API (Port 5000): Authentication (HMAC-SHA256), patient search, inference endpoints, session telemetry ingestion.
- Streamlit Presentation Dashboard (Port 8502): 17 dedicated pages across Executive, Software Research, and Hardware Monitoring workspaces.

---

## Chapter 5: Explainability & Safety Integration
### 5.1 Local Feature Attribution with Kernel SHAP
- Implementation of `ClinicalTransformerSHAPExplainer`.
- Generation of individualized Shapley force plots identifying top-5 risk drivers.

### 5.2 Radiological Interpretability with Grad-CAM
- Gradient-weighted class activation mapping on ResNet-18 layer 4.
- Generation of high-resolution pulmonary activation overlays.

### 5.3 Automated qSOFA Clinical Auditing
- Real-time calculation of RR $\ge 22$ and GCS $< 15$.
- Rigid refusal to substitute MAP for SBP.

---

## Chapter 6: Experimental Results & Comparative Evaluation
### 6.1 Baseline Comparisons on Held-Out Test Set ($N = 602$)
- Tabulation of Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUROC, AUPRC, Brier Score, and ECE across fixed benchmark thresholds.
- Benchmark: Majority-Class Baseline vs. qSOFA vs. Logistic Regression vs. Random Forest vs. FT-Transformer vs. Model A vs. Model B.
- Cohort Distinction: Documentation of ML models evaluated on the complete $N = 602$ locked test cohort, contrasting with the audited 2-point qSOFA subscale evaluated on $N = 595$ stays (7 stays excluded due to unmeasured GCS in the 6-hour admission window; not imputed).
- Threshold Provenance: Explicit separation of fixed benchmark evaluation thresholds ($\tau_{\text{Trans}}=0.14, \tau_{\text{ModA}}=0.18$ optimized for sensitivity $\ge 65\%$) from live serving thresholds ($\tau_{\text{Trans}}=0.21, \tau_{\text{ModA}}=0.18$ optimized for validation F1-score).

### 6.2 Analysis of Decision Thresholds & The Accuracy Trade-Off
- Mathematical derivation of accuracy bounds under severe class imbalance ($17.28\%$ sepsis prevalence, $82.72\%$ negative baseline).
- Mathematical demonstration that achieving $\ge 87\%$ accuracy requires sensitivity $\ge 24.77\%$ ($0.1728 \times \text{Sens} + 0.8272 \ge 0.87$) even under hypothetical perfect $100\%$ specificity.
- Empirical threshold-sweep analysis demonstrating that attempting to maximize overall accuracy pushes cutoffs to $\tau \ge 0.75 - 0.84$, reaching peak accuracy of only $83.5\% - 84.1\%$ while severely suppressing sensitivity down to $5.2\% - 9.1\%$ (missing $>90\%$ of septic stays).
- Formal justification for prevalence-aligned screening operating points ($\tau \approx 0.14 - 0.21$) that guarantee actionable screening sensitivity ($66\% - 83\%$) and calibrated probabilities (Brier score $\approx 0.13$) over artificial accuracy maximization.

### 6.3 In-Sample vs. Out-of-Sample Calibration Audit
- Examination of Brier score reductions ($0.23 \to 0.13$).
- Clear disclosure of validation in-sample calibration vs. held-out test evaluation.

---

## Chapter 7: Software Testing & Integration Verification
### 7.1 Test Suite Summary
- Unit, regression, and integration testing across 40+ automated test suites:
  - `test_hourly_inference.py`, `test_hourly_adapter.py`, `test_hourly_feature_extractor.py` (31 tests passed).
  - `test_retrospective_adapter.py` (9 tests passed).
  - Telemetry gate and patient session security tests.

### 7.2 Security & Robustness Audits
- Path traversal protection on medical image serving (`os.path.normpath`).
- Exemption of test-set ground-truth labels from clinical inference payloads.
- Telemetry silence monitoring ($> 10\,\text{min}$ latency warning).

---

## Chapter 8: Ethical, Legal & Social Implications
### 8.1 Software as a Medical Device (SaMD) Classification
- FDA 21 CFR Part 820 / EU MDR compliance considerations.
- Clear positioning as a Class II / non-device clinical decision support prototype.

### 8.2 Bias, Fairness & Cohort Representation
- Demographic distribution analysis in MIMIC-IV.
- Risks of algorithmic bias across underrepresented demographic subgroups.

### 8.3 Data Privacy & Confidentiality
- HIPAA Safe Harbor de-identification compliance.
- Absence of real patient names or plaintext identifiers.

---

## Chapter 9: Conclusions & Future Work
### 9.1 Summary of Findings
- Multimodal deep learning achieves superior discriminative power over heuristic scores.
- Calibration and threshold tuning are indispensable for imbalanced clinical workflows.
- Hardware-software decoupling guarantees safety in telemedicine architectures.

### 9.2 Limitations
- Evaluated on retrospective single-center data.
- Live hardware streaming requires active local Wi-Fi transmission.
- Full longitudinal hourly sequence model remains preparatory pending cloud export.

### 9.3 Future Directions
- Multi-center prospective clinical trials.
- Direct Bluetooth Low Energy (BLE) edge streaming to clinician mobile devices.
- Integration of free-text clinical nursing notes using ClinicalBERT.
