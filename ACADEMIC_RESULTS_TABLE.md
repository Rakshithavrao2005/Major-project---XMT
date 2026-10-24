# XMT-Sepsis-System: Academic Results & Benchmark Tables

**Document Type:** Formal Experimental Benchmark & Results Reference  
**Project Root:** `C:\Users\Varshini\.gemini\antigravity\scratch\XMT-Sepsis-System`  
**Location:** `submission/ACADEMIC_RESULTS_TABLE.md`  

---

## 1. Primary Benchmark: Stay-Level Static Prediction Task ($T = 6\text{h}, H = 24\text{h}$)

### 1.1 Task Definition
- **Observation Window:** First 6 hours of ICU admission ($[0, 6\text{h}]$).
- **Prediction Target:** Sepsis-3 onset occurring within 1–24 hours post-observation window ($T$ to $T + 24\text{h}$).
- **Evaluation Split:** Authoritative held-out test set `ids_test.csv` ($N = 602$, 553 unique patients, zero overlap with training cohort).
- **Class Balance:** Negatives = 498 ($82.72\%$), Positives = 104 ($17.28\%$).

---

### 1.2 Comprehensive Benchmark Performance Matrix

| Model Architecture | Input Modality / Features | Operating Cutoff ($\tau$) | Confusion Matrix ($TP / FP / TN / FN$) | Accuracy | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score | AUROC (95% CI) | AUPRC (95% CI) | Brier Score | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | None (*Always Neg*) | — | $0 / 0 / 498 / 104$ | **82.72%** | 0.00% | 100.00% | 0.00% | 0.0000 | 0.5000 | 0.1728 | 0.1728 | — |
| **qSOFA (Audited 2-Point)** | RR, GCS | Score $\ge 2$ | $10 / 81 / 411 / 93$ | **70.76%** | 9.71% | 83.54% | 10.99% | 0.1031 | 0.4501 (0.40–0.51) | 0.1603 (0.13–0.19) | 0.3374 | — |
| **Logistic Regression (Cal.)**| 49 Labs & Vitals | 0.16 | $75 / 192 / 306 / 29$ | **63.29%** | **72.12%** | **61.45%** | **28.09%** | **0.4043** | **0.7162** (0.66–0.77) | **0.3348** (0.26–0.43) | **0.1316** | 0.0229 |
| **Random Forest (Cal.)** | 49 Labs & Vitals | 0.40 | $34 / 56 / 442 / 70$ | **79.07%** | 32.69% | 88.76% | 37.78% | 0.3505 | 0.6887 (0.63–0.75) | 0.3314 (0.26–0.43) | 0.1467 | 0.1024 |
| **CXR Encoder (ResNet-18)** | Radiograph Only | 0.10 | $104 / 498 / 0 / 0$ | **17.28%** | 100.00% | 0.00% | 17.28% | 0.2946 | 0.6131 (0.56–0.67) | 0.2180 (0.17–0.28) | 0.2670 | 0.3528 |
| **Clinical Transformer (Raw)** | 49 Labs & Vitals | 0.60 | $58 / 119 / 379 / 46$ | **72.59%** | 55.77% | 76.10% | 32.77% | 0.4128 | 0.7069 (0.65–0.76) | 0.3281 (0.26–0.42) | 0.2226 | 0.2881 |
| **Clinical Transformer (Platt)**| 49 Labs & Vitals | **0.14** | $82 / 232 / 266 / 22$ | **57.81%** | **78.85%** | **53.41%** | **26.11%** | **0.3923** | **0.7069** (0.65–0.76) | **0.3281** (0.26–0.42) | **0.1326** | **0.0326** |
| **Model A Concat Fusion (Raw)**| 49 Labs + CXR | 0.50 | $70 / 161 / 337 / 34$ | **67.61%** | 67.31% | 67.67% | 30.30% | 0.4179 | 0.7114 (0.66–0.77) | 0.3335 (0.27–0.43) | 0.2171 | 0.2823 |
| **Model A Concat Fusion (Platt)**| 49 Labs + CXR | **0.18** | $69 / 160 / 338 / 35$ | **67.61%** | **66.35%** | **67.87%** | **30.13%** | **0.4144** | **0.7114** (0.66–0.77) | **0.3335** (0.27–0.43) | **0.1325** | **0.0210** |
| **Model B Cross-Attn (Raw)** | 49 Labs + CXR | 0.50 | $66 / 152 / 346 / 38$ | **68.44%** | 63.46% | 69.48% | 30.28% | 0.4112 | 0.7056 (0.65–0.76) | 0.3245 (0.26–0.42) | 0.2118 | 0.2664 |
| **Model B Cross-Attn (Platt)** | 49 Labs + CXR | **0.12** | $86 / 274 / 224 / 18$ | **51.50%** | **82.69%** | **44.98%** | **23.89%** | **0.3707** | **0.7056** (0.65–0.76) | **0.3245** (0.26–0.42) | **0.1334** | **0.0248** |

*Note on Cohort Totals & qSOFA Subscale Exclusion:* Machine learning model confusion matrices strictly sum to the full locked test cohort of 602 stays ($TP+FP+TN+FN = 602$). The qSOFA subscale sums to 595 stays ($10 + 81 + 411 + 93 = 595$) because exactly 7 stays in the test set had unmeasured Glasgow Coma Scale (GCS) records during the 6-hour observation window and were excluded from subscale evaluation per clinical auditing guidelines rather than falsely imputing normal mentation. Full 3-point consensus qSOFA is marked unavailable because Systolic Blood Pressure (SBP) was not extracted in this dataset version.

*Note on Threshold Selection & Production Distinctions:* Operating thresholds reported in the benchmark table above ($\tau_{\text{Trans}} = 0.14$, $\tau_{\text{ModA}} = 0.18$, $\tau_{\text{ModB}} = 0.12$) represent fixed evaluation operating points selected on internal validation data (`val_sub`) targeting clinical screening sensitivity ($\ge 65\%$). These are distinct from the production serving thresholds configured in `patient_service.py` ($\tau_{\text{Trans}} = 0.21$, $\tau_{\text{ModA}} = 0.18$, $\tau_{\text{ModB}} = 0.20$), which were chosen to maximize validation F1-score. Neither set of thresholds constitutes a clinically validated or certified diagnostic cutoff.

---

## 2. Separate Longitudinal Hourly Prediction Task (Status & Specifications)

> [!IMPORTANT]
> **Task Distinction Notice:** The stay-level benchmark above ($T=6\text{h}, H=24\text{h}$) must **NOT** be confused or mixed with the longitudinal hourly sequential prediction task. The hourly task evaluates continuous sliding windows over time.

### 2.1 Hourly Pipeline Specifications
- **Architecture:** Temporal Hourly Transformer (`hourly_transformer_best.pt` in `models/hourly/stage3_4b/`).
- **Input Representation:** Sliding sequential windows of hourly lab measurements and vital sign trends.
- **Hourly Base Prevalence:** $\sim 0.65\%$ per patient-hour (reflecting real-time ICU surveillance).
- **Platt Calibrator:** `calibrator_platt.pkl` ($a = 0.9803, b = -4.9744$).
- **Software Guardrail Status:** **PREPARATORY & GATED.**
  - `data/raw_hourly/` is currently empty awaiting authorized institutional BigQuery full-cohort export.
  - The runtime pipeline cleanly catches data absence via `DataRequirementError` and HTTP 503 (`api_hourly_status`), preventing crashes or hallucinations.
- **Test Suite Verification:** All 31 hourly adapter, feature extraction, and offline inference unit tests (`test_hourly_inference.py`, `test_hourly_adapter.py`, `test_hourly_feature_extractor.py`) are **100% passed** (passing in 23.16 seconds).

---

## 3. Disclosures & System Limitations

1. **Hardware Telemetry Streaming Status (STANDBY):**
   - The ESP32 firmware ([sepsis_risk_monitor.ino](file:///C:/Users/Varshini/.gemini/antigravity/scratch/XMT-Sepsis-System/hardware/sepsis_risk_monitor/sepsis_risk_monitor.ino)) compiles and flashes with exit code 0.
   - The backend ingestion gate and telemetry poller are fully implemented.
   - **Disclosure:** Live continuous telemetry transmission from the physical sensor over local Wi-Fi was not actively transmitting during offline audits; the system transparently reports `telemetry freshness: None` and flags `SILENCE_WARNING`.
2. **The 87% Accuracy Constraint:**
   - Under $17.28\%$ sepsis prevalence, a trivial majority-class ("always negative") baseline achieves **$82.72\%$ accuracy** ($498/602$) with **$0\%$ sensitivity**.
   - To achieve **$87.0\%$ accuracy**, a model mathematically requires at least **$24.77\%$ sensitivity** even under perfect specificity ($100\%$):
     $$\text{Accuracy} = 0.1728 \cdot \text{Sens} + 0.8272 \cdot \text{Spec} \ge 0.87 \implies \text{Sens} \ge \frac{0.87 - 0.8272}{0.1728} = 24.77\%$$
     If specificity drops below $100\%$ (e.g., $95\%$), the required sensitivity rises to $\ge 48.7\%$.
   - Documented validation threshold sweeps in `models/experiments/phase7/phase7_experiment_results.json` demonstrate that artificially elevating decision thresholds to maximize accuracy reached peak validation accuracy of $83.5\% - 84.1\%$ only by severely suppressing sensitivity down to $5.2\% - 9.1\%$, which defeats clinical sepsis surveillance.
   - Consequently, our models deliberately operate at prevalence-aligned screening thresholds ($\tau \approx 0.12 - 0.21$), prioritizing actionable sensitivity ($66\% - 83\%$) over raw accuracy.
3. **Regulatory Status:**
   - **Research prototype decision-support tool only.** Not cleared by FDA, CE, or CDSCO. Must not be used for direct clinical diagnosis or real-time patient treatment.
