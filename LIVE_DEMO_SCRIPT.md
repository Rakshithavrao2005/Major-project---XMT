# XMT-Sepsis-System: 5–7 Minute Live Demonstration Script

**System URL:** `http://127.0.0.1:8502` (Streamlit Dashboard)  
**Backend API:** `http://127.0.0.1:5000` (Flask REST API)  
**Target Duration:** 5 to 7 Minutes  
**Audience:** Academic Reviewers / Faculty Examiners / Project Evaluation Committee  
**Location:** `submission/LIVE_DEMO_SCRIPT.md`  

---

## Pre-Demo Checklist (30 Seconds Before Presentation)
1. Verify both daemons are running in the background:
   - Flask on Port 5000 (`http://127.0.0.1:5000/api/health` -> HTTP 200).
   - Streamlit on Port 8502 (`http://127.0.0.1:8502` -> HTTP 200).
2. Open Google Chrome or Microsoft Edge and navigate to `http://127.0.0.1:8502`.
3. Ensure the browser window is maximized (1080p recommended).

---

## Demonstration Timeline & Step-by-Step Script

### Minute 0:00 – 1:00 | Introduction & Executive Architecture (Page: Home)
- **Action:** Open the **Home** page on the sidebar.
- **Visual Focus:** Hospital hero presentation card, capability cards, and 4-step workflow banner.
- **Narrative:**
  > "Respected panel, welcome to the demonstration of **XMT-Sepsis** — an Explainable Multimodal Transformer for early sepsis disease prediction in a proactive telemedicine framework.
  > Sepsis accounts for over 11 million deaths globally each year, and early detection in the first hours of hospital admission is critical.
  > Today, hospital records, radiological imaging, and bedside IoT monitors remain isolated. Our system brings them together into an end-to-end clinical decision-support platform.
  > As shown on the Home screen, our 4-step workflow integrates:
  > 1. Edge IoT Bedside Acquisition,
  > 2. Multimodal Transformer Fusion,
  > 3. Transparent Explainable AI using SHAP and Grad-CAM, and
  > 4. Proactive Clinician Decision Support.
  > Notice also our permanent research disclaimer at the top and bottom of every screen: this is an experimental research prototype evaluated on retrospective ICU data."

---

### Minute 1:00 – 2:15 | Cohort Exploration & Unmeasured Data Integrity (Page: Patient Explorer)
- **Action:** Click **Patient Explorer** on the left sidebar navigation.
- **Visual Focus:** Total patient metrics (2,661 unique patients, 2,884 stays), filter controls, and the patient directory table.
- **Narrative:**
  > "Let's first explore our clinical cohort. On the **Patient Explorer** page, we have 2,661 unique ICU patients from the MIMIC-IV database, representing 2,884 admissions.
  > We have enforced strict patient-level separation: 2,108 patients in the training set and 553 patients in the locked test set, with zero patient overlap between them.
  > We can filter by split or multi-stay admissions.
  > Let's look at **Patient 10002930** (Admission Stay `35629889`).
  > Notice how our clinical summaries display vital signs and laboratory panels. A vital clinical feature of our system is that unmeasured laboratory tests — such as arterial blood gases or lactate — are explicitly flagged as 'Unmeasured' rather than falsely imputed as normal in the clinician interface. This prevents dangerous clinical blind spots."

---

### Minute 2:15 – 3:45 | Clinical AI & Local SHAP Attribution (Page: Clinical AI)
- **Action:** Click **Clinical AI** on the left sidebar.
- **Visual Focus:** Patient dropdown (select Patient `10002930`), Model Inference Card, Calibrated Probability Gauge, and SHAP Feature Attribution Chart.
- **Narrative:**
  > "Now, let's observe our deep tabular AI in action.
  > On the **Clinical AI** page, our Feature Tokenizer Transformer (FT-Transformer) evaluates 49 physiological biomarkers gathered during the first 6 hours of ICU admission.
  > When we select Patient `10002930`, the backend runs an immediate local forward pass.
  > Here, the model generates an uncalibrated raw probability, which is then mapped through our validation-fitted Platt calibrator.
  > The Platt-calibrated sepsis probability is **14.26%**, which falls below our validation-selected production threshold of **21%** (optimized on validation F1-score), categorizing this stay as **Standard Risk**. Note that this operating cutoff is a research configuration chosen to balance precision and recall on development data, not a certified clinical cutoff.
  > But critically, we don't just output a black-box probability.
  > Scrolling down to our **Local Explainability** section, our automated Kernel SHAP engine calculates exact Shapley attributions for this individual patient.
  > We can see the top biomarkers driving the prediction: elevated Heart Rate and Respiratory Rate push the risk upwards, while normal White Blood Cell count and preserved Glasgow Coma Scale pull the risk downwards.
  > This provides immediate, interpretable justification for attending clinicians."

---

### Minute 3:45 – 4:45 | Multimodal Fusion & Grad-CAM Heatmaps (Page: Chest X-ray & Model Fusion)
- **Action:** Click **Chest X-ray** on the sidebar, inspect Grad-CAM, then click **Model Fusion**.
- **Visual Focus:** Admission chest radiograph, Layer-4 Grad-CAM pulmonary heatmap overlay, and Model A vs Model B comparison.
- **Narrative:**
  > "Sepsis frequently arises from pulmonary etiologies like severe pneumonia.
  > Navigating to the **Chest X-ray** page, we see the patient's admission radiograph linked strictly via their authoritative DICOM identifier.
  > Our ResNet-18 convolutional encoder processes the radiograph and extracts visual representations.
  > By computing gradients across Layer 4, our system generates an overlaid **Grad-CAM activation heatmap**. This visually highlights bilateral lung fields where pulmonary opacities or infiltrates are concentrated.
  > Next, on the **Model Fusion** page, we examine our joint multimodal models:
  > **Model A (Concatenation Fusion)** fuses the clinical transformer embedding and visual embedding through a joint MLP head.
  > **Model B (Cross-Attention Fusion)** allows a dense visual query to attend directly across all clinical biomarker tokens.
  > On our held-out test cohort, Model A achieves an AUROC of **0.7114**, providing a substantial improvement over unimodal imaging alone."

---

### Minute 4:45 – 5:45 | Authoritative Benchmark Evaluation (Page: Model Performance)
- **Action:** Click **Model Evaluation** on the sidebar.
- **Visual Focus:** Test set ROC curves, PR curves, and comparison table against qSOFA.
- **Narrative:**
  > "Let us look at authoritative quantitative validation on our locked test set of 602 patients.
  > In critical care, consensus clinical heuristic scores like **qSOFA** achieve an AUROC of only **0.4501** on early admission data because they miss subtle non-hypotensive sepsis. (Note that 7 of the 602 test stays had unobserved GCS and were safely excluded from the 2-point subscale evaluation, yielding 595 evaluated stays).
  > In contrast, our **Clinical Transformer** achieves an AUROC of **0.7069**, and **Multimodal Model A** achieves **0.7114** with an AUPRC of **0.3335**.
  > Furthermore, our Platt probability calibration reduces the Brier score from **0.23 down to 0.13**, proving that the predicted probabilities faithfully reflect empirical risk.
  > For our benchmark reporting, we evaluate models at fixed validation-selected operating thresholds targeting $\ge 65\%$ sensitivity ($\tau = 0.14$ for Transformer, $\tau = 0.18$ for Model A), ensuring high-risk septic cases are flagged. In contrast to a trivial baseline that predicts all cases negative to get 82.72% accuracy while missing all septic patients, our screening operating points deliver clinically actionable sensitivity of 66% to 79%."

---

### Minute 5:45 – 6:45 | Edge IoT Hardware & Bedside Telemetry (Page: Hardware Overview & Live Monitoring)
- **Action:** Click **Hardware Overview**, then navigate to **Live Monitoring**.
- **Visual Focus:** ESP32 board schematic, LED state diagram, and session-scoped telemetry panel.
- **Narrative:**
  > "Finally, let us examine our edge IoT architecture.
  > On **Hardware Overview**, we have integrated an ESP32 microcontroller paired with a MAX30102 pulse oximeter and 16x2 bedside LCD.
  > The firmware running on the ESP32 implements:
  > - An optical lead-off detector ($IR < 10,000$ counts) that resets vitals when a finger is removed, lighting a Yellow LED and displaying 'CHECK SENSOR'.
  > - A 20 BPM sudden spike filter that rejects erratic motion artifacts.
  > - Tri-color status LEDs: Green for stable vitals, Yellow for sampling, and Red for critical tachycardia and hypoxia.
  > On the **Live Monitoring** page, wearable telemetry streams into authenticated, prospective patient monitoring sessions.
  > Crucially, our architecture enforces **Strict Cohort Isolation**: wearable telemetry vitals stream into dedicated ambulatory sessions and are **never mixed or merged** with retrospective hospital ICU database records.
  > As you see on the dashboard, when the physical hardware is in standby mode, our telemetry silence monitor triggers a prominent `SILENCE_WARNING` after 10 minutes, transparently reporting data as *unavailable* rather than fabricating artificial live vitals."

---

### Minute 6:45 – 7:15 | Conclusion & Panel Q&A Handover
- **Action:** Click back to **About XMT** or **Home**.
- **Narrative:**
  > "To conclude: XMT-Sepsis demonstrates a complete, transparent, and leakage-safe engineering pipeline for early sepsis surveillance, bridging tabular deep learning, medical computer vision, edge IoT embedded sensing, and local explainability.
  > The system is running live locally on our server without any external cloud dependencies.
  > Thank you, and we are now ready for the panel's questions."

---

## Panel Q&A Quick Reference Cheat Sheet

| Likely Panel Question | Authoritative Verified Response |
| :--- | :--- |
| **Why is accuracy only ~65% instead of 90%+?** | In an imbalanced dataset with 17.28% sepsis prevalence, predicting 'All Negative' achieves an 82.72% accuracy baseline with 0% sensitivity. Mathematically, reaching 87.0% accuracy requires at least 24.77% sensitivity even if specificity were a perfect 100% ($\frac{0.87 - 0.8272}{0.1728} = 24.77\%$), and higher sensitivity if false positives occur. Documented validation threshold sweeps in `phase7_experiment_results.json` show that pushing thresholds upward to maximize accuracy achieved 83.5%–84.1% accuracy only by suppressing sensitivity down to 5.2%–9.1%. In clinical practice, high sensitivity ($\ge 65\%$) is prioritized for early warning. |
| **Why not use XGBoost instead of a Transformer?** | We conducted controlled experiments in Phase 7 with scikit-learn's `HistGradientBoosting`. The decision trees severely overfitted (0.9899 Training AUROC $\to$ 0.6409 Validation AUROC), whereas regularized FT-Transformers and linear models generalized far more reliably. |
| **How did you prevent data leakage?** | We enforced patient-grouped stratified splits on `subject_id` (0% overlap across train, validation, and test). All scaling (`scaler.pkl`) was fitted strictly on `train_sub` ($N=1,828$). The observation window was locked at $T = 6\text{h}$, and the test set was locked throughout model development. |
| **Is the ESP32 streaming live right now?** | The ESP32 firmware is compiled and flashed on COM4, and the backend telemetry gate is fully implemented. When the physical board is idle or not connected to local Wi-Fi, the system safely marks telemetry as *unavailable* (`SILENCE_WARNING`) rather than generating false synthetic signals. |
| **Is this system clinically validated?** | No. This is strictly an academic research prototype evaluated on retrospective de-identified MIMIC-IV data. It requires multi-center prospective validation and regulatory clearance (FDA SaMD / CE mark) before any diagnostic use. |
