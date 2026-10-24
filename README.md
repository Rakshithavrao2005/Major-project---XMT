# XMT-Sepsis-System
## Explainable Multimodal Transformer for Sepsis Disease Prediction in a Proactive Telemedicine Framework

---

## Overview

**XMT-Sepsis-System** is a research-grade clinical decision-support system designed to predict sepsis onset early and accurately by fusing heterogeneous patient data modalities — electronic health records, physiological sensor streams, and medical imaging — through an Explainable Multimodal Transformer (XMT) architecture. The system is embedded inside a proactive telemedicine framework, enabling real-time remote monitoring and timely clinical alerts.

---

## Planned Modules

| Module | Directory | Description |
|---|---|---|
| **Data Preprocessing** | `src/preprocessing/` | Cleaning, normalisation, imputation, and alignment of raw multimodal data |
| **Clinical Data Pipeline** | `src/clinical/` | Processing of EHR / structured lab / vitals data (e.g., MIMIC-IV) |
| **Sensor Data Pipeline** | `src/sensor/` | Ingestion and processing of wearable / IoT physiological streams |
| **Imaging Pipeline** | `src/imaging/` | Loading, segmentation, and feature extraction from medical images (CXR, CT) |
| **Multimodal Fusion** | `src/fusion/` | Cross-modal attention and Transformer-based fusion of all modalities |
| **Evaluation** | `src/evaluation/` | Metrics (AUROC, AUPRC, F1), explainability (SHAP, Grad-CAM), benchmarking |
| **Trained Models** | `models/` | Saved model checkpoints (excluded from Git) |
| **Dashboard** | `dashboard/` | Interactive clinician-facing telemedicine dashboard (web UI) |
| **Hardware Interface** | `hardware/` | Edge device integration scripts for wearable / IoT sensors |
| **Backend API** | `backend/` | REST / WebSocket API layer connecting models to the dashboard |
| **Tests** | `tests/` | Unit and integration tests for all modules |

---

## Data

Raw and processed datasets are stored locally under `data/` and are **excluded from Git** (see `.gitignore`).

- `data/raw/` — Original unmodified source data (e.g., MIMIC-IV exports)
- `data/processed/` — Cleaned, feature-engineered, model-ready splits
- `data/metadata/` — Data dictionaries, cohort definitions, column maps

> ⚠️ **Patient data, credentials, and API keys must never be committed to this repository.**

---

## Getting Started

```bash
# 1. Clone the repository
git clone <repo-url>
cd XMT-Sepsis-System

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install minimal dependencies
pip install -r requirements.txt
```

---

## Repository Structure

```
XMT-Sepsis-System/
├── data/
│   ├── raw/            # Original source data (git-ignored)
│   ├── processed/      # Cleaned / feature-engineered data (git-ignored)
│   └── metadata/       # Data dictionaries and cohort definitions
├── src/
│   ├── preprocessing/  # Data cleaning, normalisation, imputation
│   ├── clinical/       # EHR / structured data pipeline
│   ├── sensor/         # Physiological sensor / IoT pipeline
│   ├── imaging/        # Medical image processing pipeline
│   ├── fusion/         # Multimodal Transformer fusion
│   └── evaluation/     # Metrics, explainability, benchmarks
├── models/             # Model checkpoints (git-ignored)
├── dashboard/          # Telemedicine UI
├── hardware/           # Edge device integration
├── backend/            # REST / WebSocket API
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore
```

---

## Roadmap

- [ ] Phase 1 — Data Preprocessing & EDA
- [ ] Phase 2 — Clinical & Sensor Data Pipelines
- [ ] Phase 3 — Imaging Pipeline
- [ ] Phase 4 — Multimodal Transformer (XMT) Architecture
- [ ] Phase 5 — Explainability (SHAP, Grad-CAM, Attention Visualisation)
- [ ] Phase 6 — Backend API & Telemedicine Dashboard
- [ ] Phase 7 — Hardware / Edge Deployment
- [ ] Phase 8 — Clinical Validation & Benchmarking

---

## License

TBD — To be determined based on institutional and data-use agreement requirements.

---

## Citation

> If you use this system in your research, please cite accordingly (citation details TBD).
