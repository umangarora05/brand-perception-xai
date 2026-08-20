# brand-perception-xai
An explainable machine learning pipeline for automated brand logo auditing, cognitive friction detection, and data-driven design prescription.


# Differential Trust Engine: Context-Adaptive Brand Perception System

[![Patent Status](https://img.shields.io/badge/Patent-Published%20(IN202641096863%20A1)-blue)](https://ipindia.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-red.svg)](https://www.tensorflow.org/)

---

## 📌 Executive Overview

The **Differential Trust Engine** is an explainable machine learning architecture designed to infer consumer trust-propensity directly from brand asset geometry, typography, and color intent. 

Moving beyond generic image classification or subjective focus groups, the system isolates exact points of cognitive friction using a **matched-pair differential pipeline** (r0 → r1), benchmarks chromatic features against a continuous psychographic emotion manifold, and produces bounded, mathematically actionable design prescriptions via Explainable AI (XAI / SHAP).

---

## ⚖️ Intellectual Property & Patent Status

The core algorithms, dynamic signal gating mechanisms, and explainability-driven diagnostic architectures in this repository are protected under the **Indian Patent Office (IPO)**:

- **Title of Invention:** CONTEXT-ADAPTIVE EXPLAINABLE ARTIFICIAL INTELLIGENCE SYSTEM FOR BRAND-LOGO TRUST-PROPENSITY INFERENCE AND AUTOMATED DESIGN PRESCRIPTION
- **Application Number:** 202641096863
- **Publication Number:** IN202641096863 A1
- **Publication Date:** 14-08-2026
- **International Classification (IPC):** G06N 5/04, G06K 9/62, G06N 20/00, G06N 5/02, G06N 3/04
- **Applicant:** Vellore Institute of Technology (VIT)

---

## 🔬 Core System Architecture & Novel Innovations

                                  [ Input Logo Asset ]
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
    [ Multi-Domain Extractor ]                           [ Matched-Pair Differential ]
    (RGB, HSV, Edges, Symmetry,                          (Baseline r0 vs. Revision r1)
     Circularity, Deep Mean)                                          │
                │                                         [ ResNet-50 Embedding Delta ]
                ▼                                         (ΔE = E(r1) - E(r0), ||ΔE||₂)
    [ Emotion-Manifold Mapping ]                                      │
    (3D RGB KDTree Search over                                        │
     745 Psychographic Palettes)                                      │
                │                                                     │
                └──────────────────────────┬──────────────────────────┘
                                           ▼
                            [ Dual-Model Trust Inference ]
                            ├── 60% Differential Engine (13 Features)
                            └── 40% Logo-2K+ Model (167k General Knowledge Base)
                                           │
                                           ▼
                         [ Dynamic Signal Calibration (XAI) ]
                         ├── Achromatic Signal Gate (Saturation S < 0.05)
                         └── Dominant Signal Affirmation (Dominance D > 0.40)
                                           │
                                           ▼
                        [ Actionable Design Prescription ]


### 1. Multi-Domain Feature Extraction (13-Pillar Vector)
The pipeline parses visual assets into a 13-dimensional interference and structural feature matrix:
* **Spectral:** Normalized `red`, `green`, `blue`, and HSV `saturation`.
* **Typographic & Geometric:** `typography_weight` (Canny edge density), `font_circularity` (contour perimeter-to-area ratio), and `harmony_score` (horizontal axis symmetry / SSIM).
* **Interference & Differential:** `occlusion_ratio`, `structural_degradation` (1 - SSIM), `neural_displacement` (||ΔE||₂ using ResNet-50 backbone), and `deep_feature_mean`.
* **Engineered Non-Linear Features:** `color_harmony_interaction` ((B - R) * harmony) and `structure_density` (circularity * (1 - weight)).

### 2. Ground-Truth Emotion Manifold Mapping
To eliminate circular heuristics, target trust labels are generated via a 3-Nearest Neighbor search using a **K-Dimensional Tree (`scipy.spatial.KDTree`)** mapped over 745 human-curated, emotion-tagged brand palettes:
`Trust Target = (Weighted Palette Score * 0.70) + (Proximity Confidence * 0.15) + (Structural Bonus)`
*Psychological Weights:* trust (1.0), reliable (0.8), professional (0.6), sophisticated (0.4), calm (0.3), strong (0.3), classic (0.3).

### 3. Dual-Model Cross-Dataset Ensembling
Final inference incorporates knowledge from two independently trained engines:
1. **Primary Differential Model (60% weight):** Trained on matched typographic pair interventions (r0/r1).
2. **Corroborating General Model (40% weight):** Pre-trained on 167,140 brand logos across 2,341 commercial categories (Logo-2K+ corpus).

### 4. Dynamic Signal Calibration (Patented XAI Layer)
Model-agnostic SHAP values are dynamically re-routed based on operational chromatic conditions:
* **Achromatic Signal Gate:** When saturation S < 0.05, spectral SHAP channels (R, G, B, Sat) are muted to 0.0, directing optimization solely to geometry and stroke weight to avoid irrelevant color advice.
* **Dominant Signal Affirmation:** When a single color channel exhibits a dominance ratio D > 0.40, negative raw SHAP attributions are contextualized into positive brand assets rather than flagged as defects.

---

## 📊 Empirical Validation & Performance Benchmarks

The inference engine was evaluated on an augmented dataset of **25,000 records** (15,000 differential matched-pair audits + 10,000 sampled general-corpus entries) with an 80/20 train-test split:

| Metric / Parameter | Value | Interpretation |
| :--- | :--- | :--- |
| **Coefficient of Determination (R²)** | **0.9599** | **95.99%** of target perceptual variance explained |
| **Root Mean Squared Error (RMSE)** | **0.0160** | Exceptionally low error on normalized [0, 1] scale |
| **Mean Absolute Error (MAE)** | **0.0108** | High precision across held-out test distributions |
| **Training Records** | **25,000** | Balanced, regularized cross-corpus dataset |
| **Held-out Test Subset** | **5,000** | Unseen evaluation pairs |
| **Inference Latency** | **< 3.0 sec** | Real-time diagnostic processing per logo asset |

---

## 🛠️ Tech Stack & Dependencies

```txt
python >= 3.12.0
lightgbm == 4.6.0
tensorflow == 2.19.0
shap >= 0.44.0
scipy >= 1.12.0
scikit-learn >= 1.4.0
scikit-image >= 0.22.0
opencv-python >= 4.9.0
pandas >= 2.2.0
numpy >= 1.26.0
matplotlib >= 3.8.0
seaborn >= 0.13.0



🚀 Quickstart & Usage
1. Ingestion & Audit Pipeline
Python
import joblib
import pandas as pd
from block12_audit import analyze_new_logo

# Load trained models & assets
model_lgb = joblib.load("Differential_Trust_Engine.pkl")
model_167k = joblib.load("logo_behavior_model.pkl")

# Execute end-to-end diagnostic audit on a target asset
analyze_new_logo("sample_logo.png")
2. Sample Diagnostic Output
Plaintext
[sample_logo.png] Initiating Perceptual Ingest...
--- SYSTEM DIAGNOSTIC REPORT ---
Asset Identity:                sample_logo.png
Engine Status:                 Chromatic Mode (Blue Dominant) Active
Differential Model Score:      0.4120  (trained on r0/r1 matched pairs)
167k Corroborating Score:      0.4350  (trained on 167,140 Logo-2K+ images)
Ensemble Trust Coefficient:    0.4212  (final output)

--- RECOMMENDED STRATEGY ---
PRIMARY FRICTION DETECTED: typography_weight (Impact: -0.0418)
STRATEGY: Action: Cognitive Density Overload. Reduce stroke weight to improve clarity.

--- RECOMMENDED STRATEGY ---
PRIMARY FRICTION DETECTED: typography_weight (Impact: -0.0418)
STRATEGY: Action: Cognitive Density Overload. Reduce stroke weight to improve clarity.



📁 Repository Structure
├── models/
│   ├── Differential_Trust_Engine.pkl    # Primary 13-feature LightGBM model
│   ├── logo_behavior_model.pkl          # Corroborating 167k general logo model
│   └── Empirical_Audit_Evidence.pkl     # Archived vector evidence records
├── data/
│   └── emotion_palette.csv              # 745 psychographic palette manifold
├── exhibits/
│   ├── Patent_Exhibit_Factor_Importance.png
│   ├── Patent_Exhibit_Behavioral_Distribution.png
│   └── Patent_Exhibit_Accuracy_Validation.png
├── patent/
│   └── IN202641096863_Specification.pdf # Published IPO patent document
├── differential_trust_engine.py         # Main execution notebook / script
├── requirements.txt
└── README.md



👥 Authors & Acknowledgments
Guide: Dr. Shynu P G – Research Guidance & System Formulation
Umang Arora – Core Architecture, Differential Pipelines, ML Modeling
Khushi – Model Validation & Data Structuring



📄 License
This repository is distributed for academic research and evaluation purposes. Commercial implementation, licensing, and derivative deployments of the core algorithms are subject to patent claims under Application No. 202641096863 A1.



<img width="480" height="575" alt="image" src="https://github.com/user-attachments/assets/51b16060-b172-45ce-a82d-34fa1f99509c" />

