# Gravity-Normalized Tremor Classification in Parkinson's Disease Using Wearable IMU

## Overview

This repository implements a wearable IMU-based pipeline for Parkinson's disease tremor severity classification using gravity-normalized feature extraction and SHAP explainability. The core methodological contribution — gravity normalization as a pre-processing step — is adapted from subject-independent activity recognition (Paper 1) and applied here to pathological tremor grading.

The pipeline separates raw accelerometer signals into gravitational and dynamic components using a low-pass Butterworth filter (cutoff 0.3 Hz, 4th order), then extracts time-domain, frequency-domain, and tremor-band features from each component independently. An SVM with RBF kernel and class-balanced weighting is trained and evaluated using leave-one-subject-out (LOSO) cross-validation.

## Results

| Metric | Value |
|--------|-------|
| Validation | Stratified 5-Fold |
| Classifier | SVM RBF (C=10, gamma=scale) |
| Overall Accuracy | 90.91% |
| Macro F1 | 0.4307  |
| Features | 95 gravity-normalized features per window |

Note: Results depend on the specific dataset split used. Run 03_classification_loso.ipynb to reproduce.

## Key Finding

Gravity-normalized dynamic acceleration features dominate SHAP attribution for tremor severity classification, consistent with the mechanistic hypothesis that pathological tremor is encoded in the dynamic (non-gravitational) acceleration component. This extends findings from subject-independent HAR (where gravity components dominated) to a pathological tremor context where the opposite holds — the dynamic residual is the clinically relevant signal.

## Dataset

- Source: Parkinson's Disease Tremor IMU Dataset
- Repository: https://github.com/jiehu01/Parkinson-s-Disease-Tremor-Dataset
- Subjects: 34 PD patients, medication on/off conditions
- Sensor: Accelerometer (wrist/hand placement)
- Sampling frequency: 50 Hz
- Window length: 128 samples (2.56 seconds), non-overlapping
- Labels: Tremor severity 0 (none) to 3 (severe), MDS-UPDRS scale
- Format: X.npy [N, 128, 3], Y.npy [N]

## Pipeline

```
Raw IMU (3-axis, 50 Hz)
        ↓
Gravity Normalization (Butterworth LPF 0.3 Hz)
        ↓
[Gravity Component]    [Dynamic Component]    [Raw Signal]
        ↓                      ↓                    ↓
Time + Frequency Domain Features (10 per axis per component)
        ↓
Feature Matrix [N × 95]
        ↓
StandardScaler → SVM RBF (class-balanced)
        ↓
LOSO / 5-Fold Validation
        ↓
SHAP KernelExplainer Attribution
```

## Repository Structure

```
.
├── data/
│   ├── X.npy                  # Raw IMU windows [N, 128, 3]
│   ├── Y.npy                  # Severity labels [N]
│   ├── subject_ids.npy        # Subject IDs for LOSO (if available)
│   ├── X_features.npy         # Extracted feature matrix [N, 95]
│   └── feature_names.npy      # Feature name strings
├── figures/                   # All output figures (auto-generated)
├── 01_data_exploration.ipynb
├── 02_feature_extraction.ipynb
├── 03_classification_loso.ipynb
├── 04_shap_analysis.ipynb
├── requirements.txt
└── README.md
```

## How to Reproduce

```bash
git clone https://github.com/Kritika-bme/parkinson-tremor-classification
cd parkinson-tremor-classification

pip install -r requirements.txt

mkdir -p data figures
```

Download X.npy and Y.npy from https://github.com/jiehu01/Parkinson-s-Disease-Tremor-Dataset and place in `data/`. If subject_ids.npy is available, place it there too for LOSO validation; otherwise the pipeline falls back to stratified 5-fold.

Run notebooks sequentially:
1. 01_data_exploration.ipynb
2. 02_feature_extraction.ipynb
3. 03_classification_loso.ipynb
4. 04_shap_analysis.ipynb

All figures save automatically to `figures/`.

## Limitations

- Severe class imbalance (Score 0 dominates at ~91.5% in some dataset splits). Addressed via class-balanced SVM weighting; macro F1 is the primary reported metric.
- SHAP computation via KernelExplainer is approximation-based (nsamples=100 per window). Exact Shapley values would require exponentially more compute.
- Results are dataset-specific. Generalization to other PD tremor datasets requires re-validation.

## Author

Kritika Patidar
Department of Biomedical Engineering, SGSITS Indore
BS Data Science & Applications, Indian Institute of Technology Madras
SMARTIE Tech, Nirmaan Pre-Incubator, IIT Madras
kritika.bme@gmail.com
