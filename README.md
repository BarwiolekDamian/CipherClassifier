# CipherClassifier: Cipher Family Classification Using Statistical Features and Random Forest

## Overview
This project investigates whether statistical features of 1024-byte ciphertext samples (Shannon entropy, NIST SP 800-22 Monobit test, Index of Coincidence, and autocorrelation) combined with a Random Forest classifier can distinguish block and stream cipher families (AES, DES, 3DES, RC4) under a Ciphertext-Only Attack (COA) scenario, and where the limits of this methodology lie with respect to round count.

## Repository Structure

```text
CipherClassifier/
├── data/
│   ├── raw/                 # Natural, low-entropy plaintexts (e.g., natural language texts)
│   ├── ciphertexts/         # Generated fixed-size 1024 B ciphertext samples (.npz / raw binary)
│   └── features.csv         # Extracted statistical feature vectors with class labels
├── models/                  # Serialized machine learning models and scalers (.joblib)
├── figures/                 # Evaluation figures (Confusion Matrix, ROC curves, Feature Importance)
├── ciphers_reduced.py       # Educational pure Python/NumPy implementation of reduced-round ciphers
├── generate_data.py         # Phase 1: Dataset generation (20,000 samples + adversarial set)
├── extract_features.py      # Phase 2: Statistical feature extraction pipeline
├── train_model.py           # Phase 3 & 4: Baselines, MVP binary classifier, 4-class RF tuning & COA test
├── predict.py               # Phase 5: Production CLI inference script with latency and uncertainty monitoring
├── requirements.txt         # Pinned project dependencies
└── README.md                # Project documentation and architecture overview
```

## Module Descriptions

- **`ciphers_reduced.py`**: Implements educational reduced-round versions of AES (1–2 rounds), DES (2 rounds), and 3DES (2–3 rounds) in ECB mode without external high-level round restrictions.
- **`generate_data.py`**: Generates balanced 1024-byte ciphertext datasets (5,000 samples per class across 4 families) and full-round control samples for adversarial evaluation.
- **`extract_features.py`**: Computes mandatory and optional statistical features from raw byte arrays (entropy, NIST frequency test, Index of Coincidence, autocorrelations, Chi-square, Runs test).
- **`train_model.py`**: Handles stratified data splitting, trains baseline models (Logistic Regression, Decision Tree), trains binary MVP and multiclass Random Forest models with cross-validation, and performs the Adversarial COA test.
- **`predict.py`**: CLI tool for SOC environments that extracts features on the fly, predicts cipher families, measures latency in milliseconds, and flags low-confidence samples (`max(P) < 0.50`) for L2 analyst review.