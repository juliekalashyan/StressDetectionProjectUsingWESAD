# WESAD Stress Detection — Assessment of Human Stress Level Using Machine Learning Techniques

A web application that assesses human **stress levels** using **multiple
machine learning techniques** on wearable sensor data from the
[WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
(Wearable Stress and Affect Detection) dataset.
Upload a subject's `.pkl` file through a browser-based UI and the system will
classify every time window as baseline, stress, amusement, or meditation —
and compare the performance of six different ML classifiers.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [ML Classifiers](#ml-classifiers)
4. [Prerequisites](#prerequisites)
5. [Setup & Installation](#setup--installation)
6. [Running the Application](#running-the-application)
7. [Usage Guide](#usage-guide)
8. [REST API](#rest-api)
9. [Project Structure](#project-structure)
10. [How It Works – Full Pipeline](#how-it-works--full-pipeline)
11. [Feature Extraction](#feature-extraction)
12. [Sensor Channels](#sensor-channels)
13. [WESAD Labels](#wesad-labels)
14. [Training a Model from the Command Line](#training-a-model-from-the-command-line)
15. [Running the Tests](#running-the-tests)
16. [Troubleshooting](#troubleshooting)
17. [Dataset Attribution](#dataset-attribution)

---

## Overview

The **WESAD** dataset contains physiological signals collected from 15
subjects wearing two devices:

* **RespiBAN (chest)** — ECG, EMG, EDA, temperature, respiration, and
  3-axis accelerometer (all sampled at **700 Hz**).
* **Empatica E4 (wrist)** — BVP (64 Hz), EDA (4 Hz), temperature (4 Hz),
  and 3-axis accelerometer (32 Hz).

Each subject went through a controlled protocol with phases of *baseline*,
*stress* (Trier Social Stress Test), *amusement* (funny videos), and
*meditation*.  The data for each subject is stored in a single Python pickle
file (e.g. `S2.pkl`).

This project provides a **Flask web application** that:

1. Reads a subject's `.pkl` file.
2. Aligns all sensor channels to 700 Hz.
3. Splits the time-series into 5-second windows.
4. Computes **154 features** per window (time-domain + frequency-domain).
5. Classifies each window using one of **6 ML classifiers**.
6. Displays the results — including stress level, confusion matrix, and
   feature importance — in the browser.

---

## Features

| Feature | Description |
|---------|-------------|
| **File upload** | Upload a WESAD `.pkl` file; the system extracts 14 sensor channels, computes window-level features, and classifies each window. |
| **Manual input** | Enter average sensor readings by hand for a quick single-window prediction. |
| **Auto-train** | If no pre-trained model exists for the uploaded subject, a classifier is trained on-the-fly and saved for future use. |
| **6-Model comparison** | Upload a `.pkl` and compare all 6 classifiers side-by-side with accuracy, F1, precision, recall, confusion matrices, and radar charts. |
| **Pre-train CLI** | Use `train_model.py` to train and evaluate any classifier offline. Use `--compare` to benchmark all 6. |
| **Results dashboard** | Stress verdict, stress gauge, per-label distribution, timeline, confusion matrix heatmap, and top-15 feature importance chart. |
| **REST API** | JSON endpoints for programmatic predictions (`/api/predict`) and model listing (`/api/models`). |

---

## ML Classifiers

The system supports six machine learning techniques, all from scikit-learn:

| # | Classifier | Key Hyperparameters |
|---|-----------|-------------------|
| 1 | **Random Forest** | 300 trees, max depth 25, balanced class weights |
| 2 | **Support Vector Machine (SVM)** | RBF kernel, C=10, balanced class weights |
| 3 | **K-Nearest Neighbours (KNN)** | k=5, distance-weighted |
| 4 | **Decision Tree** | Max depth 20, balanced class weights |
| 5 | **Gradient Boosting** | 200 estimators, max depth 5, learning rate 0.1 |
| 6 | **Logistic Regression** | Multinomial, L-BFGS solver, balanced class weights |

All classifiers are evaluated with the same train/test split and optional
stratified k-fold cross-validation.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| **Python** | 3.9 or newer |
| **pip** | latest recommended |
| **OS** | Windows, macOS, or Linux |

> **Visual Studio Code users:** open the project folder in VS Code, then use
> the integrated terminal (**Ctrl + `` ` ``**) to run all commands below. It
> is recommended to create a virtual environment first (VS Code will detect it
> automatically).

### Obtaining the WESAD dataset

The dataset is **not** included in this repository. Download it from the
official source:

<https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>

After downloading and extracting, you will find one `.pkl` file per subject
inside folders named `S2/`, `S3/`, … `S17/` (e.g. `WESAD/S2/S2.pkl`).

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/juliekalashyan/stress-detection.git
cd stress-detection

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (cmd):
venv\Scripts\activate.bat
# On macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

The server starts at **<http://127.0.0.1:5000>**.  Open that URL in your
browser.

> To enable debug mode (auto-reload on code changes), set the environment
> variable before running:
>
> ```bash
> # Linux / macOS
> export FLASK_DEBUG=1
> # Windows PowerShell
> $env:FLASK_DEBUG="1"
> ```

---

## Usage Guide

### Option 1 — Upload a `.pkl` file

1. Open <http://127.0.0.1:5000> in your browser.
2. In the **Upload WESAD Subject File** section, click "Choose File" and
   select a WESAD subject pickle file (e.g. `S2.pkl`).
3. Click **Upload & Analyse**.
4. Wait while the system:
   * extracts chest and wrist sensor data,
   * computes 154 window-level features (time-domain + frequency-domain),
   * trains a model (if one hasn't been trained yet) or loads an existing
     model,
   * classifies every 5-second window.
5. View the **Results** page showing:
   * **Overall verdict** — Stress Detected / No Significant Stress.
   * **Stress gauge** — animated percentage of stress windows.
   * **Label distribution table and bar chart.**
   * **Confusion matrix heatmap** (when trained on-the-fly).
   * **Top 15 feature importance chart** (when trained on-the-fly).

### Option 2 — Compare 6 ML classifiers

1. In the **Upload** section, select your `.pkl` file.
2. Click **Compare 6 ML Models** instead of "Upload & Analyse".
3. The system trains all six classifiers on the same data and shows:
   * **Ranked comparison table** (accuracy, F1, precision, recall, CV, time).
   * **Accuracy bar chart**.
   * **Multi-metric radar chart**.
   * **Confusion matrix** for each classifier.

### Option 3 — Manual sensor input

1. First, make sure a model for the subject has already been trained (either
   by uploading that subject's `.pkl` file once, or by running
   `train_model.py`).
2. In the **Manual Sensor Input** section, select a model from the dropdown.
3. Fill in average sensor values for all 14 channels (Chest and Wrist).
4. Click **Predict Stress Level** to see the single-window prediction.

---

## REST API

The application provides JSON endpoints for programmatic access.

### `GET /api/models`

Returns a list of available trained models.

```json
{ "models": ["S2", "S5", "S11", "S13", "S14", "S15", "S16", "S17"] }
```

### `POST /api/predict`

Predict stress from sensor values.

**Request body:**
```json
{
    "subject_id": "S2",
    "sensors": {
        "Chest Acc X": 0.1,
        "Chest Acc Y": 0.2,
        "Chest Acc Z": 9.8,
        "Chest ECG": 0.5,
        "Chest EMG": 0.01,
        "Chest EDA": 2.5,
        "Chest Temp": 34.0,
        "Chest Resp": 0.3,
        "Wrist Acc X": 0.1,
        "Wrist Acc Y": 0.1,
        "Wrist Acc Z": 9.8,
        "Wrist BVP": 50.0,
        "Wrist EDA": 1.2,
        "Wrist Temp": 33.0
    }
}
```

**Response:**
```json
{
    "subject_id": "S2",
    "predictions": [1],
    "label_names": ["Baseline"],
    "is_stressed": [false],
    "stress_ratio": 0.0,
    "stress_level": "Low Stress"
}
```

---

## Project Structure

```
stress-detection/
├── app.py                   # Flask web application (routes, API & logic)
├── train_model.py           # CLI script to pre-train / compare models
├── model/                   # Core ML pipeline
│   ├── __init__.py
│   ├── data_processor.py    # Reads WESAD .pkl files, extracts & aligns sensors
│   ├── feature_extractor.py # Sliding-window feature computation (time + freq domain)
│   ├── predictor.py         # Loads saved models & runs predictions
│   └── trainer.py           # Training logic with 6 classifiers & comparison
├── static/                  # Single-page app front-end
│   ├── index.html           # HTML shell — all rendering done client-side
│   ├── app.js               # SPA logic — fetch API, render results & charts
│   └── style.css            # UI styling (glassmorphism dark theme)
├── trained_models/          # Saved .joblib model & scaler files
├── tests/
│   └── test_app.py          # Unit, integration & API tests (27 tests)
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

### Key files explained

| File | What it does |
|------|-------------|
| `app.py` | Defines Flask routes: `/` (serves SPA), `/api/config` (feature & model metadata), `/api/upload` (file upload + prediction), `/api/compare` (6-model comparison), `/api/predict` (manual sensor input). On upload it loads the `.pkl`, extracts features, trains or loads a model, and returns JSON results. |
| `static/index.html` | HTML shell for the single-page application. All dynamic rendering is handled by `app.js`. |
| `static/app.js` | Client-side SPA logic — fetches `/api/config`, builds sensor input forms, renders result & comparison pages with Chart.js charts, gauge animations, and tab navigation. |
| `static/style.css` | Glassmorphism dark theme with responsive layout, animated orbs, and accessibility support (`prefers-reduced-motion`, `focus-visible`). |
| `model/data_processor.py` | `load_subject(pkl_path)` opens the `.pkl` file, validates its structure, extracts 8 chest channels and 6 wrist channels, resamples wrist signals to 700 Hz via linear interpolation, concatenates everything into a (samples × 14) matrix, filters out non-standard labels, and returns the feature matrix together with the label array. |
| `model/feature_extractor.py` | `extract_windows(features, labels)` splits the 700 Hz time-series into non-overlapping 5-second windows (3,500 samples each) and computes **8 time-domain statistics** (mean, std, min, max, median, IQR, skewness, kurtosis) and **3 frequency-domain statistics** (dominant frequency, spectral energy, spectral entropy) per channel → 14 × 11 = **154 features** per window. |
| `model/trainer.py` | `train_model()` trains any of the 6 supported classifiers. `compare_classifiers()` trains all 6 and returns results sorted by accuracy. Returns metrics including confusion matrix, feature importances, F1 scores, and training time. |
| `model/predictor.py` | `predict(model, scaler, X)` scales the features, runs the classifier, and returns predictions, human-readable label names, per-window stress flags, and the overall stress ratio. |
| `train_model.py` | Standalone script with `--classifier` flag to pick any of the 6 classifiers and `--compare` flag to benchmark all of them. |

---

## How It Works – Full Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  WESAD .pkl  │────▸│ data_processor.py │────▸│ 14-channel matrix │
│  (subject)   │     │  extract chest    │     │  (n_samples × 14) │
└──────────────┘     │  extract wrist    │     │  + labels array   │
                     │  align to 700 Hz  │     └─────────┬─────────┘
                     └──────────────────┘               │
                                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │         feature_extractor.py                 │
                     │  Split into 5-second windows (3,500 samples) │
                     │  Time-domain: 8 statistics per channel       │
                     │  Freq-domain: 3 statistics per channel (FFT) │
                     │  → 154 features per window                   │
                     │  Majority-vote label per window               │
                     └─────────────────────┬────────────────────────┘
                                           │
                                           ▼
                     ┌──────────────────────────────────────────────┐
                     │         trainer.py / predictor.py            │
                     │  StandardScaler → Classifier → predictions   │
                     │  6 classifiers: RF, SVM, KNN, DT, GB, LR    │
                     │  Compute stress_ratio = #stress / #windows   │
                     └─────────────────────┬────────────────────────┘
                                           │
                                           ▼
                     ┌──────────────────────────────────────────────┐
                     │               Results Page                   │
                     │  Verdict: Stressed / Not Stressed            │
                     │  Gauge, distribution, confusion matrix       │
                     │  Feature importance, timeline                │
                     └──────────────────────────────────────────────┘
```

### Step-by-step detail

1. **Load `.pkl`** — `pickle.load()` reads the WESAD file, which contains a
   Python dictionary with `signal → chest / wrist` sensor arrays and a
   `label` array.

2. **Extract chest data** — 8 channels at 700 Hz are flattened directly:
   Acc X/Y/Z, ECG, EMG, EDA, Temp, Resp.

3. **Extract wrist data** — 6 channels originally sampled at 4–64 Hz are
   resampled to 700 Hz using linear interpolation (`np.interp`).

4. **Align** — chest and wrist DataFrames (now the same length) are
   concatenated into a single 14-column matrix.  Samples with non-standard
   labels (outside 0–4) are filtered out.

5. **Window** — the matrix is split into non-overlapping windows of 5 seconds
   (5 × 700 = 3,500 rows each).

6. **Feature computation** — for each window and each of the 14 channels:
   * **Time-domain (8 per channel):** mean, standard deviation, minimum,
     maximum, median, IQR, skewness, kurtosis.
   * **Frequency-domain (3 per channel):** dominant frequency, spectral
     energy, spectral entropy (computed via FFT).
   * Total: 14 × (8 + 3) = **154 features per window**.

7. **Label assignment** — each window gets the majority non-transient label
   (transient label 0 is ignored during voting).

8. **Classification** — one of 6 classifiers (default: Random Forest) is
   trained on 80% of the windows and evaluated on 20%.  The model is saved for
   future use.  For comparison, all 6 classifiers can be trained and ranked.

9. **Result** — if more than 30% of windows are classified as *Stress*
   (label 2), the overall verdict is **"Stress Detected"**.

---

## Feature Extraction

Each 5-second window produces **154 features** (per-channel statistics):

### Time-domain features (8 × 14 channels = 112)

| Statistic | Description |
|-----------|-------------|
| Mean | Average signal value in the window |
| Std | Standard deviation |
| Min | Minimum value |
| Max | Maximum value |
| Median | Median value |
| IQR | Interquartile range (Q3 − Q1) |
| Skewness | Asymmetry of the distribution |
| Kurtosis | Tailedness of the distribution |

### Frequency-domain features (3 × 14 channels = 42)

| Statistic | Description |
|-----------|-------------|
| Dominant frequency | Frequency with highest FFT magnitude |
| Spectral energy | Sum of squared FFT magnitudes |
| Spectral entropy | Shannon entropy of the power spectral density |

---

## Sensor Channels

The 14 input features correspond to these sensor signals:

| # | Column Name | Sensor | Device | Native Rate | Unit | Typical Range |
|---|-------------|--------|--------|-------------|------|---------------|
| 1 | Chest Acc X | Accelerometer X-axis | RespiBAN (chest) | 700 Hz | g | −3 to +3 |
| 2 | Chest Acc Y | Accelerometer Y-axis | RespiBAN (chest) | 700 Hz | g | −3 to +3 |
| 3 | Chest Acc Z | Accelerometer Z-axis | RespiBAN (chest) | 700 Hz | g | −3 to +3 |
| 4 | Chest ECG | Electrocardiogram | RespiBAN (chest) | 700 Hz | mV | −1 to +2 |
| 5 | Chest EMG | Electromyogram | RespiBAN (chest) | 700 Hz | mV | −0.5 to +0.5 |
| 6 | Chest EDA | Electrodermal Activity | RespiBAN (chest) | 700 Hz | µS | 0 to 40 |
| 7 | Chest Temp | Skin Temperature | RespiBAN (chest) | 700 Hz | °C | 30 to 40 |
| 8 | Chest Resp | Respiration | RespiBAN (chest) | 700 Hz | a.u. | −3 to +3 |
| 9 | Wrist Acc X | Accelerometer X-axis | Empatica E4 (wrist) | 32 Hz → 700 Hz | g | −3 to +3 |
| 10 | Wrist Acc Y | Accelerometer Y-axis | Empatica E4 (wrist) | 32 Hz → 700 Hz | g | −3 to +3 |
| 11 | Wrist Acc Z | Accelerometer Z-axis | Empatica E4 (wrist) | 32 Hz → 700 Hz | g | −3 to +3 |
| 12 | Wrist BVP | Blood Volume Pulse | Empatica E4 (wrist) | 64 Hz → 700 Hz | a.u. | −200 to +200 |
| 13 | Wrist EDA | Electrodermal Activity | Empatica E4 (wrist) | 4 Hz → 700 Hz | µS | 0 to 5 |
| 14 | Wrist Temp | Skin Temperature | Empatica E4 (wrist) | 4 Hz → 700 Hz | °C | 28 to 40 |

---

## WESAD Labels

Each time point (and each derived window) is labelled with one of:

| Label | State | Description |
|-------|-------|-------------|
| 0 | Transient | Transition period between conditions (ignored during training) |
| 1 | Baseline | Neutral, relaxed state |
| 2 | **Stress** | Trier Social Stress Test (public speaking + mental arithmetic) |
| 3 | Amusement | Watching funny video clips |
| 4 | Meditation | Guided meditation session |

---

## Training a Model from the Command Line

You can pre-train a model before starting the web app:

```bash
# Train with default Random Forest
python train_model.py path/to/WESAD/S2/S2.pkl --subject-id S2

# Train with a specific classifier
python train_model.py path/to/WESAD/S2/S2.pkl --classifier "SVM"

# Compare all 6 classifiers and print a ranking table
python train_model.py path/to/WESAD/S2/S2.pkl --compare
```

This will:

1. Load and process the subject's data.
2. Extract 5-second window features (154 features per window).
3. Train the selected classifier(s) with an 80/20 train/test split.
4. Print accuracy, F1, precision, recall, CV scores, and a classification report.
5. Save the model and scaler to `trained_models/`.

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--subject-id` | Inferred from filename | Override the subject ID |
| `--window-sec` | 5 | Window size in seconds |
| `--classifier` | Random Forest | One of: Random Forest, SVM, KNN, Decision Tree, Gradient Boosting, Logistic Regression |
| `--compare` | off | Train all 6 classifiers and print a ranked comparison table |

---

## Running the Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_app.py::test_flask_index -v
```

The test suite includes **27 tests**:

| # | Test | What it checks |
|---|------|---------------|
| 1 | `test_extract_chest_data_shape` | Chest data extraction produces (n, 8) DataFrame |
| 2 | `test_compute_window_features` | Window feature vector has correct length (154) |
| 3 | `test_extract_windows_basic` | Window segmentation produces correct number of windows |
| 4 | `test_features_from_manual_input` | Manual input builds a valid (1, 154) feature vector |
| 5 | `test_features_from_manual_input_means_only` | Means-only manual input builds a valid (1, 14) feature vector |
| 6 | `test_feature_names_length` | `FEATURE_NAMES` list matches 154 features |
| 7 | `test_label_map_contains_stress` | Label map correctly maps 2 → "Stress" |
| 8 | `test_train_model_returns_metrics` | `train_model()` returns dict with accuracy, f1, confusion_matrix, etc. |
| 9 | `test_compare_classifiers_returns_sorted` | `compare_classifiers()` returns list sorted by accuracy descending |
| 10 | `test_train_model_svm` | Training with `classifier_name="SVM"` succeeds and returns metrics |
| 11 | `test_train_general_model` | General model training on synthetic multi-subject data succeeds |
| 12 | `test_flask_index` | Home page loads (HTTP 200) |
| 13 | `test_api_config` | `/api/config` returns correct feature columns, labels, and sensor metadata |
| 14 | `test_api_upload_no_file` | Upload without file shows error message |
| 15 | `test_api_upload_wrong_extension` | Upload of non-.pkl file returns 400 |
| 16 | `test_api_predict_no_json` | `POST /api/predict` without JSON body returns 400 |
| 17 | `test_api_predict_missing_subject` | `POST /api/predict` without `subject_id` uses general model (200/503) |
| 18 | `test_api_predict_unknown_model` | `POST /api/predict` with unknown subject returns 404 |
| 19 | `test_api_models` | `GET /api/models` returns JSON list of trained models |
| 20 | `test_end_to_end_upload` | Full upload flow with synthetic data returns results |
| 21 | `test_end_to_end_compare` | Full compare flow with synthetic data returns ranked 6-classifier results |
| 22 | `test_stress_level_info_file_upload_mode` | Stress level thresholds work for multi-window ratios |
| 23 | `test_stress_level_info_manual_mode` | Stress level thresholds work for manual probability values |
| 24 | `test_build_result_dict_manual_baseline_forces_low_stress` | Baseline prediction in manual mode forces "Low Stress" |
| 25 | `test_build_result_dict_manual_stress_is_stressed` | Stress prediction in manual mode sets `overall_stress=True` |
| 26 | `test_upload_malformed_pkl_missing_signal` | `.pkl` without 'signal' key returns error |
| 27 | `test_upload_malformed_pkl_not_dict` | `.pkl` containing non-dict returns error |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `Address already in use` (port 5000) | Another process is using port 5000. Stop it or run `python app.py` with a different port by editing `app.py`. |
| `Error processing file: ...` after upload | Make sure you are uploading a valid WESAD `.pkl` file (e.g. `S2.pkl` from the WESAD dataset). |
| `No trained model found for subject` (manual input) | Upload the subject's `.pkl` file first (or run `train_model.py`) to create the model. |
| Slow first upload | The first time a subject file is uploaded, the model is trained on-the-fly. This may take 10–30 seconds depending on your machine. Subsequent uploads for the same subject will be faster. |

---

## Dataset Attribution

This project uses the **WESAD** dataset:

> Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and
> Kristof Van Laerhoven. 2018. **Introducing WESAD, a Multimodal Dataset for
> Wearable Stress and Affect Detection.** In *Proceedings of the 20th ACM
> International Conference on Multimodal Interaction (ICMI '18).*
>
> <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>