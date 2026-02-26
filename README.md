# WESAD Stress Detection

A web application that detects whether a person is experiencing **stress**
using wearable sensor data from the
[WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
(Wearable Stress and Affect Detection) dataset.
Upload a subject's `.pkl` file through a browser-based UI and the system will
tell you whether that person was stressed, relaxed, or amused.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Setup & Installation](#setup--installation)
5. [Running the Application](#running-the-application)
6. [Usage Guide](#usage-guide)
7. [Project Structure](#project-structure)
8. [How It Works – Full Pipeline](#how-it-works--full-pipeline)
9. [Sensor Channels](#sensor-channels)
10. [WESAD Labels](#wesad-labels)
11. [Training a Model from the Command Line](#training-a-model-from-the-command-line)
12. [Running the Tests](#running-the-tests)
13. [Troubleshooting](#troubleshooting)
14. [Dataset Attribution](#dataset-attribution)

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
4. Computes statistical features per window.
5. Classifies each window using a **Random Forest** model.
6. Displays the results — including whether the person was stressed — in the
   browser.

---

## Features

| Feature | Description |
|---------|-------------|
| **File upload** | Upload a WESAD `.pkl` file; the system extracts 14 sensor channels, computes window-level features, and classifies each window. |
| **Manual input** | Enter average sensor readings by hand for a quick single-window prediction. |
| **Auto-train** | If no pre-trained model exists for the uploaded subject, a Random Forest is trained on-the-fly and saved for future use. |
| **Pre-train CLI** | Use `train_model.py` to train and evaluate a model offline before starting the web app. |
| **Results dashboard** | See the overall stress verdict, per-label distribution, stress ratio, and a bar chart — all in the browser. |

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
   * computes 56 window-level features,
   * trains a model (if one hasn't been trained yet) or loads an existing
     model,
   * classifies every 5-second window.
5. View the **Results** page showing:
   * **Overall verdict** — Stress Detected / No Significant Stress.
   * **Stress ratio** — percentage of windows classified as Stress.
   * **Label distribution table and bar chart.**

### Option 2 — Manual sensor input

1. First, make sure a model for the subject has already been trained (either
   by uploading that subject's `.pkl` file once, or by running
   `train_model.py`).
2. In the **Manual Sensor Input** section, enter the Subject ID (e.g. `S2`).
3. Fill in average sensor values for all 14 channels (Chest and Wrist).
4. Click **Predict Stress** to see the single-window prediction.

---

## Project Structure

```
stress-detection/
├── app.py                   # Flask web application (routes & logic)
├── train_model.py           # CLI script to pre-train a model offline
├── model/                   # Core ML pipeline
│   ├── __init__.py
│   ├── data_processor.py    # Reads WESAD .pkl files, extracts & aligns sensors
│   ├── feature_extractor.py # Sliding-window feature computation
│   └── predictor.py         # Loads saved models & runs predictions
├── templates/               # Jinja2 HTML templates
│   ├── index.html           # Home page — upload form & manual input
│   └── result.html          # Results page — verdict, stats, bar chart
├── static/
│   └── style.css            # UI styling
├── trained_models/          # Saved .joblib model & scaler files (git-ignored)
├── tests/
│   └── test_app.py          # Unit & integration tests
├── requirements.txt         # Python package dependencies
├── .gitignore
├── diplom                   # Original Colab notebook (reference only)
└── README.md                # This file
```

### Key files explained

| File | What it does |
|------|-------------|
| `app.py` | Defines three Flask routes: `/` (home page), `/upload` (file upload + prediction), `/manual` (manual input prediction). On upload it loads the `.pkl` file, extracts features, trains or loads a model, and renders the results page. |
| `model/data_processor.py` | `load_subject(pkl_path)` opens the `.pkl` file, extracts 8 chest channels and 6 wrist channels, up-samples the wrist signals to 700 Hz, concatenates everything into a (samples × 14) matrix and returns it together with the label array. |
| `model/feature_extractor.py` | `extract_windows(features, labels)` splits the 700 Hz time-series into non-overlapping 5-second windows (3 500 samples each) and computes **mean, std, min, max** for every channel — producing 14 × 4 = **56 features** per window. A majority-vote assigns one label to each window. |
| `model/predictor.py` | `predict(model, scaler, X)` scales the features, runs the Random Forest, and returns predictions, human-readable label names, per-window stress flags, and the overall stress ratio. |
| `train_model.py` | Standalone script. Loads a subject, extracts windows, trains a `RandomForestClassifier` (200 trees, max depth 20, balanced class weights), prints a classification report, and saves the model + scaler to `trained_models/`. |

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
                     │  Split into 5-second windows (3 500 samples) │
                     │  Compute mean, std, min, max per channel     │
                     │  → 56 features per window                    │
                     │  Majority-vote label per window.              │
                     └─────────────────────┬────────────────────────┘
                                           │
                                           ▼
                     ┌──────────────────────────────────────────────┐
                     │              predictor.py                    │
                     │  StandardScaler → RandomForest → predictions │
                     │  Compute stress_ratio = #stress / #windows   │
                     └─────────────────────┬────────────────────────┘
                                           │
                                           ▼
                     ┌──────────────────────────────────────────────┐
                     │               Results Page                   │
                     │  Verdict: Stressed / Not Stressed            │
                     │  Label distribution table & bar chart        │
                     └──────────────────────────────────────────────┘
```

### Step-by-step detail

1. **Load `.pkl`** — `pickle.load()` reads the WESAD file, which contains a
   Python dictionary with `signal → chest / wrist` sensor arrays and a
   `label` array.

2. **Extract chest data** — 8 channels at 700 Hz are flattened directly:
   Acc X/Y/Z, ECG, EMG, EDA, Temp, Resp.

3. **Extract wrist data** — 6 channels originally sampled at 4–64 Hz are
   up-sampled to 700 Hz by inserting NaN values between real samples, then
   NaNs are filled with zeros.

4. **Align** — both DataFrames are truncated to the shorter length and
   concatenated into a single 14-column matrix.

5. **Window** — the matrix is split into non-overlapping windows of 5 seconds
   (5 × 700 = 3,500 rows each).

6. **Feature computation** — for each window and each of the 14 channels,
   four statistics are computed: **mean**, **standard deviation**,
   **minimum**, and **maximum** → 14 × 4 = **56 features**.

7. **Label assignment** — each window gets the majority non-transient label
   (transient label 0 is ignored during voting).

8. **Classification** — a `RandomForestClassifier` (scikit-learn) is used.
   If no saved model exists, one is trained on 80% of the windows and saved.
   The model then predicts labels for all windows.

9. **Result** — if more than 30% of windows are classified as *Stress*
   (label 2), the overall verdict is **"Stress Detected"**.

---

## Sensor Channels

The 14 input features correspond to these sensor signals:

| # | Column Name | Sensor | Device | Native Rate |
|---|-------------|--------|--------|-------------|
| 1 | Chest Acc X | Accelerometer X-axis | RespiBAN (chest) | 700 Hz |
| 2 | Chest Acc Y | Accelerometer Y-axis | RespiBAN (chest) | 700 Hz |
| 3 | Chest Acc Z | Accelerometer Z-axis | RespiBAN (chest) | 700 Hz |
| 4 | Chest ECG | Electrocardiogram | RespiBAN (chest) | 700 Hz |
| 5 | Chest EMG | Electromyogram | RespiBAN (chest) | 700 Hz |
| 6 | Chest EDA | Electrodermal Activity | RespiBAN (chest) | 700 Hz |
| 7 | Chest Temp | Skin Temperature | RespiBAN (chest) | 700 Hz |
| 8 | Chest Resp | Respiration | RespiBAN (chest) | 700 Hz |
| 9 | Wrist Acc X | Accelerometer X-axis | Empatica E4 (wrist) | 32 Hz → 700 Hz |
| 10 | Wrist Acc Y | Accelerometer Y-axis | Empatica E4 (wrist) | 32 Hz → 700 Hz |
| 11 | Wrist Acc Z | Accelerometer Z-axis | Empatica E4 (wrist) | 32 Hz → 700 Hz |
| 12 | Wrist BVP | Blood Volume Pulse | Empatica E4 (wrist) | 64 Hz → 700 Hz |
| 13 | Wrist EDA | Electrodermal Activity | Empatica E4 (wrist) | 4 Hz → 700 Hz |
| 14 | Wrist Temp | Skin Temperature | Empatica E4 (wrist) | 4 Hz → 700 Hz |

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
python train_model.py path/to/WESAD/S2/S2.pkl --subject-id S2
```

This will:

1. Load and process the subject's data.
2. Extract 5-second window features.
3. Train a Random Forest with an 80/20 train/test split.
4. Print a classification report (precision, recall, F1-score).
5. Save the model and scaler to `trained_models/model_S2.joblib` and
   `trained_models/scaler_S2.joblib`.

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--subject-id` | Inferred from filename | Override the subject ID |
| `--window-sec` | 5 | Window size in seconds |

---

## Running the Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_app.py::test_flask_index -v
```

The test suite includes:

| Test | What it checks |
|------|---------------|
| `test_extract_chest_data_shape` | Chest data extraction produces (n, 8) DataFrame |
| `test_compute_window_features` | Window feature vector has correct shape and values |
| `test_extract_windows_basic` | Window segmentation produces correct number of windows |
| `test_features_from_manual_input` | Manual input builds a valid (1, 56) feature vector |
| `test_label_map_contains_stress` | Label map correctly maps 2 → "Stress" |
| `test_flask_index` | Home page loads (HTTP 200) |
| `test_flask_upload_no_file` | Upload without file shows error message |
| `test_flask_manual_no_model` | Manual predict without model shows error |
| `test_end_to_end_upload` | Full upload flow with synthetic data returns results |

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