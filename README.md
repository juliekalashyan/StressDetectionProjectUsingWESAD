# WESAD Stress Detection — Full-Stack ML Web Application

**Real-time physiological stress detection** using wearable sensor data from the [WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) dataset.

A production-ready **Flask** backend serves a **single-page vanilla JavaScript** frontend that enables:
- **Instant analysis** of pre-trained subjects (< 50 ms response)
- **File upload** with automated feature extraction and model training
- **Multi-classifier comparison** across 7 ML algorithms with visualizations
- **Manual sensor input** for ad-hoc stress predictions
- **Longitudinal history tracking** with date filtering and CSV export
- **Persistent storage** across server restarts via SQLite
- **Rate limiting** protection and 30+ automated test coverage

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Initial Setup](#initial-setup)
6. [Running the Application](#running-the-application)
7. [Understanding the ML Pipeline](#understanding-the-ml-pipeline)
8. [API Reference](#api-reference)
9. [Training Models](#training-models)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is a **stress detection system** that analyzes physiological sensor data collected from wearable devices (chest and wrist accelerometers, ECG, EMG, EDA, temperature, respiration). The system:

1. **Loads WESAD data** — pickle files containing synchronized sensor streams
2. **Extracts features** — 159 time-domain, frequency-domain, and heart-rate-variability (HRV) features per 5-second window
3. **Trains ML classifiers** — 7 different models (Random Forest, SVM, KNN, Decision Tree, Gradient Boosting, Logistic Regression, LightGBM)
4. **Makes predictions** — classifies sensor windows as baseline, stress, or amusement
5. **Serves results via REST API** — Flask backend with multi-tier caching
6. **Visualizes outcomes** — interactive frontend with chart.js visualizations, confusion matrices, radar plots

---

## How It Works

### Data Flow

```
WESAD .pkl File (User Upload)
    ↓
[Data Processor] — Safe loading, signal filtering, resampling to 700 Hz
    ↓
[Feature Extractor] — Extract 159 features per 5-sec window
    ↓
[Model Trainer] — Train 7 classifiers with k-fold cross-validation
    ↓
[SQLite Cache] — Store model, results, and history
    ↓
[Flask API] — Serve JSON responses to frontend
    ↓
[Web Frontend] — Interactive charts, confusion matrices, radar plots
```

### Caching Strategy (3-tier)

1. **Memory Cache** — Fast in-process dictionary for frequently accessed predictions
2. **SQLite Database** — Persistent storage; survives server restarts
3. **Recompute** — On-demand feature extraction and prediction if not cached

Example: Clicking a pre-trained subject chip loads cached results from SQLite in < 50 ms.

### Key ML Concepts

- **Window-based features** — Sensor data split into overlapping 5-second windows
- **LOSO cross-validation** — General model trained leaving one subject out at a time for unbiased performance
- **Feature scaling** — StandardScaler normalizes features before model input
- **Class balancing** — RF classifier uses balanced class weights for imbalanced stress labels

---

## Features

| Feature | Details |
|---|---|
| **Pre-trained subjects** | 15 subjects (S2, S3, ..., S17) with instant predictions from disk cache |
| **File upload** | Drag-and-drop WESAD `.pkl` files; auto-trains per-subject model on first use |
| **Classifier comparison** | Side-by-side metrics: accuracy, F1, precision, recall, ROC-AUC, confusion matrix, radar chart |
| **Manual sensor input** | Enter 14 sensor values directly with quick-fill presets (Low/High/Critical stress levels) |
| **General model** | Subject-independent model for new subjects (trained with LOSO) |
| **History tracking** | Optional `tracking_id` links multiple uploads to the same person for longitudinal analysis |
| **Date filtering** | Filter history by date range; export as CSV for external analysis |
| **Persistent storage** | SQLite database (`wesad.db`) stores all predictions, history, and comparisons |
| **Rate limiting** | Per-IP sliding-window limiter (10 expensive requests / 60 sec) prevents abuse |
| **Responsive UI** | Dark glassmorphism theme; works on desktop, tablet, mobile |
| **30+ tests** | Comprehensive pytest suite: data loading, feature extraction, training, API, end-to-end |

---

## Project Structure

```
Project/
├── app.py                       # Flask web server and REST API
├── database.py                  # SQLite layer for history, results, comparisons
├── train_model.py               # CLI for training and model comparison
├── requirements.txt             # Python 3.10+ dependencies
├── README.md                    # This file
├── wesad.db                     # SQLite database (auto-created, ~10 MB)
│
├── model/                       # Python module: ML pipeline
│   ├── __init__.py
│   ├── data_processor.py        # Load .pkl, filter signals (ECG, EMG, EDA, etc.)
│   ├── feature_extractor.py     # Extract 159 features per window (time, freq, HRV)
│   ├── predictor.py             # Load models, predict, cache results (LRU)
│   └── trainer.py               # Train 7 classifiers, compare, save to disk
│
├── static/                      # Web frontend
│   ├── index.html               # Single-page app shell
│   ├── app.js                   # Client-side routing, API calls, Chart.js
│   └── style.css                # Responsive dark theme (glassmorphism)
│
├── tests/
│   └── test_app.py              # 30 pytest tests
│
├── trained_models/              # Pre-trained models and scalers
│   ├── model_S2.joblib, scaler_S2.joblib, features_S2.npz  # Per-subject
│   ├── model_S3.joblib, ...
│   ├── model_general.joblib     # General (LOSO) model
│   ├── model_general_manual.joblib  # For manual input (means only)
│   └── ... (15 subjects × 3 files = 45 files)
│
├── saved_files/                 # User-uploaded .pkl files (persisted)
│
└── ProjectWESAD.ipynb           # Exploratory Jupyter notebook
```

---

## Initial Setup

### Prerequisites

- **Python 3.10+**
- **Git** (optional, for version control)

### Installation Steps

1. **Create virtual environment**
   ```bash
   cd Project
   python -m venv .venv
   ```

2. **Activate environment**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   *Dependencies:*
   - `flask` (3.0+) — Web framework
   - `scikit-learn` (1.3+) — ML classifiers, preprocessing
   - `pandas` (2.0+) — Data manipulation
   - `numpy` (1.24+) — Numerical computing
   - `scipy` (1.11+) — Signal processing (FFT, filtering)
   - `joblib` (1.3+) — Model serialization
   - `lightgbm` (4.0+) — Gradient boosting
   - `pytest` (7.0+) — Testing framework

### Verify Installation

```bash
python -c "import flask, sklearn, lightgbm; print('✓ All dependencies installed')"
```

---

## Running the Application

### Start the Server

```bash
python app.py
```

**Output:**
```
 * Serving Flask app ...
 * Running on http://127.0.0.1:5000
```

### Access the Frontend

Open **http://127.0.0.1:5000** in your web browser.

### Stop the Server

Press **Ctrl+C** in the terminal (or **Cmd+C** on macOS).

---

## Understanding the ML Pipeline

### 1. Data Processing (`model/data_processor.py`)

**Input:** WESAD pickle file containing synchronized sensor streams

**Processing:**
- Safe pickle loading (handles corrupted files gracefully)
- **Signal filtering:**
  - **ECG:** 0.5–40 Hz bandpass (removes baseline wander + high-freq noise)
  - **EMG:** 20–300 Hz bandpass (muscle activity band)
  - **EDA:** 5 Hz lowpass (slow electrodermal response)
  - **Respiration:** 1 Hz lowpass (breathing rate)
- **Resampling:** All sensors aligned to 700 Hz using polyphase anti-aliasing filter

**Output:** NumPy array of shape `(n_samples, 14)` where columns are:
```
Chest: Acc-X, Acc-Y, Acc-Z, ECG, EMG, EDA, Temperature, Respiration
Wrist: Acc-X, Acc-Y, Acc-Z, BVP, EDA, Temperature
```

### 2. Feature Extraction (`model/feature_extractor.py`)

**Input:** Filtered sensor data → sliding windows (5-sec, 50% overlap)

**Feature Count:** 159 per window

**Feature Categories:**

1. **Time-domain** (112 features = 14 sensors × 8 stats)
   - Mean, std, min, max, median, IQR, skewness, kurtosis
   - Computed independently per sensor channel

2. **Frequency-domain** (42 features = 14 sensors × 3 stats)
   - Dominant frequency, spectral energy, spectral entropy
   - Extracted via FFT up to 350 Hz (Nyquist at 700 Hz)

3. **Heart Rate Variability** (5 features, computed from ECG)
   - **RMSSD:** Root mean square of successive differences (parasympathetic tone)
   - **SDNN:** Standard deviation of RR intervals (overall HRV)
   - **Mean RR:** Average interval between heartbeats
   - **pNN50:** Percentage of RR intervals > 50 ms apart (stress indicator)
   - **HR:** Heart rate in bpm (derived from mean RR)

**HRV Computation:**
- R-peak detection on ECG using signal peaks + physiological constraints
- RR intervals filtered to 300–1500 ms (40–200 bpm range)
- Missing or invalid HRV → fills with zeros

**Output:** Array of shape `(n_windows, 159)` + labels

### 3. Model Training (`model/trainer.py`)

**Input:** Features (X) and stress labels (y)

**Classifiers (7 total):**

| Classifier | Type | Key Hyperparameters |
|---|---|---|
| **Random Forest** | Ensemble | 300 trees, max_depth=25, balanced class weights |
| **SVM** | Kernel-based | RBF kernel, C=1.0 |
| **K-Nearest Neighbors** | Instance-based | k=5, Euclidean distance |
| **Decision Tree** | Tree | max_depth=15 |
| **Gradient Boosting** | Ensemble | 100 estimators, learning_rate=0.1 |
| **Logistic Regression** | Linear | L2 penalty, max_iter=1000 |
| **LightGBM** | Gradient Boosting | 100 leaves, 0.05 learning rate |

**Training Pipeline:**
1. Impute missing values (SimpleImputer)
2. Scale features (StandardScaler)
3. Train classifier with k-fold cross-validation (default k=5)
4. Compute metrics: accuracy, F1 (weighted), precision, recall, ROC-AUC

**Output:** Serialized model + scaler (joblib format)

### 4. Prediction (`model/predictor.py`)

**Input:** Sensor features + trained model

**Process:**
1. **Model loading:** LRU cache prevents redundant disk I/O
2. **Feature scaling:** Apply fitted scaler
3. **Prediction:** Forward pass → log probabilities for 3 classes
4. **Result caching:** Store in SQLite for future requests

**Output:** Predicted class (1=Baseline, 2=Stress, 3=Amusement) + confidence scores

### 5. Stress Classification

Three stress levels:
- **Baseline (1):** Normal, calm state
- **Stress (2):** Induced stress (task-based in WESAD)
- **Amusement (3):** Humor/entertainment state

The system focuses on detecting **Stress vs. Non-Stress** for practical applications.

---

## API Reference

All endpoints return JSON responses.

### Pre-trained Subject Prediction

```http
GET /api/predict/<subject_id>
```

**Example:** `GET /api/predict/S2`

**Response:**
```json
{
  "subject_id": "S2",
  "windows": 1250,
  "stress_count": 380,
  "stress_ratio": 0.304,
  "accuracy": 0.85,
  "f1": 0.82,
  "confusion_matrix": [[500, 20], [30, 700]],
  "timestamp": "2025-04-10T14:30:00Z"
}
```

### File Upload & Train

```http
POST /api/upload
```

**Parameters (multipart/form-data):**
- `file` (binary) — WESAD `.pkl` file
- `subject_id` (string) — Subject identifier (e.g., "S99")

**Response:** Same as prediction

### Classifier Comparison

```http
POST /api/compare
```

**Parameters (JSON):**
```json
{
  "file": "<binary data>",
  "subject_id": "S99"
}
```

**Response:**
```json
{
  "comparisons": [
    {
      "classifier_name": "Random Forest",
      "accuracy": 0.87,
      "f1_weighted": 0.85,
      "precision": 0.84,
      "recall": 0.86,
      "roc_auc": 0.91,
      "train_time_sec": 3.2
    },
    ...
  ]
}
```

### Manual Input Prediction

```http
POST /api/predict_manual
```

**Body (JSON):**
```json
{
  "values": [
    -0.5, 0.3, -0.1,     // Chest Acc X, Y, Z
    45.2, -30.5, 0.8, 36.5, -0.2,  // Chest ECG, EMG, EDA, Temp, Resp
    0.1, 0.2, -0.05,     // Wrist Acc X, Y, Z
    25.1, 0.3, 35.8     // Wrist BVP, EDA, Temp
  ],
  "tracking_id": "patient_001"
}
```

**Response:**
```json
{
  "predicted_class": 2,
  "predicted_label": "Stress",
  "confidence": 0.78,
  "timestamp": "2025-04-10T14:35:00Z"
}
```

### History

```http
GET /api/history?tracking_id=patient_001&start_date=2025-04-01&end_date=2025-04-10
```

**Response:**
```json
{
  "entries": [
    {
      "timestamp": "2025-04-10T14:30:00Z",
      "predicted_label": "Stress",
      "confidence": 0.78,
      "source": "manual_input"
    },
    ...
  ]
}
```

---

## Training Models

### Pre-trained Models

15 subject models are included in `trained_models/` (S2, S3, ..., S17). These are ready to use immediately.

### Train a New Subject

```bash
python train_model.py <path_to_file.pkl> --subject-id S99
```

**Example:**
```bash
python train_model.py /path/to/data/S99.pkl --subject-id S99
```

**Output:**
- `trained_models/model_S99.joblib` — Trained Random Forest
- `trained_models/scaler_S99.joblib` — Feature scaler
- `trained_models/features_S99.npz` — Cached features (for quick re-analysis)

### Compare All Classifiers

```bash
python train_model.py <path_to_file.pkl> --compare --subject-id S99
```

**Output:**
```
Rank  Classifier              Accuracy    F1(w)   F1(m)    Prec    Rec   CV Mean    Time
=========================================================================================
1     Random Forest             87.50%    85.23%  84.12%  84.50%  85.80%  85.30%   3.21s
2     Gradient Boosting         86.80%    84.90%  83.50%  84.20%  85.30%  84.80%   2.15s
3     SVM                       85.20%    82.10%  80.90%  82.50%  81.20%  83.40%   5.67s
...
```

### Train General Model (LOSO)

Requires the full WESAD dataset. Set `WESAD_DATA_DIR`:

```bash
WESAD_DATA_DIR=/path/to/WESAD python train_model.py --general
```

This trains 15 separate models (one per subject excluded) and produces:
- `trained_models/model_general.joblib`
- `trained_models/scaler_general.joblib`

---

## Testing

Run the automated test suite:

```bash
pytest tests/test_app.py -v
```

**Test coverage (30+ tests):**
- Data loading and filtering
- Feature extraction (time, frequency, HRV)
- Model training and evaluation
- Flask API endpoints
- Caching behavior
- Rate limiting
- History and persistence

**Expected output:**
```
tests/test_app.py::test_load_subject PASSED                    [  3%]
tests/test_app.py::test_extract_features PASSED                [  6%]
tests/test_app.py::test_train_model PASSED                     [  9%]
...
============================ 30 passed in 12.34s ============================
```

---

## Architecture Highlights

### Backend (`app.py`)

- **Flask** web server with 10+ RESTful endpoints
- **3-tier caching:** in-memory → SQLite → recompute
- **Rate limiter:** per-IP sliding-window (10 req/60 sec)
- **Async file handling:** uploads stored in temp directory, moved to `saved_files/` on success
- **Error handling:** graceful fallbacks for corrupted data, missing models

### Database (`database.py`)

- **SQLite** with Write-Ahead Logging (WAL) for concurrent access
- **Tables:**
  - `history` — prediction records with optional `tracking_id`
  - `results` — cached model predictions
  - `comparisons` — cached classifier comparison results
- **Thread-safe:** uses thread-local connections

### Frontend (`static/`)

- **HTML5** SPA shell
- **Vanilla JavaScript** (no jQuery, React, Vue)
- **Chart.js** for visualizations: confusion matrix, radar plot, bar chart
- **Responsive CSS** with glassmorphism theme (dark mode)
- **Drag-and-drop** file upload
- **Real-time** input validation and quick-fill presets

---

## Configuration

### Environment Variables

```bash
# Override WESAD dataset path (for training general model)
WESAD_DATA_DIR=/path/to/WESAD

# Default: http://127.0.0.1:5000
FLASK_ENV=development  # or 'production'
```

### Tweakable Parameters

**In `model/trainer.py`:**
- `DEFAULT_RF_PARAMS` — Random Forest hyperparameters
- `STRESS_RATIO_THRESHOLD` — Threshold for "overall stress" flag (default 0.30)

**In `model/feature_extractor.py`:**
- `DEFAULT_WINDOW_SEC` — Window size in seconds (default 5)
- `FS` — Sampling frequency (fixed at 700 Hz)

**In `app.py`:**
- `_COMPARE_CACHE_TTL` — Comparison cache lifetime (default 3600 sec)
- `_rate_limiter` — Max requests per window

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'lightgbm'"

**Solution:** `pip install lightgbm` (optional dependency). The app will use Random Forest instead if LightGBM is unavailable.

### Issue: Database locked error

**Solution:** SQLite WAL mode handles concurrency. Ensure no other process is writing to `wesad.db`. Restart the Flask server if the lock persists.

### Issue: Out of memory when uploading large `.pkl` files

**Solution:** Increase virtual memory or reduce window size (change `DEFAULT_WINDOW_SEC` in feature_extractor.py).

### Issue: Pre-trained models not loading

**Solution:** Verify `trained_models/` folder exists and contains `.joblib` files. Check Flask logs for file I/O errors.

### Issue: Port 5000 already in use

**Solution:** Change the port in `app.py` or kill the process using port 5000:
```bash
# Windows
netstat -ano | find ":5000"
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :5000
kill -9 <PID>
```

---

## Future Enhancements

- Real-time streaming prediction from wearable devices
- Deep learning models (LSTM, TCN) for temporal patterns
- Mobile app for live stress monitoring
- Integration with wearable device SDKs (Empatica, Apple Watch, Fitbit)
- Multi-language frontend
- User authentication and multi-tenant support

---

## License & Attribution

Built on the **WESAD** dataset:

> Schuller, B., Pirker, H., et al. (2018). WESAD: A Publicly Available Dataset for Wearable Stress and Affect Detection. 2018 International Conference on Affective Computing and Intelligent Interaction (ACII). IEEE.

Download the dataset: https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29

---

## Questions?

Refer to the code comments in `model/`, `app.py`, and `database.py` for implementation details.
