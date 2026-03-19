# WESAD Stress Detection — Full-Stack ML Web Application

Real-time physiological stress detection using the
[WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
dataset. A **Flask** back-end serves a single-page **vanilla JS** front-end
that lets you upload subject `.pkl` files, view predictions, compare up to
**7 ML classifiers**, and enter sensor readings manually — all with
**instant, single-click results** for pre-trained subjects.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [How It Works](#how-it-works)
5. [REST API](#rest-api)
6. [ML Classifiers](#ml-classifiers)
7. [Feature Engineering](#feature-engineering)
8. [Training](#training)
9. [Testing](#testing)
10. [Configuration](#configuration)

---

## Features

| Category | Details |
|---|---|
| **Instant analysis** | Pre-computed results served from 2-tier cache (memory → SQLite → recompute). Single-click subject chips load in &lt;50 ms. |
| **File upload** | Drag-and-drop a WESAD `.pkl` — the app trains a per-subject model on the fly if none exists. |
| **Classifier comparison** | Compare all 7 classifiers (6 base + LightGBM) side-by-side with accuracy, F1, precision, recall, ROC-AUC, confusion matrices, and a radar chart. |
| **Manual input** | Enter 14 sensor readings manually (with quick-fill presets for Low / High / Critical stress). Uses a compact means-only general model. |
| **History tracking** | Attach an optional `tracking_id` to uploads or manual readings to build a longitudinal stress history for the same person across times and days, then filter by date range or export as CSV. |
| **General model** | Subject-independent model trained with Leave-One-Subject-Out (LOSO) cross-validation across all available subjects. |
| **Persistent storage** | All analysis results, history, and comparisons stored in a single `wesad.db` SQLite database. Uploaded `.pkl` files saved in `saved_files/`. Data survives server restarts. |
| **Rate limiting** | Sliding-window per-IP rate limiter (10 requests / 60 s) protects expensive endpoints. |
| **30 automated tests** | Comprehensive pytest suite covering data processing, feature extraction, training, Flask API, and end-to-end flows. |

---

## Project Structure

```
Project/
├── app.py                    # Flask application — all routes and caching logic
├── database.py               # SQLite database layer (history, results, comparisons)
├── train_model.py            # CLI tool for training / comparing models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── wesad.db                  # SQLite database (created on first run)
├── model/
│   ├── __init__.py
│   ├── data_processor.py     # Safe pickle loading, signal filtering, resampling
│   ├── feature_extractor.py  # 159-feature extraction with HRV (50 % overlap)
│   ├── predictor.py          # Model loading (LRU-cached), prediction with auto-truncation
│   └── trainer.py            # 7 classifiers, Pipeline-based training, general model + LOSO
├── static/
│   ├── index.html            # SPA HTML shell (dark glassmorphism theme)
│   ├── app.js                # Client-side JS — routing, Chart.js visualisations
│   └── style.css             # CSS — glassmorphism, animations, responsive layout
├── tests/
│   └── test_app.py           # 30 pytest tests
├── trained_models/           # Saved .joblib models, scalers, and feature caches (.npz)
├── saved_files/              # Uploaded .pkl files (persisted for re-analysis)
└── ProjectWESAD.ipynb        # Exploratory Jupyter notebook
```

---

## Quick Start

### 1. Clone & install

```bash
cd Project
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the server

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 3. Analyse a subject

- **Pre-trained subjects** — click a subject chip (e.g. S2, S3, …) for
  instant results.
- **Upload** — drag a WESAD `.pkl` file onto the upload area. The app
  extracts features, trains a model (if needed), and displays results.
- **Manual input** — enter 14 sensor values (or use a quick-fill preset)
  and click **Analyse**.
- **Compare models** — hold **Shift** and click a subject chip (or use the
  Compare button) to see all 7 classifiers evaluated side-by-side.

### 4. Train a general model (optional)

```bash
python train_model.py --general --data-dir /path/to/WESAD
```

Or set the `WESAD_DATA_DIR` environment variable and let the server
auto-train on first startup.

---

## How It Works

### Data Pipeline

```
.pkl file
  │
  ├─ Safe unpickling (_SafeUnpickler — allows only numpy/pandas/dict types)
  │
  ├─ Signal extraction: 8 chest channels (700 Hz) + 6 wrist channels (mixed Hz)
  │
  ├─ Filtering
  │   ├─ ECG:  bandpass 0.5–40 Hz (Butterworth order 4)
  │   ├─ EMG:  bandpass 20–300 Hz (Butterworth order 4)
  │   ├─ EDA:  lowpass 5 Hz (Butterworth order 4)
  │   └─ Resp: lowpass 1 Hz (Butterworth order 4)
  │
  ├─ Resampling: wrist signals upsampled to 700 Hz via polyphase resampling
  │   (scipy.signal.resample_poly with anti-aliasing filter)
  │
  ├─ Windowing: 60-second windows, 50 % overlap (configurable)
  │
  ├─ Feature extraction: 159 features per window
  │   ├─ 112 time-domain  (8 statistics × 14 channels)
  │   ├─  42 frequency-domain (3 PSD statistics × 14 channels)
  │   └─   5 HRV features (RMSSD, SDNN, mean RR, pNN50, heart rate from ECG)
  │
  └─ Classification: scikit-learn Pipeline (SimpleImputer → StandardScaler → Classifier)
```

### 2-Tier Caching

1. **Client-side** — `app.js` fetches `/api/batch-results` on page load and
   caches all pre-computed results in a JS object. Clicking a chip reads
   local data with zero server round-trip.
2. **Server memory** — `_prediction_cache` dict populated by `_warmup_cache()`
   at startup. Falls back to SQLite (`wesad.db`) for persistence across restarts.

### Backward Compatibility

Models trained with an older feature count (e.g. 154 features before HRV was
added) are still loaded — `load_model()` accepts any model expecting ≥ 100
features. At prediction time, the feature matrix is auto-truncated to match
the model's expected column count.

---

## REST API

All endpoints return JSON. Base URL: `http://127.0.0.1:5000`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/config` | Front-end config: feature columns, subjects, label map, sensor metadata, classifier names, general-model status. |
| `GET` | `/api/batch-results` | All pre-computed prediction results (single response for instant client-side access). |
| `GET` | `/api/models` | List available trained model subject IDs. |
| `GET` | `/api/saved-files` | List previously saved `.pkl` files with metadata. |
| `POST` | `/api/upload` | Upload a `.pkl` file → train (if needed) → predict → return results. Rate-limited. |
| `POST` | `/api/compare` | Upload a `.pkl` file → compare all classifiers → return comparison table. Rate-limited. |
| `POST` | `/api/predict` | Manual sensor input → predict with general or per-subject model. JSON body: `{"sensors": {...}}`; add `tracking_id` to append the result to a longitudinal history. |
| `GET` | `/api/history/<tracking_id>` | Return the stored chronological history and trend summary for a tracking ID. |
| `GET` | `/api/history/<tracking_id>/export` | Download history for a tracking ID as CSV. Supports the same `start` and `end` filters as the history endpoint. |
| `POST` | `/api/analyze-pretrained/<sid>` | Analyse a pre-trained subject (instant from cache, or re-compute). |
| `POST` | `/api/compare-pretrained/<sid>` | Compare all classifiers for a pre-trained subject. Rate-limited. |
| `POST` | `/api/analyze/<sid>` | Analyse a saved `.pkl` by subject ID (no upload needed). |
| `POST` | `/api/compare-saved/<sid>` | Compare classifiers on a saved `.pkl`. Rate-limited. |
| `DELETE` | `/api/saved-files/<sid>` | Delete a saved `.pkl`, its cached features, and saved results. |

---

## ML Classifiers

The system uses **scikit-learn Pipelines** with `SimpleImputer` (median) →
`StandardScaler` → classifier. Seven classifiers are available:

| # | Classifier | Key Hyperparameters |
|---|---|---|
| 1 | **Random Forest** | 200 trees, max depth 20, min samples split 5, class weight balanced |
| 2 | **SVM (RBF)** | C = 10, gamma = scale, class weight balanced |
| 3 | **K-Nearest Neighbours** | k = 7, distance-weighted, ball tree |
| 4 | **Decision Tree** | max depth 15, min samples split 10, class weight balanced |
| 5 | **Gradient Boosting** | 200 estimators, learning rate 0.1, max depth 5, subsample 0.8 |
| 6 | **Logistic Regression** | C = 1.0, max iter 1000, class weight balanced |
| 7 | **LightGBM** *(optional)* | 200 estimators, learning rate 0.05, 31 leaves, class weight balanced |

> LightGBM is included automatically when the `lightgbm` package is installed
> (listed in `requirements.txt`). If not installed, the remaining 6 classifiers
> are used without error.

---

## Feature Engineering

### 159 Features per Window

| Category | Count | Description |
|---|---|---|
| **Time-domain** | 112 | 8 statistics (mean, std, min, max, median, IQR, skewness, kurtosis) × 14 channels |
| **Frequency-domain** | 42 | 3 PSD statistics (total power, peak frequency, spectral entropy) × 14 channels |
| **HRV (Heart Rate Variability)** | 5 | RMSSD, SDNN, mean RR interval, pNN50, heart rate — derived from ECG R-peak detection |

### 14 Sensor Channels

| # | Channel | Source | Native Rate |
|---|---|---|---|
| 1–3 | Chest Acc X / Y / Z | RespiBAN (chest) | 700 Hz |
| 4 | Chest ECG | RespiBAN | 700 Hz |
| 5 | Chest EMG | RespiBAN | 700 Hz |
| 6 | Chest EDA | RespiBAN | 700 Hz |
| 7 | Chest Temp | RespiBAN | 700 Hz |
| 8 | Chest Resp | RespiBAN | 700 Hz |
| 9–11 | Wrist Acc X / Y / Z | Empatica E4 (wrist) | 32 Hz → 700 Hz |
| 12 | Wrist BVP | Empatica E4 | 64 Hz → 700 Hz |
| 13 | Wrist EDA | Empatica E4 | 4 Hz → 700 Hz |
| 14 | Wrist Temp | Empatica E4 | 4 Hz → 700 Hz |

Wrist signals are upsampled to 700 Hz using `scipy.signal.resample_poly`
(polyphase anti-aliasing filter) to align with chest signals before windowing.

### Signal Filtering

| Signal | Filter | Details |
|---|---|---|
| ECG | Bandpass 0.5–40 Hz | Butterworth order 4 |
| EMG | Bandpass 20–300 Hz | Butterworth order 4 |
| EDA | Lowpass 5 Hz | Butterworth order 4 |
| Resp | Lowpass 1 Hz | Butterworth order 4 |

### Labels

| Code | Label | Description |
|---|---|---|
| 1 | Baseline | Neutral / relaxed |
| 2 | Stress | Trier Social Stress Test (TSST) |
| 3 | Amusement | Funny video clips |

Labels 4–7 (meditation, ignore, etc.) are excluded during windowing.

---

## Training

### Per-Subject Model

```bash
# Train from a .pkl via CLI
python train_model.py --subject S2 --data-dir /path/to/WESAD

# Or simply upload the .pkl in the web UI — training happens automatically
```

### General (Subject-Independent) Model

```bash
python train_model.py --general --data-dir /path/to/WESAD
```

This trains on **all** subjects using stratified 5-fold cross-validation and
also evaluates Leave-One-Subject-Out (LOSO) accuracy. Two models are saved:

- `model_general.joblib` — full 159-feature model for file-upload analysis.
- `model_general_manual.joblib` — compact 14-feature (means-only) model for
  manual sensor input.

### Compare Classifiers

```bash
python train_model.py --compare --data-dir /path/to/WESAD --subject S3
```

Or use the web UI: hold **Shift** + click a subject chip to compare all
classifiers on that subject's data.

---

## Testing

```bash
python -m pytest tests/ -v
```

**30 tests** covering:

| # | Test | Description |
|---|---|---|
| 1 | `test_load_subject_valid` | Load & align signals from a real/mock `.pkl` |
| 2 | `test_load_subject_filters_labels` | Only labels 1, 2, 3 retained |
| 3 | `test_load_subject_invalid_path` | Graceful error on missing file |
| 4 | `test_load_subject_invalid_pkl` | Graceful error on malformed pickle |
| 5 | `test_extract_windows_shape` | Window count and feature shape (159 cols) |
| 6 | `test_extract_windows_no_overlap` | `overlap=0.0` produces non-overlapping windows |
| 7 | `test_compute_window_features` | Single window → 159 features |
| 8 | `test_feature_names_length` | `FEATURE_NAMES` has exactly 159 entries |
| 9 | `test_features_from_manual_input` | Manual input → shape (1, 159) |
| 10 | `test_train_model_returns_pipeline` | `train_model()` returns a Pipeline |
| 11 | `test_save_and_load_model` | Round-trip save → load succeeds |
| 12 | `test_predict_output_format` | `predict()` returns expected keys |
| 13 | `test_compare_classifiers` | All classifiers return results |
| 14 | `test_flask_index` | `GET /` serves `index.html` |
| 15 | `test_api_config_keys` | `/api/config` returns required keys |
| 16 | `test_api_models` | `/api/models` returns list |
| 17 | `test_api_upload_no_file` | Upload without file → 400 |
| 18 | `test_api_upload_wrong_ext` | Non-.pkl upload → 400 |
| 19 | `test_api_predict_no_json` | Missing JSON body → 400 |
| 20 | `test_api_batch_results` | `/api/batch-results` returns dict |
| 21 | `test_api_config_has_classifier_names` | Config includes classifier names list |
| 22 | `test_api_config_has_sensor_meta` | Config includes sensor metadata |
| 23 | `test_rate_limiter_basic` | Rate limiter allows up to max then blocks |
| 24 | `test_build_result_dict_low_stress` | Low ratio → "Low Stress" |
| 25 | `test_build_result_dict_critical_stress` | High ratio → "Critical Stress" |
| 26 | `test_build_result_dict_manual_not_stressed` | Manual baseline → forced "Low Stress" |
| 27 | `test_build_result_dict_manual_stress_is_stressed` | Manual stress → "High Stress" or above |
| 28 | `test_upload_malformed_pkl_not_dict` | Upload corrupt pkl → 500 with message |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `WESAD_DATA_DIR` | `../WESAD` | Path to raw WESAD dataset (for training general model) |
| `FLASK_DEBUG` | `0` | Set to `1` to enable Flask debug mode |
| `MAX_CONTENT_LENGTH` | 200 MB | Maximum upload file size |

---

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for packages:
  `flask`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`, `pytest`,
  `lightgbm`

---

## License

This project is for educational and research purposes. The WESAD dataset is
available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29).
