#!/usr/bin/env python3
"""Flask web application for WESAD stress detection.

Serves a static SPA front-end and exposes JSON API endpoints.
No Jinja2 templates — all rendering happens client-side.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import tempfile

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from model.data_processor import load_subject, FEATURE_COLUMNS
from model.feature_extractor import (
    extract_windows,
    features_from_manual_input,
    features_from_manual_input_means_only,
    LABEL_MAP,
    DEFAULT_WINDOW_SEC,
    FEATURE_NAMES,
)
from model.predictor import (
    model_exists, load_model, predict, invalidate_model_cache, load_manual_model,
)
from model.trainer import (
    train_model,
    compare_classifiers,
    save_model,
    train_general_model,
    general_model_exists,
    general_manual_model_exists,
    GENERAL_ID,
    GENERAL_MANUAL_ID,
    CLASSIFIER_CATALOGUE,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "wesad_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")

# Path to WESAD dataset root (for training the general model).
# Set env var WESAD_DATA_DIR to override.
WESAD_DATA_DIR = os.environ.get(
    "WESAD_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "WESAD"),
)

# Sensor metadata: placeholder hint, unit label, and preset values per scenario.
# low = Baseline centroid, high = Stress centroid, critical = extrapolated
# beyond stress.  Values derived from real WESAD data across all 15 subjects.
SENSOR_META = {
    "Chest Acc X":  {"unit": "g",      "hint": "~0.77–0.85",   "low": 0.77,    "high": 0.85,    "critical": 0.89},
    "Chest Acc Y":  {"unit": "g",      "hint": "~−0.05–0.0",   "low": -0.049,  "high": -0.009,  "critical": 0.01},
    "Chest Acc Z":  {"unit": "g",      "hint": "~−0.39–−0.27", "low": -0.391,  "high": -0.266,  "critical": -0.20},
    "Chest ECG":    {"unit": "mV",     "hint": "~0.001",       "low": 0.00115, "high": 0.00102, "critical": 0.00095},
    "Chest EMG":    {"unit": "mV",     "hint": "~−0.003",      "low": -0.00292,"high": -0.00291,"critical": -0.00314},
    "Chest EDA":    {"unit": "\u00b5S","hint": "3.9–6.0",      "low": 3.9,     "high": 6.0,     "critical": 7.0},
    "Chest Temp":   {"unit": "\u00b0C","hint": "33–34",        "low": 33.4,    "high": 34.1,    "critical": 34.5},
    "Chest Resp":   {"unit": "a.u.",   "hint": "~0.05",        "low": 0.056,   "high": 0.050,   "critical": 0.046},
    "Wrist Acc X":  {"unit": "1/64g",  "hint": "14–17",        "low": 14.3,    "high": 17.3,    "critical": 19.0},
    "Wrist Acc Y":  {"unit": "1/64g",  "hint": "−9 – −4",      "low": -4.4,    "high": -9.4,    "critical": -12.0},
    "Wrist Acc Z":  {"unit": "1/64g",  "hint": "3–12",         "low": 12.4,    "high": 3.1,     "critical": -2.0},
    "Wrist BVP":    {"unit": "a.u.",   "hint": "~0.0",         "low": 0.005,   "high": 0.018,   "critical": 0.025},
    "Wrist EDA":    {"unit": "\u00b5S","hint": "1.3–3.4",      "low": 1.3,     "high": 3.4,     "critical": 4.5},
    "Wrist Temp":   {"unit": "\u00b0C","hint": "32.6–33.3",    "low": 33.3,    "high": 32.7,    "critical": 32.3},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stress_level_info(ratio: float, *, is_manual: bool = False) -> dict:
    """Return stress level name, color, and gauge CSS class from a ratio.

    When *is_manual* is True the value is a class probability (0-1) where
    0.50 is the natural decision boundary, so wider thresholds are used.
    """
    if is_manual:
        # Thresholds calibrated for single-prediction class probabilities.
        if ratio < 0.35:
            return {"stress_level": "Low Stress", "stress_level_color": "#34d399", "gauge_class": "gauge-low"}
        elif ratio < 0.50:
            return {"stress_level": "Moderate Stress", "stress_level_color": "#fbbf24", "gauge_class": "gauge-moderate"}
        elif ratio < 0.75:
            return {"stress_level": "High Stress", "stress_level_color": "#fbbf24", "gauge_class": "gauge-high"}
        else:
            return {"stress_level": "Critical Stress", "stress_level_color": "#dc2626", "gauge_class": "gauge-critical"}

    # Multi-window stress ratio thresholds (file-upload mode).
    if ratio < 0.20:
        return {"stress_level": "Low Stress", "stress_level_color": "#34d399", "gauge_class": "gauge-low"}
    elif ratio < 0.40:
        return {"stress_level": "Moderate Stress", "stress_level_color": "#fbbf24", "gauge_class": "gauge-moderate"}
    elif ratio < 0.65:
        return {"stress_level": "High Stress", "stress_level_color": "#f87171", "gauge_class": "gauge-high"}
    else:
        return {"stress_level": "Critical Stress", "stress_level_color": "#dc2626", "gauge_class": "gauge-critical"}


def _build_result_dict(subject_id: str, method: str, results: dict,
                       metrics: dict | None = None) -> dict:
    """Build a JSON-serialisable result dict."""
    valid_labels = {"Baseline", "Stress", "Amusement"}

    filtered_preds = []
    filtered_names = []
    for pred, name in zip(results["predictions"], results["label_names"]):
        if name in valid_labels:
            filtered_preds.append(pred)
            filtered_names.append(name)

    label_counts = {}
    for name in filtered_names:
        label_counts[name] = label_counts.get(name, 0) + 1

    effective_ratio = (
        results["stress_confidence"] if results.get("stress_confidence") is not None
        else results["stress_ratio"]
    )
    is_manual = len(filtered_preds) == 1 and results.get("stress_confidence") is not None
    level_info = _stress_level_info(effective_ratio, is_manual=is_manual)

    # For manual input, base the verdict on the actual predicted label.
    # If the model predicts Baseline (not stressed), force "Low Stress"
    # so the label never contradicts the prediction.
    if is_manual:
        overall_stress = bool(results["is_stressed"][0])
        if not overall_stress:
            level_info = {
                "stress_level": "Low Stress",
                "stress_level_color": "#34d399",
                "gauge_class": "gauge-low",
            }
    else:
        overall_stress = effective_ratio > 0.30

    ctx = {
        "subject_id": subject_id,
        "method": method,
        "total_windows": len(filtered_preds),
        "label_counts": label_counts,
        "stress_ratio": effective_ratio,
        "overall_stress": overall_stress,
        "predictions": [int(p) for p in filtered_preds],
        "label_names": filtered_names,
        "window_sec": DEFAULT_WINDOW_SEC,
        "is_manual": is_manual,
        **level_info,
    }

    if metrics:
        ctx["accuracy"] = metrics.get("accuracy")
        ctx["f1_weighted"] = metrics.get("f1_weighted")
        ctx["precision"] = metrics.get("precision")
        ctx["recall"] = metrics.get("recall")
        ctx["cv_mean"] = metrics.get("cv_mean")
        ctx["cv_std"] = metrics.get("cv_std")
        ctx["confusion_matrix"] = metrics.get("confusion_matrix", [])
        ctx["target_names"] = metrics.get("target_names", [])
        ctx["classifier_name"] = metrics.get("classifier_name", "Random Forest")

        importances = metrics.get("feature_importances")
        if importances:
            feat_names = FEATURE_NAMES if len(importances) == len(FEATURE_NAMES) else [
                f"F{i}" for i in range(len(importances))
            ]
            pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:15]
            ctx["feature_importance_labels"] = [p[0] for p in pairs]
            ctx["feature_importance_values"] = [round(p[1], 5) for p in pairs]

    return ctx


def _available_subjects():
    """Return a sorted list of subject IDs that have trained models.

    The general model is excluded — it is surfaced separately via
    ``general_model_ready`` in the config endpoint.
    """
    if not os.path.isdir(TRAINED_MODELS_DIR):
        return []
    subjects = set()
    for fname in os.listdir(TRAINED_MODELS_DIR):
        if fname.startswith("model_") and fname.endswith(".joblib"):
            sid = fname[len("model_"):-len(".joblib")]
            if sid not in (GENERAL_ID, GENERAL_MANUAL_ID):
                subjects.add(sid)
    return sorted(subjects, key=lambda s: [int(t) if t.isdigit() else t for t in __import__('re').split(r'(\d+)', s)])


def _ensure_general_model():
    """Train the general (subject-independent) model if it doesn't exist yet.

    Also ensures the companion means-only manual model exists.
    """
    if general_model_exists() and general_manual_model_exists():
        logger.info("General model (full + manual) already exists.")
        return
    if not os.path.isdir(WESAD_DATA_DIR):
        logger.warning(
            "WESAD data directory not found at %s — cannot auto-train general model. "
            "Set WESAD_DATA_DIR env var or run: python train_model.py --general --data-dir <path>",
            WESAD_DATA_DIR,
        )
        return
    logger.info("Training general model from %s …", WESAD_DATA_DIR)
    try:
        _, _, metrics = train_general_model(
            WESAD_DATA_DIR, cv_folds=5, verbose=True,
        )
        logger.info(
            "General model ready — %d subjects, %d windows, accuracy %.2f%%",
            metrics["n_subjects"], metrics["n_windows_total"], metrics["accuracy"] * 100,
        )
    except Exception as exc:
        logger.error("Failed to train general model: %s", exc)


# ---------------------------------------------------------------------------
# Routes – static SPA
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page application."""
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/api/config", methods=["GET"])
def api_config():
    """Return front-end configuration: feature columns, models, labels, etc."""
    return jsonify({
        "feature_columns": FEATURE_COLUMNS,
        "subjects": _available_subjects(),
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "classifier_names": list(CLASSIFIER_CATALOGUE.keys()),
        "sensor_meta": SENSOR_META,
        "general_model_ready": general_model_exists() and general_manual_model_exists(),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Process a .pkl file upload and return prediction results as JSON."""
    file = request.files.get("pkl_file")
    if not file or file.filename == "":
        return jsonify({"error": "Please select a WESAD subject .pkl file."}), 400

    if not file.filename.endswith(".pkl"):
        return jsonify({"error": "Only .pkl files are supported."}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename."}), 400
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)
    subject_id = os.path.splitext(safe_name)[0]

    try:
        features, labels = load_subject(save_path)
        X, y = extract_windows(features, labels)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        model, scaler, metrics = None, None, None
        if model_exists(subject_id):
            model, scaler = load_model(subject_id)
            if model is None:
                invalidate_model_cache(subject_id)
                model, scaler = None, None

        if model is None:
            model, scaler, metrics = train_model(
                X, y, cv_folds=5, verbose=True,
            )
            save_model(model, scaler, subject_id)

        results = predict(model, scaler, X)
        method = "pre-trained" if metrics is None else "trained on-the-fly"
        ctx = _build_result_dict(subject_id, method, results, metrics=metrics)
        return jsonify(ctx)

    except Exception as exc:
        logger.exception("Error processing upload")
        return jsonify({"error": f"Error processing file: {exc}"}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Train all classifiers on an uploaded .pkl and return comparison JSON."""
    file = request.files.get("pkl_file")
    if not file or file.filename == "":
        return jsonify({"error": "Please select a WESAD subject .pkl file."}), 400

    if not file.filename.endswith(".pkl"):
        return jsonify({"error": "Only .pkl files are supported."}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename."}), 400
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)
    subject_id = os.path.splitext(safe_name)[0]

    try:
        features, labels = load_subject(save_path)
        X, y = extract_windows(features, labels)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        results = compare_classifiers(X, y, cv_folds=5, verbose=True)
        return jsonify({"subject_id": subject_id, "results": results})

    except Exception as exc:
        logger.exception("Error during comparison")
        return jsonify({"error": f"Error during model comparison: {exc}"}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: predict stress from manual sensor values.

    Uses the **general** (subject-independent) model by default.
    An optional ``subject_id`` field overrides this to use a specific
    per-subject model instead.

    Expects JSON body::

        {
            "sensors": { "Chest ECG": 0.5, ... }
        }

    Or, to force a specific model::

        {
            "subject_id": "S2",
            "sensors": { ... }
        }
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    subject_id = data.get("subject_id", "").strip()
    sensors = data.get("sensors", {})

    # Decide which model to use
    if subject_id:
        # Explicit per-subject override
        if not model_exists(subject_id):
            return jsonify({
                "error": f"No model for '{subject_id}'.",
                "available": _available_subjects(),
            }), 404
        model_label = f"Manual (model: {subject_id})"
    else:
        # Default: general model
        if not general_model_exists():
            return jsonify({
                "error": "General model not available. "
                         "Run: python train_model.py --general --data-dir <WESAD_ROOT>",
            }), 503
        subject_id = GENERAL_ID
        model_label = "Manual (general model)"

    sensor_values = {}
    for col in FEATURE_COLUMNS:
        try:
            sensor_values[col] = float(sensors.get(col, 0.0))
        except (TypeError, ValueError):
            sensor_values[col] = 0.0

    # Choose the right model & feature vector.
    # When using the general model → use the compact means-only model that
    # was trained on just the 14 mean features per window.  This avoids the
    # synthetic-feature problem (the full 154-feature model can't
    # meaningfully classify fabricated std/iqr/skew/freq values).
    if subject_id == GENERAL_ID:
        model, scaler = load_manual_model()
        if model is None:
            return jsonify({
                "error": "Means-only manual model not available. "
                         "Retrain with: python train_model.py --general --data-dir <WESAD_ROOT>",
            }), 503
        X = features_from_manual_input_means_only(sensor_values)
    else:
        model, scaler = load_model(subject_id)
        X = features_from_manual_input(sensor_values)

    results = predict(model, scaler, X)

    ctx = _build_result_dict(model_label, "manual input", results)
    return jsonify(ctx)


@app.route("/api/models", methods=["GET"])
def api_models():
    """List available trained models."""
    return jsonify({"models": _available_subjects()})


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum upload size is 2 GB."}), 413


if __name__ == "__main__":
    _ensure_general_model()
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1",
            host="127.0.0.1", port=5000, use_reloader=False)
