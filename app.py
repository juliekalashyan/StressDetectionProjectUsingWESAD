#!/usr/bin/env python3
"""Flask web application for StressAware stress detection.

Serves a static SPA front-end and exposes JSON API endpoints.
No Jinja2 templates — all rendering happens client-side.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import tempfile
import time
import threading
import csv
import io
import json
from datetime import datetime, timezone

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
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
    features_cache_exists, save_features_cache, load_features_cache,
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

import database as db

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")

DEFAULT_MAX_UPLOAD_MB = 2048
env_max_content = os.environ.get("MAX_CONTENT_LENGTH")
if env_max_content is not None:
    try:
        app.config["MAX_CONTENT_LENGTH"] = int(env_max_content)
    except ValueError:
        app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
else:
    env_max_upload_mb = os.environ.get("MAX_UPLOAD_MB")
    if env_max_upload_mb is not None:
        try:
            app.config["MAX_CONTENT_LENGTH"] = int(env_max_upload_mb) * 1024 * 1024
        except ValueError:
            app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    else:
        app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "wesad_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
logger.info("Effective upload max content length: %s", app.config["MAX_CONTENT_LENGTH"])


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (per-IP, sliding window)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by IP."""

    def __init__(self, max_requests: int = 10, window_sec: int = 60):
        self._max = max_requests
        self._window = window_sec
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            history = self._requests.get(key, [])
            # Prune entries outside the window
            history = [t for t in history if now - t < self._window]
            if len(history) >= self._max:
                self._requests[key] = history
                return False
            history.append(now)
            self._requests[key] = history
            return True


# 10 expensive requests (upload/compare/train) per minute per IP
_rate_limiter = _RateLimiter(max_requests=10, window_sec=60)


def _check_rate_limit():
    """Return a 429 response if the client has exceeded the rate limit."""
    ip = request.remote_addr or "unknown"
    if not _rate_limiter.is_allowed(ip):
        return jsonify({"error": "Too many requests. Please wait and try again."}), 429
    return None


# ---------------------------------------------------------------------------
# Comparison result cache (avoids re-training 6+ classifiers)
# ---------------------------------------------------------------------------
_compare_cache: dict[str, tuple[float, list]] = {}   # sid -> (timestamp, results)
_COMPARE_CACHE_TTL = 3600  # 1 hour

# Pre-computed prediction results — populated at startup for instant responses.
_prediction_cache: dict[str, dict] = {}  # sid -> result dict


def _get_cached_comparison(subject_id: str):
    # 1. In-memory cache
    entry = _compare_cache.get(subject_id)
    if entry is not None:
        ts, results = entry
        if time.time() - ts <= _COMPARE_CACHE_TTL:
            return results
        del _compare_cache[subject_id]
    # 2. Database
    disk = db.load_comparison(subject_id)
    if disk is not None:
        _compare_cache[subject_id] = (time.time(), disk)
        return disk
    return None


def _set_cached_comparison(subject_id: str, results: list):
    _compare_cache[subject_id] = (time.time(), results)
    db.save_comparison(subject_id, results)

# Persistent directory for saved .pkl files the user wants to keep.
SAVED_FILES_DIR = os.path.join(os.path.dirname(__file__), "saved_files")
os.makedirs(SAVED_FILES_DIR, exist_ok=True)

TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")

# Initialise the SQLite database (creates tables if needed).
db.init_db()

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
        "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
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
        ctx["roc_auc"] = metrics.get("roc_auc")
        ctx["cv_mean"] = metrics.get("cv_mean")
        ctx["cv_std"] = metrics.get("cv_std")
        ctx["confusion_matrix"] = metrics.get("confusion_matrix", [])
        ctx["target_names"] = metrics.get("target_names", [])
        ctx["classifier_name"] = metrics.get("classifier_name", "Random Forest")
        ctx["n_features_used"] = metrics.get("n_features_used")
        ctx["feature_selection_applied"] = metrics.get("feature_selection_applied", False)

        importances = metrics.get("feature_importances")
        if importances:
            feat_names = FEATURE_NAMES if len(importances) == len(FEATURE_NAMES) else [
                f"F{i}" for i in range(len(importances))
            ]
            pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:15]
            ctx["feature_importance_labels"] = [p[0] for p in pairs]
            ctx["feature_importance_values"] = [round(p[1], 5) for p in pairs]

    return ctx


# ---------------------------------------------------------------------------
# Persistent results cache — survives server restarts
# ---------------------------------------------------------------------------


def _normalise_tracking_id(raw_tracking_id: str | None) -> str:
    if raw_tracking_id is None:
        return ""
    return secure_filename(str(raw_tracking_id).strip())


def _history_entry_from_result(tracking_id: str, result: dict) -> dict:
    latest_label = result["label_names"][0] if result.get("label_names") else None
    return {
        "tracking_id": tracking_id,
        "captured_at": result.get("captured_at"),
        "subject_id": result.get("subject_id"),
        "method": result.get("method"),
        "is_manual": result.get("is_manual", False),
        "stress_ratio": result.get("stress_ratio", 0.0),
        "stress_level": result.get("stress_level"),
        "overall_stress": result.get("overall_stress", False),
        "predicted_label": latest_label,
        "total_windows": result.get("total_windows", 0),
        "label_counts": result.get("label_counts", {}),
    }


def _build_history_summary(entries: list[dict]) -> dict:
    if not entries:
        return {"count": 0, "latest": None, "previous": None, "delta": None, "trend": "stable"}

    latest = entries[-1]
    previous = entries[-2] if len(entries) > 1 else None
    delta = None
    trend = "stable"
    if previous is not None:
        delta = round(float(latest.get("stress_ratio", 0.0)) - float(previous.get("stress_ratio", 0.0)), 4)
        if delta > 0.02:
            trend = "up"
        elif delta < -0.02:
            trend = "down"

    return {
        "count": len(entries),
        "latest": latest,
        "previous": previous,
        "delta": delta,
        "trend": trend,
    }


def _append_history_entry(tracking_id: str, result: dict) -> dict:
    entry = _history_entry_from_result(tracking_id, result)
    db.append_history_entry(tracking_id, entry)
    entries = db.load_history(tracking_id)
    return _build_history_summary(entries)


def _history_entries_to_csv(entries: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "tracking_id",
        "captured_at",
        "subject_id",
        "method",
        "is_manual",
        "stress_ratio",
        "stress_level",
        "overall_stress",
        "predicted_label",
        "total_windows",
        "label_counts",
    ])
    for entry in entries:
        writer.writerow([
            entry.get("tracking_id", ""),
            entry.get("captured_at", ""),
            entry.get("subject_id", ""),
            entry.get("method", ""),
            entry.get("is_manual", False),
            entry.get("stress_ratio", 0.0),
            entry.get("stress_level", ""),
            entry.get("overall_stress", False),
            entry.get("predicted_label", ""),
            entry.get("total_windows", 0),
            json.dumps(entry.get("label_counts", {}), ensure_ascii=False),
        ])
    return buffer.getvalue()


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


def _warmup_cache():
    """Load saved results from disk into the in-memory prediction cache.

    Also pre-compute results for subjects that have a model + feature cache
    but no saved result yet (first startup after training).
    Called once at server start so every click is instant.
    """
    # 1. Load all previously saved results from database
    loaded = 0
    for sid in db.list_result_ids():
        if sid in _prediction_cache:
            continue
        result = db.load_result(sid)
        if result is not None:
            _prediction_cache[sid] = result
            loaded += 1

    # 2. Pre-compute for subjects that have model + features but no result yet
    computed = 0
    for sid in _available_subjects():
        if sid in _prediction_cache:
            continue
        try:
            if not model_exists(sid) or not features_cache_exists(sid):
                continue
            model, scaler = load_model(sid)
            if model is None:
                continue
            X, y = load_features_cache(sid)
            if X is None:
                continue
            results = predict(model, scaler, X)
            ctx = _build_result_dict(sid, "pre-trained", results)
            _prediction_cache[sid] = ctx
            db.save_result(sid, ctx)
            computed += 1
        except Exception as exc:
            logger.warning("Warmup compute failed for %s: %s", sid, exc)
    logger.info(
        "Warmup complete: %d loaded from disk, %d computed fresh. "
        "%d total subjects ready.",
        loaded, computed, len(_prediction_cache),
    )


# ---------------------------------------------------------------------------
# Routes – static SPA
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page application."""
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    """Lightweight health-check endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "models_loaded": len(_prediction_cache),
        "general_model_ready": general_model_exists() and general_manual_model_exists(),
    })


@app.after_request
def add_cache_headers(response):
    """Add cache-control headers to static assets."""
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/api/config", methods=["GET"])
def api_config():
    """Return front-end configuration: feature columns, models, labels, etc."""
    subjects = _available_subjects()
    subject_cache = {sid: features_cache_exists(sid) for sid in subjects}
    # Subjects with saved results (superset of subjects with models)
    result_subjects = sorted(
        set(subjects) | set(_prediction_cache.keys()),
        key=lambda s: [int(t) if t.isdigit() else t for t in __import__('re').split(r'(\d+)', s)],
    )
    max_upload_mb = None
    if app.config["MAX_CONTENT_LENGTH"] is not None:
        max_upload_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({
        "feature_columns": FEATURE_COLUMNS,
        "subjects": result_subjects,
        "subject_cache": subject_cache,
        "instant_subjects": list(_prediction_cache.keys()),
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "classifier_names": list(CLASSIFIER_CATALOGUE.keys()),
        "sensor_meta": SENSOR_META,
        "general_model_ready": general_model_exists() and general_manual_model_exists(),
        "max_upload_mb": max_upload_mb,
    })


@app.route("/api/batch-results", methods=["GET"])
def api_batch_results():
    """Return all pre-computed prediction results in a single response.

    The frontend calls this once on page load to enable instant
    single-click analysis with zero latency.
    """
    return jsonify(_prediction_cache)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Process a .pkl file upload and return prediction results as JSON."""
    rl = _check_rate_limit()
    if rl:
        return rl
    file = request.files.get("pkl_file")
    if not file or file.filename == "":
        return jsonify({"error": "Please select a WESAD subject .pkl file."}), 400

    if not file.filename.endswith(".pkl"):
        return jsonify({"error": "Only .pkl files are supported."}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename."}), 400
    tracking_id = _normalise_tracking_id(request.form.get("tracking_id"))
    if request.form.get("tracking_id") and not tracking_id:
        return jsonify({"error": "tracking_id must contain at least one valid character."}), 400
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)
    subject_id = os.path.splitext(safe_name)[0]

    try:
        model, scaler, metrics = None, None, None
        X, y = None, None

        # Fast path: if a pre-trained model AND cached features exist,
        # skip the expensive pickle-load + feature extraction entirely.
        if model_exists(subject_id) and features_cache_exists(subject_id):
            model, scaler = load_model(subject_id)
            if model is not None:
                X, y = load_features_cache(subject_id)

        # Slow path: need to extract features from the raw .pkl
        if X is None:
            features, labels = load_subject(save_path)
            X, y = extract_windows(features, labels)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            save_features_cache(subject_id, X, y)

        if model is None:
            if model_exists(subject_id):
                model, scaler = load_model(subject_id)
                if model is None:
                    invalidate_model_cache(subject_id)

        if model is None:
            model, scaler, metrics = train_model(
                X, y, cv_folds=5, verbose=True,
            )
            save_model(model, scaler, subject_id)

        # Persist the .pkl so the user can re-analyse without uploading again.
        saved_pkl = os.path.join(SAVED_FILES_DIR, safe_name)
        if not os.path.exists(saved_pkl) and os.path.exists(save_path):
            import shutil
            shutil.copy2(save_path, saved_pkl)

        results = predict(model, scaler, X)
        method = "pre-trained" if metrics is None else "trained on-the-fly"
        ctx = _build_result_dict(subject_id, method, results, metrics=metrics)
        # Cache result in memory and persist to disk
        _prediction_cache[subject_id] = ctx
        db.save_result(subject_id, ctx)
        response_ctx = dict(ctx)
        if tracking_id:
            history_summary = _append_history_entry(tracking_id, ctx)
            response_ctx["tracking_id"] = tracking_id
            response_ctx["history_length"] = history_summary["count"]
        return jsonify(response_ctx)

    except Exception as exc:
        logger.exception("Error processing upload")
        return jsonify({"error": f"Error processing file: {exc}"}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Train all classifiers on an uploaded .pkl and return comparison JSON."""
    rl = _check_rate_limit()
    if rl:
        return rl
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
        # Check comparison cache first
        cached = _get_cached_comparison(subject_id)
        if cached is not None:
            return jsonify({"subject_id": subject_id, "results": cached})

        # Use cached features if available, otherwise extract from pkl
        X, y = load_features_cache(subject_id)
        if X is None:
            features, labels = load_subject(save_path)
            X, y = extract_windows(features, labels)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            save_features_cache(subject_id, X, y)

        results = compare_classifiers(X, y, cv_folds=5, verbose=True)
        _set_cached_comparison(subject_id, results)
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
    tracking_id = _normalise_tracking_id(data.get("tracking_id"))
    sensors = data.get("sensors", {})

    if data.get("tracking_id") and not tracking_id:
        return jsonify({"error": "tracking_id must contain at least one valid character."}), 400

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
    # synthetic-feature problem (the full 159-feature model can't
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
    response_ctx = dict(ctx)
    if tracking_id:
        history_summary = _append_history_entry(tracking_id, ctx)
        response_ctx["tracking_id"] = tracking_id
        response_ctx["history_length"] = history_summary["count"]
    return jsonify(response_ctx)


@app.route("/api/models", methods=["GET"])
def api_models():
    """List available trained models."""
    return jsonify({"models": _available_subjects()})


@app.route("/api/history/<tracking_id>", methods=["GET"])
def api_history(tracking_id):
    """Return the longitudinal history stored for a tracking ID."""
    safe_id = _normalise_tracking_id(tracking_id)
    if not safe_id:
        return jsonify({"error": "Invalid tracking ID."}), 400

    entries = db.filter_history(
        safe_id,
        start=request.args.get("start"),
        end=request.args.get("end"),
    )
    if not entries:
        return jsonify({"error": f"No history found for '{tracking_id}'."}), 404

    return jsonify({
        "tracking_id": safe_id,
        "filters": {
            "start": request.args.get("start"),
            "end": request.args.get("end"),
        },
        "entries": entries,
        "summary": _build_history_summary(entries),
    })


@app.route("/api/history/<tracking_id>/export", methods=["GET"])
def api_history_export(tracking_id):
    """Export the stored history for a tracking ID as CSV."""
    safe_id = _normalise_tracking_id(tracking_id)
    if not safe_id:
        return jsonify({"error": "Invalid tracking ID."}), 400

    entries = db.filter_history(
        safe_id,
        start=request.args.get("start"),
        end=request.args.get("end"),
    )
    if not entries:
        return jsonify({"error": f"No history found for '{tracking_id}'."}), 404

    csv_text = _history_entries_to_csv(entries)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="history_{safe_id}.csv"',
        },
    )


# ---------------------------------------------------------------------------
# Pre-trained model analysis — click a subject chip, get results
# ---------------------------------------------------------------------------

def _find_pkl(subject_id: str) -> str | None:
    """Locate a .pkl file for *subject_id* across known directories.

    Search order:
      1. saved_files/<sid>.pkl
      2. WESAD_DATA_DIR/<sid>/<sid>.pkl
    Returns the path if found, else ``None``.
    """
    safe = secure_filename(subject_id)
    # 1. saved uploads
    p = os.path.join(SAVED_FILES_DIR, f"{safe}.pkl")
    if os.path.isfile(p):
        return p
    # 2. WESAD raw dataset
    p = os.path.join(WESAD_DATA_DIR, safe, f"{safe}.pkl")
    if os.path.isfile(p):
        return p
    return None


@app.route("/api/analyze-pretrained/<subject_id>", methods=["POST"])
def api_analyze_pretrained(subject_id):
    """Analyse a subject — serves saved results instantly when available.

    Priority order:
      1. In-memory prediction cache (fastest)
      2. On-disk saved JSON result (survives restarts)
      3. Model + cached features (re-predict)
      4. Model + raw .pkl (extract + predict)
    """
    safe_id = secure_filename(subject_id)

    try:
        # 1. Instant path: in-memory cache
        cached_result = _prediction_cache.get(safe_id)
        if cached_result is not None:
            return jsonify(cached_result)

        # 2. Disk-saved result (from a previous upload/analysis)
        disk_result = db.load_result(safe_id)
        if disk_result is not None:
            _prediction_cache[safe_id] = disk_result
            return jsonify(disk_result)

        # 3. Need a model to compute — check it exists
        if not model_exists(safe_id):
            return jsonify({
                "error": f"No results or model found for '{subject_id}'. "
                         "Please upload the .pkl file first.",
            }), 404

        model, scaler = None, None
        X, y = None, None

        # Fast path: cached features
        if features_cache_exists(safe_id):
            model, scaler = load_model(safe_id)
            if model is not None:
                X, y = load_features_cache(safe_id)

        # Slow path: extract from pkl
        if X is None:
            pkl_path = _find_pkl(safe_id)
            if pkl_path is None:
                return jsonify({
                    "error": f"No data found for '{subject_id}'. "
                             "Please upload the .pkl file once so results can be saved.",
                }), 404
            features, labels = load_subject(pkl_path)
            X, y = extract_windows(features, labels)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            save_features_cache(safe_id, X, y)

        if model is None:
            model, scaler = load_model(safe_id)
            if model is None:
                invalidate_model_cache(safe_id)
                return jsonify({
                    "error": f"Model for '{subject_id}' is outdated. "
                             "Please re-upload the .pkl file to retrain.",
                }), 500

        results = predict(model, scaler, X)
        ctx = _build_result_dict(safe_id, "pre-trained", results)
        # Persist to memory + disk
        _prediction_cache[safe_id] = ctx
        db.save_result(safe_id, ctx)
        return jsonify(ctx)

    except Exception as exc:
        logger.exception("Error analysing pretrained subject")
        return jsonify({"error": f"Error processing: {exc}"}), 500


@app.route("/api/compare-pretrained/<subject_id>", methods=["POST"])
def api_compare_pretrained(subject_id):
    """Compare all classifiers for a subject with a pre-trained model."""
    rl = _check_rate_limit()
    if rl:
        return rl
    safe_id = secure_filename(subject_id)

    try:
        cached = _get_cached_comparison(safe_id)
        if cached is not None:
            return jsonify({"subject_id": safe_id, "results": cached})

        X, y = load_features_cache(safe_id)
        if X is None:
            pkl_path = _find_pkl(safe_id)
            if pkl_path is None:
                return jsonify({
                    "error": f"Feature cache not found for '{subject_id}'. "
                             "Please upload the .pkl file once so features can be cached.",
                }), 404
            features, labels = load_subject(pkl_path)
            X, y = extract_windows(features, labels)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            save_features_cache(safe_id, X, y)

        results = compare_classifiers(X, y, cv_folds=5, verbose=True)
        _set_cached_comparison(safe_id, results)
        return jsonify({"subject_id": safe_id, "results": results})

    except Exception as exc:
        logger.exception("Error during pretrained comparison")
        return jsonify({"error": f"Error during model comparison: {exc}"}), 500


# ---------------------------------------------------------------------------
# Saved-files endpoints — upload once, analyse many times
# ---------------------------------------------------------------------------

@app.route("/api/saved-files", methods=["GET"])
def api_saved_files():
    """Return a list of previously saved .pkl files."""
    files = []
    if os.path.isdir(SAVED_FILES_DIR):
        for fname in sorted(os.listdir(SAVED_FILES_DIR)):
            if fname.endswith(".pkl"):
                fpath = os.path.join(SAVED_FILES_DIR, fname)
                sid = os.path.splitext(fname)[0]
                files.append({
                    "filename": fname,
                    "subject_id": sid,
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 1),
                    "has_model": model_exists(sid),
                    "has_cache": features_cache_exists(sid),
                })
    return jsonify({"files": files})


@app.route("/api/analyze/<subject_id>", methods=["POST"])
def api_analyze_saved(subject_id):
    """Analyse a previously saved .pkl file by subject_id (no upload needed)."""
    safe_id = secure_filename(subject_id)
    pkl_path = os.path.join(SAVED_FILES_DIR, f"{safe_id}.pkl")
    if not os.path.isfile(pkl_path):
        return jsonify({"error": f"No saved file for '{subject_id}'."}), 404

    try:
        # Instant path: return pre-computed result from memory/disk cache
        cached_result = _prediction_cache.get(safe_id)
        if cached_result is not None:
            return jsonify(cached_result)
        disk_result = db.load_result(safe_id)
        if disk_result is not None:
            _prediction_cache[safe_id] = disk_result
            return jsonify(disk_result)

        model, scaler, metrics = None, None, None
        X, y = None, None

        if model_exists(safe_id) and features_cache_exists(safe_id):
            model, scaler = load_model(safe_id)
            if model is not None:
                X, y = load_features_cache(safe_id)

        if X is None:
            features, labels = load_subject(pkl_path)
            X, y = extract_windows(features, labels)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            save_features_cache(safe_id, X, y)

        if model is None:
            if model_exists(safe_id):
                model, scaler = load_model(safe_id)
                if model is None:
                    invalidate_model_cache(safe_id)

        if model is None:
            model, scaler, metrics = train_model(X, y, cv_folds=5, verbose=True)
            save_model(model, scaler, safe_id)

        results = predict(model, scaler, X)
        method = "pre-trained" if metrics is None else "trained on-the-fly"
        ctx = _build_result_dict(safe_id, method, results, metrics=metrics)
        # Persist to memory + disk
        _prediction_cache[safe_id] = ctx
        db.save_result(safe_id, ctx)
        return jsonify(ctx)

    except Exception as exc:
        logger.exception("Error analysing saved file")
        return jsonify({"error": f"Error processing file: {exc}"}), 500


@app.route("/api/compare-saved/<subject_id>", methods=["POST"])
def api_compare_saved(subject_id):
    """Compare classifiers on a previously saved .pkl file."""
    rl = _check_rate_limit()
    if rl:
        return rl
    safe_id = secure_filename(subject_id)
    pkl_path = os.path.join(SAVED_FILES_DIR, f"{safe_id}.pkl")
    if not os.path.isfile(pkl_path):
        return jsonify({"error": f"No saved file for '{subject_id}'."}), 404

    try:
        cached = _get_cached_comparison(safe_id)
        if cached is not None:
            return jsonify({"subject_id": safe_id, "results": cached})

        X, y = load_features_cache(safe_id)
        if X is None:
            features, labels = load_subject(pkl_path)
            X, y = extract_windows(features, labels)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            save_features_cache(safe_id, X, y)

        results = compare_classifiers(X, y, cv_folds=5, verbose=True)
        _set_cached_comparison(safe_id, results)
        return jsonify({"subject_id": safe_id, "results": results})

    except Exception as exc:
        logger.exception("Error during comparison of saved file")
        return jsonify({"error": f"Error during model comparison: {exc}"}), 500


@app.route("/api/saved-files/<subject_id>", methods=["DELETE"])
def api_delete_saved(subject_id):
    """Delete a saved .pkl file, cached features, and saved results."""
    safe_id = secure_filename(subject_id)
    pkl_path = os.path.join(SAVED_FILES_DIR, f"{safe_id}.pkl")
    if not os.path.isfile(pkl_path):
        return jsonify({"error": f"No saved file for '{subject_id}'."}), 404
    os.remove(pkl_path)
    # Also clean up saved results
    _prediction_cache.pop(safe_id, None)
    db.delete_result(safe_id)
    return jsonify({"deleted": safe_id})


@app.errorhandler(413)
def request_entity_too_large(error):
    max_content = app.config["MAX_CONTENT_LENGTH"]
    if max_content is None:
        limit_desc = "unlimited"
    else:
        limit_mb = max_content // (1024 * 1024)
        limit_desc = f"{limit_mb} MB" if limit_mb < 1024 else f"{limit_mb // 1024} GB"
    return jsonify({"error": f"File too large. Maximum upload size is {limit_desc}."}), 413


if __name__ == "__main__":



    _ensure_general_model()
    _warmup_cache()
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1",
            host="127.0.0.1", port=5000, use_reloader=False)
