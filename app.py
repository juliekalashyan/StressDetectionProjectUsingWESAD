#!/usr/bin/env python3
"""Flask web application for WESAD stress detection.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import tempfile

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from model.data_processor import load_subject, FEATURE_COLUMNS
from model.feature_extractor import (
    extract_windows,
    features_from_manual_input,
    LABEL_MAP,
    DEFAULT_WINDOW_SEC,
)
from model.predictor import model_exists, load_model, predict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB – WESAD .pkl files can be ~1 GB
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "wesad-stress-dev-key")

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "wesad_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Pre-trained model directory
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")


def _stress_level_info(ratio: float) -> dict:
    """Return stress level name, color, and gauge CSS class from a ratio."""
    if ratio < 0.15:
        return {"stress_level": "Low Stress", "stress_level_color": "#34d399", "gauge_class": "gauge-low"}
    elif ratio < 0.35:
        return {"stress_level": "Moderate Stress", "stress_level_color": "#fbbf24", "gauge_class": "gauge-moderate"}
    elif ratio < 0.60:
        return {"stress_level": "High Stress", "stress_level_color": "#f87171", "gauge_class": "gauge-high"}
    else:
        return {"stress_level": "Critical Stress", "stress_level_color": "#dc2626", "gauge_class": "gauge-critical"}


def _build_result_context(subject_id: str, method: str, results: dict) -> dict:
    """Build the full template context dict for result.html."""
    # Only keep valid WESAD labels (filter out Unknown / legacy labels like Meditation-2)
    valid_labels = {"Transient", "Baseline", "Stress", "Amusement", "Meditation"}

    # Filter predictions + label_names in sync so timeline/table/chart stay consistent
    filtered_preds = []
    filtered_names = []
    for pred, name in zip(results["predictions"], results["label_names"]):
        if name in valid_labels:
            filtered_preds.append(pred)
            filtered_names.append(name)

    label_counts = {}
    for name in filtered_names:
        label_counts[name] = label_counts.get(name, 0) + 1

    level_info = _stress_level_info(results["stress_ratio"])
    return {
        "subject_id": subject_id,
        "method": method,
        "total_windows": len(filtered_preds),
        "label_counts": label_counts,
        "stress_ratio": results["stress_ratio"],
        "overall_stress": results["stress_ratio"] > 0.30,
        "predictions": filtered_preds,
        "label_names": filtered_names,
        "window_sec": DEFAULT_WINDOW_SEC,
        **level_info,
    }


def _available_subjects():
    """Return a sorted list of subject IDs that have trained models."""
    if not os.path.isdir(TRAINED_MODELS_DIR):
        return []
    subjects = set()
    for fname in os.listdir(TRAINED_MODELS_DIR):
        if fname.startswith("model_") and fname.endswith(".joblib"):
            sid = fname[len("model_"):-len(".joblib")]
            subjects.add(sid)
    return sorted(subjects)


@app.route("/")
def index():
    """Landing page with upload form and manual input."""
    subjects = _available_subjects()
    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS,
        subjects=subjects,
        label_map=LABEL_MAP,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Handle .pkl file upload, train-on-the-fly or use existing model."""
    file = request.files.get("pkl_file")
    if not file or file.filename == "":
        flash("Please select a WESAD subject .pkl file.", "danger")
        return redirect(url_for("index"))

    if not file.filename.endswith(".pkl"):
        flash("Only .pkl files are supported.", "danger")
        return redirect(url_for("index"))

    # Save uploaded file to a temp location
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    # Infer subject ID from filename (e.g. S2.pkl -> S2)
    subject_id = os.path.splitext(file.filename)[0]

    try:
        features, labels = load_subject(save_path)
        X, y = extract_windows(features, labels)

        # Remove transient windows
        mask = y != 0
        X_valid = X[mask]
        y_valid = y[mask]

        if model_exists(subject_id):
            # Use pre-trained model
            model, scaler = load_model(subject_id)
            results = predict(model, scaler, X_valid)
            method = "pre-trained"
        else:
            # Train on-the-fly
            

            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            clf.fit(X_train_s, y_train)

            # Predict on ALL valid windows
            results = predict(clf, scaler, X_valid)
            method = "trained on-the-fly"

            # Optionally save the model for future use
            import joblib

            os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
            joblib.dump(
                clf,
                os.path.join(TRAINED_MODELS_DIR, f"model_{subject_id}.joblib"),
            )
            joblib.dump(
                scaler,
                os.path.join(TRAINED_MODELS_DIR, f"scaler_{subject_id}.joblib"),
            )

        ctx = _build_result_context(subject_id, method, results)
        return render_template("result.html", **ctx)

    except Exception as exc:
        flash(f"Error processing file: {exc}", "danger")
        return redirect(url_for("index"))
    finally:
        # Clean up uploaded file
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/manual", methods=["POST"])
def manual():
    """Handle manual sensor value input."""
    subject_id = request.form.get("subject_id", "").strip()
    if not subject_id:
        flash("Please select a model.", "danger")
        return redirect(url_for("index"))

    # Allow the user to pick any available model
    if not model_exists(subject_id):
        available = _available_subjects()
        if available:
            flash(
                f"No model for '{subject_id}'. "
                f"Available models: {', '.join(available)}. "
                "Please select one from the dropdown.",
                "danger",
            )
        else:
            flash(
                "No trained models found. "
                "Please upload a WESAD .pkl file first to train a model.",
                "danger",
            )
        return redirect(url_for("index"))

    sensor_values = {}
    for col in FEATURE_COLUMNS:
        val = request.form.get(col, "0")
        try:
            sensor_values[col] = float(val)
        except ValueError:
            sensor_values[col] = 0.0

    X = features_from_manual_input(sensor_values)
    model, scaler = load_model(subject_id)
    results = predict(model, scaler, X)

    ctx = _build_result_context(f"Manual (model: {subject_id})", "pre-trained", results)
    return render_template("result.html", **ctx)


@app.errorhandler(413)
def request_entity_too_large(error):
    flash("File is too large. Maximum upload size is 2 GB.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", host="127.0.0.1", port=5000)
