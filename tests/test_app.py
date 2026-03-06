"""Tests for the model and Flask application (API-only, no Jinja2)."""

import os
import sys
import pickle
import tempfile
import io
import json

import numpy as np
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.data_processor import extract_chest_data, FEATURE_COLUMNS
from model.feature_extractor import (
    compute_window_features,
    extract_windows,
    features_from_manual_input,
    features_from_manual_input_means_only,
    LABEL_MAP,
    FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Data processor tests
# ---------------------------------------------------------------------------

def _make_chest_dict(n=100):
    """Create a minimal chest sensor dictionary."""
    return {
        "ACC": np.random.randn(n, 3).astype(np.float32),
        "ECG": np.random.randn(n, 1).astype(np.float32),
        "EMG": np.random.randn(n, 1).astype(np.float32),
        "EDA": np.random.randn(n, 1).astype(np.float32),
        "Temp": np.random.randn(n, 1).astype(np.float32),
        "Resp": np.random.randn(n, 1).astype(np.float32),
    }


def test_extract_chest_data_shape():
    chest = _make_chest_dict(200)
    df = extract_chest_data(chest)
    assert df.shape == (200, 8)
    assert list(df.columns) == FEATURE_COLUMNS[:8]


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------

def test_compute_window_features():
    window = np.ones((50, 14), dtype=np.float32)
    feats = compute_window_features(window)
    # 8 time-domain stats * 14 channels + 3 freq-domain stats * 14 channels = 154
    assert feats.shape == (154,)
    # mean should be 1, std 0 for the time-domain part
    np.testing.assert_allclose(feats[:14], 1.0)
    np.testing.assert_allclose(feats[14:28], 0.0, atol=1e-6)


def test_extract_windows_basic():
    n = 7000  # 10 windows at window_sec=1, fs=700
    features = np.random.randn(n, 14).astype(np.float32)
    labels = np.ones(n, dtype=np.int64)
    labels[:3500] = 2  # First half stressed

    X, y = extract_windows(features, labels, window_sec=1, fs=700)
    assert X.shape[0] == 10
    assert X.shape[1] == 154
    assert y.shape[0] == 10


def test_features_from_manual_input():
    vals = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
    X = features_from_manual_input(vals)
    assert X.shape == (1, 154)
    # mean part should equal ordered values
    expected = np.arange(14, dtype=np.float32)
    np.testing.assert_allclose(X[0, :14], expected)


def test_features_from_manual_input_means_only():
    vals = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
    X = features_from_manual_input_means_only(vals)
    assert X.shape == (1, 14)
    expected = np.arange(14, dtype=np.float32)
    np.testing.assert_allclose(X[0], expected)


def test_feature_names_length():
    """FEATURE_NAMES should match the total feature count."""
    assert len(FEATURE_NAMES) == 154


def test_label_map_contains_stress():
    assert 2 in LABEL_MAP
    assert LABEL_MAP[2] == "Stress"


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def test_train_model_returns_metrics():
    from model.trainer import train_model
    X = np.random.randn(200, 154).astype(np.float32)
    y = np.random.choice([1, 2, 3], size=200).astype(np.int64)
    clf, scaler, metrics = train_model(X, y, cv_folds=2, verbose=False)
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert "train_time_sec" in metrics
    assert "classifier_name" in metrics
    assert metrics["classifier_name"] == "Random Forest"


def test_compare_classifiers_returns_sorted():
    from model.trainer import compare_classifiers
    X = np.random.randn(200, 154).astype(np.float32)
    y = np.random.choice([1, 2, 3], size=200).astype(np.int64)
    results = compare_classifiers(
        X, y, cv_folds=2,
        classifiers=["Random Forest", "KNN", "Decision Tree"],
    )
    assert len(results) == 3
    # Should be sorted by accuracy descending
    for i in range(len(results) - 1):
        assert results[i]["accuracy"] >= results[i + 1]["accuracy"]


def test_train_model_svm():
    from model.trainer import train_model
    X = np.random.randn(200, 154).astype(np.float32)
    y = np.random.choice([1, 2, 3], size=200).astype(np.int64)
    clf, scaler, metrics = train_model(
        X, y, cv_folds=0, classifier_name="SVM",
    )
    assert metrics["classifier_name"] == "SVM"
    assert metrics["accuracy"] > 0


def test_train_general_model():
    """Train a general model from a fake multi-subject dataset."""
    from model.trainer import (
        train_general_model, general_model_exists,
        general_manual_model_exists, GENERAL_ID,
    )

    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as model_dir:
        # Create two fake subject folders with .pkl files
        for sid in ("S99", "S98"):
            subdir = os.path.join(data_dir, sid)
            os.makedirs(subdir)
            pkl_path = os.path.join(subdir, f"{sid}.pkl")
            _make_fake_pkl(pkl_path, duration_sec=120)

        pipeline, _, metrics = train_general_model(
            data_dir, cv_folds=2, verbose=False, model_dir=model_dir,
        )
        assert metrics["accuracy"] > 0
        assert metrics["n_subjects"] == 2
        assert metrics["n_windows_total"] > 0
        assert general_model_exists(model_dir=model_dir)
        assert general_manual_model_exists(model_dir=model_dir)
        assert "manual_model_accuracy" in metrics


# ---------------------------------------------------------------------------
# Flask app tests (API-only, no Jinja2)
# ---------------------------------------------------------------------------

def test_flask_index():
    from app import app
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"WESAD Stress Detection" in resp.data


def test_api_config():
    from app import app
    client = app.test_client()
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "feature_columns" in data
    assert "subjects" in data
    assert "label_map" in data
    assert "sensor_meta" in data
    assert "classifier_names" in data
    assert "general_model_ready" in data
    assert isinstance(data["feature_columns"], list)
    assert len(data["feature_columns"]) == 14


def test_api_upload_no_file():
    from app import app
    client = app.test_client()
    resp = client.post("/api/upload", data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_api_upload_wrong_extension():
    from app import app
    client = app.test_client()
    resp = client.post(
        "/api/upload",
        data={"pkl_file": (io.BytesIO(b"fake"), "test.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "pkl" in data["error"].lower()


def test_api_predict_no_json():
    from app import app
    client = app.test_client()
    resp = client.post("/api/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_api_predict_missing_subject():
    """No subject_id means use general model; 503 if it doesn't exist."""
    from app import app
    from model.trainer import general_model_exists
    client = app.test_client()
    resp = client.post(
        "/api/predict",
        data=json.dumps({"sensors": {}}),
        content_type="application/json",
    )
    # If the general model exists → 200; if not → 503
    if general_model_exists():
        assert resp.status_code == 200
    else:
        assert resp.status_code == 503


def test_api_predict_unknown_model():
    from app import app
    client = app.test_client()
    resp = client.post(
        "/api/predict",
        data=json.dumps({"subject_id": "NONEXISTENT", "sensors": {}}),
        content_type="application/json",
    )
    assert resp.status_code == 404


def test_api_models():
    from app import app
    client = app.test_client()
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "models" in data
    assert isinstance(data["models"], list)


# ---------------------------------------------------------------------------
# End-to-end: create a fake .pkl, upload via API, get JSON results
# ---------------------------------------------------------------------------

def _make_fake_pkl(path, duration_sec=10):
    """Create a minimal WESAD-like .pkl file for testing."""
    n_chest = duration_sec * 700
    # Ensure balanced labels so stratified split always works
    labels = np.tile([1, 2, 3], n_chest // 3 + 1)[:n_chest].astype(np.int64)
    data = {
        "signal": {
            "chest": {
                "ACC": np.random.randn(n_chest, 3).astype(np.float32),
                "ECG": np.random.randn(n_chest, 1).astype(np.float32),
                "EMG": np.random.randn(n_chest, 1).astype(np.float32),
                "EDA": np.random.randn(n_chest, 1).astype(np.float32),
                "Temp": np.random.randn(n_chest, 1).astype(np.float32),
                "Resp": np.random.randn(n_chest, 1).astype(np.float32),
            },
            "wrist": {
                "ACC": np.random.randn(duration_sec * 32, 3).astype(np.float32),
                "BVP": np.random.randn(duration_sec * 64, 1).astype(np.float32),
                "EDA": np.random.randn(duration_sec * 4, 1).astype(np.float32),
                "TEMP": np.random.randn(duration_sec * 4, 1).astype(np.float32),
            },
        },
        "label": labels,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_end_to_end_upload():
    """Upload a fake .pkl file via /api/upload and verify JSON results."""
    from app import app

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        _make_fake_pkl(tmp.name, duration_sec=120)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            data = f.read()

        client = app.test_client()
        resp = client.post(
            "/api/upload",
            data={"pkl_file": (io.BytesIO(data), "TEST.pkl")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert "predictions" in result
        assert "label_names" in result
        assert "stress_ratio" in result
        assert "stress_level" in result
        assert result["total_windows"] > 0
    finally:
        os.unlink(tmp_path)
        # Clean up trained model artifacts
        for f in ("model_TEST.joblib", "scaler_TEST.joblib"):
            p = os.path.join(os.path.dirname(__file__), "..", "trained_models", f)
            if os.path.exists(p):
                os.unlink(p)


def test_end_to_end_compare():
    """Upload a fake .pkl file via /api/compare and verify JSON results."""
    from app import app

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        _make_fake_pkl(tmp.name, duration_sec=120)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            data = f.read()

        client = app.test_client()
        resp = client.post(
            "/api/compare",
            data={"pkl_file": (io.BytesIO(data), "TEST.pkl")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert "subject_id" in result
        assert "results" in result
        assert len(result["results"]) == 6  # 6 classifiers
    finally:
        os.unlink(tmp_path)
        # Clean up trained model artifacts
        for f in ("model_TEST.joblib", "scaler_TEST.joblib"):
            p = os.path.join(os.path.dirname(__file__), "..", "trained_models", f)
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Stress-level classification tests
# ---------------------------------------------------------------------------

def test_stress_level_info_file_upload_mode():
    """_stress_level_info returns correct levels for file-upload (multi-window) ratios."""
    from app import _stress_level_info

    low = _stress_level_info(0.10)
    assert low["stress_level"] == "Low Stress"

    moderate = _stress_level_info(0.25)
    assert moderate["stress_level"] == "Moderate Stress"

    high = _stress_level_info(0.50)
    assert high["stress_level"] == "High Stress"

    critical = _stress_level_info(0.80)
    assert critical["stress_level"] == "Critical Stress"


def test_stress_level_info_manual_mode():
    """_stress_level_info returns correct levels for manual (probability) ratios."""
    from app import _stress_level_info

    low = _stress_level_info(0.20, is_manual=True)
    assert low["stress_level"] == "Low Stress"

    moderate = _stress_level_info(0.40, is_manual=True)
    assert moderate["stress_level"] == "Moderate Stress"

    high = _stress_level_info(0.60, is_manual=True)
    assert high["stress_level"] == "High Stress"

    critical = _stress_level_info(0.90, is_manual=True)
    assert critical["stress_level"] == "Critical Stress"


def test_build_result_dict_manual_baseline_forces_low_stress():
    """When manual input predicts Baseline, stress_level must be Low Stress."""
    from app import _build_result_dict

    results = {
        "predictions": [1],
        "label_names": ["Baseline"],
        "is_stressed": [False],
        "stress_ratio": 0.0,
        "stress_confidence": 0.1,
    }
    ctx = _build_result_dict("test", "manual", results)
    assert ctx["stress_level"] == "Low Stress"
    assert ctx["overall_stress"] is False


def test_build_result_dict_manual_stress_is_stressed():
    """When manual input predicts Stress, overall_stress must be True."""
    from app import _build_result_dict

    results = {
        "predictions": [2],
        "label_names": ["Stress"],
        "is_stressed": [True],
        "stress_ratio": 1.0,
        "stress_confidence": 0.85,
    }
    ctx = _build_result_dict("test", "manual", results)
    assert ctx["overall_stress"] is True
    assert ctx["stress_level"] != "Low Stress"


# ---------------------------------------------------------------------------
# Malformed .pkl rejection tests
# ---------------------------------------------------------------------------

def test_upload_malformed_pkl_missing_signal():
    """A .pkl file without 'signal' key should return a clear error."""
    from app import app

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump({"label": np.ones(100)}, tmp)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            data = f.read()

        client = app.test_client()
        resp = client.post(
            "/api/upload",
            data={"pkl_file": (io.BytesIO(data), "BAD.pkl")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500
        result = resp.get_json()
        assert "error" in result
    finally:
        os.unlink(tmp_path)


def test_upload_malformed_pkl_not_dict():
    """A .pkl file that doesn't contain a dict should return an error."""
    from app import app

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump([1, 2, 3], tmp)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            data = f.read()

        client = app.test_client()
        resp = client.post(
            "/api/upload",
            data={"pkl_file": (io.BytesIO(data), "BAD2.pkl")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500
        result = resp.get_json()
        assert "error" in result
    finally:
        os.unlink(tmp_path)
