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
    # 8 time-domain stats * 14 channels + 3 freq-domain stats * 14 channels + 5 HRV = 159
    assert feats.shape == (159,)
    # mean should be 1, std 0 for the time-domain part
    np.testing.assert_allclose(feats[:14], 1.0)
    np.testing.assert_allclose(feats[14:28], 0.0, atol=1e-6)


def test_extract_windows_basic():
    n = 7000  # with 50% overlap at window_sec=1, fs=700 -> ~19 windows
    features = np.random.randn(n, 14).astype(np.float32)
    labels = np.ones(n, dtype=np.int64)
    labels[:3500] = 2  # First half stressed

    X, y = extract_windows(features, labels, window_sec=1, fs=700)
    assert X.shape[0] >= 10  # at least 10 windows (overlap produces more)
    assert X.shape[1] == 159
    assert y.shape[0] == X.shape[0]


def test_extract_windows_no_overlap():
    """Legacy non-overlapping mode still works."""
    n = 7000
    features = np.random.randn(n, 14).astype(np.float32)
    labels = np.ones(n, dtype=np.int64)
    labels[:3500] = 2

    X, y = extract_windows(features, labels, window_sec=1, fs=700, overlap=0.0)
    assert X.shape[0] == 10
    assert X.shape[1] == 159


def test_features_from_manual_input():
    vals = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
    X = features_from_manual_input(vals)
    assert X.shape == (1, 159)
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
    assert len(FEATURE_NAMES) == 159


def test_label_map_contains_stress():
    assert 2 in LABEL_MAP
    assert LABEL_MAP[2] == "Stress"


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def test_train_model_returns_metrics():
    from model.trainer import train_model
    X = np.random.randn(200, 159).astype(np.float32)
    y = np.random.choice([1, 2, 3], size=200).astype(np.int64)
    clf, scaler, metrics = train_model(X, y, cv_folds=2, verbose=False)
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert "train_time_sec" in metrics
    assert "classifier_name" in metrics
    assert metrics["classifier_name"] == "Random Forest"
    assert "roc_auc" in metrics
    assert "n_features_used" in metrics


def test_compare_classifiers_returns_sorted():
    from model.trainer import compare_classifiers
    X = np.random.randn(200, 159).astype(np.float32)
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
    X = np.random.randn(200, 159).astype(np.float32)
    y = np.random.choice([1, 2, 3], size=200).astype(np.int64)
    clf, scaler, metrics = train_model(
        X, y, cv_folds=0, classifier_name="SVM",
    )
    assert metrics["classifier_name"] == "SVM"
    assert metrics["accuracy"] > 0
    assert "roc_auc" in metrics


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


def test_api_predict_tracking_history(monkeypatch):
    import app as app_module
    import database as db_module

    tracking_id = "TRACK_TEST"
    # Clean any leftover history rows
    conn = db_module.get_connection()
    conn.execute("DELETE FROM history WHERE tracking_id = ?", (tracking_id,))
    conn.commit()

    monkeypatch.setattr(app_module, "general_model_exists", lambda: True)
    monkeypatch.setattr(app_module, "load_manual_model", lambda: (object(), object()))

    def fake_predict(_model, _scaler, X):
        stressed = float(X[0, 0]) >= 1.0
        label = "Stress" if stressed else "Baseline"
        return {
            "predictions": [2 if stressed else 1],
            "label_names": [label],
            "is_stressed": [stressed],
            "stress_ratio": 1.0 if stressed else 0.0,
            "stress_confidence": 0.85 if stressed else 0.15,
        }

    monkeypatch.setattr(app_module, "predict", fake_predict)

    client = app_module.app.test_client()

    try:
        resp1 = client.post(
            "/api/predict",
            data=json.dumps({
                "tracking_id": tracking_id,
                "sensors": {"Chest Acc X": 0.0},
            }),
            content_type="application/json",
        )
        assert resp1.status_code == 200
        first = resp1.get_json()
        assert first["tracking_id"] == tracking_id
        assert first["history_length"] == 1

        resp2 = client.post(
            "/api/predict",
            data=json.dumps({
                "tracking_id": tracking_id,
                "sensors": {"Chest Acc X": 1.0},
            }),
            content_type="application/json",
        )
        assert resp2.status_code == 200
        second = resp2.get_json()
        assert second["history_length"] == 2

        history_resp = client.get(f"/api/history/{tracking_id}")
        assert history_resp.status_code == 200
        history = history_resp.get_json()
        assert history["tracking_id"] == tracking_id
        assert len(history["entries"]) == 2
        assert history["summary"]["count"] == 2
        assert history["summary"]["delta"] > 0
    finally:
        conn = db_module.get_connection()
        conn.execute("DELETE FROM history WHERE tracking_id = ?", (tracking_id,))
        conn.commit()


def test_api_history_filter_and_export():
    import app as app_module
    import database as db_module

    tracking_id = "TRACK_FILTER"
    # Clean any leftover rows
    conn = db_module.get_connection()
    conn.execute("DELETE FROM history WHERE tracking_id = ?", (tracking_id,))
    conn.commit()

    entries = [
        {
            "tracking_id": tracking_id,
            "captured_at": "2026-03-16T09:00:00Z",
            "subject_id": "Manual (general model)",
            "method": "manual input",
            "is_manual": True,
            "stress_ratio": 0.15,
            "stress_level": "Low Stress",
            "overall_stress": False,
            "predicted_label": "Baseline",
            "total_windows": 1,
            "label_counts": {"Baseline": 1},
        },
        {
            "tracking_id": tracking_id,
            "captured_at": "2026-03-17T12:00:00Z",
            "subject_id": "Manual (general model)",
            "method": "manual input",
            "is_manual": True,
            "stress_ratio": 0.55,
            "stress_level": "High Stress",
            "overall_stress": True,
            "predicted_label": "Stress",
            "total_windows": 1,
            "label_counts": {"Stress": 1},
        },
        {
            "tracking_id": tracking_id,
            "captured_at": "2026-03-18T18:30:00Z",
            "subject_id": "Manual (general model)",
            "method": "manual input",
            "is_manual": True,
            "stress_ratio": 0.35,
            "stress_level": "Moderate Stress",
            "overall_stress": True,
            "predicted_label": "Stress",
            "total_windows": 1,
            "label_counts": {"Stress": 1},
        },
    ]
    for entry in entries:
        db_module.append_history_entry(tracking_id, entry)

    client = app_module.app.test_client()

    try:
        filtered_resp = client.get(
            f"/api/history/{tracking_id}?start=2026-03-17T00:00:00Z&end=2026-03-17T23:59:59Z"
        )
        assert filtered_resp.status_code == 200
        filtered = filtered_resp.get_json()
        assert len(filtered["entries"]) == 1
        assert filtered["entries"][0]["captured_at"] == "2026-03-17T12:00:00Z"

        csv_resp = client.get(
            f"/api/history/{tracking_id}/export?format=csv&start=2026-03-17T00:00:00Z"
        )
        assert csv_resp.status_code == 200
        assert csv_resp.mimetype == "text/csv"
        csv_text = csv_resp.get_data(as_text=True)
        assert "tracking_id,captured_at,subject_id" in csv_text
        assert "2026-03-17T12:00:00Z" in csv_text
        assert "2026-03-16T09:00:00Z" not in csv_text
    finally:
        conn = db_module.get_connection()
        conn.execute("DELETE FROM history WHERE tracking_id = ?", (tracking_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# End-to-end: create a fake .pkl, upload via API, get JSON results
# ---------------------------------------------------------------------------

def _make_fake_pkl(path, duration_sec=10):
    """Create a minimal WESAD-like .pkl file for testing."""
    rng = np.random.RandomState(42)  # deterministic for reproducible tests
    n_chest = duration_sec * 700
    # Ensure balanced labels so stratified split always works
    labels = np.tile([1, 2, 3], n_chest // 3 + 1)[:n_chest].astype(np.int64)
    data = {
        "signal": {
            "chest": {
                "ACC": rng.randn(n_chest, 3).astype(np.float32),
                "ECG": rng.randn(n_chest, 1).astype(np.float32),
                "EMG": rng.randn(n_chest, 1).astype(np.float32),
                "EDA": rng.randn(n_chest, 1).astype(np.float32),
                "Temp": rng.randn(n_chest, 1).astype(np.float32),
                "Resp": rng.randn(n_chest, 1).astype(np.float32),
            },
            "wrist": {
                "ACC": rng.randn(duration_sec * 32, 3).astype(np.float32),
                "BVP": rng.randn(duration_sec * 64, 1).astype(np.float32),
                "EDA": rng.randn(duration_sec * 4, 1).astype(np.float32),
                "TEMP": rng.randn(duration_sec * 4, 1).astype(np.float32),
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
        for f in ("model_TEST.joblib", "scaler_TEST.joblib", "features_TEST.npz"):
            p = os.path.join(os.path.dirname(__file__), "..", "trained_models", f)
            if os.path.exists(p):
                os.unlink(p)
        # Clean up saved file
        sp = os.path.join(os.path.dirname(__file__), "..", "saved_files", "TEST.pkl")
        if os.path.exists(sp):
            os.unlink(sp)
        # Clean up cached result in DB
        import database as db_module
        db_module.delete_result("TEST")
        from app import _prediction_cache
        _prediction_cache.pop("TEST", None)


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
        assert len(result["results"]) >= 6  # 6 base + optional LightGBM
    finally:
        os.unlink(tmp_path)
        # Clean up trained model artifacts
        for f in ("model_TEST.joblib", "scaler_TEST.joblib", "features_TEST.npz"):
            p = os.path.join(os.path.dirname(__file__), "..", "trained_models", f)
            if os.path.exists(p):
                os.unlink(p)
        # Clean up saved file
        sp = os.path.join(os.path.dirname(__file__), "..", "saved_files", "TEST.pkl")
        if os.path.exists(sp):
            os.unlink(sp)
        # Clean up cached result in DB
        import database as db_module
        db_module.delete_result("TEST")
        # Clean up cached comparison in DB
        conn = db_module.get_connection()
        conn.execute("DELETE FROM comparisons WHERE subject_id = ?", ("TEST",))
        conn.commit()
        from app import _prediction_cache, _compare_cache
        _prediction_cache.pop("TEST", None)
        _compare_cache.pop("TEST", None)


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
