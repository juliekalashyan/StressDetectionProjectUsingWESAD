"""Tests for the model and Flask application."""

import os
import sys
import pickle
import tempfile
import io

import numpy as np
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.data_processor import extract_chest_data, FEATURE_COLUMNS
from model.feature_extractor import (
    compute_window_features,
    extract_windows,
    features_from_manual_input,
    LABEL_MAP,
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
    assert feats.shape == (56,)
    # mean should be 1, std 0, min 1, max 1
    np.testing.assert_allclose(feats[:14], 1.0)
    np.testing.assert_allclose(feats[14:28], 0.0)


def test_extract_windows_basic():
    n = 7000  # 10 windows at window_sec=1, fs=700
    features = np.random.randn(n, 14).astype(np.float32)
    labels = np.ones(n, dtype=np.int64)
    labels[:3500] = 2  # First half stressed

    X, y = extract_windows(features, labels, window_sec=1, fs=700)
    assert X.shape[0] == 10
    assert X.shape[1] == 56
    assert y.shape[0] == 10


def test_features_from_manual_input():
    vals = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
    X = features_from_manual_input(vals)
    assert X.shape == (1, 56)
    # mean part should equal ordered values
    expected = np.arange(14, dtype=np.float32)
    np.testing.assert_allclose(X[0, :14], expected)


def test_label_map_contains_stress():
    assert 2 in LABEL_MAP
    assert LABEL_MAP[2] == "Stress"


# ---------------------------------------------------------------------------
# Flask app tests
# ---------------------------------------------------------------------------

def test_flask_index():
    from app import app
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"WESAD Stress Detection" in resp.data


def test_flask_upload_no_file():
    from app import app
    client = app.test_client()
    resp = client.post("/upload", data={}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Please select" in resp.data


def test_flask_manual_no_model():
    from app import app
    client = app.test_client()
    resp = client.post(
        "/manual",
        data={"subject_id": "NONEXISTENT"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"No trained model" in resp.data


# ---------------------------------------------------------------------------
# End-to-end: create a fake .pkl, upload, get results
# ---------------------------------------------------------------------------

def _make_fake_pkl(path, duration_sec=10):
    """Create a minimal WESAD-like .pkl file for testing.

    The WESAD chest sensors are at 700 Hz and wrist sensors at
    32 Hz (ACC), 64 Hz (BVP), and 4 Hz (EDA, TEMP).
    """
    n_chest = duration_sec * 700
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
        "label": np.random.choice([1, 2, 3], size=n_chest).astype(np.int64),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_end_to_end_upload():
    """Upload a fake .pkl file and verify the results page is returned."""
    from app import app

    # Create a fake pkl in a temp file (20 seconds of data = 2+ windows at 5s)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        _make_fake_pkl(tmp.name, duration_sec=20)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            data = f.read()

        client = app.test_client()
        resp = client.post(
            "/upload",
            data={"pkl_file": (io.BytesIO(data), "TEST.pkl")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Results" in resp.data or b"Stress" in resp.data
    finally:
        os.unlink(tmp_path)
