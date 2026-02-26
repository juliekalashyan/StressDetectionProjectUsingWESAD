"""Prediction helpers – load a trained model and classify new data."""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from .feature_extractor import LABEL_MAP, STRESS_LABELS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")


def _model_path(subject_id):
    return os.path.join(MODEL_DIR, f"model_{subject_id}.joblib")


def _scaler_path(subject_id):
    return os.path.join(MODEL_DIR, f"scaler_{subject_id}.joblib")


def model_exists(subject_id):
    """Return True if a trained model exists for the given subject."""
    return os.path.isfile(_model_path(subject_id)) and os.path.isfile(
        _scaler_path(subject_id)
    )


from functools import lru_cache

@lru_cache(maxsize=16)
def load_model(subject_id):
    """Load and return (model, scaler) for a subject."""
    model = joblib.load(_model_path(subject_id))
    scaler = joblib.load(_scaler_path(subject_id))
    return model, scaler


def predict(model, scaler, X):
    """Run prediction on window features.

    Parameters
    ----------
    model : sklearn estimator
    scaler : StandardScaler
    X : np.ndarray of shape (n_windows, 56)

    Returns
    -------
    dict with keys:
        predictions : list[int]
        label_names : list[str]
        is_stressed : list[bool]
        stress_ratio : float  (fraction of windows predicted as Stress)
    """
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    label_names = [LABEL_MAP.get(int(p), f"Unknown({p})") for p in preds]
    is_stressed = [int(p) in STRESS_LABELS for p in preds]
    stress_count = sum(is_stressed)
    stress_ratio = stress_count / len(preds) if len(preds) > 0 else 0.0

    return {
        "predictions": [int(p) for p in preds],
        "label_names": label_names,
        "is_stressed": is_stressed,
        "stress_ratio": stress_ratio,
    }
