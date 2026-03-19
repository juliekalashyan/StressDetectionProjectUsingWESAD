"""Prediction helpers – load a trained model and classify new data."""

import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

from .feature_extractor import LABEL_MAP, STRESS_LABELS, FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")
EXPECTED_N_FEATURES = len(FEATURE_NAMES)

# Minimum feature count we'll still accept (older models trained before HRV
# features were added).  Models with fewer features than this are truly stale.
_MIN_ACCEPTED_FEATURES = 100


def _model_path(subject_id):
    return os.path.join(MODEL_DIR, f"model_{subject_id}.joblib")


def _scaler_path(subject_id):
    return os.path.join(MODEL_DIR, f"scaler_{subject_id}.joblib")


def _features_cache_path(subject_id):
    return os.path.join(MODEL_DIR, f"features_{subject_id}.npz")


def model_exists(subject_id):
    """Return True if a trained model exists for the given subject."""
    return os.path.isfile(_model_path(subject_id)) and os.path.isfile(
        _scaler_path(subject_id)
    )


def features_cache_exists(subject_id):
    """Return True if cached features exist for the given subject."""
    return os.path.isfile(_features_cache_path(subject_id))


def save_features_cache(subject_id, X, y):
    """Save extracted features to disk so future uploads skip feature extraction."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez_compressed(_features_cache_path(subject_id), X=X, y=y)


def load_features_cache(subject_id):
    """Load cached features. Returns (X, y) or (None, None) if unavailable."""
    path = _features_cache_path(subject_id)
    if not os.path.isfile(path):
        return None, None
    data = np.load(path)
    return data["X"], data["y"]


from functools import lru_cache


def invalidate_model_cache(subject_id: str) -> None:
    """Delete stale model files and clear the LRU cache for *subject_id*."""
    load_model.cache_clear()
    for path in (_model_path(subject_id), _scaler_path(subject_id), _features_cache_path(subject_id)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@lru_cache(maxsize=16)
def load_model(subject_id):
    """Load and return (model, scaler) for a subject.

    Accepts models trained on fewer features than the current
    EXPECTED_N_FEATURES (e.g. 154 vs 159) — the predict function
    will truncate the feature matrix to match.  Only truly stale
    models (< _MIN_ACCEPTED_FEATURES) are rejected.
    """
    model = joblib.load(_model_path(subject_id))
    scaler = joblib.load(_scaler_path(subject_id))

    n_feat = _model_n_features(model, scaler)
    if n_feat is not None and n_feat < _MIN_ACCEPTED_FEATURES:
        return None, None  # truly stale — caller will retrain

    return model, scaler


def load_manual_model():
    """Load the means-only general model used for manual input predictions.

    This model was trained on just the 14 mean features per window, so
    it accepts 14 raw sensor values directly — no synthetic statistics
    needed.  Feature-count validation is skipped because this model
    intentionally has fewer features than the full model.

    Returns (model, scaler) or (None, None) if not available.
    """
    from .trainer import GENERAL_MANUAL_ID

    mp = _model_path(GENERAL_MANUAL_ID)
    sp = _scaler_path(GENERAL_MANUAL_ID)
    if not os.path.isfile(mp) or not os.path.isfile(sp):
        return None, None
    return joblib.load(mp), joblib.load(sp)


def _model_n_features(model, scaler) -> int | None:
    """Return the number of input features the model expects, or None if unknown."""
    # New-style: full sklearn Pipeline
    if isinstance(model, Pipeline):
        # The StandardScaler step stores n_features_in_
        for name, step in model.steps:
            if hasattr(step, "n_features_in_"):
                return step.n_features_in_
    # Legacy: separate scaler
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        return scaler.n_features_in_
    return None


def predict(model, scaler, X):
    """Run prediction on window features.

    Parameters
    ----------
    model : sklearn estimator
    scaler : StandardScaler
    X : np.ndarray of shape (n_windows, n_features)

    Returns
    -------
    dict with keys:
        predictions : list[int]
        label_names : list[str]
        is_stressed : list[bool]
        stress_ratio : float  (fraction of windows predicted as Stress)
    """
    # Auto-truncate features if the model was trained on fewer columns
    # (e.g. 154-feature models vs current 159-feature extraction).
    model_n = _model_n_features(model, scaler)
    if model_n is not None and X.shape[1] > model_n:
        X = X[:, :model_n]

    # model may be a full sklearn Pipeline (imputer+scaler+clf) or a plain
    # estimator paired with a separate scaler (legacy saved models).
    if scaler is None:
        # New-style: pipeline handles all preprocessing internally
        preds = model.predict(X)
    else:
        # Legacy: apply scaler then predict
        X_scaled = np.nan_to_num(
            scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        preds = model.predict(X_scaled)

    label_names = [LABEL_MAP.get(int(p), f"Unknown({p})") for p in preds]
    is_stressed = [int(p) in STRESS_LABELS for p in preds]
    stress_count = sum(is_stressed)
    baseline_count = sum(1 for p in preds if int(p) == 1)

    # Ratio of stress vs (stress + baseline) — the meaningful clinical comparison.
    # Amusement windows are excluded so they don't dilute the score.
    denom = stress_count + baseline_count
    stress_ratio = stress_count / denom if denom > 0 else 0.0

    # For single-window predictions (manual input), use class probabilities
    # instead of the binary 0/1 ratio so we get a meaningful confidence score.
    stress_confidence = None
    if len(preds) == 1:
        clf = model[-1] if isinstance(model, Pipeline) else model
        try:
            if isinstance(model, Pipeline):
                proba = model.predict_proba(X)
            elif scaler is not None:
                X_s = np.nan_to_num(
                    scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                proba = model.predict_proba(X_s)
            else:
                proba = model.predict_proba(X)
            # Find the index of the Stress class (label 2) in the model's classes
            classes = list(clf.classes_)
            if 2 in classes:
                stress_idx = classes.index(2)
                stress_confidence = float(proba[0, stress_idx])
        except Exception:
            pass  # classifier doesn't support predict_proba (e.g. SVM without probability=True)

    return {
        "predictions": [int(p) for p in preds],
        "label_names": label_names,
        "is_stressed": is_stressed,
        "stress_ratio": stress_ratio,
        "stress_confidence": stress_confidence,
    }
