"""Shared training logic for WESAD stress-detection models.

Used by both ``train_model.py`` (CLI) and ``app.py`` (on-the-fly training).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_extractor import LABEL_MAP

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")

# ---------------------------------------------------------------------------
# Hyper-parameters (single source of truth)
# ---------------------------------------------------------------------------
DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}

STRESS_RATIO_THRESHOLD: float = 0.30  # ratio above which "overall stress" flag is set


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.20,
    cv_folds: int = 5,
    verbose: bool = False,
) -> tuple[RandomForestClassifier, StandardScaler, dict[str, Any]]:
    """Train a Random Forest classifier and return model, scaler, metrics.

    Parameters
    ----------
    X : array of shape (n_windows, n_features)
    y : array of shape (n_windows,)
    test_size : float
        Fraction reserved for evaluation.
    cv_folds : int
        Number of stratified k-fold CV splits (0 to skip).
    verbose : bool
        If *True*, print/log classification report.

    Returns
    -------
    model : RandomForestClassifier
    scaler : StandardScaler
    metrics : dict  – keys ``accuracy``, ``cv_mean``, ``cv_std``, ``classification_report``
    """
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(**DEFAULT_RF_PARAMS)
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    acc = float(accuracy_score(y_test, y_pred))
    target_names = [LABEL_MAP.get(u, str(u)) for u in sorted(np.unique(y))]
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0,
    )

    # Optional cross-validation
    cv_mean, cv_std = 0.0, 0.0
    if cv_folds > 0:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, scaler.transform(X), y, cv=cv, scoring="accuracy")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "classification_report": report,
    }

    if verbose:
        logger.info("Accuracy: %.4f", acc)
        if cv_folds > 0:
            logger.info("CV accuracy: %.4f ± %.4f", cv_mean, cv_std)
        logger.info("\n%s", report)

    return clf, scaler, metrics


def save_model(
    clf: RandomForestClassifier,
    scaler: StandardScaler,
    subject_id: str,
    model_dir: str = MODEL_DIR,
) -> tuple[str, str]:
    """Persist model and scaler to *model_dir*. Returns saved paths."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{subject_id}.joblib")
    scaler_path = os.path.join(model_dir, f"scaler_{subject_id}.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved model  → %s", model_path)
    logger.info("Saved scaler → %s", scaler_path)
    return model_path, scaler_path
