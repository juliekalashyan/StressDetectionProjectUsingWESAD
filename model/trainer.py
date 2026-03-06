"""Shared training logic for WESAD stress-detection models.

Used by both ``train_model.py`` (CLI) and ``app.py`` (on-the-fly training).

Supports multiple ML classifiers for comparison:
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- Decision Tree
- Gradient Boosting
- Logistic Regression
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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

# ---------------------------------------------------------------------------
# Classifier catalogue – each entry can be used for comparison
# ---------------------------------------------------------------------------
CLASSIFIER_CATALOGUE: dict[str, dict[str, Any]] = {
    "Random Forest": {
        "class": RandomForestClassifier,
        "params": DEFAULT_RF_PARAMS,
    },
    "SVM": {
        "class": SVC,
        "params": {
            "kernel": "rbf",
            "C": 10,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
            "class_weight": "balanced",
        },
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "params": {
            "n_neighbors": 5,
            "weights": "distance",
            "n_jobs": -1,
        },
    },
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "params": {
            "max_depth": 20,
            "random_state": 42,
            "class_weight": "balanced",
        },
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    },
    "Logistic Regression": {
        "class": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
    },
}


def _build_pipeline(cat: dict[str, Any]) -> Any:
    """Return a fresh Pipeline: SimpleImputer → StandardScaler → classifier."""
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        cat["class"](**cat["params"]),
    )


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.20,
    cv_folds: int = 5,
    verbose: bool = False,
    classifier_name: str = "Random Forest",
) -> tuple[Any, None, dict[str, Any]]:
    """Train a full Pipeline (imputer → scaler → classifier) and return metrics.

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
    classifier_name : str
        Key from ``CLASSIFIER_CATALOGUE``.

    Returns
    -------
    pipeline : fitted sklearn Pipeline (imputer + scaler + classifier)
    None     : placeholder for backward-compat (scaler is inside pipeline)
    metrics  : dict
    """
    cat = CLASSIFIER_CATALOGUE.get(classifier_name, CLASSIFIER_CATALOGUE["Random Forest"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    pipeline = _build_pipeline(cat)
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    target_names = [LABEL_MAP.get(u, str(u)) for u in sorted(np.unique(y))]
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

    # Optional cross-validation using a fresh pipeline per fold
    cv_mean, cv_std = 0.0, 0.0
    if cv_folds > 0:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(_build_pipeline(cat), X, y, cv=cv, scoring="accuracy")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    # Feature importance (if available on the final estimator)
    importances = None
    final_clf = pipeline.steps[-1][1]
    if hasattr(final_clf, "feature_importances_"):
        importances = final_clf.feature_importances_.tolist()

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "precision": prec,
        "recall": rec,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "confusion_matrix": cm,
        "target_names": target_names,
        "classification_report": report,
        "train_time_sec": round(train_time, 3),
        "feature_importances": importances,
        "classifier_name": classifier_name,
    }

    if verbose:
        logger.info("[%s] Accuracy: %.4f  F1(w): %.4f  Time: %.2fs",
                     classifier_name, acc, f1_w, train_time)
        if cv_folds > 0:
            logger.info("CV accuracy: %.4f ± %.4f", cv_mean, cv_std)
        logger.info("\n%s", report)

    return pipeline, None, metrics


def compare_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.20,
    cv_folds: int = 5,
    classifiers: list[str] | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Train all (or selected) classifiers and return comparison metrics.

    Parameters
    ----------
    classifiers : list[str] | None
        Subset of ``CLASSIFIER_CATALOGUE`` keys; *None* = all.

    Returns
    -------
    List of metric dicts, one per classifier, sorted by accuracy descending.
    """
    names = classifiers or list(CLASSIFIER_CATALOGUE.keys())
    results: list[dict[str, Any]] = []

    for name in names:
        logger.info("Training %s …", name)
        try:
            _pipeline, _none, metrics = train_model(
                X, y,
                test_size=test_size,
                cv_folds=cv_folds,
                verbose=verbose,
                classifier_name=name,
            )
            results.append(metrics)
        except Exception as exc:
            logger.warning("Classifier %s failed: %s", name, exc)
            results.append({
                "classifier_name": name,
                "accuracy": 0.0,
                "f1_weighted": 0.0,
                "f1_macro": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "cv_mean": 0.0,
                "cv_std": 0.0,
                "confusion_matrix": [],
                "target_names": [],
                "classification_report": f"Error: {exc}",
                "train_time_sec": 0.0,
                "feature_importances": None,
                "error": str(exc),
            })

    results.sort(key=lambda m: m["accuracy"], reverse=True)
    return results


def save_model(
    pipeline: Any,
    _scaler: Any,  # kept for call-site compat; scaler is inside the pipeline
    subject_id: str,
    model_dir: str = MODEL_DIR,
) -> tuple[str, str]:
    """Persist the fitted pipeline to *model_dir*. Returns saved paths."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{subject_id}.joblib")
    scaler_path = os.path.join(model_dir, f"scaler_{subject_id}.joblib")
    joblib.dump(pipeline, model_path)
    # Store a sentinel so load_model still finds a scaler file
    joblib.dump(None, scaler_path)
    logger.info("Saved pipeline → %s", model_path)
    return model_path, scaler_path


# ---------------------------------------------------------------------------
# General (subject-independent) model
# ---------------------------------------------------------------------------
GENERAL_ID = "general"
GENERAL_MANUAL_ID = "general_manual"

N_MEAN_FEATURES = 14  # one mean per sensor channel


def general_model_exists(model_dir: str = MODEL_DIR) -> bool:
    """Return True if a trained general model exists."""
    return (
        os.path.isfile(os.path.join(model_dir, f"model_{GENERAL_ID}.joblib"))
        and os.path.isfile(os.path.join(model_dir, f"scaler_{GENERAL_ID}.joblib"))
    )


def general_manual_model_exists(model_dir: str = MODEL_DIR) -> bool:
    """Return True if the means-only manual model exists."""
    return (
        os.path.isfile(os.path.join(model_dir, f"model_{GENERAL_MANUAL_ID}.joblib"))
        and os.path.isfile(os.path.join(model_dir, f"scaler_{GENERAL_MANUAL_ID}.joblib"))
    )


def train_general_model(
    data_dir: str,
    *,
    window_sec: int = 5,
    classifier_name: str = "Random Forest",
    cv_folds: int = 5,
    verbose: bool = False,
    model_dir: str = MODEL_DIR,
) -> tuple[Any, None, dict[str, Any]]:
    """Train a single model on **all** WESAD subjects combined.

    Parameters
    ----------
    data_dir : str
        Path to the WESAD dataset root (contains S2/, S3/, … sub-folders
        each with a ``<SID>.pkl`` file).
    window_sec : int
        Window size in seconds for feature extraction.
    classifier_name : str
        Key from ``CLASSIFIER_CATALOGUE``.
    cv_folds : int
        Number of stratified CV folds (0 to skip).
    verbose : bool
        Print progress.
    model_dir : str
        Where to save the resulting ``model_general.joblib``.

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    None     : placeholder
    metrics  : dict
    """
    from .data_processor import load_subject
    from .feature_extractor import extract_windows

    all_X, all_y = [], []
    pkl_files = sorted(
        os.path.join(data_dir, d, f)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        for f in os.listdir(os.path.join(data_dir, d))
        if f.endswith(".pkl")
    )
    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl files found under {data_dir}. "
            "Expected structure: <data_dir>/S2/S2.pkl, S3/S3.pkl, …"
        )

    for pkl in pkl_files:
        sid = os.path.splitext(os.path.basename(pkl))[0]
        if verbose:
            logger.info("Loading %s …", sid)
        features, labels = load_subject(pkl)
        X, y = extract_windows(features, labels, window_sec=window_sec)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        all_X.append(X)
        all_y.append(y)
        if verbose:
            logger.info("  %s → %d windows", sid, X.shape[0])

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    if verbose:
        logger.info(
            "Combined dataset: %d windows from %d subjects", X_all.shape[0], len(pkl_files)
        )

    pipeline, _, metrics = train_model(
        X_all, y_all,
        cv_folds=cv_folds,
        verbose=verbose,
        classifier_name=classifier_name,
    )

    save_model(pipeline, None, GENERAL_ID, model_dir=model_dir)
    metrics["n_subjects"] = len(pkl_files)
    metrics["n_windows_total"] = int(X_all.shape[0])

    # ----- Also train a means-only model for manual input predictions -----
    # Manual input provides only 14 raw sensor readings (one per channel).
    # The full 154-feature model can't meaningfully classify those because
    # the remaining 140 features (std, iqr, skew, freq, …) have to be
    # fabricated.  A compact model trained on just the 14 mean columns
    # receives exactly the same data shape at inference time.
    #
    # We deliberately drop ``class_weight="balanced"`` for this model:
    # with only 14 features the balanced weighting over-predicts the
    # minority Stress class for centroid-like inputs.  The natural class
    # prior produces better-calibrated probabilities for manual entry.
    X_means = X_all[:, :N_MEAN_FEATURES]
    manual_rf_params = {k: v for k, v in DEFAULT_RF_PARAMS.items() if k != "class_weight"}
    manual_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        RandomForestClassifier(**manual_rf_params),
    )

    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_means, y_all, test_size=0.20, random_state=42, stratify=y_all,
    )
    manual_pipeline.fit(X_m_train, y_m_train)
    manual_acc = float(accuracy_score(y_m_test, manual_pipeline.predict(X_m_test)))

    save_model(manual_pipeline, None, GENERAL_MANUAL_ID, model_dir=model_dir)
    if verbose:
        logger.info(
            "Means-only manual model accuracy: %.4f (on %d features)",
            manual_acc, N_MEAN_FEATURES,
        )
    metrics["manual_model_accuracy"] = manual_acc

    return pipeline, None, metrics
