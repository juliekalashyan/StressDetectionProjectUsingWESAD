#!/usr/bin/env python3
"""Train a stress-detection model on a WESAD subject .pkl file.

Usage
-----
    python train_model.py <path_to_subject.pkl> [--subject-id S2]

The trained Random Forest model and scaler are saved to ``trained_models/``.
"""

import argparse
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

from model.data_processor import load_subject
from model.feature_extractor import extract_windows, LABEL_MAP

MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")


def train(pkl_path, subject_id, window_sec=5):
    """Train and persist a Random Forest model for one WESAD subject."""
    print(f"Loading subject data from {pkl_path} …")
    features, labels = load_subject(pkl_path)
    print(f"  Raw samples: {len(features):,}")

    print("Extracting window features …")
    X, y = extract_windows(features, labels, window_sec=window_sec)
    print(f"  Windows: {X.shape[0]:,}  |  Features per window: {X.shape[1]}")

    # Remove transient (label 0) windows for cleaner training
    mask = y != 0
    X, y = X[mask], y[mask]
    print(f"  After removing transient windows: {X.shape[0]:,}")

    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Label {u} ({LABEL_MAP.get(u, '?')}): {c}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train Random Forest
    print("Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    target_names = [LABEL_MAP.get(u, str(u)) for u in sorted(np.unique(y))]
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
    )

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model_{subject_id}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{subject_id}.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def main():
    parser = argparse.ArgumentParser(description="Train stress detection model")
    parser.add_argument("pkl_path", help="Path to a WESAD subject .pkl file")
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Subject identifier (e.g. S2). Inferred from filename if omitted.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5,
        help="Window size in seconds (default: 5)",
    )
    args = parser.parse_args()

    subject_id = args.subject_id
    if subject_id is None:
        # Infer from filename, e.g. "S2.pkl" -> "S2"
        subject_id = os.path.splitext(os.path.basename(args.pkl_path))[0]

    train(args.pkl_path, subject_id, window_sec=args.window_sec)


if __name__ == "__main__":
    main()
