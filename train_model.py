#!/usr/bin/env python3
"""Train a stress-detection model on a WESAD subject .pkl file.

Usage
-----
    python train_model.py <path_to_subject.pkl> [--subject-id S2]
    python train_model.py <path_to_subject.pkl> --compare   # compare all classifiers
    python train_model.py --general --data-dir <WESAD_ROOT>  # train general model

The trained model and scaler are saved to ``trained_models/``.
"""

import argparse
import os
import sys

import numpy as np

from model.data_processor import load_subject
from model.feature_extractor import extract_windows, LABEL_MAP
from model.trainer import (
    train_model,
    compare_classifiers,
    save_model,
    train_general_model,
    CLASSIFIER_CATALOGUE,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")


def train(pkl_path, subject_id, window_sec=5, classifier_name="Random Forest"):
    """Train and persist a model for one WESAD subject."""
    print(f"Loading subject data from {pkl_path} …")
    features, labels = load_subject(pkl_path)
    print(f"  Raw samples: {len(features):,}")

    print("Extracting window features …")
    X, y = extract_windows(features, labels, window_sec=window_sec)
    print(f"  Windows: {X.shape[0]:,}  |  Features per window: {X.shape[1]}")

    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Label {u} ({LABEL_MAP.get(u, '?')}): {c}")

    print(f"\nTraining {classifier_name} …")
    clf, scaler, metrics = train_model(
        X, y, cv_folds=5, verbose=True, classifier_name=classifier_name,
    )

    print(f"\n{'='*50}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"CV accuracy:   {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"Train time:    {metrics['train_time_sec']:.2f}s")
    print(f"{'='*50}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")

    # Save model and scaler
    save_model(clf, scaler, subject_id)


def compare(pkl_path, subject_id, window_sec=5):
    """Train all classifiers and print comparison table."""
    print(f"Loading subject data from {pkl_path} …")
    features, labels = load_subject(pkl_path)

    print("Extracting window features …")
    X, y = extract_windows(features, labels, window_sec=window_sec)
    print(f"  Windows: {X.shape[0]:,}  |  Features: {X.shape[1]}")

    print(f"\nComparing {len(CLASSIFIER_CATALOGUE)} classifiers …\n")
    results = compare_classifiers(X, y, cv_folds=5, verbose=True)

    # Print comparison table
    header = f"{'Rank':<5} {'Classifier':<22} {'Accuracy':>9} {'F1(w)':>8} {'F1(m)':>8} {'Prec':>8} {'Rec':>8} {'CV Mean':>8} {'Time':>7}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for i, r in enumerate(results, 1):
        print(
            f"{i:<5} {r['classifier_name']:<22} "
            f"{r['accuracy']*100:>8.2f}% "
            f"{r['f1_weighted']*100:>7.2f}% "
            f"{r['f1_macro']*100:>7.2f}% "
            f"{r['precision']*100:>7.2f}% "
            f"{r['recall']*100:>7.2f}% "
            f"{r['cv_mean']*100:>7.2f}% "
            f"{r['train_time_sec']:>6.2f}s"
        )
    print(f"{'='*len(header)}")
    print(f"\nBest classifier: {results[0]['classifier_name']} "
          f"(accuracy={results[0]['accuracy']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train stress detection model")
    parser.add_argument("pkl_path", nargs="?", default=None,
                        help="Path to a WESAD subject .pkl file (not needed with --general)")
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
    parser.add_argument(
        "--classifier",
        default="Random Forest",
        choices=list(CLASSIFIER_CATALOGUE.keys()),
        help="ML classifier to use (default: Random Forest)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all classifiers and print a ranking table",
    )
    parser.add_argument(
        "--general",
        action="store_true",
        help="Train a general (subject-independent) model from all subjects",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to WESAD dataset root (contains S2/, S3/, … sub-folders). "
             "Required with --general unless WESAD_DATA_DIR env var is set.",
    )
    args = parser.parse_args()

    if args.general:
        data_dir = args.data_dir or os.environ.get("WESAD_DATA_DIR")
        if not data_dir:
            parser.error("--general requires --data-dir or WESAD_DATA_DIR env var")
        print(f"Training general model from all subjects in {data_dir} …")
        _, _, metrics = train_general_model(
            data_dir,
            window_sec=int(args.window_sec),
            classifier_name=args.classifier,
            cv_folds=5,
            verbose=True,
        )
        print(f"\n{'='*50}")
        print(f"General model trained on {metrics['n_subjects']} subjects, "
              f"{metrics['n_windows_total']:,} windows")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"CV accuracy:   {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"{'='*50}")
        return

    if not args.pkl_path:
        parser.error("pkl_path is required (unless using --general)")

    subject_id = args.subject_id
    if subject_id is None:
        subject_id = os.path.splitext(os.path.basename(args.pkl_path))[0]

    if args.compare:
        compare(args.pkl_path, subject_id, window_sec=args.window_sec)
    else:
        train(args.pkl_path, subject_id, window_sec=args.window_sec,
              classifier_name=args.classifier)


if __name__ == "__main__":
    main()
