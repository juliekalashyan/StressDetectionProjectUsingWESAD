"""Window-based feature extraction for WESAD sensor data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Sampling frequency used in the WESAD pipeline (all sensors aligned to 700 Hz)
FS: int = 700

# Default window size in seconds
DEFAULT_WINDOW_SEC: int = 5

# Human-readable label map for WESAD
LABEL_MAP: dict[int, str] = {
    0: "Transient",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation",
    5: "Meditation-2",
}

# Labels that indicate stress
STRESS_LABELS: set[int] = {2}


def compute_window_features(window: np.ndarray) -> np.ndarray:
    """Compute statistical features for a single window.

    Parameters
    ----------
    window : np.ndarray of shape (window_size, n_channels)

    Returns
    -------
    np.ndarray
        1-D array of features (8 stats x n_channels = 112 for 14 channels):
        [mean, std, min, max, median, IQR, skewness, kurtosis].
    """
    feats = np.concatenate([
        np.mean(window, axis=0),
        np.std(window, axis=0),
        np.min(window, axis=0),
        np.max(window, axis=0),
        np.median(window, axis=0),
        sp_stats.iqr(window, axis=0),
        sp_stats.skew(window, axis=0),
        sp_stats.kurtosis(window, axis=0),
    ])
    return feats.astype(np.float32)


def extract_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_sec: float = DEFAULT_WINDOW_SEC,
    fs: int = FS,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment raw sample-level data into labelled windows.

    Parameters
    ----------
    features : np.ndarray of shape (n_samples, 14)
    labels : np.ndarray of shape (n_samples,)
    window_sec : float
        Window size in seconds.
    fs : int
        Sampling frequency.

    Returns
    -------
    X : np.ndarray of shape (n_windows, n_features)
        Window-level features.
    y : np.ndarray of shape (n_windows,)
        Majority-vote label per window.
    """
    window_size = int(window_sec * fs)
    n_samples = len(features)
    n_windows = n_samples // window_size

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        win_feat = features[start:end]
        win_label = labels[start:end]

        # Majority-vote label (ignoring transient = 0)
        non_zero = win_label[win_label != 0]
        if len(non_zero) == 0:
            majority = 0
        else:
            values, counts = np.unique(non_zero, return_counts=True)
            majority = int(values[np.argmax(counts)])

        X_list.append(compute_window_features(win_feat))
        y_list.append(majority)

    logger.info("Extracted %d windows (%ds each @ %d Hz).", len(X_list), window_sec, fs)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def features_from_manual_input(sensor_values: dict[str, float]) -> np.ndarray:
    """Create a feature vector from manually entered sensor averages.

    Parameters
    ----------
    sensor_values : dict
        Mapping of sensor name to a single float value (the average reading
        during the period of interest).

    Returns
    -------
    np.ndarray of shape (1, n_features)
        Feature vector compatible with the trained model.  Because only a
        single value per channel is provided, std/IQR/skew/kurt are set to 0
        and min/max/median equal the value itself.
    """
    from .data_processor import FEATURE_COLUMNS

    ordered = np.array(
        [float(sensor_values.get(col, 0.0)) for col in FEATURE_COLUMNS],
        dtype=np.float32,
    )
    zeros = np.zeros_like(ordered)
    # mean=val, std=0, min=val, max=val, median=val, iqr=0, skew=0, kurt=0
    feats = np.concatenate([ordered, zeros, ordered, ordered, ordered, zeros, zeros, zeros])
    return feats.reshape(1, -1)
