"""Window-based feature extraction for WESAD sensor data.

Features include:
- Time-domain: mean, std, min, max, median, IQR, skewness, kurtosis
- Frequency-domain: dominant frequency, spectral energy, spectral entropy
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)

# Sampling frequency used in the WESAD pipeline (all sensors aligned to 700 Hz)
FS: int = 700

# Default window size in seconds
DEFAULT_WINDOW_SEC: int = 5

# Human-readable label map for WESAD
LABEL_MAP: dict[int, str] = {
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
}

# Labels that indicate stress
STRESS_LABELS: set[int] = {2}

# Feature names (useful for labelling / importance charts)
_TIME_STATS = ["mean", "std", "min", "max", "median", "iqr", "skew", "kurtosis"]
_FREQ_STATS = ["dom_freq", "spectral_energy", "spectral_entropy"]

CHANNEL_NAMES: list[str] = [
    "Chest Acc X", "Chest Acc Y", "Chest Acc Z",
    "Chest ECG", "Chest EMG", "Chest EDA", "Chest Temp", "Chest Resp",
    "Wrist Acc X", "Wrist Acc Y", "Wrist Acc Z",
    "Wrist BVP", "Wrist EDA", "Wrist Temp",
]

FEATURE_NAMES: list[str] = (
    [f"{ch}_{stat}" for stat in _TIME_STATS for ch in CHANNEL_NAMES]
    + [f"{ch}_{stat}" for stat in _FREQ_STATS for ch in CHANNEL_NAMES]
)


def _spectral_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute frequency-domain features for each channel.

    Returns array of shape (3 * n_channels,): [dom_freq..., energy..., entropy...].
    """
    n_channels = window.shape[1]
    dom_freqs = np.zeros(n_channels, dtype=np.float32)
    energies = np.zeros(n_channels, dtype=np.float32)
    entropies = np.zeros(n_channels, dtype=np.float32)

    freqs = rfftfreq(window.shape[0], d=1.0 / fs)

    for ch in range(n_channels):
        fft_vals = np.abs(rfft(window[:, ch]))
        # Dominant frequency
        dom_freqs[ch] = freqs[np.argmax(fft_vals[1:])] if len(fft_vals) > 1 else 0.0
        # Spectral energy
        energies[ch] = np.sum(fft_vals ** 2)
        # Spectral entropy
        psd = fft_vals ** 2
        psd_sum = psd.sum()
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            psd_norm = psd_norm[psd_norm > 0]
            entropies[ch] = -np.sum(psd_norm * np.log2(psd_norm))
        else:
            entropies[ch] = 0.0

    return np.concatenate([dom_freqs, energies, entropies])


def compute_window_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute statistical + spectral features for a single window.

    Parameters
    ----------
    window : np.ndarray of shape (window_size, n_channels)
    fs : int
        Sampling frequency (used for spectral features).

    Returns
    -------
    np.ndarray
        1-D array of features:
        Time-domain (8 stats x 14 channels = 112) +
        Frequency-domain (3 stats x 14 channels = 42) = 154 total.
    """
    time_feats = np.concatenate([
        np.mean(window, axis=0),
        np.std(window, axis=0),
        np.min(window, axis=0),
        np.max(window, axis=0),
        np.median(window, axis=0),
        sp_stats.iqr(window, axis=0),
        sp_stats.skew(window, axis=0),
        sp_stats.kurtosis(window, axis=0),
    ])
    freq_feats = _spectral_features(window, fs=fs)
    feats = np.concatenate([time_feats, freq_feats]).astype(np.float32)
    # Sanitise: replace NaN / Inf that can arise from constant-value
    # channels (skew, kurtosis) or degenerate FFT windows.
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


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

        # Majority-vote label across the window
        if len(win_label) == 0:
            continue
        values, counts = np.unique(win_label, return_counts=True)
        majority = int(values[np.argmax(counts)])

        X_list.append(compute_window_features(win_feat, fs=fs))
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
        Feature vector compatible with the trained model.  Synthetic
        statistics (std ≈ 8%, iqr ≈ 7% of value, min/max spread ±10%)
        are generated to approximate a real windowed signal.  Spectral
        features are set to 0 (no time-series available for FFT).
    """
    from .data_processor import FEATURE_COLUMNS

    ordered = np.array(
        [float(sensor_values.get(col, 0.0)) for col in FEATURE_COLUMNS],
        dtype=np.float32,
    )
    # Synthesise realistic statistics from the single reading.
    # std ≈ 5-10% of the value, iqr ≈ 7% — gives the model something
    # closer to what it saw during training instead of all-zero columns.
    abs_vals = np.abs(ordered) + 1e-6
    synth_std = abs_vals * 0.08
    synth_iqr = abs_vals * 0.07
    zeros = np.zeros_like(ordered)
    # Time-domain: mean, std, min, max, median, iqr, skew, kurt
    time_feats = np.concatenate([
        ordered,       # mean
        synth_std,     # std  (was 0 → unrealistic)
        ordered * 0.9, # min  (slightly below mean)
        ordered * 1.1, # max  (slightly above mean)
        ordered,       # median
        synth_iqr,     # iqr  (was 0 → unrealistic)
        zeros,         # skew (0 is plausible for short windows)
        zeros,         # kurtosis (0 is plausible)
    ])
    # Frequency-domain: dom_freq=0, energy=0, entropy=0
    freq_feats = np.concatenate([zeros, zeros, zeros])
    feats = np.concatenate([time_feats, freq_feats])
    return feats.reshape(1, -1)


def features_from_manual_input_means_only(sensor_values: dict[str, float]) -> np.ndarray:
    """Create a *means-only* feature vector for the compact manual model.

    Parameters
    ----------
    sensor_values : dict
        Mapping of sensor name → single float value.

    Returns
    -------
    np.ndarray of shape (1, 14)
        One value per sensor channel — exactly what the means-only model
        was trained on.
    """
    from .data_processor import FEATURE_COLUMNS

    ordered = np.array(
        [float(sensor_values.get(col, 0.0)) for col in FEATURE_COLUMNS],
        dtype=np.float32,
    )
    return ordered.reshape(1, -1)
