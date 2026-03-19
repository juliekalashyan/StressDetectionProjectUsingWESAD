"""Window-based feature extraction for WESAD sensor data.

Features include:
- Time-domain: mean, std, min, max, median, IQR, skewness, kurtosis
- Frequency-domain: dominant frequency, spectral energy, spectral entropy
- HRV features: RMSSD, SDNN, mean RR, pNN50, HR (from ECG R-peaks)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

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
_HRV_STATS = ["hrv_rmssd", "hrv_sdnn", "hrv_mean_rr", "hrv_pnn50", "hrv_hr"]

CHANNEL_NAMES: list[str] = [
    "Chest Acc X", "Chest Acc Y", "Chest Acc Z",
    "Chest ECG", "Chest EMG", "Chest EDA", "Chest Temp", "Chest Resp",
    "Wrist Acc X", "Wrist Acc Y", "Wrist Acc Z",
    "Wrist BVP", "Wrist EDA", "Wrist Temp",
]

N_HRV_FEATURES = len(_HRV_STATS)

FEATURE_NAMES: list[str] = (
    [f"{ch}_{stat}" for stat in _TIME_STATS for ch in CHANNEL_NAMES]
    + [f"{ch}_{stat}" for stat in _FREQ_STATS for ch in CHANNEL_NAMES]
    + [f"ECG_{stat}" for stat in _HRV_STATS]
)


# ---------------------------------------------------------------------------
# HRV feature helpers
# ---------------------------------------------------------------------------

def _compute_hrv_features_single(ecg_window: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extract 5 HRV features from a single ECG window.

    Returns [RMSSD, SDNN, mean_RR, pNN50, HR].
    """
    # Simple R-peak detection: find peaks with minimum distance
    min_distance = int(0.3 * fs)  # min 300 ms between beats (~200 bpm max)
    # Use a height threshold at the 70th percentile of the signal
    threshold = np.percentile(ecg_window, 70)
    peaks, _ = find_peaks(ecg_window, distance=min_distance, height=threshold)

    if len(peaks) < 2:
        return np.zeros(N_HRV_FEATURES, dtype=np.float32)

    # RR intervals in milliseconds
    rr = np.diff(peaks) / fs * 1000.0

    # Filter physiologically plausible RR intervals (300–1500 ms = 40–200 bpm)
    rr = rr[(rr > 300) & (rr < 1500)]
    if len(rr) < 2:
        return np.zeros(N_HRV_FEATURES, dtype=np.float32)

    rr_diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    sdnn = np.std(rr, ddof=1) if len(rr) > 1 else 0.0
    mean_rr = np.mean(rr)
    pnn50 = np.sum(np.abs(rr_diff) > 50.0) / len(rr_diff) * 100.0 if len(rr_diff) > 0 else 0.0
    hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0

    return np.array([rmssd, sdnn, mean_rr, pnn50, hr], dtype=np.float32)


def _batch_hrv_features(windows: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute HRV features for all windows. ECG is channel index 3.

    Parameters
    ----------
    windows : np.ndarray of shape (n_windows, window_size, n_channels)

    Returns
    -------
    np.ndarray of shape (n_windows, N_HRV_FEATURES)
    """
    n_windows = windows.shape[0]
    hrv = np.zeros((n_windows, N_HRV_FEATURES), dtype=np.float32)
    for i in range(n_windows):
        hrv[i] = _compute_hrv_features_single(windows[i, :, 3], fs=fs)
    return hrv


def _spectral_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute frequency-domain features for each channel.

    Returns array of shape (3 * n_channels,): [dom_freq..., energy..., entropy...].
    """
    # Vectorised: FFT all channels at once (axis=0 operates along samples)
    fft_vals = np.abs(rfft(window, axis=0))  # (freq_bins, n_channels)
    freqs = rfftfreq(window.shape[0], d=1.0 / fs)

    # Dominant frequency (skip DC at index 0)
    if fft_vals.shape[0] > 1:
        dom_freqs = freqs[np.argmax(fft_vals[1:], axis=0) + 1]
    else:
        dom_freqs = np.zeros(window.shape[1], dtype=np.float32)

    # Spectral energy
    psd = fft_vals ** 2
    energies = psd.sum(axis=0)

    # Spectral entropy
    psd_sum = psd.sum(axis=0, keepdims=True)
    safe_sum = np.where(psd_sum == 0, 1.0, psd_sum)
    psd_norm = psd / safe_sum
    log_psd = np.where(psd_norm > 0, np.log2(psd_norm), 0.0)
    entropies = -np.sum(psd_norm * log_psd, axis=0)
    entropies = np.where(psd_sum.squeeze() == 0, 0.0, entropies)

    return np.concatenate([dom_freqs, energies, entropies]).astype(np.float32)


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
        Frequency-domain (3 stats x 14 channels = 42) +
        HRV features (5) = 159 total.
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
    hrv_feats = _compute_hrv_features_single(window[:, 3], fs=fs)
    feats = np.concatenate([time_feats, freq_feats, hrv_feats]).astype(np.float32)
    # Sanitise: replace NaN / Inf that can arise from constant-value
    # channels (skew, kurtosis) or degenerate FFT windows.
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def _batch_spectral_features(windows: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute spectral features for ALL windows at once (fully vectorised).

    Parameters
    ----------
    windows : np.ndarray of shape (n_windows, window_size, n_channels)

    Returns
    -------
    np.ndarray of shape (n_windows, 3 * n_channels)
    """
    # FFT along the time axis for every window & channel simultaneously
    # scipy.fft with workers=-1 uses all CPU cores via pocketfft threading
    fft_vals = np.abs(rfft(windows, axis=1, workers=-1))  # (n_windows, freq_bins, n_channels)
    freqs = rfftfreq(windows.shape[1], d=1.0 / fs)

    # Dominant frequency (skip DC at index 0)
    dom_idx = np.argmax(fft_vals[:, 1:, :], axis=1)  # (n_windows, n_channels)
    dom_freqs = freqs[dom_idx + 1]

    # Spectral energy
    psd = fft_vals ** 2
    energies = psd.sum(axis=1)  # (n_windows, n_channels)

    # Spectral entropy
    psd_sum = psd.sum(axis=1, keepdims=True)  # (n_windows, 1, n_channels)
    safe_sum = np.where(psd_sum == 0, 1.0, psd_sum)
    psd_norm = psd / safe_sum
    log_psd = np.where(psd_norm > 0, np.log2(psd_norm), 0.0)
    entropies = -np.sum(psd_norm * log_psd, axis=1)  # (n_windows, n_channels)
    entropies = np.where(psd_sum.squeeze(axis=1) == 0, 0.0, entropies)

    return np.concatenate([dom_freqs, energies, entropies], axis=1).astype(np.float32)


def _batch_compute_features(windows: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute all features for every window in one vectorised pass.

    Parameters
    ----------
    windows : np.ndarray of shape (n_windows, window_size, n_channels)

    Returns
    -------
    np.ndarray of shape (n_windows, 159)
    """
    # --- Time-domain stats (all windows at once) ---
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    mins = np.min(windows, axis=1)
    maxs = np.max(windows, axis=1)

    # Single quantile call for median + IQR (much faster than separate calls)
    quantiles = np.quantile(windows, [0.25, 0.5, 0.75], axis=1)
    medians = quantiles[1]
    iqrs = quantiles[2] - quantiles[0]

    # Manual skew & kurtosis from moments (avoids slow scipy.stats overhead)
    centered = windows - means[:, np.newaxis, :]
    safe_std = np.where(stds == 0, 1.0, stds)
    skews = np.mean(centered ** 3, axis=1) / (safe_std ** 3)
    kurts = np.mean(centered ** 4, axis=1) / (safe_std ** 4) - 3.0

    time_feats = np.concatenate(
        [means, stds, mins, maxs, medians, iqrs, skews, kurts], axis=1,
    )

    # --- Frequency-domain stats (all windows at once) ---
    freq_feats = _batch_spectral_features(windows, fs=fs)

    # --- HRV features from ECG (channel 3) ---
    hrv_feats = _batch_hrv_features(windows, fs=fs)

    features = np.concatenate([time_feats, freq_feats, hrv_feats], axis=1).astype(np.float32)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_sec: float = DEFAULT_WINDOW_SEC,
    fs: int = FS,
    overlap: float = 0.5,
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
    overlap : float
        Fraction of overlap between consecutive windows (0.0–0.99).
        Default 0.5 (50%) doubles training data and preserves temporal
        continuity. Set to 0.0 for legacy non-overlapping behaviour.

    Returns
    -------
    X : np.ndarray of shape (n_windows, n_features)
        Window-level features.
    y : np.ndarray of shape (n_windows,)
        Majority-vote label per window.
    """
    overlap = max(0.0, min(overlap, 0.99))
    window_size = int(window_sec * fs)
    step_size = max(1, int(window_size * (1 - overlap)))
    n_samples = len(features)

    if n_samples < window_size:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.int64)

    # Build overlapping windows using stride tricks for zero-copy views
    n_windows = (n_samples - window_size) // step_size + 1

    if n_windows == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.int64)

    # Collect window start indices
    starts = np.arange(n_windows) * step_size

    # Fancy-index into feature and label arrays
    idx = starts[:, np.newaxis] + np.arange(window_size)[np.newaxis, :]
    windows = features[idx]                    # (n_windows, window_size, n_channels)
    label_windows = labels[idx]                # (n_windows, window_size)

    # Majority-vote labels — fully vectorised (labels are 1, 2, 3)
    max_label = int(label_windows.max()) + 1
    counts = np.zeros((n_windows, max_label), dtype=np.int32)
    for lbl in np.unique(label_windows):
        counts[:, lbl] = np.sum(label_windows == lbl, axis=1)
    y = np.argmax(counts, axis=1).astype(np.int64)

    # Batch-compute all 159 features for every window at once
    X = _batch_compute_features(windows, fs=fs)

    logger.info(
        "Extracted %d windows (%ds each, %.0f%% overlap @ %d Hz).",
        n_windows, window_sec, overlap * 100, fs,
    )
    return X, y


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
    # HRV: synthesise plausible resting values from manual input
    hrv_feats = np.array([30.0, 50.0, 800.0, 20.0, 75.0], dtype=np.float32)
    feats = np.concatenate([time_feats, freq_feats, hrv_feats])
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
