"""Data processing utilities for WESAD .pkl files.

Extracts chest and wrist sensor signals and aligns them to a common
sampling rate following the same logic used in the original notebook.
"""

import io
import logging

import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample_poly, butter, sosfiltfilt
from math import gcd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bandpass / lowpass filter helpers
# ---------------------------------------------------------------------------

def _bandpass(signal: np.ndarray, low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal, axis=0).astype(signal.dtype)


def _lowpass(signal: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth lowpass filter."""
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, signal, axis=0).astype(signal.dtype)


# Chest sensor native sampling rate
_CHEST_FS = 700


def _filter_chest_signals(chest_arr: np.ndarray) -> np.ndarray:
    """Apply appropriate bandpass/lowpass filters to each chest channel.

    Column order: AccX, AccY, AccZ, ECG, EMG, EDA, Temp, Resp
    """
    filtered = chest_arr.copy()
    n = filtered.shape[0]
    if n < 30:  # too short for filtering
        return filtered
    # ECG: 0.5–40 Hz bandpass (removes baseline wander + high-freq noise)
    filtered[:, 3] = _bandpass(filtered[:, 3], 0.5, 40.0, _CHEST_FS)
    # EMG: 20–450 Hz bandpass (captures muscle activity band)
    filtered[:, 4] = _bandpass(filtered[:, 4], 20.0, 300.0, _CHEST_FS)
    # EDA: lowpass at 5 Hz (slow electrodermal response)
    filtered[:, 5] = _lowpass(filtered[:, 5], 5.0, _CHEST_FS)
    # Resp: lowpass at 1 Hz (breathing rate)
    filtered[:, 7] = _lowpass(filtered[:, 7], 1.0, _CHEST_FS)
    return filtered

def extract_chest_data(data):
    """Extract and flatten individual chest sensor signals.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp',
        each mapping to a NumPy array of signal readings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each chest sensor axis/signal.
    """
    return pd.DataFrame({
        "Chest Acc X": data["ACC"][:, 0].flatten(),
        "Chest Acc Y": data["ACC"][:, 1].flatten(),
        "Chest Acc Z": data["ACC"][:, 2].flatten(),
        "Chest ECG": data["ECG"].flatten(),
        "Chest EMG": data["EMG"].flatten(),
        "Chest EDA": data["EDA"].flatten(),
        "Chest Temp": data["Temp"].flatten(),
        "Chest Resp": data["Resp"].flatten(),
    })


def _resample_to_length(signal, target_len):
    """Resample *signal* to exactly *target_len* samples.

    Uses scipy.signal.resample_poly (polyphase anti-aliasing filter) for
    better frequency-content preservation compared to linear interpolation.
    Falls back to np.interp for very small signals or edge cases.
    """
    n = len(signal)
    if n == target_len:
        return signal.astype(np.float64)
    # resample_poly needs integer up/down factors
    g = gcd(target_len, n)
    up, down = target_len // g, n // g
    # For very large ratios fall back to interp to avoid memory issues
    if up > 1000 or down > 1000 or n < 8:
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, signal).astype(np.float64)
    return resample_poly(signal.astype(np.float64), up, down)


def extract_wrist_data(data, target_len):
    """Extract and resample wrist sensor data to *target_len* samples.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'ACC', 'BVP', 'EDA', 'TEMP'.
    target_len : int
        Number of samples to resample each signal to (should match chest length).

    Returns
    -------
    pd.DataFrame
        DataFrame with resampled wrist signals as columns.
    """
    return pd.DataFrame({
        "Wrist Acc X": _resample_to_length(data["ACC"][:, 0].flatten(), target_len),
        "Wrist Acc Y": _resample_to_length(data["ACC"][:, 1].flatten(), target_len),
        "Wrist Acc Z": _resample_to_length(data["ACC"][:, 2].flatten(), target_len),
        "Wrist BVP":   _resample_to_length(data["BVP"].flatten(), target_len),
        "Wrist EDA":   _resample_to_length(data["EDA"].flatten(), target_len),
        "Wrist Temp":  _resample_to_length(data["TEMP"].flatten(), target_len),
    })


def _extract_chest_array(data, dtype=np.float32):
    """Extract chest sensors into a numpy array of shape (n_samples, 8).

    Fast alternative to extract_chest_data() — skips pandas entirely.
    """
    return np.column_stack([
        data["ACC"][:, 0].flatten(),
        data["ACC"][:, 1].flatten(),
        data["ACC"][:, 2].flatten(),
        data["ECG"].flatten(),
        data["EMG"].flatten(),
        data["EDA"].flatten(),
        data["Temp"].flatten(),
        data["Resp"].flatten(),
    ]).astype(dtype)


def _extract_wrist_array(data, target_len, dtype=np.float32):
    """Extract and resample wrist sensors into a numpy array of shape (target_len, 6).

    Fast alternative to extract_wrist_data() — skips pandas entirely.
    """
    return np.column_stack([
        _resample_to_length(data["ACC"][:, 0].flatten(), target_len),
        _resample_to_length(data["ACC"][:, 1].flatten(), target_len),
        _resample_to_length(data["ACC"][:, 2].flatten(), target_len),
        _resample_to_length(data["BVP"].flatten(), target_len),
        _resample_to_length(data["EDA"].flatten(), target_len),
        _resample_to_length(data["TEMP"].flatten(), target_len),
    ]).astype(dtype)


FEATURE_COLUMNS = [
    "Chest Acc X",
    "Chest Acc Y",
    "Chest Acc Z",
    "Chest ECG",
    "Chest EMG",
    "Chest EDA",
    "Chest Temp",
    "Chest Resp",
    "Wrist Acc X",
    "Wrist Acc Y",
    "Wrist Acc Z",
    "Wrist BVP",
    "Wrist EDA",
    "Wrist Temp",
]


def load_subject(pkl_path):
    """Load a WESAD subject .pkl file and return features + labels.

    Returns
    -------
    features : np.ndarray of shape (n_samples, 14)
    labels : np.ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If the .pkl file does not contain the expected WESAD structure.
    """

    # --- Safe pickle loading: restrict unpickling to numpy/array types ---
    class _SafeUnpickler(pickle.Unpickler):
        _ALLOWED_MODULES = frozenset({
            'numpy', 'numpy.core.multiarray', 'numpy.core.numeric',
            'numpy._core.multiarray', 'numpy._core.numeric',
            'numpy.ma.core', 'collections', 'builtins', '_codecs',
            'copy_reg', 'copyreg',
        })

        def find_class(self, module: str, name: str):
            if module.split('.')[0] in ('numpy', 'collections', 'builtins',
                                        '_codecs', 'copy_reg', 'copyreg'):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"Blocked unpickling of {module}.{name} — "
                f"only NumPy data structures are allowed."
            )

    with open(pkl_path, "rb") as fh:
        data = _SafeUnpickler(fh, encoding="latin1").load()

    # Validate expected WESAD structure
    if not isinstance(data, dict):
        raise ValueError("Invalid .pkl file: expected a dictionary at the top level.")
    for key in ("signal", "label"):
        if key not in data:
            raise ValueError(f"Invalid .pkl file: missing required key '{key}'.")
    if not isinstance(data["signal"], dict):
        raise ValueError("Invalid .pkl file: 'signal' must be a dictionary.")
    for sensor in ("chest", "wrist"):
        if sensor not in data["signal"]:
            raise ValueError(f"Invalid .pkl file: missing 'signal/{sensor}' data.")
    chest_sig = data["signal"]["chest"]
    for key in ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp"):
        if key not in chest_sig:
            raise ValueError(f"Invalid .pkl file: missing chest sensor '{key}'.")
    wrist_sig = data["signal"]["wrist"]
    for key in ("ACC", "BVP", "EDA", "TEMP"):
        if key not in wrist_sig:
            raise ValueError(f"Invalid .pkl file: missing wrist sensor '{key}'.")

    # Keep only meaningful WESAD labels (1=Baseline, 2=Stress, 3=Amusement).
    # Label 0 (Transient) is a protocol artefact, not a real physiological state.
    valid_mask = np.isin(data["label"], [1, 2, 3])

    # Fast path: pure numpy arrays — avoids pandas DataFrame overhead
    chest_arr = _extract_chest_array(data["signal"]["chest"])
    n_chest = len(chest_arr)

    # Apply bandpass / lowpass filters to chest signals before resampling
    chest_arr = _filter_chest_signals(chest_arr)

    wrist_arr = _extract_wrist_array(data["signal"]["wrist"], target_len=n_chest)

    features = np.nan_to_num(
        np.hstack([chest_arr, wrist_arr]), nan=0.0,
    )
    labels = data["label"][:n_chest].astype(np.int64)

    # Apply valid-label mask so only labels 1–3 (Baseline/Stress/Amusement) survive
    mask = valid_mask[:n_chest]
    features = features[mask]
    labels = labels[mask]

    return features, labels
