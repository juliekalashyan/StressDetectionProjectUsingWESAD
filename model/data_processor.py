"""Data processing utilities for WESAD .pkl files.

Extracts chest and wrist sensor signals and aligns them to a common
sampling rate following the same logic used in the original notebook.
"""

import numpy as np
import pandas as pd
import pickle

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
    """Resample *signal* to exactly *target_len* samples using linear interpolation."""
    n = len(signal)
    if n == target_len:
        return signal.astype(np.float64)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, signal).astype(np.float64)


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
    """
    

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")

    # Keep only valid WESAD labels (0-4); drop any non-standard ones
    # (e.g. label 7 = "not defined" in some subjects)
    valid_mask = np.isin(data["label"], [0, 1, 2, 3, 4])

    chest_df = extract_chest_data(data["signal"]["chest"])
    wrist_df = extract_wrist_data(data["signal"]["wrist"], target_len=len(chest_df))

    combined = pd.concat(
        [chest_df.reset_index(drop=True), wrist_df.reset_index(drop=True)],
        axis=1,
    )
    features = np.nan_to_num(combined.to_numpy(dtype=np.float32), nan=0.0)
    labels = data["label"][:len(chest_df)].astype(np.int64)

    # Apply valid-label mask so only labels 0–4 survive
    mask = valid_mask[:len(chest_df)]
    features = features[mask]
    labels = labels[mask]

    return features, labels
