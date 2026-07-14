"""
Signal front-end shared by training and inference.

Every recording is brought to a common representation before the network sees
it: single sampling rate, band-pass filtered, per-lead normalized, and cropped
or padded to a fixed window. This is the one step all ten winning teams share.
"""

import numpy as np
from scipy.signal import resample, decimate, butter, filtfilt

from config import FS, WINDOW_SIZE, FILTER_BANDWIDTH, LEADS


def parse_header(header_data):
    """Pull sampling rate, gains, age and sex out of a WFDB .hea header."""
    first = header_data[0].strip().split()
    sampling_rate = int(first[2])
    n_samples = int(first[3])

    gains = []
    for line in header_data[1:1 + LEADS]:
        parts = line.strip().split()
        # gain is the 3rd field, formatted like "1000/mV"
        gain = float(parts[2].split("/")[0]) if len(parts) > 2 else 1000.0
        gains.append(gain if gain != 0 else 1000.0)

    age, sex = 50.0, 0.0
    for line in header_data:
        if line.startswith("#Age"):
            val = line.split(":")[1].strip()
            try:
                age = float(val)
            except ValueError:
                age = 50.0
            if not np.isfinite(age):
                age = 50.0
        elif line.startswith("#Sex"):
            val = line.split(":")[1].strip().lower()
            sex = 1.0 if val.startswith("f") else 0.0

    return {
        "sampling_rate": sampling_rate,
        "n_samples": n_samples,
        "gains": np.asarray(gains, dtype=np.float32),
        "age": age,
        "sex": sex,
    }


def resample_to_fs(recording, sampling_rate, fs=FS):
    """Bring any sampling rate to the common `fs`."""
    if sampling_rate == fs:
        return recording
    if sampling_rate > fs and sampling_rate % fs == 0:
        return decimate(recording, sampling_rate // fs, axis=1, zero_phase=True)
    new_len = int(round(recording.shape[1] * fs / sampling_rate))
    return resample(recording, new_len, axis=1)


def bandpass(recording, bandwidth=FILTER_BANDWIDTH, fs=FS):
    """Zero-phase Butterworth band-pass; removes baseline drift and HF noise."""
    low, high = bandwidth
    nyq = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    # filtfilt needs signal longer than the padding it applies.
    if recording.shape[1] <= 3 * max(len(a), len(b)):
        return recording
    return filtfilt(b, a, recording, axis=1)


def normalize_per_lead(recording, eps=1e-8):
    """Robust per-lead standardization (median / IQR) — resistant to spikes."""
    med = np.median(recording, axis=1, keepdims=True)
    q75 = np.percentile(recording, 75, axis=1, keepdims=True)
    q25 = np.percentile(recording, 25, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, eps)
    return (recording - med) / iqr


def preprocess(recording, header_data):
    """Full front-end: gain -> resample -> filter -> normalize. Returns float32."""
    meta = parse_header(header_data)
    recording = np.asarray(recording, dtype=np.float32)

    # Convert ADC units to physical mV using per-lead gain.
    recording = recording / meta["gains"][:, None]
    recording = resample_to_fs(recording, meta["sampling_rate"])
    recording = bandpass(recording)
    recording = normalize_per_lead(recording)
    return np.ascontiguousarray(recording, dtype=np.float32), meta


def fixed_window(recording, window_size=WINDOW_SIZE, start=None):
    """Crop or right-pad a recording to exactly `window_size` samples."""
    n = recording.shape[1]
    if n < window_size:
        pad = window_size - n
        return np.pad(recording, ((0, 0), (0, pad)))
    if start is None:
        start = (n - window_size) // 2
    start = max(0, min(start, n - window_size))
    return recording[:, start:start + window_size]


def sliding_windows(recording, window_size=WINDOW_SIZE, n_windows=10):
    """Evenly-spaced windows for test-time averaging over a long recording."""
    n = recording.shape[1]
    if n <= window_size:
        return [fixed_window(recording, window_size)]
    starts = np.linspace(0, n - window_size, num=n_windows, dtype=int)
    return [recording[:, s:s + window_size] for s in np.unique(starts)]
