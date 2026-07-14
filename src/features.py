"""
Hand-crafted "wide" features fused with the deep representation.

Demographics (age, sex) plus a compact set of heart-rate-variability, waveform
and template statistics. These give the classifier cheap, robust signal that a
single fixed window can miss — the fusion trick from Teams 1 and 10.

The feature vector length is `N_DEMO + N_WIDE_FEATS` and its order is fixed.
Features are z-normalized at train time; the means/stds are saved with the
model and reapplied at inference.
"""

import numpy as np
from scipy.signal import find_peaks

from config import FS, N_WIDE_FEATS


def _rpeaks(lead_signal, fs=FS):
    """Lightweight R-peak detector (no external deps): normalize, threshold."""
    sig = lead_signal - np.mean(lead_signal)
    denom = np.std(sig) + 1e-8
    sig = sig / denom
    # refractory period ~ 0.25 s prevents double detections
    peaks, _ = find_peaks(sig, distance=int(0.25 * fs), height=1.0)
    return peaks


def _hrv_stats(rr):
    """Standard time-domain HRV descriptors from an RR-interval series (s)."""
    if len(rr) < 2:
        return [0.0] * 6
    diff = np.diff(rr)
    mean_rr = np.mean(rr)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(diff ** 2))
    pnn50 = np.mean(np.abs(diff) > 0.05)
    hr = 60.0 / mean_rr if mean_rr > 0 else 0.0
    cv = sdnn / mean_rr if mean_rr > 0 else 0.0
    return [mean_rr, sdnn, rmssd, pnn50, hr, cv]


def extract_wide_features(recording, meta, channel=1):
    """
    Compute the fixed-length hand-crafted feature vector for one recording.

    recording: (12, T) preprocessed signal
    meta:      dict from preprocessing.parse_header (age, sex)
    returns:   float32 vector of length N_WIDE_FEATS
    """
    lead = recording[channel]
    peaks = _rpeaks(lead)
    rr = np.diff(peaks) / FS if len(peaks) > 1 else np.array([])

    feats = []
    feats += _hrv_stats(rr)                                    # 6 HRV features

    # Full-waveform amplitude statistics on the analysis lead.
    feats += [
        float(np.mean(lead)),
        float(np.std(lead)),
        float(np.percentile(lead, 5)),
        float(np.percentile(lead, 95)),
        float(np.max(lead) - np.min(lead)),
        float(np.mean(np.abs(np.diff(lead)))),                 # mean abs slope
    ]                                                          # +6 = 12

    # QRS template statistics: beat-to-beat amplitude consistency.
    if len(peaks) >= 3:
        half = int(0.10 * FS)
        beats = [lead[max(0, p - half):p + half]
                 for p in peaks if half <= p < len(lead) - half]
        beats = [b for b in beats if len(b) == 2 * half]
        if beats:
            beats = np.stack(beats)
            template = beats.mean(axis=0)
            feats += [
                float(np.mean(np.abs(beats - template))),      # template error
                float(np.mean(np.ptp(beats, axis=1))),         # QRS amplitude
                float(np.std(np.ptp(beats, axis=1))),          # amplitude spread
                float(len(peaks)),                             # beat count
            ]
        else:
            feats += [0.0, 0.0, 0.0, float(len(peaks))]
    else:
        feats += [0.0, 0.0, 0.0, float(len(peaks))]            # +4 = 16

    # Cross-lead energy summary (rhythm information beyond the analysis lead).
    lead_energy = np.mean(recording ** 2, axis=1)
    feats += [
        float(np.mean(lead_energy)),
        float(np.std(lead_energy)),
        float(np.max(lead_energy)),
        float(np.min(lead_energy)),
    ]                                                          # +4 = 20

    feats = np.asarray(feats[:N_WIDE_FEATS], dtype=np.float32)
    if len(feats) < N_WIDE_FEATS:
        feats = np.pad(feats, (0, N_WIDE_FEATS - len(feats)))
    feats[~np.isfinite(feats)] = 0.0
    return feats


def demographic_features(meta, age_mean=50.0, age_std=20.0):
    """Age (standardized) and sex (0/1) as a length-2 vector."""
    age = (meta["age"] - age_mean) / age_std
    return np.asarray([age, meta["sex"]], dtype=np.float32)
