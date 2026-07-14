"""
Dataset, label parsing, augmentation, and multi-label stratified splitting.
"""

import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from config import (
    CLASSES, NUM_CLASSES, EQUIVALENT_CLASSES, WINDOW_SIZE, LEADS,
    AUG_RANDOM_CROP, AUG_LEAD_DROPOUT, AUG_GAUSSIAN_NOISE, AUG_BASELINE_WANDER,
    FS, N_DEMO, N_WIDE_FEATS,
)
from preprocessing import preprocess, fixed_window
from features import extract_wide_features, demographic_features

_CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
# Map every code to its canonical (representative) index so that equivalent
# diagnoses share one label column.
_EQUIV = {}
for a, b in EQUIVALENT_CLASSES:
    _EQUIV[b] = a


def parse_labels(header_data):
    """One-hot (multi-hot) label vector over the 27 scored classes."""
    label = np.zeros(NUM_CLASSES, dtype=np.float32)
    for line in header_data:
        if line.startswith("#Dx"):
            for code in line.split(":")[1].strip().split(","):
                code = code.strip()
                code = _EQUIV.get(code, code)      # collapse equivalents
                if code in _CLASS_INDEX:
                    label[_CLASS_INDEX[code]] = 1.0
                    # mirror onto the paired column so both stay consistent
                    for a, b in EQUIVALENT_CLASSES:
                        if code == a:
                            label[_CLASS_INDEX[b]] = 1.0
    return label


def load_record(mat_path):
    """Load a .mat recording and its .hea header."""
    data = np.asarray(loadmat(mat_path)["val"], dtype=np.float32)
    with open(mat_path.replace(".mat", ".hea"), "r") as f:
        header = f.readlines()
    return data, header


def find_records(input_directory):
    """List all scored .mat/.hea record stems that carry at least one label."""
    records = []
    for f in os.listdir(input_directory):
        if f.lower().endswith(".mat") and not f.startswith("."):
            records.append(os.path.join(input_directory, f))
    return sorted(records)


# --------------------------------------------------------------------------- #
#  Augmentation                                                               #
# --------------------------------------------------------------------------- #
def _augment(sig, rng):
    """In-place-safe waveform augmentation for a (12, WINDOW_SIZE) window."""
    if rng.random() < AUG_LEAD_DROPOUT:
        drop = rng.integers(0, LEADS)
        sig[drop] = 0.0
    if rng.random() < AUG_GAUSSIAN_NOISE:
        sig = sig + rng.normal(0, 0.05, sig.shape).astype(np.float32)
    if rng.random() < AUG_BASELINE_WANDER:
        t = np.arange(sig.shape[1]) / FS
        freq = rng.uniform(0.15, 0.4)
        wander = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sig = sig + wander[None]
    return sig


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #
class ECGDataset(Dataset):
    """
    Yields (signal_window, wide_features, label).

    Records are preprocessed once and cached in memory as float32; each __getitem__
    draws a (random during training, centered during eval) window and, in
    training mode, applies waveform augmentation.
    """

    def __init__(self, records, feat_stats=None, train=True, cache=True, seed=0):
        self.records = records
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.feat_mean, self.feat_std = (feat_stats or (None, None))
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.records)

    def _prepare(self, path):
        if self._cache is not None and path in self._cache:
            return self._cache[path]
        data, header = load_record(path)
        sig, meta = preprocess(data, header)
        label = parse_labels(header)
        wide = np.concatenate([
            demographic_features(meta),
            extract_wide_features(sig, meta),
        ]).astype(np.float32)
        item = (sig, wide, label)
        if self._cache is not None:
            self._cache[path] = item
        return item

    def __getitem__(self, idx):
        sig, wide, label = self._prepare(self.records[idx])

        if self.train and sig.shape[1] > WINDOW_SIZE and self.rng.random() < AUG_RANDOM_CROP:
            start = self.rng.integers(0, sig.shape[1] - WINDOW_SIZE + 1)
            window = sig[:, start:start + WINDOW_SIZE].copy()
        else:
            window = fixed_window(sig, WINDOW_SIZE).copy()

        if self.train:
            window = _augment(window, self.rng)

        wide = wide.copy()
        if self.feat_mean is not None:
            wide = (wide - self.feat_mean) / self.feat_std
            wide[~np.isfinite(wide)] = 0.0

        return (
            torch.from_numpy(window).float(),
            torch.from_numpy(wide).float(),
            torch.from_numpy(label).float(),
        )

    def all_labels(self):
        return np.stack([self._prepare(p)[2] for p in self.records])

    def all_wide(self):
        return np.stack([self._prepare(p)[1] for p in self.records])


# --------------------------------------------------------------------------- #
#  Multi-label stratified k-fold (iterative stratification)                   #
# --------------------------------------------------------------------------- #
def stratified_kfold(labels, n_folds, seed=0):
    """
    Iterative stratification (Sechidis et al. 2011): assign each sample to the
    fold that most needs its rarest active label, keeping every class balanced
    across folds. Falls back gracefully for label-free rows.
    """
    rng = np.random.default_rng(seed)
    n, n_classes = labels.shape
    fold_of = np.full(n, -1, dtype=int)

    target = np.array([n / n_folds] * n_folds)
    # desired positives per (fold, class)
    class_totals = labels.sum(axis=0)
    desired = np.outer(np.ones(n_folds) / n_folds, class_totals)
    fold_counts = np.zeros((n_folds, n_classes))
    fold_size = np.zeros(n_folds)

    # process rarest labels first for the best balance
    order = np.argsort(class_totals)
    remaining = list(rng.permutation(n))
    remaining_set = set(remaining)

    for c in order:
        members = [i for i in remaining if labels[i, c] > 0]
        for i in members:
            if fold_of[i] != -1:
                continue
            need = desired[:, c] - fold_counts[:, c]
            best = np.flatnonzero(need == need.max())
            if len(best) > 1:                      # tie-break on total size
                sizes = fold_size[best]
                best = best[np.flatnonzero(sizes == sizes.min())]
            f = int(rng.choice(best))
            fold_of[i] = f
            fold_counts[f] += labels[i]
            fold_size[f] += 1
            remaining_set.discard(i)

    # any rows with no active label
    for i in list(remaining_set):
        if fold_of[i] == -1:
            f = int(np.argmin(fold_size))
            fold_of[i] = f
            fold_size[f] += 1

    return fold_of
