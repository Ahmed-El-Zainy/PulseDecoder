"""
The official PhysioNet/CinC 2020 challenge metric plus per-class threshold search.

The challenge metric is a weighted, reward-matrix accuracy that credits partial
agreement between confusable diagnoses. We need it during training to (a) pick
the best epoch and (b) tune per-class decision thresholds — a flat 0.5 leaves
several points of score on the table.

Place the official `weights.csv` next to this file (present in several team
folders in this repo). If it is missing we fall back to a macro-F score so the
pipeline still runs.
"""

import os

import numpy as np

from config import CLASSES, NORMAL_CLASS


def load_weights(weights_path="weights.csv"):
    """Load the 27x27 reward matrix aligned to CLASSES; None if unavailable."""
    if not os.path.exists(weights_path):
        return None
    rows = []
    with open(weights_path, "r") as f:
        header = f.readline().strip().split(",")[1:]
        for line in f:
            rows.append(line.strip().split(",")[1:])
    file_classes = [c.strip() for c in header]
    matrix = np.array(rows, dtype=np.float64)

    # Reorder to our CLASSES order (file may group equivalent codes as "a|b").
    idx = []
    for c in CLASSES:
        found = None
        for j, fc in enumerate(file_classes):
            if c in fc.split("|"):
                found = j
                break
        idx.append(found)
    if any(i is None for i in idx):
        return None
    idx = np.array(idx)
    return matrix[np.ix_(idx, idx)]


def _normalizer(labels, normal_idx):
    """Denominator: score if every record were labeled with its true classes."""
    return labels


def compute_challenge_metric(weights, labels, outputs):
    """
    weights: (C, C) reward matrix
    labels:  (N, C) bool ground truth
    outputs: (N, C) bool predictions
    """
    labels = labels.astype(bool)
    outputs = outputs.astype(bool)
    n, c = labels.shape
    normal_idx = CLASSES.index(NORMAL_CLASS)

    def observed(pred):
        A = np.zeros((c, c))
        for i in range(n):
            true = np.flatnonzero(labels[i])
            pos = np.flatnonzero(pred[i])
            norm = max(len(np.union1d(true, pos)), 1)
            for t in true:
                for p in pos:
                    A[t, p] += 1.0 / norm
        return np.sum(weights * A)

    correct = observed(outputs)
    ideal = observed(labels)

    inactive = np.zeros_like(labels)
    inactive[:, normal_idx] = 1
    baseline = observed(inactive)

    if ideal - baseline == 0:
        return 0.0
    return float((correct - baseline) / (ideal - baseline))


def macro_f_measure(labels, outputs):
    """Fallback when the reward matrix is unavailable."""
    labels, outputs = labels.astype(bool), outputs.astype(bool)
    fs = []
    for j in range(labels.shape[1]):
        tp = np.sum(labels[:, j] & outputs[:, j])
        fp = np.sum(~labels[:, j] & outputs[:, j])
        fn = np.sum(labels[:, j] & ~outputs[:, j])
        denom = 2 * tp + fp + fn
        fs.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(fs))


def score(labels, probs, thresholds, weights=None):
    """Evaluate a probability matrix under given per-class thresholds."""
    preds = probs > thresholds[None]
    if weights is not None:
        return compute_challenge_metric(weights, labels, preds)
    return macro_f_measure(labels, preds)


def optimize_thresholds(labels, probs, weights=None, grid=None, rounds=2):
    """
    Coordinate-ascent search over per-class thresholds to maximize the metric.
    Repeats a few passes because thresholds interact through the reward matrix.
    """
    if grid is None:
        grid = np.arange(0.05, 0.85, 0.02)
    c = labels.shape[1]
    thresholds = np.full(c, 0.3)
    best = score(labels, probs, thresholds, weights)

    for _ in range(rounds):
        for j in range(c):
            base = thresholds.copy()
            best_t, best_j = thresholds[j], best
            for t in grid:
                base[j] = t
                s = score(labels, probs, base, weights)
                if s > best_j:
                    best_j, best_t = s, t
            thresholds[j] = best_t
            best = best_j
    return thresholds, best
