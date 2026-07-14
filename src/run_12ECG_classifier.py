"""
Inference entry point called by the challenge harness (driver.py).

    load_12ECG_model(model_dir)                 -> loaded model bundle
    run_12ECG_classifier(data, header, model)   -> (labels, scores, classes)

Ensembles every fold model, averages sigmoid probabilities over sliding windows
(test-time augmentation), applies the tuned per-class thresholds, and mirrors
predictions onto equivalent-class columns.
"""

import os

import numpy as np
import torch

from config import (
    CLASSES, NUM_CLASSES, EQUIVALENT_CLASSES, WINDOW_SIZE, TTA_WINDOWS,
)
from model import build_model
from preprocessing import preprocess, sliding_windows
from features import extract_wide_features, demographic_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def load_12ECG_model(model_dir):
    """Load fold models, thresholds, and feature normalization statistics."""
    feat_mean = np.loadtxt(os.path.join(model_dir, "feat_mean.txt"))
    feat_std = np.loadtxt(os.path.join(model_dir, "feat_std.txt"))

    models, thresholds = [], []
    for name in sorted(os.listdir(model_dir)):
        fold_dir = os.path.join(model_dir, name)
        tar = os.path.join(fold_dir, "model.tar")
        if not (name.startswith("fold_") and os.path.exists(tar)):
            continue
        net = build_model().to(DEVICE)
        state = torch.load(tar, map_location=DEVICE)
        net.load_state_dict(state["model_state_dict"])
        net.eval()
        models.append(net)
        thresholds.append(np.loadtxt(os.path.join(fold_dir, "thresholds.txt")))

    if not models:
        raise RuntimeError(f"No fold models found in {model_dir}")

    return {
        "models": models,
        "thresholds": np.mean(thresholds, axis=0),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }


def _mirror_equivalents(vec):
    """Force both members of each equivalent pair to agree (use the max)."""
    for a, b in EQUIVALENT_CLASSES:
        ia, ib = _CLASS_INDEX[a], _CLASS_INDEX[b]
        vec[ia] = vec[ib] = max(vec[ia], vec[ib])
    return vec


@torch.no_grad()
def run_12ECG_classifier(data, header_data, loaded_model):
    sig, meta = preprocess(data, header_data)

    wide = np.concatenate([
        demographic_features(meta),
        extract_wide_features(sig, meta),
    ]).astype(np.float32)
    wide = (wide - loaded_model["feat_mean"]) / loaded_model["feat_std"]
    wide[~np.isfinite(wide)] = 0.0
    wide_t = torch.from_numpy(wide).float().unsqueeze(0).to(DEVICE)

    windows = sliding_windows(sig, WINDOW_SIZE, TTA_WINDOWS)

    prob_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    n_terms = 0
    for net in loaded_model["models"]:
        for w in windows:
            x = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEVICE)
            out = net(x, wide_t)
            prob_sum += torch.sigmoid(out).cpu().numpy()[0]
            n_terms += 1

    probs = prob_sum / max(n_terms, 1)
    probs = _mirror_equivalents(probs)

    preds = (probs > loaded_model["thresholds"]).astype(int)
    # Guarantee at least one prediction: fall back to the most probable class.
    if preds.sum() == 0:
        preds[int(np.argmax(probs))] = 1
    preds = _mirror_equivalents(preds).astype(int)

    return preds, probs.astype(np.float32), CLASSES
