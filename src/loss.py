"""
Multi-label objective: class-weighted focal BCE with label smoothing.

Rare rhythm classes dominate the challenge score's difficulty, so we (1) weight
each class by inverse log-frequency and (2) add a focal term that down-weights
easy negatives. Label smoothing keeps the sigmoid from saturating.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FOCAL_GAMMA, LABEL_SMOOTHING


def compute_class_weights(labels):
    """Inverse log-frequency weights, normalized to mean 1."""
    freq = labels.sum(axis=0)
    freq = np.clip(freq, 1.0, None)
    w = 1.0 / (np.log(freq) + 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class FocalBCELoss(nn.Module):
    def __init__(self, class_weights=None, gamma=FOCAL_GAMMA,
                 smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if self.gamma > 0:
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            bce = bce * (1 - p_t).clamp(min=1e-6) ** self.gamma

        if self.class_weights is not None:
            bce = bce * self.class_weights.unsqueeze(0)

        return bce.mean()
