"""
Training entry point called by the challenge harness (train_model.py).

    train_12ECG_classifier(input_directory, output_directory)

Runs N-fold iterative-stratified cross-validation. Each fold trains an ECGNet
with AdamW + warmup-cosine schedule, mixup, and an EMA copy of the weights;
after training, per-class thresholds are tuned on the held-out fold against the
challenge metric. All fold models, thresholds, and feature statistics are saved
to `output_directory` for the inference script to ensemble.
"""

import os
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    N_FOLDS, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, WARMUP_EPOCHS, MIXUP_ALPHA,
    EMA_DECAY, GRAD_CLIP, SEED, NUM_CLASSES,
)
from dataset import ECGDataset, find_records, stratified_kfold
from model import build_model
from loss import FocalBCELoss, compute_class_weights
from metric import load_weights, optimize_thresholds, score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model parameters — a free eval boost."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def warmup_cosine(optimizer, epoch, step, steps_per_epoch):
    """Linear warmup then cosine decay of the learning rate."""
    total = EPOCHS * steps_per_epoch
    warm = WARMUP_EPOCHS * steps_per_epoch
    cur = epoch * steps_per_epoch + step
    if cur < warm:
        lr = LR * cur / max(warm, 1)
    else:
        progress = (cur - warm) / max(total - warm, 1)
        lr = 0.5 * LR * (1 + np.cos(np.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def mixup(x, wide, y, alpha, rng):
    """Convex combinations of pairs — regularizes the multi-label boundary."""
    if alpha <= 0:
        return x, wide, y, y, 1.0
    lam = rng.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[perm]
    wide = lam * wide + (1 - lam) * wide[perm]
    return x, wide, y, y[perm], lam


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for sig, wide, y in loader:
        out = model(sig.to(DEVICE), wide.to(DEVICE))
        probs.append(torch.sigmoid(out).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one_fold(train_ds, val_ds, class_weights, weights_matrix, fold, out_dir):
    rng = np.random.default_rng(SEED + fold)
    model = build_model().to(DEVICE)
    ema = EMA(model, EMA_DECAY)
    criterion = FocalBCELoss(class_weights.to(DEVICE)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    steps = len(train_loader)

    best_score, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        for step, (sig, wide, y) in enumerate(train_loader):
            lr = warmup_cosine(optimizer, epoch, step, steps)
            sig, wide, y = sig.to(DEVICE), wide.to(DEVICE), y.to(DEVICE)
            sig, wide, ya, yb, lam = mixup(sig, wide, y, MIXUP_ALPHA, rng)

            optimizer.zero_grad()
            out = model(sig, wide)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ema.update(model)

        # Evaluate the EMA weights each epoch at a coarse 0.3 threshold.
        probs, labels = predict_probs(ema.shadow, val_loader)
        coarse = np.full(NUM_CLASSES, 0.3)
        s = score(labels, probs, coarse, weights_matrix)
        print(f"  fold {fold} epoch {epoch+1:02d}/{EPOCHS}  lr={lr:.2e}  "
              f"val_score={s:.4f}")
        if s > best_score:
            best_score = s
            best_state = copy.deepcopy(ema.shadow.state_dict())

    # Restore best EMA weights, tune thresholds on the held-out fold.
    model.load_state_dict(best_state)
    probs, labels = predict_probs(model, val_loader)
    thresholds, tuned = optimize_thresholds(labels, probs, weights_matrix)
    print(f"  fold {fold} best={best_score:.4f}  tuned={tuned:.4f}")

    fold_dir = os.path.join(out_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save({"model_state_dict": best_state}, os.path.join(fold_dir, "model.tar"))
    np.savetxt(os.path.join(fold_dir, "thresholds.txt"), thresholds)
    return tuned


def train_12ECG_classifier(input_directory, output_directory):
    set_seed(SEED)
    os.makedirs(output_directory, exist_ok=True)

    records = find_records(input_directory)
    print(f"Found {len(records)} records.")

    # One pass to build label matrix + feature statistics (cached in the dataset).
    base = ECGDataset(records, feat_stats=None, train=False, cache=True, seed=SEED)
    labels = base.all_labels()
    wide = base.all_wide()
    feat_mean = wide.mean(axis=0)
    feat_std = wide.std(axis=0) + 1e-6
    np.savetxt(os.path.join(output_directory, "feat_mean.txt"), feat_mean)
    np.savetxt(os.path.join(output_directory, "feat_std.txt"), feat_std)

    class_weights = compute_class_weights(labels)
    weights_matrix = load_weights(os.path.join(os.path.dirname(__file__), "weights.csv"))
    if weights_matrix is None:
        print("weights.csv not found — using macro-F fallback for model selection.")

    fold_of = stratified_kfold(labels, N_FOLDS, seed=SEED)

    scores = []
    for fold in range(N_FOLDS):
        tr_idx = np.flatnonzero(fold_of != fold)
        va_idx = np.flatnonzero(fold_of == fold)
        tr_records = [records[i] for i in tr_idx]
        va_records = [records[i] for i in va_idx]

        train_ds = ECGDataset(tr_records, (feat_mean, feat_std), train=True,
                              cache=True, seed=SEED + fold)
        val_ds = ECGDataset(va_records, (feat_mean, feat_std), train=False,
                            cache=True, seed=SEED + fold)
        # Reuse the already-preprocessed cache from `base`.
        train_ds._cache.update({p: base._cache[p] for p in tr_records})
        val_ds._cache.update({p: base._cache[p] for p in va_records})

        print(f"[Fold {fold}] train={len(tr_records)} val={len(va_records)}")
        scores.append(
            train_one_fold(train_ds, val_ds, class_weights, weights_matrix,
                           fold, output_directory)
        )

    print(f"Cross-validation score: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
