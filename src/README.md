# SE-ResNet + Transformer — 12-lead ECG classifier

A complete, runnable model for the **PhysioNet/CinC Challenge 2020** (27-class
multi-label diagnosis from 12-lead ECG), synthesized from the design choices
that recurred across the top-10 leaderboard solutions in this repository.

It is a drop-in for the standard challenge harness: `train_model.py` and
`driver.py` are unchanged, and `train_12ECG_classifier` / `run_12ECG_classifier`
keep the required signatures.

```bash
pip install -r requirements.txt
python train_model.py  <training_data_dir>  <model_dir>     # train + cross-validate
python driver.py       <model_dir> <test_dir> <output_dir>  # classify
```

---

## Architecture

```
 12-lead ECG (any rate/length)
        │  resample→500 Hz · gain · band-pass 3–45 Hz · robust per-lead norm
        ▼
 Multi-scale stem            parallel convs k=7/15/31, stride 2 → 1×1 project → maxpool
        ▼
 SE-ResNet-34 backbone       4 stages [3,4,6,3] of SE residual blocks (squeeze-excite,
        │                    in-block dropout, zero-init last BN)
        ▼
 Transformer encoder         2 layers · 8 heads · sinusoidal positions   (optional)
        ▼
 Attention pooling           learned soft-attention over time → 512-d vector
        ▼
 Late fusion                 concat [512 deep ⊕ age,sex ⊕ 20 hand-crafted feats]
        │                    → Linear 256 → BN → dropout
        ▼
 27 sigmoid logits           equivalent classes tied to one probability
```

~21 M trainable parameters. Every stage is annotated in `model.py` with the team
it was borrowed from.

### Why each piece

| Component | Rationale | Seen in |
|---|---|---|
| **1D SE-ResNet backbone** | The single most reliable core in the field; squeeze-excitation recalibrates lead/channel importance per recording. | Teams 2, 6, 8, 10 |
| **Multi-scale wide-kernel stem** | Short kernels catch QRS morphology, long kernels catch rhythm — both at the first layer. | Teams 5, 8 |
| **Transformer temporal head** | Self-attention models long-range dependencies (e.g. intermittent ectopy) a fixed receptive field misses. Toggle with `USE_TRANSFORMER`. | Team 1 (CTN) |
| **Attention pooling** | A learned, sharper summary than global mean-pooling. | — |
| **Demographic + feature fusion** | Age/sex and HRV/template statistics are cheap, robust signal fused right before the head. | Teams 1, 10 |
| **Adaptive pooling** | One model ingests any-length recordings without retraining. | Teams 5, 6 |

---

## Training recipe (the details that move the score)

- **Data split** — iterative-stratified multi-label k-fold (`N_FOLDS=5`) so every
  rare class is balanced across folds (`dataset.stratified_kfold`).
- **Objective** — class-weighted (`1/log(freq)`) **focal BCE** with label
  smoothing; the focal term rescues rare rhythm classes (`loss.py`).
- **Optimizer** — AdamW, `lr=1e-3`, weight decay `1e-2`, **linear warmup → cosine
  decay**.
- **Regularization** — mixup (`α=0.2`), waveform augmentation (random crop, lead
  dropout, Gaussian noise, baseline wander), in-block dropout.
- **Weight averaging** — an **EMA** copy of the weights is evaluated and saved
  each epoch (consistently beats the raw weights).
- **Thresholds** — after training, **per-class decision thresholds** are tuned on
  the held-out fold by coordinate ascent against the official challenge metric
  (`metric.optimize_thresholds`) — not a flat 0.5.

## Inference

Ensembles all fold models, averages sigmoid probabilities over **sliding
windows** (test-time augmentation), applies the averaged tuned thresholds, and
mirrors predictions onto equivalent-class columns. Guarantees at least one label
per record.

---

## The challenge metric & equivalent classes

`weights.csv` (the official 27×27 reward matrix) ships with the package and is
used for model selection and threshold tuning. The three scored-as-equal pairs —
**CRBBB≡RBBB, PAC≡SVPB, PVC≡VPB** — are merged in the labels and tied at output,
matching the challenge scoring.

If `weights.csv` is absent the pipeline falls back to a macro-F score so it still
runs.

---

## Files

| File | Role |
|---|---|
| `config.py` | Every hyper-parameter, class list, equivalent pairs |
| `model.py` | **The architecture** (SE-ResNet + Transformer + fusion) |
| `preprocessing.py` | Resample, filter, normalize, windowing |
| `features.py` | Demographic + HRV/template/waveform features |
| `dataset.py` | Dataset, augmentation, label parsing, stratified k-fold |
| `loss.py` | Class-weighted focal BCE |
| `metric.py` | Official challenge metric + threshold optimization |
| `train_12ECG_classifier.py` | Cross-validated training loop (EMA, mixup, cosine) |
| `run_12ECG_classifier.py` | Ensembled, TTA inference |
| `train_model.py`, `driver.py` | Challenge harness (unchanged) |

## Tuning knobs

- Backbone depth/width: `BLOCK_LAYERS`, `BLOCK_CHANNELS` (e.g. `[3,4,6,3]` → 34,
  `[3,4,23,3]` → 101).
- `USE_TRANSFORMER=False` for a pure-CNN, faster variant.
- `WINDOW_SECONDS`, `FS` for the analysis window.
- `N_FOLDS`, `EPOCHS`, `MIXUP_ALPHA`, `FOCAL_GAMMA` for the training budget.

## Verified

The full train → save → load → predict path and the challenge metric are covered
by an integration test on synthetic WFDB records (variable sampling rates and
lengths). A perfect prediction scores exactly 1.0 under the loaded reward matrix.
