"""
Central configuration for the SE-ResNet + Transformer ECG classifier.

Every hyper-parameter that matters for reproducing state-of-the-art results on
the PhysioNet/CinC 2020 12-lead ECG task lives here so that training and
inference always agree.
"""

# --------------------------------------------------------------------------- #
#  Scored classes (SNOMED-CT codes)                                           #
# --------------------------------------------------------------------------- #
# The 27 scored diagnoses. Order is fixed and shared by training + inference.
CLASSES = sorted([
    "270492004", "164889003", "164890007", "426627000", "713427006",
    "713426002", "445118002", "39732003",  "164909002", "251146004",
    "698252002", "10370003",  "284470004", "427172004", "164947007",
    "111975006", "164917005", "47665007",  "59118001",  "427393009",
    "426177001", "426783006", "427084000", "63593006",  "164934002",
    "59931005",  "17338001",
])
NUM_CLASSES = len(CLASSES)

# Three pairs are scored as identical by the challenge metric. We train the
# network to emit the SAME probability for both members of a pair by merging
# their labels, which is what every top team did.
EQUIVALENT_CLASSES = [
    ("713427006", "59118001"),   # CRBBB  == RBBB
    ("284470004", "63593006"),   # PAC    == SVPB
    ("427172004", "17338001"),   # PVC    == VPB
]
NORMAL_CLASS = "426783006"       # sinus rhythm — used by the challenge metric


# --------------------------------------------------------------------------- #
#  Signal front-end                                                           #
# --------------------------------------------------------------------------- #
FS = 500                     # Hz — every recording is resampled to this rate
WINDOW_SECONDS = 15          # length of the analysis window
WINDOW_SIZE = FS * WINDOW_SECONDS      # 7500 samples
LEADS = 12
FILTER_BANDWIDTH = [3, 45]   # FIR band-pass (Hz); removes baseline + HF noise
GAIN_CLIP = 30_000.0         # amplitude clip used for stable normalization


# --------------------------------------------------------------------------- #
#  Model architecture                                                         #
# --------------------------------------------------------------------------- #
STEM_CHANNELS = 64
STEM_KERNELS = [7, 15, 31]   # multi-scale stem: morphology + rhythm scales
BLOCK_LAYERS = [3, 4, 6, 3]  # SE-ResNet-34 layout
BLOCK_CHANNELS = [64, 128, 256, 512]
BLOCK_KERNEL = 7             # 1D residual convolution kernel
SE_REDUCTION = 16            # squeeze-excitation bottleneck ratio
BLOCK_DROPOUT = 0.2

USE_TRANSFORMER = True       # temporal self-attention head on top of the CNN
D_MODEL = 512                # must equal BLOCK_CHANNELS[-1]
N_HEAD = 8
D_FF = 1024
N_TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.2

N_DEMO = 2                   # age, sex
N_WIDE_FEATS = 20            # hand-crafted HRV / template / waveform features
FUSION_HIDDEN = 256
HEAD_DROPOUT = 0.3


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #
N_FOLDS = 5                  # iterative-stratified multi-label folds
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.02
FOCAL_GAMMA = 2.0            # focal term for rare classes; 0 disables
MIXUP_ALPHA = 0.2           # 0 disables mixup
EMA_DECAY = 0.999           # exponential moving average of weights
GRAD_CLIP = 1.0
SEED = 2020

# Augmentation probabilities (applied only during training)
AUG_RANDOM_CROP = 0.5
AUG_LEAD_DROPOUT = 0.2
AUG_GAUSSIAN_NOISE = 0.2
AUG_BASELINE_WANDER = 0.2

# Inference
TTA_WINDOWS = 10            # sliding windows averaged at test time
