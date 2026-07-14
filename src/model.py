"""
SE-ResNet + Transformer architecture for 12-lead ECG multi-label classification.

Design rationale (synthesized from the PhysioNet/CinC 2020 top-10 solutions):

    signal ─► multi-scale stem ─► SE-ResNet backbone ─► Transformer encoder
                                                             │
                          attention pooling ◄───────────────┘
                                   │
        [ deep features ⊕ age/sex ⊕ hand-crafted features ]
                                   │
                            fusion MLP ─► 27 sigmoid logits

Every element is one that recurred among the winners:
  * 1D SE-ResNet backbone  ......... Teams 2, 6, 8, 10 (the consensus core)
  * multi-scale wide-kernel stem ... Teams 5, 8
  * self-attention temporal head ... Team 1 (CTN)
  * demographic + feature fusion ... Teams 1, 10
  * adaptive pooling (any length) .. Teams 5, 6
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    LEADS, STEM_CHANNELS, STEM_KERNELS, BLOCK_LAYERS, BLOCK_CHANNELS,
    BLOCK_KERNEL, SE_REDUCTION, BLOCK_DROPOUT, USE_TRANSFORMER, D_MODEL,
    N_HEAD, D_FF, N_TRANSFORMER_LAYERS, TRANSFORMER_DROPOUT, N_DEMO,
    N_WIDE_FEATS, FUSION_HIDDEN, HEAD_DROPOUT, NUM_CLASSES,
)


# --------------------------------------------------------------------------- #
#  Building blocks                                                            #
# --------------------------------------------------------------------------- #
class SELayer(nn.Module):
    """Squeeze-and-Excitation: recalibrate channels by global context."""

    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class SEResBlock(nn.Module):
    """1D residual block with squeeze-excitation and in-block dropout."""

    expansion = 1

    def __init__(self, in_ch, out_ch, kernel=BLOCK_KERNEL, stride=1,
                 downsample=None, dropout=BLOCK_DROPOUT):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1,
                               padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SELayer(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out = out + identity
        return self.relu(out)


class MultiScaleStem(nn.Module):
    """
    Parallel wide-kernel convolutions capture morphology (short kernels) and
    rhythm (long kernels) simultaneously, then merge to the stem width.
    """

    def __init__(self, in_ch=LEADS, out_ch=STEM_CHANNELS, kernels=STEM_KERNELS):
        super().__init__()
        branch_ch = out_ch // len(kernels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, k, stride=2, padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for k in kernels
        ])
        merged = branch_ch * len(kernels)
        self.project = nn.Sequential(
            nn.Conv1d(merged, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        out = self.project(out)
        return self.pool(out)


class AttentionPool(nn.Module):
    """Learned attention pooling over time — a sharper summary than mean-pool."""

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):                       # x: (B, C, T)
        attn = torch.softmax(self.score(x), dim=-1)   # (B, 1, T)
        return (x * attn).sum(dim=-1)                 # (B, C)


class PositionalEncoding(nn.Module):
    """Sinusoidal positions added to the convolved sequence."""

    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                       # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# --------------------------------------------------------------------------- #
#  Full model                                                                 #
# --------------------------------------------------------------------------- #
class ECGNet(nn.Module):
    """
    12-lead ECG classifier: multi-scale stem, SE-ResNet backbone, optional
    Transformer temporal head, attention pooling, and late fusion with
    demographic + hand-crafted features.
    """

    def __init__(self, num_classes=NUM_CLASSES, n_demo=N_DEMO,
                 n_wide=N_WIDE_FEATS, use_transformer=USE_TRANSFORMER):
        super().__init__()
        self.use_transformer = use_transformer

        self.stem = MultiScaleStem()

        self.in_ch = STEM_CHANNELS
        self.layer1 = self._make_layer(BLOCK_CHANNELS[0], BLOCK_LAYERS[0], stride=1)
        self.layer2 = self._make_layer(BLOCK_CHANNELS[1], BLOCK_LAYERS[1], stride=2)
        self.layer3 = self._make_layer(BLOCK_CHANNELS[2], BLOCK_LAYERS[2], stride=2)
        self.layer4 = self._make_layer(BLOCK_CHANNELS[3], BLOCK_LAYERS[3], stride=2)
        feat_dim = BLOCK_CHANNELS[-1]

        if use_transformer:
            assert feat_dim == D_MODEL, "D_MODEL must equal final block width"
            self.pos_enc = PositionalEncoding(D_MODEL, TRANSFORMER_DROPOUT)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF,
                dropout=TRANSFORMER_DROPOUT, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, N_TRANSFORMER_LAYERS)

        self.attn_pool = AttentionPool(feat_dim)

        # Late fusion of deep features with side information.
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + n_demo + n_wide, FUSION_HIDDEN),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(FUSION_HIDDEN),
            nn.Dropout(HEAD_DROPOUT),
        )
        self.classifier = nn.Linear(FUSION_HIDDEN, num_classes)

        self._init_weights()

    def _make_layer(self, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * SEResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, out_ch * SEResBlock.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch * SEResBlock.expansion),
            )
        layers = [SEResBlock(self.in_ch, out_ch, stride=stride, downsample=downsample)]
        self.in_ch = out_ch * SEResBlock.expansion
        for _ in range(1, blocks):
            layers.append(SEResBlock(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-init the last BN in each residual branch: each block starts as
        # an identity, which stabilizes early training (arXiv:1706.02677).
        for m in self.modules():
            if isinstance(m, SEResBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, wide_feats):
        """
        x:          (B, 12, T)   raw/normalized ECG window
        wide_feats: (B, n_demo + n_wide)  age, sex, hand-crafted features
        returns:    (B, num_classes) logits
        """
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)                      # (B, C, T')

        if self.use_transformer:
            z = z.permute(0, 2, 1)              # (B, T', C)
            z = self.pos_enc(z)
            z = self.transformer(z)
            z = z.permute(0, 2, 1)              # (B, C, T')

        pooled = self.attn_pool(z)              # (B, C)
        fused = torch.cat([pooled, wide_feats], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


def build_model(**kwargs):
    """Factory used by training and inference so both build identical models."""
    return ECGNet(**kwargs)


if __name__ == "__main__":
    # Smoke test: shapes must line up end-to-end.
    from config import WINDOW_SIZE, N_DEMO, N_WIDE_FEATS
    net = build_model()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    x = torch.randn(4, LEADS, WINDOW_SIZE)
    wf = torch.randn(4, N_DEMO + N_WIDE_FEATS)
    y = net(x, wf)
    print(f"output: {tuple(y.shape)}  |  trainable params: {n_params/1e6:.2f}M")
