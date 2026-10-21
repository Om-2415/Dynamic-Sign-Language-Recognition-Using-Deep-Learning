"""Models: custom CNN baseline, MobileNetV3 / EfficientNet-B0 transfer learning,
and CNN + LSTM/GRU for dynamic (sequence) recognition.

Every model exposes `.backbone` (feature extractor) and `.head`, so the training loop can
freeze / unfreeze the backbone the same way for all of them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

STATIC_ARCHS = ["custom_cnn", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"]
RNN_TYPES = ["lstm", "gru", "none"]  # "none" = average frame features (no temporal model)


# --------------------------------------------------------------------------- #
# Backbones (image -> feature vector)
# --------------------------------------------------------------------------- #
def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNNBackbone(nn.Module):
    """Small VGG-style CNN trained from scratch (the transparent baseline)."""

    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers.append(_conv_block(cin, w))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = cin

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class TorchvisionBackbone(nn.Module):
    """ImageNet-pretrained torchvision model with its classifier removed."""

    def __init__(self, arch: str, pretrained: bool):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if arch == "mobilenet_v3_small":
            m = tvm.mobilenet_v3_small(weights=weights)
        elif arch == "mobilenet_v3_large":
            m = tvm.mobilenet_v3_large(weights=weights)
        elif arch == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.features, self.pool = m.features, m.avgpool
        # channels of the last conv block (576 / 960 / 1280). Read from the module rather
        # than running a dummy forward, which would update pretrained BatchNorm statistics.
        self.out_dim = [mod for mod in self.features.modules() if isinstance(mod, nn.Conv2d)][-1].out_channels

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


def make_backbone(arch: str, pretrained: bool) -> nn.Module:
    if arch == "custom_cnn":
        return CustomCNNBackbone()
    return TorchvisionBackbone(arch, pretrained)


# --------------------------------------------------------------------------- #
# Full models
# --------------------------------------------------------------------------- #
class _FreezableMixin:
    backbone: nn.Module
    backbone_frozen: bool = False

    def set_backbone_trainable(self, trainable: bool):
        self.backbone_frozen = not trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        if not trainable:
            self.backbone.eval()  # keep BatchNorm statistics fixed while frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self


class StaticClassifier(_FreezableMixin, nn.Module):
    """Single image -> class logits."""

    def __init__(self, arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = make_backbone(arch, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.out_dim, num_classes))

    def forward(self, x):  # x: B,C,H,W
        return self.head(self.backbone(x))


class SequenceClassifier(_FreezableMixin, nn.Module):
    """Frames -> per-frame CNN features -> LSTM/GRU -> class logits.

    x: (B, T, C, H, W). rnn="none" averages frame features instead, which is the ablation
    showing what the temporal model adds (it cannot tell A->B motion from B->A).
    """

    def __init__(self, arch: str, num_classes: int, rnn: str = "lstm", hidden: int = 256,
                 layers: int = 1, bidirectional: bool = False, dropout: float = 0.3,
                 pretrained: bool = True, pool: str = "last"):
        super().__init__()
        if rnn not in RNN_TYPES:
            raise ValueError(f"rnn must be one of {RNN_TYPES}")
        self.backbone = make_backbone(arch, pretrained)
        d = self.backbone.out_dim
        self.rnn_type, self.pool = rnn, pool
        self.frame_dropout = nn.Dropout(dropout)
        if rnn == "none":
            self.rnn, out_dim = None, d
        else:
            cls = nn.LSTM if rnn == "lstm" else nn.GRU
            self.rnn = cls(d, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
            out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def frame_features(self, x):
        b, t = x.shape[:2]
        flat = x.flatten(0, 1)
        if self.backbone_frozen:
            with torch.no_grad():
                f = self.backbone(flat)
        else:
            f = self.backbone(flat)
        return f.view(b, t, -1)

    def forward(self, x):
        return self.classify_features(self.frame_features(x))

    def classify_features(self, f):
        """f: (B, T, D) per-frame features -> logits. Used directly by the live predictor,
        which caches one feature vector per frame instead of re-running the CNN on the
        whole window every frame."""
        f = self.frame_dropout(f)
        if self.rnn is None:
            z = f.mean(1)
        else:
            out, h = self.rnn(f)
            if self.pool == "mean":
                z = out.mean(1)
            else:  # final hidden state; for bidirectional, concat both directions
                h_n = h[0] if isinstance(h, tuple) else h
                z = torch.cat([h_n[-2], h_n[-1]], 1) if self.rnn.bidirectional else h_n[-1]
        return self.head(z)


# --------------------------------------------------------------------------- #
# Construction / checkpoints
# --------------------------------------------------------------------------- #
def build_model(cfg: dict) -> nn.Module:
    """cfg keys: task, arch, num_classes, pretrained, dropout, (+ rnn, hidden, layers,
    bidirectional, pool for task='dynamic')."""
    if cfg["task"] == "static":
        return StaticClassifier(cfg["arch"], cfg["num_classes"], cfg.get("pretrained", True),
                                cfg.get("dropout", 0.3))
    return SequenceClassifier(cfg["arch"], cfg["num_classes"], cfg.get("rnn", "lstm"),
                              cfg.get("hidden", 256), cfg.get("layers", 1),
                              cfg.get("bidirectional", False), cfg.get("dropout", 0.3),
                              cfg.get("pretrained", True), cfg.get("pool", "last"))


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_cfg"], pretrained=False)  # weights come from the checkpoint
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    size_mb += sum(b.numel() * b.element_size() for b in model.buffers()) / 2**20
    return {"params": int(total), "params_millions": round(total / 1e6, 3), "size_mb": round(size_mb, 2)}
