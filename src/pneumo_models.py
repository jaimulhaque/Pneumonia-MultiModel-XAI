
"""Baselines + proposed (backbone + one light attention at the stride-16 stage + GAP head)."""
import math
import torch
import torch.nn as nn
from torchvision import models as tvm

NUM_CLASSES = 3
DROPOUT = 0.3


class ECA(nn.Module):
    def __init__(self, ch, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(ch) + b) / gamma))
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)               # B,1,C
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)  # B,C,1,1
        return x * y


class CBAM(nn.Module):
    def __init__(self, ch, r=16, k=7):
        super().__init__()
        mid = max(ch // r, 8)
        self.mlp = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, ch, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        a = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        x = x * a
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.spatial(s))


ATTN = {"eca": ECA, "cbam": CBAM}


def _head(in_f):
    return nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(in_f, NUM_CLASSES))


def _w(weights_enum, pretrained):
    return weights_enum.IMAGENET1K_V1 if pretrained else None


def _tv(name, pretrained):
    if name == "efficientnet_b2":
        return tvm.efficientnet_b2(weights=_w(tvm.EfficientNet_B2_Weights, pretrained))
    if name == "efficientnet_v2_s":
        return tvm.efficientnet_v2_s(weights=_w(tvm.EfficientNet_V2_S_Weights, pretrained))
    if name == "resnet50":
        return tvm.resnet50(weights=_w(tvm.ResNet50_Weights, pretrained))
    if name == "convnext_tiny":
        return tvm.convnext_tiny(weights=_w(tvm.ConvNeXt_Tiny_Weights, pretrained))
    if name == "swin_t":
        return tvm.swin_t(weights=_w(tvm.Swin_T_Weights, pretrained))
    raise ValueError(name)


def _set_head(m, name):
    if name == "efficientnet_b2":
        m.classifier = _head(1408)
    elif name == "efficientnet_v2_s":
        m.classifier = _head(1280)
    elif name == "resnet50":
        m.fc = _head(2048)
    elif name == "convnext_tiny":
        m.classifier[2] = _head(768)
    elif name == "swin_t":
        m.head = _head(768)
    return m


BASELINES = ["efficientnet_b2", "resnet50", "convnext_tiny", "swin_t"]
EXTRA_BASELINES = ["efficientnet_v2_s"]          # backbone-matched baseline for the proposed model
BACKBONES = {"convnext_tiny": 5, "efficientnet_v2_s": 5}   # index of the stride-16 stage in .features


def build_baseline(name, pretrained=True):
    return _set_head(_tv(name, pretrained), name)


@torch.no_grad()
def _stage_channels(features, idx):
    was = features.training
    features.eval()
    c = features[: idx + 1](torch.zeros(1, 3, 64, 64)).shape[1]
    features.train(was)
    return c


def build_proposed(backbone="convnext_tiny", attn="eca", pretrained=True):
    """Backbone + attention after the stride-16 stage (24x24 at 384px) + original norm/GAP + simple head."""
    m = _set_head(_tv(backbone, pretrained), backbone)
    idx = BACKBONES[backbone]
    ch = _stage_channels(m.features, idx)
    feats = list(m.features)
    m.features = nn.Sequential(*feats[: idx + 1], ATTN[attn](ch), *feats[idx + 1:])
    return m


def build_model(spec, pretrained=True):
    """spec: 'baseline:resnet50' or 'proposed:convnext_tiny:eca'"""
    parts = spec.split(":")
    if parts[0] == "baseline":
        return build_baseline(parts[1], pretrained)
    return build_proposed(parts[1], parts[2], pretrained)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
