"""Additional Vigil-v3 heads built on top of YOLO-pose features.

The first v3 implementation keeps YOLO-pose as the person/pose teacher and
defines separate heads for pump-room specific tasks. These heads are deliberately
small so they can be trained after the pose backbone is loaded from an s/m pose
checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from Vigil.models.common import Conv


class AnomalyHead(nn.Module):
    """Fire/water detection head for P3/P4/P5 features.

    Each level predicts class logits and box deltas. The decoding/assigner will
    be wired in the training phase; the module is defined now so v3 has a stable
    architecture and state dict layout.
    """

    def __init__(self, in_channels: list[int], hidden: int = 256, num_classes: int = 2, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_towers = nn.ModuleList()
        self.reg_towers = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for channels in in_channels:
            self.cls_towers.append(nn.Sequential(Conv(channels, hidden, 3), Conv(hidden, hidden, 3)))
            self.reg_towers.append(nn.Sequential(Conv(channels, hidden, 3), Conv(hidden, hidden, 3)))
            self.cls_preds.append(nn.Conv2d(hidden, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hidden, 4 * reg_max, 1))

    def forward(self, features: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        cls_outputs, reg_outputs = [], []
        for feature, cls_tower, reg_tower, cls_pred, reg_pred in zip(
            features, self.cls_towers, self.reg_towers, self.cls_preds, self.reg_preds, strict=True
        ):
            cls_outputs.append(cls_pred(cls_tower(feature)))
            reg_outputs.append(reg_pred(reg_tower(feature)))
        return {"cls": cls_outputs, "reg": reg_outputs}


class PersonFeaturePool(nn.Module):
    """Pool per-person descriptors from multi-scale feature maps.

    This placeholder uses adaptive pooling over feature maps. During full v3
    training it can be replaced by box-aware ROI/grid sampling without changing
    downstream attribute/action heads.
    """

    def __init__(self, in_channels: list[int], out_channels: int = 256):
        super().__init__()
        self.proj = nn.ModuleList(Conv(channels, out_channels, 1) for channels in in_channels)
        self.out_channels = out_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for feature, proj in zip(features, self.proj, strict=True):
            pooled.append(proj(feature).mean(dim=(2, 3)))
        return torch.stack(pooled, dim=0).mean(dim=0)


class AttributeHead(nn.Module):
    """Helmet and smoking classifiers for person descriptors."""

    def __init__(self, in_channels: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, person_features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(person_features)
        return {"helmet": logits[:, 0], "smoking": logits[:, 1]}


class ActionStateHead(nn.Module):
    """Single-frame fall/wave state classifiers.

    Final fall/wave events should still be confirmed by temporal postprocess.
    """

    def __init__(self, in_channels: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, person_features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(person_features)
        return {"fall": logits[:, 0], "wave": logits[:, 1]}

