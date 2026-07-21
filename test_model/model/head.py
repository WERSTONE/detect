"""Detection and pose heads for the BiFPN model."""

import math

import torch
import torch.nn as nn

from test_model.model.common import Conv

CLS_PRIOR_PROB = 0.01
CLS_BIAS_INIT = math.log(CLS_PRIOR_PROB / (1 - CLS_PRIOR_PROB))


def _make_tower(in_ch, mid_ch, depth):
    if depth == 0:
        return nn.Identity()
    layers = [Conv(in_ch, mid_ch, 3)]
    for _ in range(depth - 1):
        layers.append(Conv(mid_ch, mid_ch, 3))
    return nn.Sequential(*layers)


class DetectHead(nn.Module):
    """Detection head for COCO classes.

    Output per grid cell:
        cls: [B, num_classes, H, W] classification logits
        reg: [B, 4*reg_max, H, W] DFL distribution
    """

    def __init__(self, in_ch, num_classes=80, reg_max=16, tower_depth=2):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_tower = _make_tower(in_ch, in_ch, tower_depth)
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

        self.reg_tower = _make_tower(in_ch, in_ch, tower_depth)
        self.reg_pred = nn.Conv2d(in_ch, 4 * reg_max, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_pred.bias, CLS_BIAS_INIT)
        # DFL bias: slight tilt toward center bins
        reg_bias = torch.zeros(4 * self.reg_max)
        for e in range(4):
            reg_bias[e * self.reg_max:(e + 1) * self.reg_max] = torch.linspace(1.0, -1.0, self.reg_max)
        self.reg_pred.bias.data.copy_(reg_bias)

    def forward(self, features):
        outs = {'cls': [], 'reg': []}
        for f in features:
            cls_feat = self.cls_tower(f)
            reg_feat = self.reg_tower(f)
            outs['cls'].append(self.cls_pred(cls_feat))
            outs['reg'].append(self.reg_pred(reg_feat))
        return outs


class YOLOLikeDetectHead(nn.Module):
    """YOLOv8-style per-level detection head.

    This follows the head layout used by the local yolov8m.pt checkpoint:
    each detection level owns separate box and class towers.
    """

    def __init__(self, channels, num_classes=80, reg_max=16, strides=(8, 16, 32),
                 img_size=640):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels] * len(strides)
        self.channels = list(channels)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = list(strides)
        self.img_size = int(img_size)

        c2 = max(16, self.channels[0] // 4, self.reg_max * 4)
        c3 = max(self.channels[0], min(self.num_classes, 100))
        self.reg_branches = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for ch in self.channels
        )
        self.cls_branches = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.num_classes, 1),
            )
            for ch in self.channels
        )
        self.bias_init()

    def bias_init(self):
        for reg_branch, cls_branch, stride in zip(
                self.reg_branches, self.cls_branches, self.strides):
            reg_branch[-1].bias.data[:] = 2.0
            cls_branch[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (self.img_size / stride) ** 2)

    def forward(self, features):
        outs = {'cls': [], 'reg': []}
        for feat, reg_branch, cls_branch in zip(
                features, self.reg_branches, self.cls_branches):
            outs['reg'].append(reg_branch(feat))
            outs['cls'].append(cls_branch(feat))
        return outs


class PoseHead(nn.Module):
    """Pose head with keypoint prediction for person detections.

    Output per grid cell:
        cls: [B, 1, H, W] person classification logit
        reg: [B, 4*reg_max, H, W] DFL distribution
        kpt: [B, 51, H, W] 17 keypoints x (dx, dy, vis)
    """

    def __init__(self, in_ch, num_kpts=17, reg_max=16, tower_depth=2):
        super().__init__()
        self.num_classes = 1
        self.num_kpts = num_kpts
        self.reg_max = reg_max
        self.proposal_branches_enabled = True

        self.cls_tower = _make_tower(in_ch, in_ch, tower_depth)
        self.cls_pred = nn.Conv2d(in_ch, 1, 1)

        self.reg_tower = _make_tower(in_ch, in_ch, tower_depth)
        self.reg_pred = nn.Conv2d(in_ch, 4 * reg_max, 1)

        kpt_mid = max(in_ch // 2, 64)
        self.kpt_tower = _make_tower(in_ch, kpt_mid, tower_depth)
        self.kpt_pred = nn.Conv2d(kpt_mid, num_kpts * 3, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in (self.cls_pred, self.reg_pred, self.kpt_pred):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_pred.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred.bias, CLS_BIAS_INIT)
        nn.init.normal_(self.reg_pred.weight, 0, 0.01)
        nn.init.normal_(self.kpt_pred.weight, 0, 0.001)
        nn.init.constant_(self.kpt_pred.bias, 0)
        reg_bias = torch.zeros(4 * self.reg_max)
        for e in range(4):
            reg_bias[e * self.reg_max:(e + 1) * self.reg_max] = torch.linspace(1.0, -1.0, self.reg_max)
        self.reg_pred.bias.data.copy_(reg_bias)

    def enable_proposal_branches(self, enabled=True):
        self.proposal_branches_enabled = bool(enabled)

    def forward(self, features):
        outs = {'cls': [], 'reg': [], 'kpt': []}
        for f in features:
            kpt_feat = self.kpt_tower(f)
            outs['kpt'].append(self.kpt_pred(kpt_feat))
            if self.proposal_branches_enabled:
                cls_feat = self.cls_tower(f)
                reg_feat = self.reg_tower(f)
                outs['cls'].append(self.cls_pred(cls_feat))
                outs['reg'].append(self.reg_pred(reg_feat))
        return outs



