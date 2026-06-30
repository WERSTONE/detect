"""Detection and pose heads (decoupled, YOLOv8-style).

Head types:
    DetectHead: cls(19) + reg(4*reg_max)  — non-person detection
    PoseHead:   cls(1)  + reg(4*reg_max) + kpt(17*3) — person + keypoints
    UnifiedHead: cls(20) + reg(4*reg_max) + kpt(17*3) — all-in-one
"""

import math

import torch
import torch.nn as nn

from test_model.common import Conv

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
    """Detection head for non-person classes (19 classes).

    Output per grid cell:
        cls: [B, 19, H, W]  — classification logits
        reg: [B, 4*reg_max, H, W] — DFL distribution
    """

    def __init__(self, in_ch, num_classes=19, reg_max=16, tower_depth=2):
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


class PoseHead(nn.Module):
    """Pose head for person detection + 17 keypoints.

    Output per grid cell:
        cls: [B, 1, H, W]   — person classification logit
        reg: [B, 4*reg_max, H, W] — DFL distribution
        kpt: [B, 51, H, W]  — 17 keypoints × (dx, dy, vis)
    """

    def __init__(self, in_ch, num_kpts=17, reg_max=16, tower_depth=2):
        super().__init__()
        self.num_classes = 1
        self.num_kpts = num_kpts
        self.reg_max = reg_max

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

    def forward(self, features):
        outs = {'cls': [], 'reg': [], 'kpt': []}
        for f in features:
            cls_feat = self.cls_tower(f)
            reg_feat = self.reg_tower(f)
            kpt_feat = self.kpt_tower(f)
            outs['cls'].append(self.cls_pred(cls_feat))
            outs['reg'].append(self.reg_pred(reg_feat))
            outs['kpt'].append(self.kpt_pred(kpt_feat))
        return outs


class UnifiedHead(nn.Module):
    """Unified head: all 20 classes + person keypoints from one head.

    Output per grid cell:
        cls: [B, 20, H, W]  — 20-class logits (incl. person at index 0)
        reg: [B, 4*reg_max, H, W] — DFL distribution
        kpt: [B, 51, H, W]  — only valid for person class
    """

    def __init__(self, in_ch, num_classes=20, num_kpts=17, reg_max=16, tower_depth=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_kpts = num_kpts
        self.reg_max = reg_max

        self.cls_tower = _make_tower(in_ch, in_ch, tower_depth)
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

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

    def forward(self, features):
        outs = {'cls': [], 'reg': [], 'kpt': []}
        for f in features:
            cls_feat = self.cls_tower(f)
            reg_feat = self.reg_tower(f)
            kpt_feat = self.kpt_tower(f)
            outs['cls'].append(self.cls_pred(cls_feat))
            outs['reg'].append(self.reg_pred(reg_feat))
            outs['kpt'].append(self.kpt_pred(kpt_feat))
        return outs
