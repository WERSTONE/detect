"""YOLOv8 FPN+PAN neck and BiFPN neck.

All necks output 3 feature levels (P3/8, P4/16, P5/32) at a unified channel count
so that shared-weight heads can process them.

Architecture:
  FPNPANNeck: Standard YOLOv8 top-down + bottom-up with output projections
  BiFPN: EfficientDet-style weighted fusion
  DetNeck: Lightweight detection-only neck
  PoseNeck: Lightweight pose neck with P2 injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from test_model.common import C2f, Conv, make_divisible


# ── Standard FPN+PAN neck ──

class FPNPANNeck(nn.Module):
    """Standard YOLOv8 FPN+PAN neck.

    Args:
        in_channels: [c3, c4, c5] from backbone P3, P4, P5
        depth, width: same as backbone for channel computation
    """

    def __init__(self, in_channels, depth=0.67, width=0.75):
        super().__init__()
        c3, c4, c5 = in_channels  # actual backbone channels (already scaled)

        def _ch(x):
            return make_divisible(min(int(x * width), 768))

        def _n(x):
            return max(1, int(round(x * depth)))

        # Top-down
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4 = C2f(c5 + c4, _ch(512), n=_n(3), shortcut=False)
        self.c2f_p3 = C2f(_ch(512) + c3, _ch(256), n=_n(3), shortcut=False)

        # Bottom-up
        self.down_p3 = Conv(_ch(256), _ch(256), 3, 2)
        self.c2f_n4 = C2f(_ch(256) + _ch(512), _ch(512), n=_n(3), shortcut=False)
        self.down_n4 = Conv(_ch(512), _ch(512), 3, 2)
        self.c2f_n5 = C2f(_ch(512) + c5, _ch(1024), n=_n(3), shortcut=False)

        # Project to unified channel for shared heads
        self.out_ch = _ch(256)
        self.out_proj_p3 = nn.Identity()  # _ch(256) == self.out_ch
        self.out_proj_p4 = Conv(_ch(512), self.out_ch, 1)
        self.out_proj_p5 = Conv(_ch(1024), self.out_ch, 1)

        self.out_channels = [self.out_ch] * 3

    def forward(self, feats):
        p3, p4, p5 = feats

        p5_up = self.upsample(p5)
        p4_td = self.c2f_p4(torch.cat([p5_up, p4], dim=1))
        p4_up = self.upsample(p4_td)
        p3_out = self.c2f_p3(torch.cat([p4_up, p3], dim=1))

        p3_d = self.down_p3(p3_out)
        n4_out = self.c2f_n4(torch.cat([p3_d, p4_td], dim=1))
        n4_d = self.down_n4(n4_out)
        n5_out = self.c2f_n5(torch.cat([n4_d, p5], dim=1))

        return [
            self.out_proj_p3(p3_out),
            self.out_proj_p4(n4_out),
            self.out_proj_p5(n5_out),
        ]


# ── BiFPN neck ──

class BiFPN(nn.Module):
    """BiFPN neck with learnable per-layer fusion weights.

    Projects all backbone features to a unified channel then applies
    top-down + bottom-up weighted feature fusion.
    """

    def __init__(self, in_channels, depth=0.67, width=0.75):
        super().__init__()
        c3, c4, c5 = in_channels

        def _ch(x):
            return make_divisible(min(int(x * width), 768))

        def _n(x):
            return max(1, int(round(x * depth)))

        self.out_ch = _ch(256)

        # Lateral projections to unified channel
        self.p3_lat = Conv(c3, self.out_ch, 1)
        self.p4_lat = Conv(c4, self.out_ch, 1)
        self.p5_lat = Conv(c5, self.out_ch, 1)

        # Fusion weights and conv blocks
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.p4_td_conv = C2f(self.out_ch, self.out_ch, n=_n(1), shortcut=False)
        self.w_p3_out = nn.Parameter(torch.ones(2))
        self.p3_out_conv = C2f(self.out_ch, self.out_ch, n=_n(1), shortcut=False)

        self.w_p4_bu = nn.Parameter(torch.ones(3))
        self.p4_out_conv = C2f(self.out_ch, self.out_ch, n=_n(1), shortcut=False)
        self.w_p5_bu = nn.Parameter(torch.ones(2))
        self.p5_out_conv = C2f(self.out_ch, self.out_ch, n=_n(1), shortcut=False)

        self.out_channels = [self.out_ch] * 3

    @staticmethod
    def _fuse(weights, *tensors, eps=1e-4):
        w = nn.functional.relu(weights)
        total = w.sum() + eps
        return sum((w[i] / total) * tensors[i] for i in range(len(tensors)))

    def forward(self, feats):
        p3, p4, p5 = feats
        p3 = self.p3_lat(p3)
        p4 = self.p4_lat(p4)
        p5 = self.p5_lat(p5)

        # Top-down
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_td = self._fuse(self.w_p4_td, p4, p5_up)
        p4_td = self.p4_td_conv(p4_td)

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_out = self._fuse(self.w_p3_out, p3, p4_up)
        p3_out = self.p3_out_conv(p3_out)

        # Bottom-up
        p3_d = F.avg_pool2d(p3_out, 2, 2)
        p4_out = self._fuse(self.w_p4_bu, p4_td, p4, p3_d)
        p4_out = self.p4_out_conv(p4_out)

        p4_d = F.avg_pool2d(p4_out, 2, 2)
        p5_out = self._fuse(self.w_p5_bu, p5, p4_d)
        p5_out = self.p5_out_conv(p5_out)

        return [p3_out, p4_out, p5_out]


# ── Lightweight necks for dual-neck model ──

class DetNeck(nn.Module):
    """Lightweight detection-only neck (~60% of full FPN+PAN)."""

    def __init__(self, in_channels, scale=0.6):
        super().__init__()
        c3, c4, c5 = in_channels

        def _ch(x):
            return make_divisible(int(x * scale))

        # Internal feature channels (scaled down from backbone)
        ic3, ic4, ic5 = _ch(c3), _ch(c4), _ch(c5)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Top-down: use actual backbone channels for input, scaled for output
        self.c2f_p4 = C2f(c5 + c4, ic4, n=1, shortcut=False)
        self.c2f_p3 = C2f(ic4 + c3, ic3, n=1, shortcut=False)

        # Bottom-up
        self.down_p3 = Conv(ic3, ic4, 3, 2)
        self.c2f_n4 = C2f(ic4 + ic4, ic4, n=1, shortcut=False)
        self.down_n4 = Conv(ic4, ic5, 3, 2)
        self.c2f_n5 = C2f(ic5 + c5, ic5, n=1, shortcut=False)

        # Output projection to unified channel = ic3
        self.out_ch = ic3
        self.out_proj_p4 = Conv(ic4, self.out_ch, 1)
        self.out_proj_p5 = Conv(ic5, self.out_ch, 1)

        self.out_channels = [self.out_ch] * 3

    def forward(self, feats):
        p3, p4, p5 = feats

        p5_up = self.upsample(p5)
        p4_td = self.c2f_p4(torch.cat([p5_up, p4], dim=1))
        p4_up = self.upsample(p4_td)
        p3_out = self.c2f_p3(torch.cat([p4_up, p3], dim=1))

        p3_d = self.down_p3(p3_out)
        n4_out = self.c2f_n4(torch.cat([p3_d, p4_td], dim=1))
        n4_d = self.down_n4(n4_out)
        n5_out = self.c2f_n5(torch.cat([n4_d, p5], dim=1))

        return [p3_out, self.out_proj_p4(n4_out), self.out_proj_p5(n5_out)]


class PoseNeck(nn.Module):
    """Lightweight pose-only neck (~40% of full FPN+PAN).
    Uses P2 backbone feature for fine-grained keypoint localization.
    """

    def __init__(self, in_channels, scale=0.4):
        super().__init__()
        c2, c3, c4, c5 = in_channels  # P2..P5 from backbone

        def _ch(x):
            return make_divisible(int(x * scale))

        # Internal feature channels
        ic3, ic4, ic5 = _ch(c3), _ch(c4), _ch(c5)

        self.p2_lat = Conv(c2, ic3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.c2f_p4 = C2f(c5 + c4, ic4, n=1, shortcut=False)
        self.c2f_p3 = C2f(ic4 + c3 + ic3, ic3, n=1, shortcut=False)  # +P2 (downsampled)

        self.down_p3 = Conv(ic3, ic4, 3, 2)
        self.c2f_n4 = C2f(ic4 + ic4, ic4, n=1, shortcut=False)
        self.down_n4 = Conv(ic4, ic5, 3, 2)
        self.c2f_n5 = C2f(ic5 + c5, ic5, n=1, shortcut=False)

        self.out_ch = ic3
        self.out_proj_p4 = Conv(ic4, self.out_ch, 1)
        self.out_proj_p5 = Conv(ic5, self.out_ch, 1)

        self.out_channels = [self.out_ch] * 3

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        p2_d = F.avg_pool2d(self.p2_lat(p2), 2, 2)  # P2 → P3 spatial

        p5_up = self.upsample(p5)
        p4_td = self.c2f_p4(torch.cat([p5_up, p4], dim=1))
        p4_up = self.upsample(p4_td)
        p3_out = self.c2f_p3(torch.cat([p4_up, p3, p2_d], dim=1))

        p3_d = self.down_p3(p3_out)
        n4_out = self.c2f_n4(torch.cat([p3_d, p4_td], dim=1))
        n4_d = self.down_n4(n4_out)
        n5_out = self.c2f_n5(torch.cat([n4_d, p5], dim=1))

        return [p3_out, self.out_proj_p4(n4_out), self.out_proj_p5(n5_out)]
