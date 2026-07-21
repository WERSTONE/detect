"""BiFPN neck for the COCO80 detection + person pose model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from test_model.model.common import C2f, Conv, make_divisible

class _BiFPNBlock(nn.Module):
    """One repeated BiFPN top-down + bottom-up fusion block."""

    def __init__(self, ch, c2f_depth=2, use_p2=False, downsample='avgpool'):
        super().__init__()
        self.use_p2 = use_p2
        self.downsample = downsample

        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.p4_td_conv = C2f(ch, ch, n=c2f_depth, shortcut=False)

        self.w_p3_out = nn.Parameter(torch.ones(3 if use_p2 else 2))
        self.p3_out_conv = C2f(ch, ch, n=c2f_depth, shortcut=False)

        self.w_p4_bu = nn.Parameter(torch.ones(3))
        self.p3_down = Conv(ch, ch, 3, 2) if downsample == 'conv' else nn.Identity()
        self.p4_out_conv = C2f(ch, ch, n=c2f_depth, shortcut=False)

        self.w_p5_bu = nn.Parameter(torch.ones(2))
        self.p4_down = Conv(ch, ch, 3, 2) if downsample == 'conv' else nn.Identity()
        self.p5_out_conv = C2f(ch, ch, n=c2f_depth, shortcut=False)

    @staticmethod
    def _fuse(weights, *tensors, eps=1e-4):
        w = nn.functional.relu(weights)
        total = w.sum() + eps
        return sum((w[i] / total) * tensors[i] for i in range(len(tensors)))

    def forward(self, p3, p4, p5, p2_ctx=None):
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_td = self._fuse(self.w_p4_td, p4, p5_up)
        p4_td = self.p4_td_conv(p4_td)

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        if self.use_p2 and p2_ctx is not None:
            p3_out = self._fuse(self.w_p3_out, p3, p4_up, p2_ctx)
        else:
            p3_out = self._fuse(self.w_p3_out, p3, p4_up)
        p3_out = self.p3_out_conv(p3_out)

        p3_d = self.p3_down(p3_out) if self.downsample == 'conv' else F.avg_pool2d(p3_out, 2, 2)
        p4_out = self._fuse(self.w_p4_bu, p4_td, p4, p3_d)
        p4_out = self.p4_out_conv(p4_out)

        p4_d = self.p4_down(p4_out) if self.downsample == 'conv' else F.avg_pool2d(p4_out, 2, 2)
        p5_out = self._fuse(self.w_p5_bu, p5, p4_d)
        p5_out = self.p5_out_conv(p5_out)
        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    """Stronger BiFPN-PAN neck for dual-task fusion.

    Keeps the YOLOv8m backbone fixed while widening the unified feature width,
    repeating BiFPN fusion, and injecting pooled P2 context into the highest-
    resolution output for better pose localization.
    """

    def __init__(self, in_channels, depth=0.67, width=0.75,
                 use_p2_context=True, downsample='avgpool',
                 out_channels=None):
        super().__init__()
        use_p2 = len(in_channels) == 4 and use_p2_context
        if use_p2:
            c2, c3, c4, c5 = in_channels
        else:
            c3, c4, c5 = in_channels[-3:]
            c2 = None

        def _ch(x):
            return make_divisible(min(x, 768) * width)

        def _n(x):
            return max(1, int(round(x * depth)))

        # Keep the default neck width close to YOLOv8m capacity without
        # overshooting too far. BiFPN uses weighted addition, so all levels
        # intentionally share one output width.
        self.out_ch = make_divisible(out_channels) if out_channels else make_divisible(c4 * (2.0 / 3.0))
        self.use_p2 = use_p2
        self.downsample = downsample

        if use_p2:
            self.p2_lat = Conv(c2, self.out_ch, 1)
        self.p3_lat = Conv(c3, self.out_ch, 1)
        self.p4_lat = Conv(c4, self.out_ch, 1)
        self.p5_lat = Conv(c5, self.out_ch, 1)

        c2f_depth = _n(3)
        repeat_blocks = _n(3)
        self.blocks = nn.ModuleList(
            _BiFPNBlock(self.out_ch, c2f_depth=c2f_depth, use_p2=use_p2,
                        downsample=downsample)
            for _ in range(repeat_blocks)
        )

        self.out_channels = [self.out_ch] * 3

    def forward(self, feats):
        if self.use_p2:
            p2, p3, p4, p5 = feats
            p2_ctx = F.avg_pool2d(self.p2_lat(p2), 2, 2)
        else:
            p3, p4, p5 = feats[-3:]
            p2_ctx = None

        p3 = self.p3_lat(p3)
        p4 = self.p4_lat(p4)
        p5 = self.p5_lat(p5)

        for block in self.blocks:
            p3, p4, p5 = block(p3, p4, p5, p2_ctx=p2_ctx)

        return [p3, p4, p5]


