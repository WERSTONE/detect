"""CSPDarkNet backbone — 输出 P3/P4/P5 (stride 8/16/32)."""

import torch.nn as nn

from Vigil.models.common import SPPF, C2f, Conv


class CSPDarkNetV2(nn.Module):
    """CSPDarkNet with C2f blocks, GN for deploy-friendly norm.

    Args:
        w: 宽度系数 (0.5 ≈ 2.5M backbone params, 1.0 ≈ 9M)
        max_ch: 通道上限
    """

    def __init__(self, w=0.75, max_ch=512):
        super().__init__()
        def ch(x):
            return min(int(x * w), max_ch)

        # Stem: /2 → /4
        self.stem = nn.Sequential(
            Conv(3, ch(32), 3, stride=2),
            Conv(ch(32), ch(64), 3, stride=2),
            C2f(ch(64), ch(64), n=3),
        )

        # Stage 3: /8 (80×80)
        self.stage3 = nn.Sequential(
            Conv(ch(64), ch(128), 3, stride=2),
            C2f(ch(128), ch(128), n=2),
        )

        # Stage 4: /16 (40×40)
        self.stage4 = nn.Sequential(
            Conv(ch(128), ch(256), 3, stride=2),
            C2f(ch(256), ch(256), n=2),
        )

        # Stage 5: /32 (20×20)
        self.stage5 = nn.Sequential(
            Conv(ch(256), ch(512), 3, stride=2),
            C2f(ch(512), ch(512), n=1),
            SPPF(ch(512), ch(512)),
        )

        self.out_channels = [ch(64), ch(128), ch(256), ch(512)]

    def forward(self, x):
        p2 = self.stem(x)       # /4, 160×160
        p3 = self.stage3(p2)    # /8, 80×80
        p4 = self.stage4(p3)    # /16, 40×40
        p5 = self.stage5(p4)    # /32, 20×20
        return p2, p3, p4, p5

