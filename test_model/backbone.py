"""YOLOv8 CSPDarkNet backbone (m: depth=0.67, width=0.75, max_ch=768).

Matches the official YOLOv8 architecture precisely:
  P1: Conv(3, 64, s=2) -> Conv(64, 128, s=2) -> C2f(128, 128, n=3)  [P2, /4]
  P3: Conv(128, 256, s=2) -> C2f(256, 256, n=6)                      [/8]
  P4: Conv(256, 512, s=2) -> C2f(512, 512, n=6)                      [/16]
  P5: Conv(512, 1024, s=2) -> C2f(1024, 1024, n=3) -> SPPF           [/32]

Scaled by width=0.75, depth=0.67 for m variant.
"""

import torch.nn as nn

from test_model.common import C2f, Conv, SPPF, ECA, make_divisible


class CSPDarkNet(nn.Module):
    """YOLOv8 CSPDarkNet backbone.

    Scales:
        n: depth=0.33, width=0.25, max_ch=1024
        s: depth=0.33, width=0.50, max_ch=1024
        m: depth=0.67, width=0.75, max_ch=768
        l: depth=1.00, width=1.00, max_ch=512
        x: depth=1.00, width=1.25, max_ch=512
    """

    def __init__(self, depth=0.67, width=0.75, max_ch=768, use_eca=False):
        super().__init__()

        def ch(x):
            return make_divisible(min(int(x * width), max_ch))

        def n_blocks(x):
            return max(1, int(round(x * depth)))

        self.use_eca = use_eca

        # Stem: /2 -> P1, /2 -> P2
        self.stem = nn.Sequential(
            Conv(3, ch(64), 3, 2),
            Conv(ch(64), ch(128), 3, 2),
            C2f(ch(128), ch(128), n=n_blocks(3), shortcut=True),
        )
        self.stem_ch = ch(128)  # P2/4

        # Stage 3: /8 (80x80)
        self.stage3 = nn.Sequential(
            Conv(ch(128), ch(256), 3, 2),
            C2f(ch(256), ch(256), n=n_blocks(6), shortcut=True),
        )
        self.stage3_ch = ch(256)  # P3/8

        # Stage 4: /16 (40x40)
        self.stage4 = nn.Sequential(
            Conv(ch(256), ch(512), 3, 2),
            C2f(ch(512), ch(512), n=n_blocks(6), shortcut=True),
        )
        self.stage4_ch = ch(512)  # P4/16

        # Stage 5: /32 (20x20)
        self.stage5_down = Conv(ch(512), ch(1024), 3, 2)
        self.stage5_c2f = C2f(ch(1024), ch(1024), n=n_blocks(3), shortcut=True)
        self.stage5_sppf = SPPF(ch(1024), ch(1024))
        self.stage5_ch = ch(1024)  # P5/32

        if use_eca:
            self.eca_p3 = ECA(ch(256))
            self.eca_p4 = ECA(ch(512))
            self.eca_p5 = ECA(ch(1024))

        self.out_channels = [ch(128), ch(256), ch(512), ch(1024)]

    def forward(self, x):
        p2 = self.stem(x)                    # /4, 160x160
        p3 = self.stage3(p2)                 # /8, 80x80
        p4 = self.stage4(p3)                 # /16, 40x40
        p5 = self.stage5_down(p4)
        p5 = self.stage5_c2f(p5)
        p5 = self.stage5_sppf(p5)            # /32, 20x20

        if self.use_eca:
            p3 = self.eca_p3(p3)
            p4 = self.eca_p4(p4)
            p5 = self.eca_p5(p5)

        return p2, p3, p4, p5
