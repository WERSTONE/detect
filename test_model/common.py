"""Building blocks: Conv, C2f, SPPF, ECA — exact YOLOv8 architecture."""

import torch
import torch.nn as nn


def make_divisible(x, divisor=8):
    return int((x + divisor - 1) // divisor * divisor)


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU."""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = (k - 1) // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck: 1x1 reduce -> 3x3 -> 1x1 expand, residual."""
    def __init__(self, in_ch, out_ch, shortcut=True, e=0.5):
        super().__init__()
        h = int(out_ch * e)
        self.cv1 = Conv(in_ch, h, 1)
        self.cv2 = Conv(h, out_ch, 3)
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions (YOLOv8)."""
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = Conv(in_ch, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, out_ch, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        h = in_ch // 2
        self.cv1 = Conv(in_ch, h, 1)
        self.cv2 = Conv(h * 4, out_ch, 1)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        p1 = nn.functional.max_pool2d(x, self.k, 1, self.k // 2)
        p2 = nn.functional.max_pool2d(p1, self.k, 1, self.k // 2)
        p3 = nn.functional.max_pool2d(p2, self.k, 1, self.k // 2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class ECA(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, ch, k=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
