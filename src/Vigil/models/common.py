"""基础模块: Conv, Bottleneck, SPPF, ECA."""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Conv2d + Norm + SiLU."""
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None,
                 groups=1, act=True, norm='bn', gn_groups=8):
        super().__init__()
        padding = (kernel - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        if norm == 'bn':
            self.bn = nn.BatchNorm2d(out_ch)
        elif norm == 'gn':
            self.bn = nn.GroupNorm(min(gn_groups, out_ch), out_ch)
        else:
            self.bn = nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准 Bottleneck: 1x1 降维 → 3x3 → 1x1 升维, 残差连接."""
    def __init__(self, in_ch, out_ch, shortcut=True, e=0.5, norm='bn', gn_groups=8):
        super().__init__()
        h = int(out_ch * e)
        self.cv1 = Conv(in_ch, h, 1, norm=norm, gn_groups=gn_groups)
        self.cv2 = Conv(h, out_ch, 3, norm=norm, gn_groups=gn_groups)
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions (YOLOv8)."""

    def __init__(self, in_ch, out_ch, n=1, shortcut=True, e=0.5,
                 norm='bn', gn_groups=8):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = Conv(in_ch, 2 * self.c, 1, norm=norm, gn_groups=gn_groups)
        self.cv2 = Conv((2 + n) * self.c, out_ch, 1, norm=norm, gn_groups=gn_groups)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0, norm=norm, gn_groups=gn_groups)
            for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast. 4级池化拼接."""
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        h = in_ch // 2
        self.cv1 = Conv(in_ch, h, 1)
        self.cv2 = Conv(h * 4, out_ch, 1)
        self.k = kernel

    def forward(self, x):
        x = self.cv1(x)
        p1 = nn.functional.max_pool2d(x, self.k, 1, self.k // 2)
        p2 = nn.functional.max_pool2d(p1, self.k, 1, self.k // 2)
        p3 = nn.functional.max_pool2d(p2, self.k, 1, self.k // 2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class ECA(nn.Module):
    """Efficient Channel Attention — 1D 卷积跨通道交互，参数 ≈ k 个.

    加入 ECA 后 neck 可以自动学习各通道对不同任务 (cls/reg/kpt/attr) 的重要性。
    """

    def __init__(self, ch, k=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)                     # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)      # [B, 1, C]
        y = self.conv(y)                          # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)    # [B, C, 1, 1]
        return x * self.sigmoid(y)

