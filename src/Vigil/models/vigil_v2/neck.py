"""简化版 Gather-Distribute Neck — 替代 FPN+PAN 的递归传递.

核心思想 (来自 Gold-YOLO): 将各层特征汇聚为全局信息，再分发回各层，
打破 FPN+PAN 必须逐层递归传递的限制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Vigil.models.common import ECA, C2f, Conv


class GatherDistributeNeck(nn.Module):
    """3 级 GD Neck: 输入 P3/P4/P5 (stride 8/16/32)，输出 3 级融合特征.

    使用 GroupNorm (gn_groups=8) 替代 BatchNorm，避免 B=1 时 BN 统计不稳定。

    Args:
        in_channels: [c3, c4, c5] backbone 输出通道
        out_ch: 统一输出通道
        gn_groups: GroupNorm 分组数
    """

    def __init__(self, in_channels, out_ch=128, gn_groups=8):
        super().__init__()
        c3, c4, c5 = in_channels
        n = 'gn'  # neck 统一使用 GroupNorm

        # Lateral convs — 对齐各层通道
        self.lat_p5 = Conv(c5, out_ch, 1, norm=n, gn_groups=gn_groups)
        self.lat_p4 = Conv(c4, out_ch, 1, norm=n, gn_groups=gn_groups)
        self.lat_p3 = Conv(c3, out_ch, 1, norm=n, gn_groups=gn_groups)

        # Gather: 将各级汇聚到统一尺度后 concat → 全局信息表示
        self.gather = Conv(out_ch * 3, out_ch, 1, norm=n, gn_groups=gn_groups)

        # Inject: 全局信息注入各级
        self.inject_p3 = C2f(out_ch * 2, out_ch, n=1, norm=n, gn_groups=gn_groups)
        self.inject_p4 = C2f(out_ch * 2, out_ch, n=1, norm=n, gn_groups=gn_groups)
        self.inject_p5 = C2f(out_ch * 2, out_ch, n=1, norm=n, gn_groups=gn_groups)

        # PAN bottom-up (保持双向通信)
        self.down_p3 = Conv(out_ch, out_ch, 3, stride=2, norm=n, gn_groups=gn_groups)
        self.down_p4 = Conv(out_ch, out_ch, 3, stride=2, norm=n, gn_groups=gn_groups)
        self.fuse_p4 = C2f(out_ch * 2, out_ch, n=1, norm=n, gn_groups=gn_groups)
        self.fuse_p5 = C2f(out_ch * 2, out_ch, n=1, norm=n, gn_groups=gn_groups)

        self.eca_p3 = ECA(out_ch)
        self.eca_p4 = ECA(out_ch)
        self.eca_p5 = ECA(out_ch)

        self.out_channels = [out_ch] * 3

    def forward(self, feats):
        p3, p4, p5 = feats

        # ── Lateral ──
        n3 = self.lat_p3(p3)   # [B, C, 80, 80]
        n4 = self.lat_p4(p4)   # [B, C, 40, 40]
        n5 = self.lat_p5(p5)   # [B, C, 20, 20]

        # ── Gather: 汇聚到统一尺度 (取中间尺度 40×40 减少计算) ──
        target_size = n4.shape[2:]
        g3 = F.interpolate(n3, size=target_size, mode="bilinear", align_corners=False)
        g4 = n4
        g5 = F.interpolate(n5, size=target_size, mode="bilinear", align_corners=False)
        global_info = self.gather(torch.cat([g3, g4, g5], dim=1))

        # ── Inject: 分发回各级 ──
        gi3 = F.interpolate(global_info, size=n3.shape[2:], mode="bilinear", align_corners=False)
        gi5 = F.interpolate(global_info, size=n5.shape[2:], mode="bilinear", align_corners=False)

        m3 = self.inject_p3(torch.cat([n3, gi3], dim=1))
        m4 = self.inject_p4(torch.cat([n4, global_info], dim=1))
        m5 = self.inject_p5(torch.cat([n5, gi5], dim=1))

        # ── PAN bottom-up ──
        p3_out = m3
        p4_out = self.fuse_p4(torch.cat([m4, self.down_p3(p3_out)], dim=1))
        p5_out = self.fuse_p5(torch.cat([m5, self.down_p4(p4_out)], dim=1))

        return [self.eca_p3(p3_out), self.eca_p4(p4_out), self.eca_p5(p5_out)]

