"""解耦检测头 + DFL 框回归 + 关键点 + 属性.

YOLOv8-style 解耦设计:
  - cls 分支:  独立 conv tower → person/fire/water score
  - reg 分支:  独立 conv tower → 4×reg_max DFL 边缘分布
  - kpt 分支:  轻量 tower → 17×3 关键点偏移
  - attr 分支: 轻量 tower → helmet/smoking logits

权重跨尺度共享.
"""

import torch
import torch.nn as nn

from Vigil.models.common import Conv


def _make_tower(in_ch, mid_ch, depth):
    """解耦分支的卷积 tower."""
    if depth == 0:
        return nn.Identity()
    layers = [Conv(in_ch, mid_ch, 3)]
    for _ in range(depth - 1):
        layers.append(Conv(mid_ch, mid_ch, 3))
    return nn.Sequential(*layers)


class VigilHeadV2(nn.Module):
    """解耦多任务检测头.

    每个格点输出:
        cls:  [B, 3, H, W]        — 3 类分类 logits (无背景类，用 score 阈值直接过滤)
        reg:  [B, 4*reg_max, H, W] — DFL 边缘分布
        kpt:  [B, 51, H, W]       — 17关键点 × (dx, dy, vis)
        attr: [B, 2, H, W]        — helmet + smoking logits
    """

    def __init__(self, in_ch, num_classes=3, num_kpts=17, reg_max=16,
                 tower_depth=2, kpt_attr_ch=96):
        super().__init__()
        self.num_classes = num_classes
        self.num_kpts = num_kpts
        self.reg_max = reg_max

        # 分类分支 (完全独立)
        self.cls_tower = _make_tower(in_ch, in_ch, tower_depth)
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

        # 回归分支 (完全独立)
        self.reg_tower = _make_tower(in_ch, in_ch, tower_depth)
        self.reg_pred = nn.Conv2d(in_ch, 4 * reg_max, 1)

        # 关键点分支 (轻量)
        self.kpt_tower = _make_tower(in_ch, kpt_attr_ch, tower_depth)
        self.kpt_pred = nn.Conv2d(kpt_attr_ch, num_kpts * 3, 1)

        # 属性分支 (轻量)
        self.attr_tower = _make_tower(in_ch, kpt_attr_ch, tower_depth)
        self.attr_pred = nn.Conv2d(kpt_attr_ch, 2, 1)

        # 跨任务门控: cls↔reg 互相告知"哪里可能有物体"
        self.gate_cls_from_reg = nn.Conv2d(in_ch, in_ch, 1)
        self.gate_reg_from_cls = nn.Conv2d(in_ch, in_ch, 1)
        # kpt/attr 塔用 cls 塔的特征辅助
        self.gate_kpt_from_cls = nn.Conv2d(in_ch, kpt_attr_ch, 1)
        self.gate_attr_from_cls = nn.Conv2d(in_ch, kpt_attr_ch, 1)

        self._init_weights()

        # 门控初始化为近似直通 (sigmoid(2)≈0.88 → 初始大部分保留)
        for m in [self.gate_cls_from_reg, self.gate_reg_from_cls,
                  self.gate_kpt_from_cls, self.gate_attr_from_cls]:
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, 2.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # cls bias: 初始 score ≈0.27，避免冷启动时正样本梯度太弱
        nn.init.constant_(self.cls_pred.bias, -1.0)
        # reg bias: 让 DFL 分布向低 bin 略微倾斜 → 初始框适中 → IoU 合理
        # 4 条边各 16 bins，每条边内 bin 0 略高 → 期望偏移 ≈ 4.5 (而非默认的 7.5)
        reg_bias = torch.zeros(4 * self.reg_max)
        for e in range(4):
            reg_bias[e * self.reg_max:(e + 1) * self.reg_max] = \
                torch.linspace(1.0, -1.0, self.reg_max)
        self.reg_pred.bias.data.copy_(reg_bias)

    def forward(self, features):
        """features: List[[B, C, H, W]] (3 个尺度) → dict of List[Tensor]."""
        outs = {"cls": [], "reg": [], "kpt": [], "attr": []}
        for f in features:
            cls_raw = self.cls_tower(f)
            reg_raw = self.reg_tower(f)
            kpt_raw = self.kpt_tower(f)
            attr_raw = self.attr_tower(f)

            # 跨任务门控: 各 tower 用其他 tower 的原始特征做空间注意力
            cls_feat = cls_raw * self.gate_cls_from_reg(reg_raw).sigmoid()
            reg_feat = reg_raw * self.gate_reg_from_cls(cls_raw).sigmoid()
            kpt_feat = kpt_raw * self.gate_kpt_from_cls(cls_raw).sigmoid()
            attr_feat = attr_raw * self.gate_attr_from_cls(cls_raw).sigmoid()

            outs["cls"].append(self.cls_pred(cls_feat))
            outs["reg"].append(self.reg_pred(reg_feat))
            outs["kpt"].append(self.kpt_pred(kpt_feat))
            outs["attr"].append(self.attr_pred(attr_feat))
        return outs


# ── 解码 ──

def _make_grid(nx, ny, device):
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device), indexing="ij")
    return torch.stack((xv, yv), 2).float()


def _dfl_decode(reg_pred, reg_max, stride, grid):
    """DFL 解碼: softmax 分布 → 期望偏移 → xyxy 框.

    Args:
        reg_pred: [B, 4*reg_max, H, W]
        reg_max: int
        stride: int
        grid: [N, 2] 格点坐标 (像素)

    Returns:
        boxes: [B, N, 4] xyxy
    """
    B, _, H, W = reg_pred.shape
    N = H * W

    reg = reg_pred.view(B, 4, reg_max, N)        # [B, 4, reg_max, N]
    reg = reg.softmax(dim=-2)                      # softmax over bins
    proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
    reg = (reg * proj.view(1, 1, reg_max, 1)).sum(dim=-2)  # [B, 4, N] bin indices
    reg = reg * stride                              # [B, 4, N] pixel offsets

    # grid 是左上角坐标, +0.5*stride 转为格点中心 (与 loss 解码一致)
    g = grid.view(1, N, 2) + 0.5 * stride            # [1, N, 2] 格点中心
    cx = g[..., 0:1].transpose(1, 2)                 # [1, N] → [1, 1, N]
    cy = g[..., 1:2].transpose(1, 2)

    l, t = reg[:, 0:1], reg[:, 1:2]                  # [B, 1, N]
    r, b = reg[:, 2:3], reg[:, 3:4]

    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    return torch.cat([x1, y1, x2, y2], dim=1).transpose(1, 2)  # [B, N, 4]


def decode_outputs_v2(head_outs, strides, reg_max, score_thresh=0.05):
    """多级 head 输出 → 检测候选 (DFL 版本).

    Returns:
        boxes:   [B, N, 4] xyxy
        scores:  [B, N, 3] person/fire/water
        kpts:    [B, N, 17, 3] xyv
        helmet:  [B, N] logits
        smoking: [B, N] logits
    """
    device = head_outs["cls"][0].device
    B = head_outs["cls"][0].shape[0]

    all_boxes, all_scores = [], []
    all_kpts, all_helmet, all_smoke = [], [], []

    for lvl, stride in enumerate(strides):
        _, _, H, W = head_outs["cls"][lvl].shape
        N = H * W

        # ── 分类 ──
        cls_pred = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, N, -1)
        scores = cls_pred.sigmoid()                           # [B, N, 3]

        # ── 框回归 (DFL) ──
        grid = _make_grid(W, H, device) * stride             # [N, 2] pixel coords
        boxes = _dfl_decode(head_outs["reg"][lvl], reg_max, stride, grid)  # [B, N, 4]

        # ── 关键点 ──
        kpt_pred = head_outs["kpt"][lvl]
        kpt_pred = kpt_pred.permute(0, 2, 3, 1).reshape(B, N, 17, 3)
        # xy 偏移 → 绝对坐标 (使用格点中心, 与 loss 一致)
        grid_center = grid.view(1, N, 1, 2) + 0.5 * stride
        kpt_xy = kpt_pred[..., :2] * stride + grid_center
        kpt_vis = kpt_pred[..., 2:3].sigmoid()
        kpts = torch.cat([kpt_xy, kpt_vis], dim=-1)           # [B, N, 17, 3]

        # ── 属性 ──
        attr_pred = head_outs["attr"][lvl].permute(0, 2, 3, 1).reshape(B, N, 2)
        helmet_logit = attr_pred[..., 0]                       # [B, N]
        smoke_logit  = attr_pred[..., 1]                       # [B, N]

        # ── 阈值过滤 ──
        mask = (scores > score_thresh).any(dim=-1)             # [B, N]
        for b in range(B):
            m = mask[b]
            if m.any():
                all_boxes.append(boxes[b][m].unsqueeze(0))
                all_scores.append(scores[b][m].unsqueeze(0))
                all_kpts.append(kpts[b][m].unsqueeze(0))
                all_helmet.append(helmet_logit[b][m].unsqueeze(0))
                all_smoke.append(smoke_logit[b][m].unsqueeze(0))

    if all_boxes:
        boxes_out   = torch.cat(all_boxes, dim=1)
        scores_out  = torch.cat(all_scores, dim=1)
        kpts_out    = torch.cat(all_kpts, dim=1)
        helmet_out  = torch.cat(all_helmet, dim=1)
        smoke_out   = torch.cat(all_smoke, dim=1)
    else:
        boxes_out   = torch.zeros(B, 0, 4, device=device)
        scores_out  = torch.zeros(B, 0, 3, device=device)
        kpts_out    = torch.zeros(B, 0, 17, 3, device=device)
        helmet_out  = torch.zeros(B, 0, device=device)
        smoke_out   = torch.zeros(B, 0, device=device)

    return boxes_out, scores_out, kpts_out, helmet_out, smoke_out

