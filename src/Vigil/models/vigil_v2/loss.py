"""统一损失: Focal cls + CIoU bbox + DFL + OKS kpt + Focal BCE attributes."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 关键点 OKS sigmas (COCO 17 kpts) ──
KPT_SIGMAS = torch.tensor([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072,
    0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])


# ── IoU 工具 ──

def _iou_xyxy(pred, target):
    """向量化 IoU, pred/target [N, 4] xyxy."""
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_p + area_t - inter + 1e-16
    return inter / union, inter, union


# ── 分类损失 ──

def _cls_loss(pred, target, alpha=0.5, gamma=2.0, class_weights=None):
    """分类损失，使用硬正样本标签和 focal-style 负样本权重。

    正样本 (target > 0): hard target=1.0，直接推动正样本置信度升高。
    负样本 (target == 0): Focal 压制 α·pt^γ，大量易分背景被抑制。

    Args:
        pred: [N, C] logits
        target: [N, C] ∈ {0, 1}, 0=背景, 1=正样本
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)
    pos_mask = target > 0
    # 正样本: hard target=1, 负样本: focal 权重抑制易分背景
    pos_weight = target
    # 负样本: Focal 抑制, 但对 hard negative (假阳性) 保持高权重
    neg_weight = alpha * (1 - pt).pow(gamma)
    weight = torch.where(pos_mask, pos_weight, neg_weight)
    loss = weight * bce
    if class_weights is not None:
        loss = loss * class_weights.view(1, -1).to(pred.device)
    return loss.sum()


# ── 框回归损失 ──

def _ciou_loss(pred_xyxy, target_xyxy, eps=1e-7):
    """CIoU Loss: 同时优化 IoU、中心距、长宽比.

    Args:
        pred_xyxy: [N, 4]
        target_xyxy: [N, 4]
    """
    iou, inter, union = _iou_xyxy(pred_xyxy, target_xyxy)

    # 中心距 / 外接矩形对角线
    px = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    py = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    tx = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
    ty = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    lt_e = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb_e = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    c2 = ((rb_e[:, 0] - lt_e[:, 0]) ** 2 +
          (rb_e[:, 1] - lt_e[:, 1]) ** 2).clamp(min=eps)

    # 长宽比一致性
    pw = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=eps)
    ph = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=eps)
    tw = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=eps)
    th = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=eps)

    v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return (1 - iou + rho2 / c2 + alpha * v).mean()


# ── DFL 分布损失 ──

def _dfl_loss(pred_dist, target, weight=None, reg_max=16):
    """Distribution Focal Loss: 让预测分布集中在 target 周围.

    Args:
        pred_dist: [N, reg_max] logits (before softmax)
        target: [N] float bin index ∈ [0, reg_max)
    """
    target = target.clamp(0, reg_max - 1 - 1e-6)
    tl = target.long()
    tr = (tl + 1).clamp(0, reg_max - 1)

    wl = tr.float() - target
    wr = target - tl.float()

    loss = (F.cross_entropy(pred_dist, tl, reduction="none") * wl +
            F.cross_entropy(pred_dist, tr, reduction="none") * wr)

    if weight is not None:
        loss = loss * weight
    return loss.mean()


# ── 属性损失 ──

def _focal_bce(pred_logits, target, gamma=2.0, pos_weight=1.0):
    """Focal BCE. 返回 sum，由调用方归一化（与 _cls_loss 一致，避免双重归一化）。"""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = torch.where(target > 0.5, pos_weight, 1.0)
    return (alpha_t * (1 - pt) ** gamma * bce).sum()


# ── 統一损失 ──

class VigilLossV2(nn.Module):
    """v2 统一多任务损失.

    所有损失统一按 total_pos 归一化 (per-positive-sample mean)，
    确保 cls/ciou/dfl/kpt/attr 的梯度量级一致。

    Args:
        w_box, w_cls, w_dfl, w_kpt, w_helm, w_smoke: 损失权重
    """

    def __init__(self, w_box=5.0, w_cls=1.0, w_dfl=12.0,
                 w_kpt=10.0, w_helm=10.0, w_smoke=10.0,
                 reg_max=16, kpt_sigmas=None, cls_weights=None):
        super().__init__()
        self.w_box = w_box
        self.w_cls = w_cls
        self.w_dfl = w_dfl
        self.w_kpt = w_kpt
        self.w_helm = w_helm
        self.w_smoke = w_smoke
        self.reg_max = reg_max
        if cls_weights is not None:
            cls_weights = torch.as_tensor(cls_weights, dtype=torch.float32)
        self.register_buffer("cls_weights", cls_weights)
        self.register_buffer("sigmas",
            kpt_sigmas if kpt_sigmas is not None else KPT_SIGMAS)

    def forward(self, head_outs, assign_targets, strides, feat_sizes):
        """
        Args:
            head_outs: dict with cls/reg/kpt/attr List[Tensor] per level
            assign_targets: List[dict or None] from assigner
            strides: List[int]
            feat_sizes: List[(H, W)]
        """
        device = head_outs["cls"][0].device
        B = head_outs["cls"][0].shape[0]

        loss_cls = torch.tensor(0.0, device=device)
        loss_ciou = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        loss_kpt = torch.tensor(0.0, device=device)
        loss_helm = torch.tensor(0.0, device=device)
        loss_smoke = torch.tensor(0.0, device=device)
        total_pos = 0
        total_person_pos = 0
        total_helmet_pos = 0
        total_smoke_pos = 0

        proj = torch.arange(self.reg_max, device=device, dtype=torch.float32)

        for lvl, (stride, (H, W)) in enumerate(zip(strides, feat_sizes)):
            targets = assign_targets[lvl]
            N_lvl = H * W

            # ── 分类损失 (所有格点, sum 后按 total_pos 归一化) ──
            cls_p_all = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, N_lvl, 3)
            cls_tgt_all = torch.zeros(B, N_lvl, 3, device=device)

            if targets is None:
                loss_cls += _cls_loss(
                    cls_p_all.reshape(-1, 3),
                    cls_tgt_all.reshape(-1, 3),
                    class_weights=self.cls_weights)
                continue

            N_pos = len(targets["gt_boxes"])
            total_pos += N_pos

            grid = targets["grid_xy"].to(device)     # [N_pos, 2] (gx, gy)
            gt_boxes = targets["gt_boxes"].to(device)
            gt_classes = targets["gt_classes"].to(device)
            batch_idx = targets["batch_idx"].to(device)
            gx, gy = grid[:, 0], grid[:, 1]

            # ── 提取 head 预测 (仅正样本位置) ──
            reg_p = head_outs["reg"][lvl][batch_idx, :, gy, gx]       # [N_pos, 4*reg_max]
            kpt_p = head_outs["kpt"][lvl][batch_idx, :, gy, gx]       # [N_pos, 51]
            attr_p = head_outs["attr"][lvl][batch_idx, :, gy, gx]     # [N_pos, 2]

            # ── DFL 解碼预测框 ──
            reg_rs = reg_p.view(N_pos, 4, self.reg_max)               # [N_pos, 4, reg_max]
            reg_probs = reg_rs.softmax(dim=-1)                         # [N_pos, 4, reg_max]
            reg_delta = (reg_probs * proj.view(1, 1, self.reg_max)).sum(dim=-1)  # [N_pos, 4]
            reg_delta = reg_delta * stride                              # pixel offsets

            locs_x = (gx.float() + 0.5) * stride
            locs_y = (gy.float() + 0.5) * stride

            l, t = reg_delta[:, 0], reg_delta[:, 1]
            r, b = reg_delta[:, 2], reg_delta[:, 3]

            pred_xyxy = torch.stack([
                locs_x - l, locs_y - t,
                locs_x + r, locs_y + b,
            ], dim=-1)  # [N_pos, 4]

            # ── IoU ──
            iou, _, _ = _iou_xyxy(pred_xyxy, gt_boxes)

            # ── 填充正样本 cls target ──
            flat_idx = gy * W + gx  # [N_pos]
            cls_tgt_all[batch_idx, flat_idx, gt_classes] = 1.0

            # cls loss on ALL positions (正+负), sum-based
            loss_cls += _cls_loss(
                cls_p_all.reshape(-1, 3),
                cls_tgt_all.reshape(-1, 3),
                class_weights=self.cls_weights)

            # ── 框回归损失 (CIoU, 转为 sum 以便按 total_pos 归一化) ──
            loss_ciou += _ciou_loss(pred_xyxy, gt_boxes) * N_pos

            # ── DFL 分布损失 ──
            # GT ltrb → bin indices
            gt_l = ((locs_x - gt_boxes[:, 0]) / stride).clamp(0, self.reg_max - 1e-6)
            gt_t = ((locs_y - gt_boxes[:, 1]) / stride).clamp(0, self.reg_max - 1e-6)
            gt_r = ((gt_boxes[:, 2] - locs_x) / stride).clamp(0, self.reg_max - 1e-6)
            gt_b = ((gt_boxes[:, 3] - locs_y) / stride).clamp(0, self.reg_max - 1e-6)
            gt_bins = torch.stack([gt_l, gt_t, gt_r, gt_b], dim=1)  # [N_pos, 4]

            # IoU 权重 (线性, floor=0.2 保证初始梯度不过度衰减)
            iou_d = iou.detach().clamp(min=0.2)
            loss_dfl += _dfl_loss(
                reg_rs.reshape(-1, self.reg_max),          # [N_pos*4, reg_max]
                gt_bins.reshape(-1),                        # [N_pos*4]
                weight=iou_d.repeat_interleave(4),          # [N_pos*4]
                reg_max=self.reg_max,
            ) * N_pos  # 转为 sum 与 cls 统一归一化

            # ── 人体属性 (仅 person) ──
            person_mask = gt_classes == 0
            if person_mask.any():
                n_person = person_mask.sum().item()
                total_person_pos += n_person
                p_idx = person_mask.nonzero(as_tuple=True)[0].to(device)
                p_boxes = gt_boxes[p_idx]
                p_locs = torch.stack([locs_x[p_idx], locs_y[p_idx]], dim=1)

                # 关键点 (OKS) — per-sample sum, 统一按 total_person_pos 归一化
                if targets["gt_kpts"] is not None:
                    gt_k_idx = targets["gt_kpts"].to(device)
                    pk = kpt_p[p_idx].view(-1, 17, 3)
                    pk_xy = pk[..., :2] * stride + p_locs.unsqueeze(1)
                    gk_xy = gt_k_idx[..., :2]

                    area = ((p_boxes[:, 2] - p_boxes[:, 0]) *
                            (p_boxes[:, 3] - p_boxes[:, 1])).clamp(min=1).sqrt()
                    sigmas = self.sigmas.view(1, 17).to(device)
                    d2 = (pk_xy - gk_xy).pow(2).sum(dim=-1)
                    k2 = sigmas.pow(2) * area.pow(2).unsqueeze(-1) + 1e-8
                    oks = (d2 / (-2 * k2)).exp()
                    visible = (gt_k_idx[..., 2] > 0).float()
                    # per-sample mean OKS → per-sample loss → sum over level
                    per_sample_oks = (oks * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1)
                    loss_kpt += (1 - per_sample_oks).sum()

                    # 关键点可见度监督 (BCE, 量级与 OKS loss 对齐)
                    pk_vis = pk[..., 2]
                    gt_vis = (gt_k_idx[..., 2] > 0).float()
                    loss_kpt += 0.1 * F.binary_cross_entropy_with_logits(
                        pk_vis.reshape(-1), gt_vis.reshape(-1), reduction="sum")

                # 头盔
                if targets["gt_helmet"] is not None:
                    gt_h = targets["gt_helmet"].to(device).float()
                    valid_h = gt_h >= 0
                    if valid_h.any():
                        total_helmet_pos += valid_h.sum().item()
                        loss_helm += _focal_bce(
                            attr_p[p_idx[valid_h], 0],
                            1 - gt_h[valid_h],
                            gamma=2.0,
                            pos_weight=4.0)

                # 吸烟
                if targets["gt_smoking"] is not None:
                    gt_s = targets["gt_smoking"].to(device).float()
                    valid_s = gt_s >= 0
                    if valid_s.any():
                        total_smoke_pos += valid_s.sum().item()
                        loss_smoke += _focal_bce(
                            attr_p[p_idx[valid_s], 1],
                            gt_s[valid_s],
                            gamma=2.0,
                            pos_weight=6.0)

        return {
            "cls":    self.w_cls   * (loss_cls / max(total_pos, 1)),
            "ciou":   self.w_box   * loss_ciou / max(total_pos, 1),
            "dfl":    self.w_dfl   * loss_dfl / max(total_pos, 1),
            "kpt":    self.w_kpt   * (loss_kpt / max(total_person_pos, 1)),
            "helmet": self.w_helm  * (loss_helm / max(total_helmet_pos, 1)),
            "smoke":  self.w_smoke * (loss_smoke / max(total_smoke_pos, 1)),
            "num_pos": total_pos,
        }

