"""Multi-task loss functions.

Supports:
- CIoU box loss
- BCE focal classification loss
- DFL distribution loss
- OKS keypoint loss
- Keypoint objectness loss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO 17 keypoint sigmas
KPT_SIGMAS = torch.tensor([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072,
    0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])


def _iou_xyxy(pred, target):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
    target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=1e-7) * (pred[:, 3] - pred[:, 1]).clamp(min=1e-7)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=1e-7) * (target[:, 3] - target[:, 1]).clamp(min=1e-7)
    iou = inter / (area_p + area_t - inter + 1e-16)
    return iou, inter, area_p + area_t - inter


def _ciou_loss(pred_xyxy, target_xyxy, eps=1e-7):
    pred_xyxy = torch.nan_to_num(pred_xyxy.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    target_xyxy = torch.nan_to_num(target_xyxy.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    iou, inter, union = _iou_xyxy(pred_xyxy, target_xyxy)

    px = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    py = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    tx = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
    ty = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    lt_e = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb_e = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    c2 = ((rb_e[:, 0] - lt_e[:, 0]) ** 2 + (rb_e[:, 1] - lt_e[:, 1]) ** 2).clamp(min=eps)

    pw = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=eps)
    ph = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=eps)
    tw = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=eps)
    th = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=eps)

    v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return (1 - iou + rho2 / c2 + alpha * v).mean()


def _cls_loss(pred, target, alpha=0.5, gamma=2.0):
    """Focal BCE classification loss (sum, not mean)."""
    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    target = torch.nan_to_num(target.float(), nan=0.0).clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    pos_mask = target > 0
    neg_weight = alpha * (1 - pt).pow(gamma)
    weight = torch.where(pos_mask, target, neg_weight)
    return (weight * bce).sum()


def _dfl_loss(pred_dist, target, weight=None, reg_max=16):
    """Distribution Focal Loss."""
    pred_dist = torch.nan_to_num(pred_dist.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=reg_max - 1, neginf=0.0)
    target = target.clamp(0, reg_max - 1 - 1e-6)
    tl = target.long()
    tr = (tl + 1).clamp(0, reg_max - 1)

    wl = tr.float() - target
    wr = target - tl.float()

    loss = (F.cross_entropy(pred_dist, tl, reduction='none') * wl +
            F.cross_entropy(pred_dist, tr, reduction='none') * wr)

    if weight is not None:
        loss = loss * weight
    return loss.mean()


class MultiTaskLoss(nn.Module):
    """Unified multi-task loss for all model variants.

    Supports dual-head and unified-head modes.

    Args:
        w_box: Box loss weight
        w_cls: Classification loss weight
        w_dfl: DFL loss weight
        w_pose: Keypoint OKS loss weight
        w_kobj: Keypoint objectness loss weight
        reg_max: DFL max bin
        num_det_classes: Number of detection classes (19 for dual, 20 for unified)
        unified_head: If True, uses unified head mode
    """

    def __init__(self, w_box=7.5, w_cls=0.5, w_dfl=1.5,
                 w_pose=12.0, w_kobj=1.0, reg_max=16,
                 num_det_classes=19, unified_head=False):
        super().__init__()
        self.w_box = w_box
        self.w_cls = w_cls
        self.w_dfl = w_dfl
        self.w_pose = w_pose
        self.w_kobj = w_kobj
        self.reg_max = reg_max
        self.num_det_classes = num_det_classes
        self.unified_head = unified_head
        self.register_buffer('sigmas', KPT_SIGMAS)

    def forward(self, head_outs, assign_targets, strides, feat_sizes,
                head_type='det', norm_pos=None, cls_norm=None,
                box_norm=None, dfl_norm=None, kpt_norm=None):
        """Compute loss for one head.

        Args:
            head_outs: dict with cls/reg[/kpt] per level
            assign_targets: List[dict] from assigner
            strides: List[int]
            feat_sizes: List[(H, W)]
            head_type: 'det', 'pose', or 'unified'
            norm_pos: Optional shared normalizer for cls/box/dfl. This keeps
                dual-head losses comparable with unified-head losses.
        """
        device = head_outs['cls'][0].device
        B = head_outs['cls'][0].shape[0]
        num_cls = self.num_det_classes
        if head_type == 'pose':
            num_cls = 1
        elif self.unified_head:
            num_cls = 20

        proj = torch.arange(self.reg_max, device=device, dtype=torch.float32)

        total_pos = 0
        total_person_pos = 0
        total_cls_items = 0
        loss_cls = torch.tensor(0.0, device=device)
        loss_ciou = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        loss_kpt = torch.tensor(0.0, device=device)
        loss_kobj = torch.tensor(0.0, device=device)

        for lvl, (stride, (H, W)) in enumerate(zip(strides, feat_sizes)):
            targets = assign_targets[lvl]
            N_lvl = H * W

            cls_p_all = head_outs['cls'][lvl].permute(0, 2, 3, 1).reshape(B, N_lvl, -1)
            cls_tgt_all = torch.zeros(B, N_lvl, cls_p_all.shape[-1], device=device)
            total_cls_items += cls_tgt_all.numel()

            if targets is None:
                loss_cls += _cls_loss(cls_p_all.reshape(-1, cls_p_all.shape[-1]),
                                      cls_tgt_all.reshape(-1, cls_p_all.shape[-1]))
                continue

            N_pos = len(targets['gt_boxes'])
            total_pos += N_pos

            grid = targets['grid_xy'].to(device)
            gt_boxes = targets['gt_boxes'].to(device)
            gt_classes = targets['gt_classes'].to(device)
            batch_idx = targets['batch_idx'].to(device)
            gx, gy = grid[:, 0], grid[:, 1]

            # Extract predictions at positive positions
            reg_p = head_outs['reg'][lvl][batch_idx, :, gy, gx]
            kpt_p = head_outs['kpt'][lvl][batch_idx, :, gy, gx] if 'kpt' in head_outs else None

            # DFL decode
            reg_rs = reg_p.view(N_pos, 4, self.reg_max)
            reg_probs = reg_rs.softmax(dim=-1)
            reg_delta = (reg_probs * proj.view(1, 1, self.reg_max)).sum(dim=-1) * stride

            locs_x = (gx.float() + 0.5) * stride
            locs_y = (gy.float() + 0.5) * stride

            l, t = reg_delta[:, 0], reg_delta[:, 1]
            r, b = reg_delta[:, 2], reg_delta[:, 3]

            pred_xyxy = torch.stack([locs_x - l, locs_y - t, locs_x + r, locs_y + b], dim=-1)

            iou, _, _ = _iou_xyxy(pred_xyxy, gt_boxes)

            # Classification targets
            cls_tgt_flat = cls_tgt_all.view(-1, cls_tgt_all.shape[-1])
            flat_idx = batch_idx * N_lvl + gy * W + gx
            if head_type == 'det':
                # Dual-head detection classes are already shifted to 0..18.
                valid_cls = (gt_classes >= 0) & (gt_classes < cls_tgt_all.shape[-1])
                if valid_cls.any():
                    cls_tgt_flat[flat_idx[valid_cls], gt_classes[valid_cls]] = 1.0
            elif head_type == 'pose':
                person_pos = gt_classes == 0
                if person_pos.any():
                    cls_tgt_flat[flat_idx[person_pos], 0] = 1.0
            elif self.unified_head:
                valid_cls = (gt_classes >= 0) & (gt_classes < cls_tgt_all.shape[-1])
                if valid_cls.any():
                    cls_tgt_flat[flat_idx[valid_cls], gt_classes[valid_cls]] = 1.0

            loss_cls += _cls_loss(cls_p_all.reshape(-1, cls_p_all.shape[-1]),
                                  cls_tgt_all.reshape(-1, cls_tgt_all.shape[-1]))

            # CIoU loss
            loss_ciou += _ciou_loss(pred_xyxy, gt_boxes) * N_pos

            # DFL loss
            gt_l = ((locs_x - gt_boxes[:, 0]) / stride).clamp(0, self.reg_max - 1e-6)
            gt_t = ((locs_y - gt_boxes[:, 1]) / stride).clamp(0, self.reg_max - 1e-6)
            gt_r = ((gt_boxes[:, 2] - locs_x) / stride).clamp(0, self.reg_max - 1e-6)
            gt_b = ((gt_boxes[:, 3] - locs_y) / stride).clamp(0, self.reg_max - 1e-6)
            gt_bins = torch.stack([gt_l, gt_t, gt_r, gt_b], dim=1)

            iou_d = iou.detach().clamp(min=0.2)
            loss_dfl += _dfl_loss(
                reg_rs.reshape(-1, self.reg_max),
                gt_bins.reshape(-1),
                weight=iou_d.repeat_interleave(4),
                reg_max=self.reg_max,
            ) * N_pos

            # Keypoint loss (only for person class)
            person_mask = gt_classes == 0
            if person_mask.any() and kpt_p is not None:
                p_idx = person_mask.nonzero(as_tuple=True)[0].to(device)
                n_person = p_idx.numel()
                total_person_pos += n_person
                p_boxes = gt_boxes[p_idx]
                p_locs = torch.stack([locs_x[p_idx], locs_y[p_idx]], dim=1)

                if targets['gt_kpts'] is not None:
                    gt_k_idx = targets['gt_kpts'].to(device)
                    if gt_k_idx.shape[0] != n_person:
                        if gt_k_idx.shape[0] > n_person:
                            gt_k_idx = gt_k_idx[:n_person]
                        else:
                            pad = torch.zeros(
                                n_person - gt_k_idx.shape[0], 17, 3,
                                device=device, dtype=gt_k_idx.dtype)
                            gt_k_idx = torch.cat([gt_k_idx, pad], dim=0)
                    pk = torch.nan_to_num(
                        kpt_p[p_idx].view(-1, 17, 3).float(),
                        nan=0.0, posinf=1e4, neginf=-1e4)
                    pk_xy = pk[..., :2] * stride + p_locs.unsqueeze(1)
                    pk_vis = pk[..., 2].clamp(-20.0, 20.0)
                    gt_k_idx = torch.nan_to_num(gt_k_idx.float(), nan=0.0, posinf=1e4, neginf=-1e4)
                    gk_xy = gt_k_idx[..., :2]
                    gk_vis = (gt_k_idx[..., 2] > 0).float()

                    # OKS-based keypoint loss
                    area = ((p_boxes[:, 2] - p_boxes[:, 0]) *
                            (p_boxes[:, 3] - p_boxes[:, 1])).clamp(min=1).sqrt()
                    sigmas = self.sigmas.view(1, 17).to(device)
                    d2 = (pk_xy - gk_xy).pow(2).sum(dim=-1)
                    k2 = sigmas.pow(2) * area.pow(2).unsqueeze(-1) + 1e-8
                    oks = (d2 / (-2 * k2)).exp()
                    visible = gk_vis
                    per_sample_oks = (oks * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1)
                    has_visible = visible.sum(dim=1) > 0
                    if has_visible.any():
                        loss_kpt += (1 - per_sample_oks[has_visible]).sum()

                    # Keypoint visibility supervision
                    loss_kobj += 0.1 * F.binary_cross_entropy_with_logits(
                        pk_vis.reshape(-1), gk_vis.reshape(-1), reduction='sum')

        # Normalize
        shared_norm = norm_pos if norm_pos is not None and norm_pos > 0 else None
        cls_norm = cls_norm or shared_norm
        box_norm = box_norm or shared_norm
        dfl_norm = dfl_norm or shared_norm
        kpt_norm = kpt_norm if kpt_norm is not None and kpt_norm > 0 else total_person_pos

        if cls_norm is None:
            cls_norm = max(total_pos, 1) if total_pos > 0 else max(total_cls_items, 1)
        else:
            cls_norm = max(cls_norm, 1)
        box_norm = max(box_norm if box_norm is not None else total_pos, 1)
        dfl_norm = max(dfl_norm if dfl_norm is not None else total_pos, 1)
        kpt_norm = max(kpt_norm, 1)

        out = {
            'cls': self.w_cls * (loss_cls / cls_norm),
            'ciou': self.w_box * loss_ciou / box_norm,
            'dfl': self.w_dfl * loss_dfl / dfl_norm,
            'num_pos': total_pos,
        }

        if head_type in ('pose', 'unified'):
            out['kpt'] = self.w_pose * (loss_kpt / kpt_norm)
            out['kobj'] = self.w_kobj * (loss_kobj / kpt_norm)

        return out
