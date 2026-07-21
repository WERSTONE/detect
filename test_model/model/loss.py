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

try:
    from ultralytics.utils.metrics import bbox_iou as _ultralytics_bbox_iou
except Exception:  # pragma: no cover - fallback keeps local smoke tests usable.
    _ultralytics_bbox_iou = None

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


def _bbox_ciou(pred_xyxy, target_xyxy, eps=1e-7):
    if _ultralytics_bbox_iou is not None:
        return _ultralytics_bbox_iou(
            target_xyxy.float(),
            pred_xyxy.float(),
            xywh=False,
            CIoU=True,
            eps=eps,
        ).squeeze(-1).clamp(min=-1.0, max=1.0)

    pred_xyxy = torch.nan_to_num(pred_xyxy.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    target_xyxy = torch.nan_to_num(target_xyxy.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    iou, _, _ = _iou_xyxy(pred_xyxy, target_xyxy)

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
    return (iou - rho2 / c2 - alpha * v).clamp(min=-1.0, max=1.0)


def _ciou_loss(pred_xyxy, target_xyxy, eps=1e-7):
    return (1 - _bbox_ciou(pred_xyxy, target_xyxy, eps=eps)).mean()


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


def _dist2bbox(distance, anchor_points):
    lt, rb = distance[..., :2], distance[..., 2:]
    return torch.cat((anchor_points - lt, anchor_points + rb), dim=-1)


def _bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = bbox[..., :2], bbox[..., 2:]
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), dim=-1).clamp_(0, reg_max - 0.01)


class YOLOTaskAlignedAssigner(nn.Module):
    """YOLOv8 TaskAlignedAssigner for detection-only training."""

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 stride=(8, 16, 32), eps=1e-9, topk2=None):
        super().__init__()
        self.topk = int(topk)
        self.topk2 = int(topk2) if topk2 is not None else self.topk
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stride = list(stride)
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = float(eps)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        bs, n_anchors, _ = pd_scores.shape
        n_max_boxes = gt_bboxes.shape[1]
        if n_max_boxes == 0:
            return (
                torch.full((bs, n_anchors), self.num_classes, device=pd_scores.device, dtype=torch.long),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros((bs, n_anchors), device=pd_scores.device, dtype=torch.bool),
                torch.zeros((bs, n_anchors), device=pd_scores.device, dtype=torch.long),
            )

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(
            mask_pos, overlaps, n_max_boxes, align_metric)
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                      anchor_points, mask_gt):
        mask_in_gts = self._select_candidates_in_gts(anchor_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self._get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self._select_topk_candidates(
            align_metric,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        bs, n_max_boxes = gt_labels.shape[:2]
        n_anchors = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            (bs, n_max_boxes, n_anchors), dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros(
            (bs, n_max_boxes, n_anchors), dtype=pd_scores.dtype, device=pd_scores.device)

        ind0 = torch.arange(bs, device=pd_scores.device).view(-1, 1).expand(-1, n_max_boxes)
        ind1 = gt_labels.squeeze(-1).long().clamp(0, self.num_classes - 1)
        bbox_scores[mask_gt] = pd_scores[ind0, :, ind1][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask_gt]
        if pd_boxes.numel() > 0:
            overlaps[mask_gt] = _bbox_ciou(pd_boxes, gt_boxes).clamp_(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def _select_topk_candidates(self, metrics, topk_mask=None):
        topk = min(self.topk, metrics.shape[-1])
        topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        else:
            topk_mask = topk_mask[..., :topk]
        topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=metrics.device)
        for k in range(topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        bs, n_max_boxes = gt_labels.shape[:2]
        batch_ind = torch.arange(bs, dtype=torch.int64, device=gt_labels.device)[:, None]
        target_gt_idx = target_gt_idx + batch_ind * n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx].clamp_(0)
        target_bboxes = gt_bboxes.reshape(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    def _select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
        gt_xywh = torch.cat(
            ((gt_bboxes[..., :2] + gt_bboxes[..., 2:]) * 0.5,
             (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp(min=0)),
            dim=-1,
        )
        wh_mask = gt_xywh[..., 2:] < self.stride[0]
        gt_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_xywh.dtype, device=gt_xywh.device),
            gt_xywh[..., 2:],
        )
        gt_bboxes = torch.cat(
            (gt_xywh[..., :2] - gt_xywh[..., 2:] * 0.5,
             gt_xywh[..., :2] + gt_xywh[..., 2:] * 0.5),
            dim=-1,
        )
        bs, n_boxes = gt_bboxes.shape[:2]
        n_anchors = xy_centers.shape[0]
        lt, rb = gt_bboxes.reshape(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2)
        return deltas.reshape(bs, n_boxes, n_anchors, -1).amin(3).gt_(self.eps)

    def _select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi, is_max, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            max_metric_idx = torch.topk(
                align_metric, min(self.topk2, align_metric.shape[-1]),
                dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_metric_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


class YOLODetectionLoss(nn.Module):
    """YOLOv8-style bbox/class/DFL loss for pure detection."""

    def __init__(self, num_classes=80, reg_max=16, strides=(8, 16, 32),
                 w_box=7.5, w_cls=0.5, w_dfl=1.5,
                 assigner_topk=10, assigner_alpha=0.5, assigner_beta=6.0,
                 assigner_eps=1e-9):
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.strides = list(strides)
        self.w_box = float(w_box)
        self.w_cls = float(w_cls)
        self.w_dfl = float(w_dfl)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = YOLOTaskAlignedAssigner(
            topk=assigner_topk,
            num_classes=num_classes,
            alpha=assigner_alpha,
            beta=assigner_beta,
            stride=strides,
            eps=assigner_eps,
        )
        self.register_buffer('proj', torch.arange(self.reg_max, dtype=torch.float32))

    def _make_anchors(self, feats, device, dtype):
        anchor_points = []
        stride_tensor = []
        for feat, stride in zip(feats, self.strides):
            h, w = feat.shape[2:]
            yv, xv = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing='ij',
            )
            points = torch.stack((xv, yv), dim=-1).reshape(-1, 2) + 0.5
            anchor_points.append(points)
            stride_tensor.append(torch.full((h * w, 1), float(stride), device=device, dtype=dtype))
        return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)

    def _bbox_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        distances = pred_dist.matmul(self.proj.to(pred_dist.dtype))
        return _dist2bbox(distances, anchor_points)

    def _preprocess_targets(self, gt_dict_list, batch_size, device):
        max_gt = max((len(gt.get('boxes', [])) for gt in gt_dict_list), default=0)
        gt_labels = torch.zeros(batch_size, max_gt, 1, device=device)
        gt_bboxes = torch.zeros(batch_size, max_gt, 4, device=device)
        mask_gt = torch.zeros(batch_size, max_gt, 1, device=device)
        for b, gt in enumerate(gt_dict_list):
            boxes = gt['boxes'].to(device, non_blocking=True).float()
            classes = gt['classes'].to(device, non_blocking=True).float().view(-1, 1)
            n = min(len(boxes), max_gt)
            if n:
                gt_bboxes[b, :n] = boxes[:n]
                gt_labels[b, :n] = classes[:n]
                mask_gt[b, :n] = 1.0
        return gt_labels, gt_bboxes, mask_gt

    def forward(self, head_outs, gt_dict_list):
        cls_feats = head_outs['cls']
        reg_feats = head_outs['reg']
        device = cls_feats[0].device
        dtype = cls_feats[0].dtype
        batch_size = cls_feats[0].shape[0]

        pred_scores = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
             for x in cls_feats],
            dim=1,
        )
        pred_distri = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * self.reg_max)
             for x in reg_feats],
            dim=1,
        )
        anchor_points, stride_tensor = self._make_anchors(cls_feats, device, dtype)
        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)
        gt_labels, gt_bboxes, mask_gt = self._preprocess_targets(gt_dict_list, batch_size, device)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = torch.clamp(target_scores.sum(), min=1.0)

        loss_cls = self.bce(pred_scores.float(), target_scores.to(dtype).float()).sum() / target_scores_sum
        loss_box = torch.zeros((), device=device)
        loss_dfl = torch.zeros((), device=device)

        if fg_mask.sum():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            pred_boxes_pix = pred_bboxes * stride_tensor
            ciou = _bbox_ciou(pred_boxes_pix[fg_mask], target_bboxes[fg_mask])
            loss_box = ((1.0 - ciou).unsqueeze(-1) * weight).sum() / target_scores_sum

            target_ltrb = _bbox2dist(anchor_points, target_bboxes / stride_tensor, self.reg_max - 1)
            dfl_raw = _dfl_loss_per_anchor(
                pred_distri[fg_mask].reshape(-1, self.reg_max),
                target_ltrb[fg_mask],
                self.reg_max,
            )
            loss_dfl = (dfl_raw * weight).sum() / target_scores_sum

        box = self.w_box * loss_box
        cls = self.w_cls * loss_cls
        dfl = self.w_dfl * loss_dfl
        total = (box + cls + dfl) * batch_size
        return {
            'total': total,
            'det_total': (box + cls + dfl).detach(),
            'det_ciou': box.detach(),
            'det_cls': cls.detach(),
            'det_dfl': dfl.detach(),
            'num_pos': fg_mask.sum().detach().float(),
            'target_scores_sum': target_scores_sum.detach(),
        }


def _dfl_loss_per_anchor(pred_dist, target_ltrb, reg_max):
    target = target_ltrb.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()
    tr = tl + 1
    wl = tr.float() - target
    wr = 1 - wl
    loss = (
        F.cross_entropy(pred_dist, tl.reshape(-1), reduction='none').view(tl.shape) * wl +
        F.cross_entropy(pred_dist, tr.reshape(-1), reduction='none').view(tl.shape) * wr
    )
    return loss.mean(-1, keepdim=True)


class MultiTaskLoss(nn.Module):
    """Multi-task loss for the BiFPN detection and pose heads.

    Args:
        w_box: Box loss weight
        w_cls: Classification loss weight
        w_dfl: DFL loss weight
        w_pose: Keypoint OKS loss weight
        w_kobj: Keypoint objectness loss weight
        reg_max: DFL max bin
        num_det_classes: Number of detection classes
    """

    def __init__(self, w_box=7.5, w_cls=0.5, w_dfl=1.5,
                 w_pose=12.0, w_kobj=1.0, reg_max=16,
                 num_det_classes=80):
        super().__init__()
        self.w_box = w_box
        self.w_cls = w_cls
        self.w_dfl = w_dfl
        self.w_pose = w_pose
        self.w_kobj = w_kobj
        self.reg_max = reg_max
        self.num_det_classes = num_det_classes
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
            head_type: 'det' or 'pose'
            norm_pos: Optional shared normalizer for cls/box/dfl. This keeps
                dual-head losses comparable with unified-head losses.
        """
        first_out = head_outs['cls'][0] if head_outs.get('cls') else head_outs['kpt'][0]
        device = first_out.device
        B = first_out.shape[0]
        num_cls = self.num_det_classes
        if head_type == 'pose':
            num_cls = 1
        need_box_terms = not (
            head_type == 'pose' and
            self.w_box == 0 and self.w_cls == 0 and self.w_dfl == 0
        )

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

            cls_p_all = None
            cls_tgt_all = None
            if need_box_terms:
                cls_p_all = head_outs['cls'][lvl].permute(0, 2, 3, 1).reshape(B, N_lvl, -1)
                cls_tgt_all = torch.zeros(B, N_lvl, cls_p_all.shape[-1], device=device)
                total_cls_items += cls_tgt_all.numel()

            if targets is None:
                if need_box_terms:
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

            kpt_p = head_outs['kpt'][lvl][batch_idx, :, gy, gx] if 'kpt' in head_outs else None

            locs_x = (gx.float() + 0.5) * stride
            locs_y = (gy.float() + 0.5) * stride

            if need_box_terms:
                # Extract predictions at positive positions
                reg_p = head_outs['reg'][lvl][batch_idx, :, gy, gx]

                # DFL decode
                reg_rs = reg_p.view(N_pos, 4, self.reg_max)
                reg_probs = reg_rs.softmax(dim=-1)
                reg_delta = (reg_probs * proj.view(1, 1, self.reg_max)).sum(dim=-1) * stride

                l, t = reg_delta[:, 0], reg_delta[:, 1]
                r, b = reg_delta[:, 2], reg_delta[:, 3]

                pred_xyxy = torch.stack([locs_x - l, locs_y - t, locs_x + r, locs_y + b], dim=-1)

                iou, _, _ = _iou_xyxy(pred_xyxy, gt_boxes)

                # Classification targets
                cls_tgt_flat = cls_tgt_all.view(-1, cls_tgt_all.shape[-1])
                flat_idx = batch_idx * N_lvl + gy * W + gx
                if head_type == 'det':
                    valid_cls = (gt_classes >= 0) & (gt_classes < cls_tgt_all.shape[-1])
                    if valid_cls.any():
                        cls_tgt_flat[flat_idx[valid_cls], gt_classes[valid_cls]] = 1.0
                elif head_type == 'pose':
                    person_pos = gt_classes == 0
                    if person_pos.any():
                        cls_tgt_flat[flat_idx[person_pos], 0] = 1.0

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
                            (p_boxes[:, 3] - p_boxes[:, 1])).clamp(min=1)
                    scale = area.sqrt()
                    sigmas = self.sigmas.view(1, 17).to(device)
                    d2 = (pk_xy - gk_xy).pow(2).sum(dim=-1)
                    # Match the OKS scale used by evaluation: (2 * sqrt(area))^2.
                    k2 = sigmas.pow(2) * (2 * scale).pow(2).unsqueeze(-1) + 1e-8
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

