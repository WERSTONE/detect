"""TaskAlignedAssigner — multi-head version.

Adapted from YOLOv8's TaskAlignedAssigner. Supports:
- Dual-head: separate det cls(19) + pose cls(1) heads
- Unified-head: single cls(20) head with person at index 0

Alignment score = cls_score^alpha * IoU^beta
"""

import torch


def _box_iou(box1, box2):
    """Batch IoU: box1 [N,4], box2 [M,4] -> [N,M]."""
    box1 = torch.nan_to_num(box1.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    box2 = torch.nan_to_num(box2.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=1e-7) * (box1[:, 3] - box1[:, 1]).clamp(min=1e-7)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=1e-7) * (box2[:, 3] - box2[:, 1]).clamp(min=1e-7)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-16)


class TaskAlignedAssigner:
    """Dynamic top-k positive sample assigner across all FPN levels.

    Args:
        topk: Number of positive samples per GT.
        alpha, beta: Alignment score exponents.
        center_radius: Center constraint in stride units (-1 = inside box only).
    """

    def __init__(self, topk=13, alpha=1.0, beta=3.0, center_radius=-1):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius

    def _build_level_ranges(self, strides, num_levels):
        ranges = []
        for i, s in enumerate(strides):
            if i == 0:
                ranges.append((0.0, s * 8))
            else:
                ranges.append((strides[i - 1] * 8, s * 8))
        return ranges

    @torch.no_grad()
    def __call__(self, pred_scores, pred_boxes, gt_boxes, gt_classes,
                 gt_kpts, feat_sizes, strides, batch_indices,
                 num_det_classes=19):
        """Assign targets.

        Args:
            pred_scores: List[[B, H*W, num_cls]] per level
            pred_boxes:  List[[B, H*W, 4]] per level
            gt_boxes:    [M, 4] xyxy
            gt_classes:  [M] 0=person, 1..19=det classes
            gt_kpts:     [M, 17, 3] or None (only for class 0)
            feat_sizes:  List[(H, W)]
            strides:     List[int]
            batch_indices: [M] which image each GT belongs to
            num_det_classes: Total detection classes (19 for dual-head, 20 for unified)

        Returns:
            List[dict or None] per level
        """
        device = gt_boxes.device
        num_levels = len(feat_sizes)
        num_gts = len(gt_boxes)

        if num_gts == 0:
            return [None] * num_levels

        batch_indices = (
            batch_indices.to(device=device, dtype=torch.long)
            if batch_indices is not None else
            torch.zeros(num_gts, device=device, dtype=torch.long)
        )

        offsets = []
        level_ids = []
        level_w = []
        total_n = 0
        for lvl, (H, W) in enumerate(feat_sizes):
            offsets.append(total_n)
            n_lvl = H * W
            total_n += n_lvl
            level_ids.append(torch.full((n_lvl,), lvl, device=device, dtype=torch.long))
            level_w.append(W)

        offsets_t = torch.tensor(offsets, device=device, dtype=torch.long)
        level_w_t = torch.tensor(level_w, device=device, dtype=torch.long)
        level_ids = torch.cat(level_ids, dim=0)

        all_scores = torch.cat(
            [s.reshape(s.shape[0], -1, s.shape[-1]) for s in pred_scores], dim=1)
        all_boxes = torch.cat(
            [b.reshape(b.shape[0], -1, 4) for b in pred_boxes], dim=1)
        B = all_scores.shape[0]

        all_centers = []
        for lvl, (H, W) in enumerate(feat_sizes):
            stride = strides[lvl]
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device), indexing='ij')
            cx = (xv.float() + 0.5) * stride
            cy = (yv.float() + 0.5) * stride
            all_centers.append(torch.stack([cx.flatten(), cy.flatten()], dim=1))
        all_centers = torch.cat(all_centers, dim=0)

        if self.center_radius >= 0:
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
            max_dist = self.center_radius * strides[0]
            center_mask = (all_centers.unsqueeze(0) - gt_centers.unsqueeze(1)).norm(dim=2) <= max_dist
        else:
            center_x = all_centers[:, 0].unsqueeze(0)
            center_y = all_centers[:, 1].unsqueeze(0)
            center_mask = (
                (center_x >= gt_boxes[:, 0:1]) &
                (center_x <= gt_boxes[:, 2:3]) &
                (center_y >= gt_boxes[:, 1:2]) &
                (center_y <= gt_boxes[:, 3:4])
            )

        level_ranges = self._build_level_ranges(strides, num_levels)
        wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=0)
        max_side = wh.max(dim=1).values
        valid_levels = torch.zeros(num_gts, num_levels, device=device, dtype=torch.bool)
        for lvl, (lo, hi) in enumerate(level_ranges):
            if lvl == num_levels - 1:
                valid_levels[:, lvl] = max_side >= lo
            else:
                valid_levels[:, lvl] = (max_side >= lo) & (max_side < hi)

        base_levels = valid_levels.clone()
        valid_levels[:, 1:] |= base_levels[:, :-1]
        valid_levels[:, :-1] |= base_levels[:, 1:]
        no_level = ~valid_levels.any(dim=1)
        fallback = torch.where(
            max_side >= level_ranges[-1][0],
            torch.full_like(batch_indices, num_levels - 1),
            torch.zeros_like(batch_indices),
        )
        valid_levels[no_level, fallback[no_level]] = True

        valid_mask = center_mask & valid_levels[:, level_ids]
        valid_count = valid_mask.sum(dim=1)
        valid_mask = torch.where((valid_count == 0).unsqueeze(1), center_mask, valid_mask)
        valid_count = valid_mask.sum(dim=1)
        active = valid_count > 0
        if active.nonzero(as_tuple=True)[0].numel() == 0:
            return [None] * num_levels

        score_idx = gt_classes.clamp(0, all_scores.shape[-1] - 1)
        if all_scores.shape[-1] == 1:
            score_idx = torch.zeros_like(score_idx)
        gt_scores = all_scores[batch_indices].gather(
            2, score_idx.view(num_gts, 1, 1).expand(-1, total_n, 1)).squeeze(-1)

        gt_pred_boxes = all_boxes[batch_indices]
        lt = torch.max(gt_pred_boxes[..., :2], gt_boxes[:, None, :2])
        rb = torch.min(gt_pred_boxes[..., 2:], gt_boxes[:, None, 2:])
        inter_wh = (rb - lt).clamp(min=0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]
        pred_area = (
            (gt_pred_boxes[..., 2] - gt_pred_boxes[..., 0]).clamp(min=1e-7) *
            (gt_pred_boxes[..., 3] - gt_pred_boxes[..., 1]).clamp(min=1e-7)
        )
        gt_area = (
            (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-7) *
            (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-7)
        ).unsqueeze(1)
        ious = inter / (pred_area + gt_area - inter + 1e-16)

        align = gt_scores.pow(self.alpha) * ious.pow(self.beta)
        align = align.masked_fill(~valid_mask, -1.0)
        topk = min(self.topk, total_n)
        topk_score, topk_idx = align.topk(topk, dim=1)
        keep = active.unsqueeze(1) & (topk_score >= 0)
        if keep.nonzero(as_tuple=True)[0].numel() == 0:
            return [None] * num_levels

        gt_sel = torch.arange(num_gts, device=device).view(-1, 1).expand(-1, topk)[keep]
        global_idx = topk_idx[keep]
        align_sel = topk_score[keep]
        level_sel = level_ids[global_idx]
        local_idx = global_idx - offsets_t[level_sel]
        gx = local_idx % level_w_t[level_sel]
        gy = local_idx // level_w_t[level_sel]
        batch_sel = batch_indices[gt_sel]
        boxes_sel = gt_boxes[gt_sel]
        classes_sel = gt_classes[gt_sel]

        merged = []
        for lvl in range(num_levels):
            lvl_pos = (level_sel == lvl).nonzero(as_tuple=True)[0]
            if lvl_pos.numel() == 0:
                merged.append(None)
                continue

            H, W = feat_sizes[lvl]
            lvl_local = local_idx[lvl_pos]
            cell_key = batch_sel[lvl_pos] * (H * W) + lvl_local
            best_score = torch.full((B * H * W,), -1.0, device=device, dtype=align_sel.dtype)
            best_score.scatter_reduce_(0, cell_key, align_sel[lvl_pos], reduce='amax', include_self=True)
            keep_pos = lvl_pos[align_sel[lvl_pos] == best_score[cell_key]]

            grid_xy = torch.stack([gx[keep_pos], gy[keep_pos]], dim=1)
            cls_keep = classes_sel[keep_pos]
            person_pos = (cls_keep == 0).nonzero(as_tuple=True)[0]

            merged.append({
                'grid_xy': grid_xy,
                'gt_boxes': boxes_sel[keep_pos],
                'gt_classes': cls_keep.long(),
                'gt_kpts': (gt_kpts[gt_sel[keep_pos][person_pos]]
                            if gt_kpts is not None and person_pos.numel() > 0 else None),
                'batch_idx': batch_sel[keep_pos].long(),
            })
        return merged
