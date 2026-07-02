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

        level_ranges = self._build_level_ranges(strides, num_levels)

        # Initialize per-level targets
        targets = [{
            'grid_xy': [], 'gt_boxes': [], 'gt_classes': [],
            'gt_kpts': [], 'batch_idx': [], 'align_score': [],
        } for _ in range(num_levels)]

        # Concatenate all levels
        offsets = []
        level_W = []
        total_N = 0
        for lvl, (H, W) in enumerate(feat_sizes):
            offsets.append(total_N)
            level_W.append(W)
            total_N += H * W

        all_scores = torch.cat([s.view(s.shape[0], -1, s.shape[-1]) for s in pred_scores], dim=1)
        all_boxes = torch.cat([b.view(b.shape[0], -1, 4) for b in pred_boxes], dim=1)

        # Build all grid center coordinates
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

        for gt_i in range(num_gts):
            gt_box = gt_boxes[gt_i]
            gt_cls_i = gt_classes[gt_i].item()
            gt_batch = batch_indices[gt_i].item() if batch_indices is not None else 0

            # Center constraint
            if self.center_radius >= 0:
                gt_cx = (gt_box[0] + gt_box[2]) / 2
                gt_cy = (gt_box[1] + gt_box[3]) / 2
                max_dist = self.center_radius * strides[0]
                dist = (all_centers - torch.tensor([gt_cx, gt_cy], device=device)).norm(dim=1)
                center_mask = dist <= max_dist
            else:
                center_mask = (
                    (all_centers[:, 0] >= gt_box[0]) &
                    (all_centers[:, 0] <= gt_box[2]) &
                    (all_centers[:, 1] >= gt_box[1]) &
                    (all_centers[:, 1] <= gt_box[3])
                )

            # Level filtering by box size
            gt_w, gt_h = (gt_box[2] - gt_box[0]).item(), (gt_box[3] - gt_box[1]).item()
            max_side = max(gt_w, gt_h)

            valid_levels = set()
            for lvl, (lo, hi) in enumerate(level_ranges):
                if lvl == num_levels - 1:
                    if max_side >= lo:
                        valid_levels.add(lvl)
                elif lo <= max_side < hi:
                    valid_levels.add(lvl)
            adjacent = set()
            for lvl in valid_levels:
                if lvl > 0:
                    adjacent.add(lvl - 1)
                if lvl < num_levels - 1:
                    adjacent.add(lvl + 1)
            valid_levels |= adjacent
            if not valid_levels:
                valid_levels.add(num_levels - 1 if max_side >= level_ranges[-1][0] else 0)

            level_mask = torch.zeros(total_N, dtype=torch.bool, device=device)
            for lvl in valid_levels:
                start = offsets[lvl]
                end = start + feat_sizes[lvl][0] * feat_sizes[lvl][1]
                level_mask[start:end] = True

            valid_mask = center_mask & level_mask
            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                valid_mask = center_mask
                valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue

            # Alignment score
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            # Map class index to head's class index space
            # For dual-head: person(0) -> PoseHead cls_idx=0, det classes(1..19) -> DetectHead cls_idx=0..18
            # For unified-head: class 0..19 direct
            if all_scores.shape[-1] == 1:
                # Pose head: only person class
                score_idx = 0
            elif all_scores.shape[-1] <= num_det_classes:
                # Detect head or unified head
                score_idx = gt_cls_i
            else:
                score_idx = gt_cls_i

            score_idx = min(score_idx, all_scores.shape[-1] - 1)
            cls_valid = all_scores[gt_batch, valid_indices, score_idx]
            ious_valid = _box_iou(all_boxes[gt_batch, valid_indices], gt_box.unsqueeze(0)).squeeze(-1)
            align_valid = cls_valid.pow(self.alpha) * ious_valid.pow(self.beta)

            topk = min(self.topk, valid_count)
            _, topk_local = align_valid.topk(topk)

            gt_kpt_val = None
            if gt_cls_i == 0 and gt_kpts is not None and gt_i < len(gt_kpts):
                gt_kpt_val = gt_kpts[gt_i]

            for k in range(topk):
                global_idx = valid_indices[topk_local[k]].item()

                for lvl in range(num_levels - 1, -1, -1):
                    if global_idx >= offsets[lvl]:
                        break

                W_l = level_W[lvl]
                local_idx = global_idx - offsets[lvl]
                gx = local_idx % W_l
                gy = local_idx // W_l

                t = targets[lvl]
                t['grid_xy'].append(torch.tensor([gx, gy], device=device))
                t['gt_boxes'].append(gt_box)
                t['gt_classes'].append(gt_cls_i)
                t['batch_idx'].append(gt_batch)
                t['align_score'].append(align_valid[topk_local[k]])
                t['gt_kpts'].append(gt_kpt_val)

        # Resolve conflicts: keep the assignment with highest alignment per grid cell
        merged = []
        for lvl in range(num_levels):
            t = targets[lvl]
            if len(t['grid_xy']) == 0:
                merged.append(None)
                continue

            best_by_cell = {}
            for i, (grid, batch_idx, align_score) in enumerate(
                    zip(t['grid_xy'], t['batch_idx'], t['align_score'])):
                key = (int(batch_idx), int(grid[0]), int(grid[1]))
                prev = best_by_cell.get(key)
                if prev is None or align_score > t['align_score'][prev]:
                    best_by_cell[key] = i
            keep = sorted(best_by_cell.values())
            for key in t:
                t[key] = [t[key][i] for i in keep]

            merged.append({
                'grid_xy': torch.stack(t['grid_xy'], dim=0),
                'gt_boxes': torch.stack(t['gt_boxes'], dim=0),
                'gt_classes': torch.tensor(t['gt_classes'], device=device, dtype=torch.long),
                'gt_kpts': (torch.stack([x for x in t['gt_kpts'] if x is not None])
                            if any(x is not None for x in t['gt_kpts']) else None),
                'batch_idx': torch.tensor(t['batch_idx'], device=device, dtype=torch.long),
            })
        return merged
