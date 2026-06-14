"""TaskAlignedAssigner — 基于预测质量的动态 top-k 正样本分配.

原理 (TOOD/YOLOv8):
    alignment = cls_score^α × IoU^β
    对每个 GT，跨所有尺度选 alignment 最高的 top-k 个预测作为正样本。
    这替代了 v1 的 FCOS center-based 固定半径分配，
    解决了正样本过少 (每 GT ~9 个) 和分配质量低的问题。
"""

import torch


def _box_iou(box1, box2):
    """批量 IoU: box1 [N, 4], box2 [M, 4] → [N, M]."""
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-16)


class TaskAlignedAssigner:
    """动态 top-k 正样本分配器. 跨所有 FPN 尺度联合选择.

    与 v1 的 CenterAssigner 不同, 此分配器使用预测质量 (alignment) 动态选择
    正样本. 添加了 center-radius 约束确保网格中心在 GT 框内,
    以及尺寸级别过滤确保每个 GT 只分配到合适的 FPN 层级.

    Args:
        topk: 每个 GT 选择的正样本数 (建议 10-13)
        alpha: 分类权重指数
        beta: IoU 权重指数
        center_radius: 中心点半径约束 (stride units), 负值=仅框内
        level_ranges: 各层级负责的 ltrb max 范围, None=auto from strides
    """

    def __init__(self, topk=13, alpha=1.0, beta=3.0, center_radius=-1):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius

    def _build_level_mask(self, num_levels, strides):
        """每个 level 负责的 max(ltrb) 像素范围."""
        ranges = []
        for i, s in enumerate(strides):
            if i == 0:
                ranges.append((0, s * 8))
            else:
                ranges.append((strides[i - 1] * 8, s * 8))
        return ranges

    def __call__(self, pred_scores, pred_boxes,
                 gt_boxes, gt_classes, gt_attrs,
                 feat_sizes, strides, batch_indices=None):
        """
        Args:
            pred_scores: List[[B, H*W, 3]]  各层预测得分 (after sigmoid)
            pred_boxes:  List[[B, H*W, 4]]  各层预测框 xyxy (DFL decoded)
            gt_boxes:    [M, 4] xyxy
            gt_classes:  [M]
            gt_attrs:    dict with kpts/helmet/smoking (or empty)
            feat_sizes:  List[(H, W)]
            strides:     List[int]
            batch_indices: [M] 每个 GT 属于 batch 中的第几张图, None=全为0

        Returns:
            targets: List[dict or None] per level
        """
        device = gt_boxes.device
        num_levels = len(feat_sizes)
        num_gts = len(gt_boxes)

        if num_gts == 0:
            return [None] * num_levels

        level_ranges = self._build_level_mask(num_levels, strides)

        # ── 初始化 targets ──
        targets = [{
            "grid_xy": [], "gt_boxes": [], "gt_classes": [],
            "gt_kpts": [], "gt_helmet": [], "gt_smoking": [], "batch_idx": [],
            "align_score": [],
        } for _ in range(num_levels)]

        # ── 1. 拼接所有 level ──
        offsets = []
        level_W = []
        total_N = 0
        for lvl, (H, W) in enumerate(feat_sizes):
            offsets.append(total_N)
            level_W.append(W)
            total_N += H * W

        all_scores = torch.cat([s.view(s.shape[0], -1, 3) for s in pred_scores], dim=1)  # [B, total_N, 3]
        all_boxes = torch.cat([b.view(b.shape[0], -1, 4) for b in pred_boxes], dim=1)    # [B, total_N, 4]

        # ── 2. 构建全局格点中心坐标 ──
        all_centers = []  # [total_N, 2] xy 像素坐标
        for lvl, (H, W) in enumerate(feat_sizes):
            stride = strides[lvl]
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device), indexing="ij")
            cx = (xv.float() + 0.5) * stride
            cy = (yv.float() + 0.5) * stride
            all_centers.append(torch.stack([cx.flatten(), cy.flatten()], dim=1))
        all_centers = torch.cat(all_centers, dim=0)  # [total_N, 2]

        gt_cls = gt_classes.long()

        # ── 3. 每个 GT 独立分配 ──
        person_count = 0
        for gt_i in range(num_gts):
            gt_box = gt_boxes[gt_i]     # [4]
            gt_cls_i = gt_cls[gt_i]
            gt_batch = batch_indices[gt_i].item() if batch_indices is not None else 0

            # ── 3a. Center-radius 约束: 网格中心必须在 GT 框内 ──
            if self.center_radius >= 0:
                # 宽松模式: 中心距 GT 中心 ≤ center_radius * stride
                gt_cx = (gt_box[0] + gt_box[2]) / 2
                gt_cy = (gt_box[1] + gt_box[3]) / 2
                # 使用最细粒度 stride 计算距离阈值
                min_stride = strides[0]
                max_dist = self.center_radius * min_stride
                dist = (all_centers - torch.tensor([gt_cx, gt_cy], device=device)).norm(dim=1)
                center_mask = dist <= max_dist
            else:
                # 严格模式 (center_radius < 0): 中心必须在 GT 框内部
                center_mask = (
                    (all_centers[:, 0] >= gt_box[0]) &
                    (all_centers[:, 0] <= gt_box[2]) &
                    (all_centers[:, 1] >= gt_box[1]) &
                    (all_centers[:, 1] <= gt_box[3])
                )

            # ── 3b. 尺寸级别过滤 ──
            gt_w = (gt_box[2] - gt_box[0]).item()
            gt_h = (gt_box[3] - gt_box[1]).item()
            max_side = max(gt_w, gt_h)

            # 匹配当前级别 + 相邻一个级别 (增加正样本数，改善框回归梯度)
            valid_levels = set()
            for lvl, (lo, hi) in enumerate(level_ranges):
                if lvl == num_levels - 1:
                    if max_side >= lo:
                        valid_levels.add(lvl)
                else:
                    if lo <= max_side < hi:
                        valid_levels.add(lvl)
            # 加入相邻级别
            adjacent = set()
            for lvl in valid_levels:
                if lvl > 0:
                    adjacent.add(lvl - 1)
                if lvl < num_levels - 1:
                    adjacent.add(lvl + 1)
            valid_levels |= adjacent
            # 至少保留一个级别
            if not valid_levels:
                valid_levels.add(num_levels - 1 if max_side >= level_ranges[-1][0] else 0)

            # 构建该 GT 的 level mask
            level_mask = torch.zeros(total_N, dtype=torch.bool, device=device)
            for lvl in valid_levels:
                start = offsets[lvl]
                end = start + feat_sizes[lvl][0] * feat_sizes[lvl][1]
                level_mask[start:end] = True

            # ── 3c. 合并约束 ──
            valid_mask = center_mask & level_mask
            valid_count = valid_mask.sum().item()

            if valid_count == 0:
                # 如果约束太过严格, 回退到仅 center_mask
                valid_mask = center_mask
                valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue  # 无法分配此 GT

            # ── 3d. 计算 alignment ──
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            cls_valid = all_scores[gt_batch, valid_indices, gt_cls_i]  # [K]
            ious_valid = _box_iou(all_boxes[gt_batch, valid_indices], gt_box.unsqueeze(0)).squeeze(-1)  # [K]
            align_valid = cls_valid.pow(self.alpha) * ious_valid.pow(self.beta)

            # ── 3e. Top-k 选择 ──
            topk = min(self.topk, valid_count)
            _, topk_local = align_valid.topk(topk)

            # 提取该 GT 的人体属性 (每个 GT 一份, topk 个 assignment 共享)
            gt_kpt_val = None
            gt_helm_val = None
            gt_smoke_val = None
            if gt_cls_i == 0:
                if gt_attrs and "kpts" in gt_attrs:
                    gt_kpt_val = gt_attrs["kpts"][person_count]
                if gt_attrs and "helmet" in gt_attrs:
                    gt_helm_val = gt_attrs["helmet"][person_count]
                if gt_attrs and "smoking" in gt_attrs:
                    gt_smoke_val = gt_attrs["smoking"][person_count]
                person_count += 1

            for k in range(topk):
                global_idx = valid_indices[topk_local[k]].item()

                # 确定属于哪一层
                for lvl in range(num_levels - 1, -1, -1):
                    if global_idx >= offsets[lvl]:
                        break

                W_l = level_W[lvl]
                local_idx = global_idx - offsets[lvl]
                gx = local_idx % W_l
                gy = local_idx // W_l

                t = targets[lvl]
                t["grid_xy"].append(torch.tensor([gx, gy], device=device))
                t["gt_boxes"].append(gt_box)
                t["gt_classes"].append(gt_cls_i)
                t["batch_idx"].append(gt_batch)
                t["align_score"].append(align_valid[topk_local[k]])

                t["gt_kpts"].append(gt_kpt_val)
                t["gt_helmet"].append(gt_helm_val)
                t["gt_smoking"].append(gt_smoke_val)

        # ── 5. 合并每层 ──
        merged = []
        for lvl in range(num_levels):
            t = targets[lvl]
            if len(t["grid_xy"]) == 0:
                merged.append(None)
                continue

            # Resolve conflicts where multiple GTs select the same grid.
            # Keep the assignment with the strongest task-aligned score.
            best_by_cell = {}
            for i, (grid, batch_idx, align_score) in enumerate(zip(
                    t["grid_xy"], t["batch_idx"], t["align_score"])):
                key = (int(batch_idx), int(grid[0]), int(grid[1]))
                prev = best_by_cell.get(key)
                if prev is None or align_score > t["align_score"][prev]:
                    best_by_cell[key] = i
            keep = sorted(best_by_cell.values())
            for key in t:
                t[key] = [t[key][i] for i in keep]

            merged.append({
                "grid_xy": torch.stack(t["grid_xy"], dim=0),                    # [K, 2]
                "gt_boxes": torch.stack(t["gt_boxes"], dim=0),                  # [K, 4]
                "gt_classes": torch.stack(t["gt_classes"], dim=0),              # [K]
                "gt_kpts": (torch.stack([x for x in t["gt_kpts"] if x is not None])
                            if any(x is not None for x in t["gt_kpts"]) else None),
                "gt_helmet": (torch.stack([x for x in t["gt_helmet"] if x is not None])
                              if any(x is not None for x in t["gt_helmet"]) else None),
                "gt_smoking": (torch.stack([x for x in t["gt_smoking"] if x is not None])
                               if any(x is not None for x in t["gt_smoking"]) else None),
                "batch_idx": torch.tensor(t["batch_idx"], device=device),
            })
        return merged

