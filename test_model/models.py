"""Five candidate multi-head YOLOv8m-based models.

M-A: Standard dual-head   (backbone + FPN+PAN + DetectHead + PoseHead)
M-B: Unified head         (backbone + FPN+PAN + UnifiedHead)
M-C: Dual-neck dual-head  (backbone + DetNeck + PoseNeck + DetectHead + PoseHead)
M-D: ECA backbone + dual-head  (backbone+ECA + FPN+PAN + DetectHead + PoseHead)
M-E: BiFPN neck + dual-head    (backbone + BiFPN + DetectHead + PoseHead)

Each model implements:
- forward(): raw head outputs
- compute_loss(images, gt_dict_list): training loss
- predict_val(images): validation predictions for mAP computation
"""

import math

import torch
import torch.nn as nn

from test_model.backbone import CSPDarkNet
from test_model.neck import FPNPANNeck, BiFPN, DetNeck, PoseNeck
from test_model.heads import DetectHead, PoseHead, UnifiedHead
from test_model.assigner import TaskAlignedAssigner
from test_model.common import Conv
from test_model.loss import MultiTaskLoss

try:
    from torchvision.ops import batched_nms
except Exception:  # pragma: no cover - fallback for minimal environments
    batched_nms = None


# ── DFL decode helpers ──

def _make_grid(nx, ny, device):
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device), indexing='ij')
    return torch.stack((xv, yv), 2).float()


def _dfl_decode(reg_pred, reg_max, stride, grid):
    """DFL decode: reg distribution -> xyxy boxes."""
    B, _, H, W = reg_pred.shape
    N = H * W
    reg = reg_pred.view(B, 4, reg_max, N)
    reg = reg.softmax(dim=-2)
    proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
    reg = (reg * proj.view(1, 1, reg_max, 1)).sum(dim=-2) * stride

    g = grid.view(1, N, 2) + 0.5 * stride
    cx = g[..., 0:1].transpose(1, 2)
    cy = g[..., 1:2].transpose(1, 2)

    l, t = reg[:, 0:1], reg[:, 1:2]
    r, b = reg[:, 2:3], reg[:, 3:4]
    x1 = cx - l; y1 = cy - t
    x2 = cx + r; y2 = cy + b
    return torch.cat([x1, y1, x2, y2], dim=1).transpose(1, 2)


def _nms(boxes, scores, iou_thresh=0.6):
    """Vectorized per-class NMS."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item()); break
        i = order[0]; keep.append(i.item())
        box_i = boxes[i]; rest = boxes[order[1:]]
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        lt = torch.max(box_i[:2], rest[:, :2])
        rb = torch.min(box_i[2:], rest[:, 2:])
        wh = (rb - lt).clamp(min=0)
        iou = wh[:, 0] * wh[:, 1] / (area_i + area_rest - wh[:, 0] * wh[:, 1] + 1e-8)
        order = order[1:][iou <= iou_thresh]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _batched_nms(boxes, scores, classes, iou_thresh=0.6, max_det=300):
    """Class-aware NMS with a small fallback if torchvision is unavailable."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    if batched_nms is not None:
        try:
            keep = batched_nms(boxes, scores, classes, iou_thresh)
        except RuntimeError:
            keep = None
    else:
        keep = None

    if keep is None:
        keep_parts = []
        for cls_id in classes.unique():
            cls_idx = (classes == cls_id).nonzero(as_tuple=True)[0]
            cls_keep = _nms(boxes[cls_idx], scores[cls_idx], iou_thresh)
            if cls_keep.numel() > 0:
                keep_parts.append(cls_idx[cls_keep])
        keep = torch.cat(keep_parts) if keep_parts else torch.empty(0, dtype=torch.long, device=boxes.device)
        if keep.numel() > 1:
            keep = keep[scores[keep].argsort(descending=True)]

    if max_det and keep.numel() > max_det:
        keep = keep[:max_det]
    return keep


# ═══════════════════════════════════════════════════════════════
# Base model with shared training logic
# ═══════════════════════════════════════════════════════════════

class _BaseModel(nn.Module):
    """Common training and inference logic for all candidate models."""

    def __init__(self, strides=(8, 16, 32), reg_max=16):
        super().__init__()
        self.strides = list(strides)
        self.reg_max = reg_max
        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=3.0)
        self.det_task_weight = 1.0
        self.pose_task_weight = 1.0
        # Loss function set in subclass init

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _forward_backbone_neck(self, x):
        raise NotImplementedError

    def set_task_weights(self, det_weight=1.0, pose_weight=1.0):
        self.det_task_weight = float(det_weight)
        self.pose_task_weight = float(pose_weight)

    @torch.no_grad()
    def _get_decoded_preds(self, head_outs, feat_sizes):
        """Decode head outputs for assigner consumption.

        Returns:
            pred_scores: List[[B, H*W, C]] per level (after sigmoid)
            pred_boxes: List[[B, H*W, 4]] per level (xyxy)
        """
        device = head_outs['cls'][0].device
        pred_scores, pred_boxes = [], []
        B = head_outs['cls'][0].shape[0]

        for lvl, ((H, W), stride) in enumerate(zip(feat_sizes, self.strides)):
            cls_p = head_outs['cls'][lvl].permute(0, 2, 3, 1).reshape(B, H * W,
                                         head_outs['cls'][lvl].shape[1])
            scores = cls_p.sigmoid()
            pred_scores.append(scores)

            grid = _make_grid(W, H, device) * stride
            boxes = _dfl_decode(head_outs['reg'][lvl], self.reg_max, stride, grid)
            pred_boxes.append(boxes)

        return pred_scores, pred_boxes

    @torch.no_grad()
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6, max_det=300):
        """Run detection on normalized images [B, 3, 640, 640].

        Returns: List[dict] per image with 'boxes'[K,4], 'scores'[K], 'classes'[K],
                 optionally 'kpts'[K,17,3]
        """
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device)

        head_outs = self._forward_head(images)
        cls_list = head_outs['cls']
        reg_list = head_outs['reg']

        return self._decode_predictions(cls_list, reg_list, head_outs.get('kpt'),
                                        score_thresh, iou_thresh, max_det=max_det)

    def _decode_predictions(self, cls_list, reg_list, kpt_list,
                            score_thresh=0.01, iou_thresh=0.6, cls_offset=0,
                            max_det=300, max_nms=30000):
        """Decode raw outputs -> final predictions.

        Args:
            cls_offset: Added to class index (1 for det head, 0 for pose/unified)
        """
        device = cls_list[0].device
        B = cls_list[0].shape[0]
        num_cls = cls_list[0].shape[1]

        results = []
        for b in range(B):
            all_boxes, all_scores, all_cls = [], [], []
            all_kpts = []

            for lvl, stride in enumerate(self.strides):
                _, _, H, W = cls_list[lvl].shape
                cls_l = cls_list[lvl][b:b+1].permute(0, 2, 3, 1).reshape(H * W, num_cls)
                scores_l = cls_l.sigmoid()

                grid = _make_grid(W, H, device) * stride
                reg_l = reg_list[lvl][b:b+1]
                boxes_l = _dfl_decode(reg_l, self.reg_max, stride, grid)[0]

                score_mask = scores_l > score_thresh
                if not score_mask.any():
                    continue

                anchor_idx, cls_idx = score_mask.nonzero(as_tuple=True)
                selected_scores = scores_l[anchor_idx, cls_idx]
                if max_nms and selected_scores.numel() > max_nms:
                    topk = selected_scores.argsort(descending=True)[:max_nms]
                    anchor_idx = anchor_idx[topk]
                    cls_idx = cls_idx[topk]
                    selected_scores = selected_scores[topk]

                selected_boxes = boxes_l[anchor_idx]
                selected_classes = cls_idx + cls_offset
                all_boxes.append(selected_boxes)
                all_scores.append(selected_scores)
                all_cls.append(selected_classes.long())

                if kpt_list is not None:
                    kpts_aligned = torch.zeros((len(selected_boxes), 17, 3), device=device)
                    person_mask = (cls_idx == 0) & (cls_offset == 0)
                    if person_mask.any():
                        kpt_l = kpt_list[lvl][b:b+1].permute(0, 2, 3, 1).reshape(H * W, 17, 3)
                        person_anchor_idx = anchor_idx[person_mask]
                        kpt_selected = kpt_l[person_anchor_idx]
                        grid_center = grid.view(H * W, 1, 2) + 0.5 * stride
                        kpt_xy = kpt_selected[..., :2] * stride + grid_center[person_anchor_idx]
                        kpt_vis = kpt_selected[..., 2:3].sigmoid()
                        kpts_aligned[person_mask] = torch.cat([kpt_xy, kpt_vis], dim=-1)
                    all_kpts.append(kpts_aligned)

            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                classes = torch.cat(all_cls)
                kpts = torch.cat(all_kpts) if all_kpts else torch.zeros((len(boxes), 17, 3), device=device)
                keep = _batched_nms(boxes, scores, classes, iou_thresh=iou_thresh, max_det=max_det)
                result = {
                    'boxes': boxes[keep],
                    'scores': scores[keep],
                    'classes': classes[keep],
                    'kpts': kpts[keep],
                }
            else:
                result = {
                    'boxes': torch.zeros(0, 4, device=device),
                    'scores': torch.zeros(0, device=device),
                    'classes': torch.zeros(0, dtype=torch.long, device=device),
                    'kpts': torch.zeros(0, 17, 3, device=device),
                }
            results.append(result)

        return results


# ═══════════════════════════════════════════════════════════════
# Dual-head base (shared by M-A, M-C, M-D, M-E)
# ═══════════════════════════════════════════════════════════════

class _DualHeadModel(_BaseModel):
    """Shared logic for all dual-head models."""

    def __init__(self, num_det_classes=80, num_kpts=17, reg_max=16):
        super().__init__(reg_max=reg_max)
        self.num_det_classes = num_det_classes
        self.num_kpts = num_kpts

        # Two-stage training controls
        self.train_det = True
        self.train_pose = True
        self.det_weight_mult = 1.0
        self.det_weight_warmup_epochs = 0
        self.use_uncertainty_weighting = False
        self.log_var_det = nn.Parameter(torch.zeros(()))
        self.log_var_pose = nn.Parameter(torch.zeros(()))
        self.det_uncertainty_weight_min = None
        self.det_uncertainty_weight_max = None
        self.pose_uncertainty_weight_min = None
        self.pose_uncertainty_weight_max = None

        self.det_loss = MultiTaskLoss(
            w_box=7.5, w_cls=0.5, w_dfl=1.5,
            w_pose=0.0, w_kobj=0.0, reg_max=reg_max,
            num_det_classes=num_det_classes, unified_head=False)
        self.pose_loss = MultiTaskLoss(
            w_box=0.0, w_cls=0.0, w_dfl=0.0,
            w_pose=12.0, w_kobj=1.0, reg_max=reg_max,
            num_det_classes=1, unified_head=False)

    def enable_uncertainty_weighting(self, enabled=True):
        self.use_uncertainty_weighting = bool(enabled)

    def set_uncertainty_weight_bounds(self, det_min=None, det_max=None,
                                      pose_min=None, pose_max=None):
        self.det_uncertainty_weight_min = det_min
        self.det_uncertainty_weight_max = det_max
        self.pose_uncertainty_weight_min = pose_min
        self.pose_uncertainty_weight_max = pose_max

    @staticmethod
    def _bounded_log_var(log_var, weight_min=None, weight_max=None):
        min_log_var = None if weight_max is None else -math.log(float(weight_max))
        max_log_var = None if weight_min is None else -math.log(float(weight_min))
        if min_log_var is None and max_log_var is None:
            return log_var
        if min_log_var is None:
            return torch.clamp(log_var, max=max_log_var)
        if max_log_var is None:
            return torch.clamp(log_var, min=min_log_var)
        return torch.clamp(log_var, min=min_log_var, max=max_log_var)

    def disable_pose_proposal_training(self):
        """Freeze unused pose cls/reg branches; detection head owns boxes/classes."""
        for name in ('cls_tower', 'cls_pred', 'reg_tower', 'reg_pred'):
            module = getattr(self.pose_head, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

    def freeze_head(self, name):
        """Freeze a head. name: 'det' or 'pose'."""
        head = getattr(self, name + '_head')
        for p in head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def update_det_weight(self, epoch):
        """Linear warmup: det_weight_mult 0→1 over det_weight_warmup_epochs."""
        if self.det_weight_warmup_epochs <= 0:
            self.det_weight_mult = 1.0
        else:
            self.det_weight_mult = min(1.0, epoch / self.det_weight_warmup_epochs)

    @torch.no_grad()
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6, max_det=300):
        """Decode boxes/classes from det head and attach pose keypoints to person detections."""
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device)
        det_out, pose_out = self._forward_head(images)
        return self._decode_predictions(
            det_out['cls'], det_out['reg'], pose_out.get('kpt'),
            score_thresh, iou_thresh, cls_offset=0, max_det=max_det)

    def compute_loss(self, images, gt_dict_list):
        device = next(self.parameters()).device
        images = images.to(device)
        B = images.shape[0]

        det_out, pose_out = self._forward_head(images)

        feat_sizes = [(t.shape[2], t.shape[3]) for t in det_out['cls']]

        # Decode for assigner
        det_scores, det_boxes = self._get_decoded_preds(det_out, feat_sizes)

        # Pose head only predicts keypoints. Person anchors are assigned from the
        # detector's person scores/boxes so keypoints align with final detections.
        pose_scores = [scores[..., 0:1] for scores in det_scores]
        pose_boxes = det_boxes

        # Separate GTs by head
        det_boxes_list = []
        det_cls_list = []
        det_batch_list = []
        pose_boxes_list = []
        pose_cls_list = []  # always 0
        pose_kpts_list = []
        pose_batch_list = []

        for b in range(B):
            gt = gt_dict_list[b]
            boxes = gt['boxes']
            classes = gt['classes']
            kpts = gt.get('kpts', torch.zeros(0, 17, 3))
            if len(boxes) == 0:
                continue

            if len(boxes):
                det_gt_boxes_batch = boxes
                det_boxes_list.append(det_gt_boxes_batch.to(device, non_blocking=True))
                det_cls_list.append(classes.to(device, non_blocking=True))
                det_batch_list.append(torch.full(
                    (len(det_gt_boxes_batch),), b, device=device, dtype=torch.long))

            person_mask = classes == 0
            if person_mask.any():
                pose_gt_boxes_batch = boxes[person_mask]
                pose_boxes_list.append(pose_gt_boxes_batch.to(device, non_blocking=True))
                pose_cls_list.append(torch.zeros(
                    len(pose_gt_boxes_batch), device=device, dtype=torch.long))
                if len(kpts) == len(boxes):
                    pose_kpts = kpts[person_mask]
                else:
                    pose_kpts = torch.zeros(len(pose_gt_boxes_batch), 17, 3, dtype=boxes.dtype)
                pose_kpts_list.append(pose_kpts.to(device, non_blocking=True))
                pose_batch_list.append(torch.full(
                    (len(pose_gt_boxes_batch),), b, device=device, dtype=torch.long))

        # Build GT tensors
        if det_boxes_list:
            det_gt_boxes = torch.cat(det_boxes_list, dim=0)
            det_gt_classes = torch.cat(det_cls_list, dim=0).long()
            det_gt_batch = torch.cat(det_batch_list, dim=0)
        else:
            det_gt_boxes = torch.empty(0, 4, device=device)
            det_gt_classes = torch.empty(0, device=device, dtype=torch.long)
            det_gt_batch = torch.empty(0, device=device, dtype=torch.long)

        if pose_boxes_list:
            pose_gt_boxes = torch.cat(pose_boxes_list, dim=0)
            pose_gt_classes = torch.cat(pose_cls_list, dim=0)
            pose_gt_kpts = torch.cat(pose_kpts_list, dim=0)
            pose_gt_batch = torch.cat(pose_batch_list, dim=0)
        else:
            pose_gt_boxes = torch.empty(0, 4, device=device)
            pose_gt_classes = torch.empty(0, device=device, dtype=torch.long)
            pose_gt_kpts = torch.empty(0, 17, 3, device=device)
            pose_gt_batch = torch.empty(0, device=device, dtype=torch.long)

        # Assigner (skip det assigner if not training det)
        if self.train_det:
            det_targets = self.assigner(
                det_scores, det_boxes, det_gt_boxes, det_gt_classes,
                None, feat_sizes, self.strides, det_gt_batch,
                num_det_classes=self.num_det_classes)
        else:
            det_targets = [None] * len(feat_sizes)

        if self.train_pose:
            pose_targets = self.assigner(
                pose_scores, pose_boxes, pose_gt_boxes, pose_gt_classes,
                pose_gt_kpts, feat_sizes, self.strides, pose_gt_batch,
                num_det_classes=1)
        else:
            pose_targets = [None] * len(feat_sizes)

        # Loss
        if self.train_det:
            det_l = self.det_loss(
                det_out, det_targets, self.strides, feat_sizes,
                head_type='det')
        else:
            det_l = {'cls': torch.tensor(0.0, device=device),
                     'ciou': torch.tensor(0.0, device=device),
                     'dfl': torch.tensor(0.0, device=device)}

        if self.train_pose:
            pose_l = self.pose_loss(
                pose_out, pose_targets, self.strides, feat_sizes,
                head_type='pose')
        else:
            pose_l = {
                'kpt': torch.tensor(0.0, device=device),
                'kobj': torch.tensor(0.0, device=device),
            }

        # Apply det_weight_mult during training only (val uses full weight)
        mult = self.det_weight_mult if self.training else 1.0
        det_cls = det_l['cls'] * mult
        det_ciou = det_l['ciou'] * mult
        det_dfl = det_l['dfl'] * mult

        det_total = det_cls + det_ciou + det_dfl
        pose_total = torch.tensor(0.0, device=device)
        if 'kpt' in pose_l:
            pose_total = pose_l['kpt'] + pose_l.get('kobj', 0.0)
        if self.use_uncertainty_weighting and self.training:
            total = torch.tensor(0.0, device=device)
            det_log_var = self._bounded_log_var(
                self.log_var_det,
                self.det_uncertainty_weight_min,
                self.det_uncertainty_weight_max,
            )
            pose_log_var = self._bounded_log_var(
                self.log_var_pose,
                self.pose_uncertainty_weight_min,
                self.pose_uncertainty_weight_max,
            )
            if self.train_det:
                total = total + torch.exp(-det_log_var) * det_total + det_log_var
            if self.train_pose:
                total = total + torch.exp(-pose_log_var) * pose_total + pose_log_var
        else:
            det_log_var = self.log_var_det
            pose_log_var = self.log_var_pose
            total = (self.det_task_weight * det_total +
                     self.pose_task_weight * pose_total)

        loss_dict = {
            'total': total,
            'det_total': det_total.detach(),
            'pose_total': pose_total.detach(),
            'task_w_det': torch.tensor(self.det_task_weight, device=device),
            'task_w_pose': torch.tensor(self.pose_task_weight, device=device),
            'log_var_det': self.log_var_det.detach(),
            'log_var_pose': self.log_var_pose.detach(),
            'uncertainty_w_det': torch.exp(-det_log_var.detach()),
            'uncertainty_w_pose': torch.exp(-pose_log_var.detach()),
            'det_cls': det_cls.detach(),
            'det_ciou': det_ciou.detach(),
            'det_dfl': det_dfl.detach(),
        }
        if 'kpt' in pose_l:
            loss_dict['pose_kpt'] = pose_l['kpt'].detach()
        if 'kobj' in pose_l:
            loss_dict['pose_kobj'] = pose_l['kobj'].detach()

        return loss_dict


# ═══════════════════════════════════════════════════════════════
# Individual models
# ═══════════════════════════════════════════════════════════════

class ModelA_DualHead(_DualHeadModel):
    """M-A: Standard dual-head."""

    def __init__(self, num_det_classes=80, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.neck = FPNPANNeck(self.backbone.out_channels[1:],
                               depth=backbone_depth, width=backbone_width)
        ch = self.neck.out_channels
        self.det_head = DetectHead(ch[0], num_classes=num_det_classes, reg_max=reg_max)
        self.pose_head = PoseHead(ch[0], num_kpts=num_kpts, reg_max=reg_max)
        self.disable_pose_proposal_training()

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.det_head(neck_feats), self.pose_head(neck_feats)

    def forward(self, x):
        return self._forward_head(x)


class ModelB_UnifiedHead(_BaseModel):
    """M-B: Unified head."""

    def __init__(self, num_classes=80, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(reg_max=reg_max)
        self.num_classes = num_classes
        self.num_kpts = num_kpts

        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.neck = FPNPANNeck(self.backbone.out_channels[1:],
                               depth=backbone_depth, width=backbone_width)
        self.head = UnifiedHead(self.neck.out_channels[0],
                                num_classes=num_classes, num_kpts=num_kpts, reg_max=reg_max)

        self.loss_fn = MultiTaskLoss(
            w_box=7.5, w_cls=0.5, w_dfl=1.5,
            w_pose=12.0, w_kobj=1.0, reg_max=reg_max,
            num_det_classes=num_classes, unified_head=True)

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.head(neck_feats)

    def forward(self, x):
        return self._forward_head(x)

    def compute_loss(self, images, gt_dict_list):
        device = next(self.parameters()).device
        images = images.to(device)
        B = images.shape[0]

        head_outs = self._forward_head(images)
        feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs['cls']]

        pred_scores, pred_boxes = self._get_decoded_preds(head_outs, feat_sizes)

        # Collect all GTs
        all_boxes, all_classes, all_kpts, all_batch = [], [], [], []
        for b in range(B):
            gt = gt_dict_list[b]
            boxes = gt['boxes']
            classes = gt['classes']
            kpts = gt.get('kpts', torch.zeros(0, 17, 3))
            if len(boxes) == 0:
                continue

            kpts_full = torch.zeros(len(boxes), 17, 3, dtype=boxes.dtype)
            person_mask = classes == 0
            if len(kpts) == len(boxes) and person_mask.any():
                kpts_full[person_mask] = kpts[person_mask]

            all_boxes.append(boxes.to(device, non_blocking=True))
            all_classes.append(classes.to(device, non_blocking=True))
            all_kpts.append(kpts_full.to(device, non_blocking=True))
            all_batch.append(torch.full((len(boxes),), b, device=device, dtype=torch.long))

        if all_boxes:
            gt_boxes = torch.cat(all_boxes, dim=0)
            gt_classes = torch.cat(all_classes, dim=0).long()
            gt_kpts = torch.cat(all_kpts, dim=0)
            gt_batch = torch.cat(all_batch, dim=0)
        else:
            gt_boxes = torch.empty(0, 4, device=device)
            gt_classes = torch.empty(0, device=device, dtype=torch.long)
            gt_kpts = torch.empty(0, 17, 3, device=device)
            gt_batch = torch.empty(0, device=device, dtype=torch.long)

        targets = self.assigner(
            pred_scores, pred_boxes, gt_boxes, gt_classes,
            gt_kpts, feat_sizes, self.strides, gt_batch,
            num_det_classes=self.num_classes)

        losses = self.loss_fn(head_outs, targets, self.strides, feat_sizes, head_type='unified')

        det_total = losses['cls'] + losses['ciou'] + losses['dfl']
        pose_total = torch.tensor(0.0, device=device)
        if 'kpt' in losses:
            pose_total = losses['kpt'] + losses.get('kobj', 0.0)
        total = (self.det_task_weight * det_total +
                 self.pose_task_weight * pose_total)

        result = {
            'total': total,
            'det_total': det_total.detach(),
            'pose_total': pose_total.detach(),
            'task_w_det': torch.tensor(self.det_task_weight, device=device),
            'task_w_pose': torch.tensor(self.pose_task_weight, device=device),
        }
        result.update({k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in losses.items()})
        return result


class ModelC_DualNeck(_DualHeadModel):
    """M-C: Dual-neck dual-head."""

    def __init__(self, num_det_classes=80, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.det_neck = DetNeck(self.backbone.out_channels[1:], scale=0.6)
        self.pose_neck = PoseNeck(self.backbone.out_channels, scale=0.4)
        self.det_head = DetectHead(self.det_neck.out_channels[0],
                                   num_classes=num_det_classes, reg_max=reg_max)
        self.pose_head = PoseHead(self.pose_neck.out_channels[0],
                                   num_kpts=num_kpts, reg_max=reg_max)
        self.disable_pose_proposal_training()

    def _forward_head(self, x):
        p2, p3, p4, p5 = self.backbone(x)
        det_feats = self.det_neck([p3, p4, p5])
        pose_feats = self.pose_neck([p2, p3, p4, p5])
        return self.det_head(det_feats), self.pose_head(pose_feats)

    def forward(self, x):
        return self._forward_head(x)


class ModelD_AttentionDual(_DualHeadModel):
    """M-D: ECA backbone + dual-head."""

    def __init__(self, num_det_classes=80, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width, use_eca=True)
        self.neck = FPNPANNeck(self.backbone.out_channels[1:],
                               depth=backbone_depth, width=backbone_width)
        ch = self.neck.out_channels
        self.det_head = DetectHead(ch[0], num_classes=num_det_classes, reg_max=reg_max)
        self.pose_head = PoseHead(ch[0], num_kpts=num_kpts, reg_max=reg_max)
        self.disable_pose_proposal_training()

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.det_head(neck_feats), self.pose_head(neck_feats)

    def forward(self, x):
        return self._forward_head(x)


class ModelE_BiFPN(_DualHeadModel):
    """M-E: BiFPN neck + dual-head."""

    def __init__(self, num_det_classes=80, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.neck = BiFPN(self.backbone.out_channels,
                          depth=backbone_depth, width=backbone_width)
        ch = self.neck.out_channels
        self.det_adapter = nn.ModuleList(Conv(c, c, 1) for c in ch)
        self.pose_adapter = nn.ModuleList(Conv(c, c, 1) for c in ch)
        self.det_head = DetectHead(
            ch[0], num_classes=num_det_classes, reg_max=reg_max, tower_depth=3)
        self.pose_head = PoseHead(
            ch[0], num_kpts=num_kpts, reg_max=reg_max, tower_depth=3)
        self.disable_pose_proposal_training()

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        det_feats = [adapter(feat) for adapter, feat in zip(self.det_adapter, neck_feats)]
        pose_feats = [adapter(feat) for adapter, feat in zip(self.pose_adapter, neck_feats)]
        return self.det_head(det_feats), self.pose_head(pose_feats)

    def forward(self, x):
        return self._forward_head(x)


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

MODEL_FACTORY = {
    'dual_head': ModelA_DualHead,
    'unified_head': ModelB_UnifiedHead,
    'dual_neck': ModelC_DualNeck,
    'attn_dual': ModelD_AttentionDual,
    'bifpn_dual': ModelE_BiFPN,
}


def create_model(name, **kwargs):
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_FACTORY.keys())}")
    model = MODEL_FACTORY[name](**kwargs)
    print(f"  Created {name}: {model.num_params / 1e6:.2f}M params")
    return model
