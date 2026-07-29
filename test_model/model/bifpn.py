"""Shared BiFPN dual-head model for COCO non-person detection and person pose."""

import math

import torch
import torch.nn as nn

from test_model.model.backbone import CSPDarkNet
from test_model.model.common import Conv
from test_model.model.head import YOLOLikeDetectHead
from test_model.model.loss import YOLODetectionLoss, YOLOPoseLoss
from test_model.model.neck import BiFPN

try:
    from torchvision.ops import batched_nms
except Exception:  # pragma: no cover - fallback for minimal environments
    batched_nms = None


def _make_grid(nx, ny, device):
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing='ij',
    )
    return torch.stack((xv, yv), 2).float()


def _dfl_decode(reg_pred, reg_max, stride, grid):
    """DFL decode: reg distribution -> xyxy boxes in input-image pixels."""
    bsz, _, h, w = reg_pred.shape
    n = h * w
    reg = reg_pred.view(bsz, 4, reg_max, n).softmax(dim=-2)
    proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
    reg = (reg * proj.view(1, 1, reg_max, 1)).sum(dim=-2) * stride

    g = grid.view(1, n, 2) + 0.5 * stride
    cx = g[..., 0:1].transpose(1, 2)
    cy = g[..., 1:2].transpose(1, 2)
    l, t = reg[:, 0:1], reg[:, 1:2]
    r, b = reg[:, 2:3], reg[:, 3:4]
    return torch.cat([cx - l, cy - t, cx + r, cy + b], dim=1).transpose(1, 2)


def _nms(boxes, scores, iou_thresh=0.6):
    """Small per-class NMS fallback used when torchvision NMS is unavailable."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i.item())
        box_i = boxes[i]
        rest = boxes[order[1:]]
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        lt = torch.max(box_i[:2], rest[:, :2])
        rb = torch.min(box_i[2:], rest[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        iou = inter / (area_i + area_rest - inter + 1e-8)
        order = order[1:][iou <= iou_thresh]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _batched_nms(boxes, scores, classes, iou_thresh=0.6, max_det=300):
    """Class-aware NMS with a fallback if torchvision is unavailable."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    keep = None
    if batched_nms is not None:
        try:
            keep = batched_nms(boxes, scores, classes, iou_thresh)
        except RuntimeError:
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


class ModelE_BiFPN(nn.Module):
    """Dual-head BiFPN model.

    Shared trunk:
        YOLOv8m-style CSPDarkNet backbone + BiFPN neck.

    Heads:
        det_head: 79 non-person COCO classes. Internal class ids 0..78 map
            back to COCO80 ids 1..79 at inference.
        pose_head: person detection (class 0) + COCO17 keypoints.
    """

    def __init__(self, num_det_classes=79, num_kpts=17, reg_max=16,
                 strides=(8, 16, 32), input_size=640,
                 backbone_depth=0.67, backbone_width=0.75,
                 neck_use_p2_context=False, neck_downsample='conv',
                 neck_out_channels=None,
                 assigner_topk=10, assigner_alpha=0.5, assigner_beta=6.0,
                 assigner_eps=1.0e-9):
        super().__init__()
        from test_model.model.pose import YOLOLikePoseHead

        self.num_det_classes = int(num_det_classes)
        if self.num_det_classes != 79:
            print(
                f"[bifpn_dual] Overriding num_det_classes={self.num_det_classes} "
                "to 79 because the detection head excludes person."
            )
            self.num_det_classes = 79
        self.num_classes = 80
        self.num_kpts = int(num_kpts)
        self.reg_max = int(reg_max)
        self.strides = list(strides)
        self.input_size = int(input_size)

        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.neck = BiFPN(
            self.backbone.out_channels,
            depth=backbone_depth,
            width=backbone_width,
            use_p2_context=neck_use_p2_context,
            downsample=neck_downsample,
            out_channels=neck_out_channels,
        )
        ch = self.neck.out_channels
        self.det_adapter = nn.ModuleList(Conv(c, c, 1) for c in ch)
        self.pose_adapter = nn.ModuleList(Conv(c, c, 1) for c in ch)
        self.det_head = YOLOLikeDetectHead(
            ch,
            num_classes=self.num_det_classes,
            reg_max=self.reg_max,
            strides=self.strides,
            img_size=self.input_size,
        )
        self.pose_head = YOLOLikePoseHead(
            ch,
            num_kpts=self.num_kpts,
            reg_max=self.reg_max,
            strides=self.strides,
            img_size=self.input_size,
        )

        self.det_loss = YOLODetectionLoss(
            num_classes=self.num_det_classes,
            reg_max=self.reg_max,
            strides=self.strides,
            assigner_topk=assigner_topk,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            assigner_eps=assigner_eps,
        )
        self.pose_loss = YOLOPoseLoss(
            num_kpts=self.num_kpts,
            reg_max=self.reg_max,
            strides=self.strides,
            assigner_topk=assigner_topk,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            assigner_eps=assigner_eps,
        )

        self.train_det = True
        self.train_pose = True
        self.det_weight_mult = 1.0
        self.det_weight_warmup_epochs = 0
        self.det_task_weight = 1.0
        self.pose_task_weight = 1.0
        self.use_uncertainty_weighting = False
        self.log_var_det = nn.Parameter(torch.zeros(()))
        self.log_var_pose = nn.Parameter(torch.zeros(()))
        self.det_uncertainty_weight_min = None
        self.det_uncertainty_weight_max = None
        self.pose_uncertainty_weight_min = None
        self.pose_uncertainty_weight_max = None

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self._forward_head(x)

    def _forward_features(self, x):
        return self.neck(self.backbone(x))

    def _forward_head(self, x):
        return self._forward_selected_heads(x, need_det=True, need_pose=True)

    def _forward_selected_heads(self, x, need_det=True, need_pose=True):
        neck_feats = self._forward_features(x)
        det_out = None
        pose_out = None
        if need_det:
            det_feats = [adapter(feat) for adapter, feat in zip(self.det_adapter, neck_feats)]
            det_out = self.det_head(det_feats)
        if need_pose:
            pose_feats = [adapter(feat) for adapter, feat in zip(self.pose_adapter, neck_feats)]
            pose_out = self.pose_head(pose_feats)
        return det_out, pose_out

    def set_task_weights(self, det_weight=1.0, pose_weight=1.0):
        self.det_task_weight = float(det_weight)
        self.pose_task_weight = float(pose_weight)

    def enable_uncertainty_weighting(self, enabled=True):
        self.use_uncertainty_weighting = bool(enabled)

    def set_uncertainty_weight_bounds(self, det_min=None, det_max=None,
                                      pose_min=None, pose_max=None):
        self.det_uncertainty_weight_min = det_min
        self.det_uncertainty_weight_max = det_max
        self.pose_uncertainty_weight_min = pose_min
        self.pose_uncertainty_weight_max = pose_max

    @staticmethod
    def _log_var_bounds(weight_min=None, weight_max=None):
        min_log_var = None if weight_max is None else -math.log(float(weight_max))
        max_log_var = None if weight_min is None else -math.log(float(weight_min))
        return min_log_var, max_log_var

    def uncertainty_log_var_bounds(self):
        return {
            'log_var_det': self._log_var_bounds(
                self.det_uncertainty_weight_min,
                self.det_uncertainty_weight_max,
            ),
            'log_var_pose': self._log_var_bounds(
                self.pose_uncertainty_weight_min,
                self.pose_uncertainty_weight_max,
            ),
        }

    def clamp_uncertainty_parameters(self):
        if not self.use_uncertainty_weighting:
            return
        bounds = self.uncertainty_log_var_bounds()
        with torch.no_grad():
            for name, param in (
                    ('log_var_det', self.log_var_det),
                    ('log_var_pose', self.log_var_pose)):
                min_log_var, max_log_var = bounds[name]
                if min_log_var is not None or max_log_var is not None:
                    param.clamp_(
                        min=-float('inf') if min_log_var is None else min_log_var,
                        max=float('inf') if max_log_var is None else max_log_var,
                    )

    @staticmethod
    def _bounded_log_var(log_var, weight_min=None, weight_max=None):
        min_log_var, max_log_var = ModelE_BiFPN._log_var_bounds(weight_min, weight_max)
        if min_log_var is None and max_log_var is None:
            return log_var
        if min_log_var is None:
            bounded = torch.clamp(log_var, max=max_log_var)
            return log_var + (bounded - log_var).detach()
        if max_log_var is None:
            bounded = torch.clamp(log_var, min=min_log_var)
            return log_var + (bounded - log_var).detach()
        bounded = torch.clamp(log_var, min=min_log_var, max=max_log_var)
        return log_var + (bounded - log_var).detach()

    def update_det_weight(self, epoch):
        if self.det_weight_warmup_epochs <= 0:
            self.det_weight_mult = 1.0
        else:
            self.det_weight_mult = min(1.0, epoch / self.det_weight_warmup_epochs)

    def freeze_head(self, name):
        head = getattr(self, f'{name}_head')
        for p in head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def disable_pose_proposal_training(self):
        """Compatibility no-op.

        The new dual model needs the pose head's own cls/reg branches because
        that head owns person detection and keypoint localization.
        """
        return

    @staticmethod
    def _filter_non_person_gt(gt_dict_list, device):
        mapped = []
        for gt in gt_dict_list:
            boxes = gt['boxes'].to(device, non_blocking=True).float()
            classes = gt['classes'].to(device, non_blocking=True).long()
            keep = classes > 0
            mapped.append({
                'boxes': boxes[keep],
                'classes': (classes[keep] - 1).long(),
                'kpts': torch.zeros((int(keep.sum().item()), 17, 3), device=device),
            })
        return mapped

    @staticmethod
    def _zero_loss(device):
        zero = torch.zeros((), device=device)
        return {
            'total': zero,
            'det_total': zero.detach(),
            'det_ciou': zero.detach(),
            'det_cls': zero.detach(),
            'det_dfl': zero.detach(),
            'num_pos': zero.detach(),
            'target_scores_sum': zero.detach(),
        }

    def _weighted_total(self, det_raw_total, pose_raw_total, device, batch_size=1):
        batch_size = max(int(batch_size), 1)
        det_raw_total = det_raw_total * (self.det_weight_mult if self.training else 1.0)
        if self.use_uncertainty_weighting and self.training:
            # YOLO losses are multiplied by batch size for optimizer scaling.
            # Learn task uncertainty from the unscaled task loss so the learned
            # weights are not driven down simply because the batch is large,
            # then multiply the objective back to keep model-gradient scale.
            det_loss_scale = det_raw_total / batch_size
            pose_loss_scale = pose_raw_total / batch_size
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
            det_obj = (
                (torch.exp(-det_log_var) * det_loss_scale + det_log_var) *
                batch_size
                if self.train_det else torch.zeros((), device=device))
            pose_obj = (
                (torch.exp(-pose_log_var) * pose_loss_scale + pose_log_var) *
                batch_size
                if self.train_pose else torch.zeros((), device=device))
        else:
            det_log_var = self.log_var_det
            pose_log_var = self.log_var_pose
            det_obj = self.det_task_weight * det_raw_total if self.train_det else torch.zeros((), device=device)
            pose_obj = self.pose_task_weight * pose_raw_total if self.train_pose else torch.zeros((), device=device)
        return det_obj, pose_obj, det_log_var, pose_log_var

    def compute_loss(self, images, gt_dict_list):
        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)
        det_out, pose_out = self._forward_selected_heads(
            images,
            need_det=self.train_det,
            need_pose=self.train_pose,
        )

        if self.train_det:
            det_losses = self.det_loss(det_out, self._filter_non_person_gt(gt_dict_list, device))
        else:
            det_losses = self._zero_loss(device)

        if self.train_pose:
            pose_losses = self.pose_loss(pose_out, gt_dict_list)
        else:
            pose_losses = self._zero_loss(device)
            pose_losses['pose_total'] = torch.zeros((), device=device)
            pose_losses['pose_kpt'] = torch.zeros((), device=device)
            pose_losses['pose_kobj'] = torch.zeros((), device=device)

        det_obj, pose_obj, det_log_var, pose_log_var = self._weighted_total(
            det_losses['total'], pose_losses['total'], device, images.shape[0])
        total = det_obj + pose_obj
        raw_total = det_losses['total'] + pose_losses['total']

        det_scale = self.det_weight_mult if self.training else 1.0
        if self.use_uncertainty_weighting and self.training:
            dyn_w_det = torch.exp(-det_log_var.detach()) if self.train_det else torch.zeros((), device=device)
            dyn_w_pose = torch.exp(-pose_log_var.detach()) if self.train_pose else torch.zeros((), device=device)
        else:
            dyn_w_det = torch.tensor(self.det_task_weight if self.train_det else 0.0, device=device)
            dyn_w_pose = torch.tensor(self.pose_task_weight if self.train_pose else 0.0, device=device)
        loss_dict = {
            'total': total,
            'loss_total': raw_total.detach(),
            '_gp_det_loss': det_obj,
            '_gp_pose_loss': pose_obj,
            'det_total': (det_losses['det_total'] * det_scale).detach(),
            'det_ciou': (det_losses['det_ciou'] * det_scale).detach(),
            'det_cls': (det_losses['det_cls'] * det_scale).detach(),
            'det_dfl': (det_losses['det_dfl'] * det_scale).detach(),
            'det_num_pos': det_losses.get('num_pos', torch.zeros((), device=device)).detach().float(),
            'pose_total': (
                pose_losses.get('det_total', torch.zeros((), device=device)) +
                pose_losses.get('pose_total', torch.zeros((), device=device))
            ).detach(),
            'pose_det_total': pose_losses.get('det_total', torch.zeros((), device=device)).detach(),
            'pose_det_ciou': pose_losses.get('det_ciou', torch.zeros((), device=device)).detach(),
            'pose_det_cls': pose_losses.get('det_cls', torch.zeros((), device=device)).detach(),
            'pose_det_dfl': pose_losses.get('det_dfl', torch.zeros((), device=device)).detach(),
            'pose_kpt_total': pose_losses.get('pose_total', torch.zeros((), device=device)).detach(),
            'pose_kpt': pose_losses.get('pose_kpt', torch.zeros((), device=device)).detach(),
            'pose_kobj': pose_losses.get('pose_kobj', torch.zeros((), device=device)).detach(),
            'pose_num_pos': pose_losses.get('num_pos', torch.zeros((), device=device)).detach().float(),
            'target_scores_sum_det': det_losses.get('target_scores_sum', torch.zeros((), device=device)).detach(),
            'target_scores_sum_pose': pose_losses.get('target_scores_sum', torch.zeros((), device=device)).detach(),
            'num_pos': (
                det_losses.get('num_pos', torch.zeros((), device=device)).detach().float() +
                pose_losses.get('num_pos', torch.zeros((), device=device)).detach().float()
            ),
            'target_scores_sum': (
                det_losses.get('target_scores_sum', torch.zeros((), device=device)).detach() +
                pose_losses.get('target_scores_sum', torch.zeros((), device=device)).detach()
            ),
            'task_w_det': torch.tensor(self.det_task_weight, device=device),
            'task_w_pose': torch.tensor(self.pose_task_weight, device=device),
            'dyn_w_det': dyn_w_det,
            'dyn_w_pose': dyn_w_pose,
            'log_var_det': self.log_var_det.detach(),
            'log_var_pose': self.log_var_pose.detach(),
            'uncertainty_w_det': torch.exp(-det_log_var.detach()),
            'uncertainty_w_pose': torch.exp(-pose_log_var.detach()),
        }
        return loss_dict

    @staticmethod
    def _decode_kpts(kpt_raw, anchor_idx, grid, stride, num_kpts):
        raw = kpt_raw[anchor_idx].view(-1, num_kpts, 3)
        anchor_points = grid.view(-1, 1, 2)[anchor_idx] + 0.5
        xy = (raw[..., :2] * 2.0 + anchor_points - 0.5) * stride
        conf = raw[..., 2:3].sigmoid()
        return torch.cat((xy, conf), dim=-1)

    def _decode_one_head(self, cls_list, reg_list, kpt_list=None, cls_offset=0,
                         score_thresh=0.01, max_nms=30000):
        device = cls_list[0].device
        bsz = cls_list[0].shape[0]
        num_cls = cls_list[0].shape[1]
        decoded = []
        for b in range(bsz):
            all_boxes, all_scores, all_cls, all_kpts = [], [], [], []
            for lvl, stride in enumerate(self.strides):
                _, _, h, w = cls_list[lvl].shape
                cls_l = cls_list[lvl][b:b + 1].permute(0, 2, 3, 1).reshape(h * w, num_cls)
                scores_l = cls_l.sigmoid()
                grid_cells = _make_grid(w, h, device)
                boxes_l = _dfl_decode(reg_list[lvl][b:b + 1], self.reg_max, stride, grid_cells * stride)[0]
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

                all_boxes.append(boxes_l[anchor_idx])
                all_scores.append(selected_scores)
                all_cls.append((cls_idx + cls_offset).long())
                if kpt_list is not None:
                    kpt_l = kpt_list[lvl][b:b + 1].permute(0, 2, 3, 1).reshape(h * w, self.num_kpts * 3)
                    all_kpts.append(self._decode_kpts(kpt_l, anchor_idx, grid_cells, stride, self.num_kpts))
                else:
                    all_kpts.append(torch.zeros((anchor_idx.numel(), self.num_kpts, 3), device=device))

            if all_boxes:
                decoded.append({
                    'boxes': torch.cat(all_boxes),
                    'scores': torch.cat(all_scores),
                    'classes': torch.cat(all_cls),
                    'kpts': torch.cat(all_kpts),
                })
            else:
                decoded.append({
                    'boxes': torch.zeros(0, 4, device=device),
                    'scores': torch.zeros(0, device=device),
                    'classes': torch.zeros(0, dtype=torch.long, device=device),
                    'kpts': torch.zeros(0, self.num_kpts, 3, device=device),
                })
        return decoded

    @torch.no_grad()
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6, max_det=300):
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)
        det_out, pose_out = self._forward_head(images)
        det_preds = self._decode_one_head(
            det_out['cls'], det_out['reg'],
            kpt_list=None,
            cls_offset=1,
            score_thresh=score_thresh,
        )
        pose_preds = self._decode_one_head(
            pose_out['cls'], pose_out['reg'],
            kpt_list=pose_out['kpt'],
            cls_offset=0,
            score_thresh=score_thresh,
        )

        results = []
        for det, pose in zip(det_preds, pose_preds):
            boxes = torch.cat([pose['boxes'], det['boxes']], dim=0)
            scores = torch.cat([pose['scores'], det['scores']], dim=0)
            classes = torch.cat([pose['classes'], det['classes']], dim=0)
            kpts = torch.cat([pose['kpts'], det['kpts']], dim=0)
            if boxes.numel() == 0:
                results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes,
                    'kpts': kpts,
                })
                continue
            keep = _batched_nms(boxes, scores, classes, iou_thresh=iou_thresh, max_det=max_det)
            results.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'classes': classes[keep],
                'kpts': kpts[keep],
            })
        return results
