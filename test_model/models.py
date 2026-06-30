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

import torch
import torch.nn as nn

from test_model.backbone import CSPDarkNet
from test_model.neck import FPNPANNeck, BiFPN, DetNeck, PoseNeck
from test_model.heads import DetectHead, PoseHead, UnifiedHead
from test_model.assigner import TaskAlignedAssigner
from test_model.loss import MultiTaskLoss


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
        # Loss function set in subclass init

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _forward_backbone_neck(self, x):
        raise NotImplementedError

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
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6):
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
                                        score_thresh, iou_thresh)

    def _decode_predictions(self, cls_list, reg_list, kpt_list,
                            score_thresh=0.01, iou_thresh=0.6, cls_offset=0):
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
            kpt_outs = []

            for lvl, stride in enumerate(self.strides):
                _, _, H, W = cls_list[lvl].shape
                cls_l = cls_list[lvl][b:b+1].permute(0, 2, 3, 1).reshape(H * W, num_cls)
                scores_l = cls_l.sigmoid()

                grid = _make_grid(W, H, device) * stride
                reg_l = reg_list[lvl][b:b+1]
                boxes_l = _dfl_decode(reg_l, self.reg_max, stride, grid)[0]

                # Per-class threshold + NMS
                for c in range(num_cls):
                    sc = scores_l[:, c]
                    keep_mask = sc > score_thresh
                    if keep_mask.any():
                        c_boxes = boxes_l[keep_mask]
                        c_scores = sc[keep_mask]
                        nms_k = _nms(c_boxes, c_scores, iou_thresh)
                        if nms_k.numel() > 0:
                            all_boxes.append(c_boxes[nms_k])
                            all_scores.append(c_scores[nms_k])
                            all_cls.append(torch.full((len(nms_k),), c + cls_offset, device=device, dtype=torch.long))

                            if kpt_list is not None and c == 0:
                                keep_indices = keep_mask.nonzero(as_tuple=True)[0]
                                final_indices = keep_indices[nms_k]
                                kpt_l = kpt_list[lvl][b:b+1].permute(0, 2, 3, 1).reshape(H * W, 17, 3)
                                kpt_selected = kpt_l[final_indices]
                                grid_center = grid.view(H * W, 1, 2) + 0.5 * stride
                                kpt_xy = kpt_selected[..., :2] * stride + grid_center[final_indices]
                                kpt_vis = kpt_selected[..., 2:3].sigmoid()
                                kpt_outs.append(torch.cat([kpt_xy, kpt_vis], dim=-1))

            if all_boxes:
                result = {
                    'boxes': torch.cat(all_boxes),
                    'scores': torch.cat(all_scores),
                    'classes': torch.cat(all_cls),
                }
                if kpt_outs:
                    result['kpts'] = torch.cat(kpt_outs)
                else:
                    result['kpts'] = torch.zeros(0, 17, 3, device=device)
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

    def __init__(self, num_det_classes=19, num_kpts=17, reg_max=16):
        super().__init__(reg_max=reg_max)
        self.num_det_classes = num_det_classes
        self.num_kpts = num_kpts

        self.det_loss = MultiTaskLoss(
            w_box=7.5, w_cls=0.5, w_dfl=1.5,
            w_pose=0.0, w_kobj=0.0, reg_max=reg_max,
            num_det_classes=num_det_classes, unified_head=False)
        self.pose_loss = MultiTaskLoss(
            w_box=7.5, w_cls=0.5, w_dfl=1.5,
            w_pose=12.0, w_kobj=1.0, reg_max=reg_max,
            num_det_classes=1, unified_head=False)

    @torch.no_grad()
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6):
        """Merge det and pose head predictions."""
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device)
        B = images.shape[0]

        det_out, pose_out = self._forward_head(images)

        # Decode detection head (classes 1..19 → shifted to 1..19)
        det_results = self._decode_predictions(
            det_out['cls'], det_out['reg'], None, score_thresh, iou_thresh,
            cls_offset=1)  # det classes are 0..18 internally, output as 1..19

        # Decode pose head (class 0 = person)
        pose_results = self._decode_predictions(
            pose_out['cls'], pose_out['reg'], pose_out.get('kpt'),
            score_thresh, iou_thresh, cls_offset=0)

        # Merge
        merged = []
        for b in range(B):
            d = det_results[b]
            p = pose_results[b]
            merged.append({
                'boxes': torch.cat([p['boxes'], d['boxes']]) if p['boxes'].numel() + d['boxes'].numel() > 0 else torch.zeros(0, 4, device=device),
                'scores': torch.cat([p['scores'], d['scores']]) if p['scores'].numel() + d['scores'].numel() > 0 else torch.zeros(0, device=device),
                'classes': torch.cat([p['classes'], d['classes']]) if p['classes'].numel() + d['classes'].numel() > 0 else torch.zeros(0, dtype=torch.long, device=device),
                'kpts': p.get('kpts', torch.zeros(0, 17, 3, device=device)),
            })
        return merged

    def compute_loss(self, images, gt_dict_list):
        device = next(self.parameters()).device
        images = images.to(device)
        B = images.shape[0]

        det_out, pose_out = self._forward_head(images)

        feat_sizes = [(t.shape[2], t.shape[3]) for t in det_out['cls']]

        # Decode for assigner
        det_scores, det_boxes = self._get_decoded_preds(det_out, feat_sizes)

        pose_scores, pose_boxes = [], []
        for lvl in range(len(feat_sizes)):
            # Pose head has only 1 class
            cls_p = pose_out['cls'][lvl].permute(0, 2, 3, 1).reshape(B, -1, 1)
            pose_scores.append(cls_p.sigmoid())
            H, W = feat_sizes[lvl]
            grid = _make_grid(W, H, device) * self.strides[lvl]
            pose_boxes.append(_dfl_decode(pose_out['reg'][lvl], self.reg_max, self.strides[lvl], grid))

        # Separate GTs by head
        det_boxes_list = []
        det_cls_list = []   # 0..18 (shifted)
        det_batch_list = []
        pose_boxes_list = []
        pose_cls_list = []  # always 0
        pose_kpts_list = []
        pose_batch_list = []

        for b in range(B):
            gt = gt_dict_list[b]
            boxes = gt['boxes'].to(device)
            classes = gt['classes'].to(device)
            kpts = gt.get('kpts', torch.zeros(0, 17, 3)).to(device)

            for i in range(len(boxes)):
                cls = classes[i].item()
                if cls == 0:  # person
                    pose_boxes_list.append(boxes[i])
                    pose_cls_list.append(0)
                    if i < len(kpts):
                        pose_kpts_list.append(kpts[i])
                    else:
                        pose_kpts_list.append(torch.zeros(17, 3, device=device))
                    pose_batch_list.append(b)
                else:  # detection classes
                    det_boxes_list.append(boxes[i])
                    det_cls_list.append(cls - 1)  # shift 1..19 -> 0..18
                    det_batch_list.append(b)

        # Build GT tensors
        if det_boxes_list:
            det_gt_boxes = torch.stack(det_boxes_list)
            det_gt_classes = torch.tensor(det_cls_list, device=device, dtype=torch.long)
            det_gt_batch = torch.tensor(det_batch_list, device=device, dtype=torch.long)
        else:
            det_gt_boxes = torch.empty(0, 4, device=device)
            det_gt_classes = torch.empty(0, device=device, dtype=torch.long)
            det_gt_batch = torch.empty(0, device=device, dtype=torch.long)

        if pose_boxes_list:
            pose_gt_boxes = torch.stack(pose_boxes_list)
            pose_gt_classes = torch.zeros(len(pose_boxes_list), device=device, dtype=torch.long)
            pose_gt_kpts = torch.stack(pose_kpts_list)
            pose_gt_batch = torch.tensor(pose_batch_list, device=device, dtype=torch.long)
        else:
            pose_gt_boxes = torch.empty(0, 4, device=device)
            pose_gt_classes = torch.empty(0, device=device, dtype=torch.long)
            pose_gt_kpts = torch.empty(0, 17, 3, device=device)
            pose_gt_batch = torch.empty(0, device=device, dtype=torch.long)

        # Assigner
        det_targets = self.assigner(
            det_scores, det_boxes, det_gt_boxes, det_gt_classes,
            None, feat_sizes, self.strides, det_gt_batch,
            num_det_classes=self.num_det_classes)

        pose_targets = self.assigner(
            pose_scores, pose_boxes, pose_gt_boxes, pose_gt_classes,
            pose_gt_kpts, feat_sizes, self.strides, pose_gt_batch,
            num_det_classes=1)

        # Loss. Use one shared assigned-positive normalizer for det/pose box,
        # cls, and DFL terms so dual-head variants stay comparable with
        # unified_head, whose loss sees all classes in one assigner.
        def _count_pos(targets):
            return sum(len(t['gt_boxes']) for t in targets if t is not None)

        shared_pos_norm = max(_count_pos(det_targets) + _count_pos(pose_targets), 1)
        pose_pos_norm = max(_count_pos(pose_targets), 1)
        det_l = self.det_loss(
            det_out, det_targets, self.strides, feat_sizes,
            head_type='det', norm_pos=shared_pos_norm)
        pose_l = self.pose_loss(
            pose_out, pose_targets, self.strides, feat_sizes,
            head_type='pose', norm_pos=shared_pos_norm, kpt_norm=pose_pos_norm)

        total = det_l['cls'] + det_l['ciou'] + det_l['dfl']
        if 'kpt' in pose_l:
            total = total + pose_l['kpt'] + pose_l.get('kobj', 0.0)
        total = total + pose_l['cls'] + pose_l['ciou'] + pose_l['dfl']

        loss_dict = {
            'total': total,
            'det_cls': det_l['cls'].item(),
            'det_ciou': det_l['ciou'].item(),
            'det_dfl': det_l['dfl'].item(),
            'pose_cls': pose_l['cls'].item(),
            'pose_ciou': pose_l['ciou'].item(),
            'pose_dfl': pose_l['dfl'].item(),
        }
        if 'kpt' in pose_l:
            loss_dict['pose_kpt'] = pose_l['kpt'].item()
        if 'kobj' in pose_l:
            loss_dict['pose_kobj'] = pose_l['kobj'].item()

        return loss_dict


# ═══════════════════════════════════════════════════════════════
# Individual models
# ═══════════════════════════════════════════════════════════════

class ModelA_DualHead(_DualHeadModel):
    """M-A: Standard dual-head."""

    def __init__(self, num_det_classes=19, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.neck = FPNPANNeck(self.backbone.out_channels[1:],
                               depth=backbone_depth, width=backbone_width)
        ch = self.neck.out_channels
        self.det_head = DetectHead(ch[0], num_classes=num_det_classes, reg_max=reg_max)
        self.pose_head = PoseHead(ch[0], num_kpts=num_kpts, reg_max=reg_max)

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.det_head(neck_feats), self.pose_head(neck_feats)

    def forward(self, x):
        return self._forward_head(x)


class ModelB_UnifiedHead(_BaseModel):
    """M-B: Unified head."""

    def __init__(self, num_classes=20, num_kpts=17, reg_max=16,
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
            boxes = gt['boxes'].to(device)
            classes = gt['classes'].to(device)
            kpts = gt.get('kpts', torch.zeros(0, 17, 3)).to(device)

            for i in range(len(boxes)):
                all_boxes.append(boxes[i])
                all_classes.append(classes[i].item())
                if classes[i] == 0 and i < len(kpts):
                    all_kpts.append(kpts[i])
                else:
                    all_kpts.append(torch.zeros(17, 3, device=device))
                all_batch.append(b)

        if all_boxes:
            gt_boxes = torch.stack(all_boxes)
            gt_classes = torch.tensor(all_classes, device=device, dtype=torch.long)
            gt_kpts = torch.stack(all_kpts)
            gt_batch = torch.tensor(all_batch, device=device, dtype=torch.long)
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

        total = losses['cls'] + losses['ciou'] + losses['dfl']
        if 'kpt' in losses:
            total = total + losses['kpt'] + losses.get('kobj', 0.0)

        result = {'total': total}
        result.update({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()})
        return result


class ModelC_DualNeck(_DualHeadModel):
    """M-C: Dual-neck dual-head."""

    def __init__(self, num_det_classes=19, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.det_neck = DetNeck(self.backbone.out_channels[1:], scale=0.6)
        self.pose_neck = PoseNeck(self.backbone.out_channels, scale=0.4)
        self.det_head = DetectHead(self.det_neck.out_channels[0],
                                   num_classes=num_det_classes, reg_max=reg_max)
        self.pose_head = PoseHead(self.pose_neck.out_channels[0],
                                   num_kpts=num_kpts, reg_max=reg_max)

    def _forward_head(self, x):
        p2, p3, p4, p5 = self.backbone(x)
        det_feats = self.det_neck([p3, p4, p5])
        pose_feats = self.pose_neck([p2, p3, p4, p5])
        return self.det_head(det_feats), self.pose_head(pose_feats)

    def forward(self, x):
        return self._forward_head(x)


class ModelD_AttentionDual(_DualHeadModel):
    """M-D: ECA backbone + dual-head."""

    def __init__(self, num_det_classes=19, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width, use_eca=True)
        self.neck = FPNPANNeck(self.backbone.out_channels[1:],
                               depth=backbone_depth, width=backbone_width)
        ch = self.neck.out_channels
        self.det_head = DetectHead(ch[0], num_classes=num_det_classes, reg_max=reg_max)
        self.pose_head = PoseHead(ch[0], num_kpts=num_kpts, reg_max=reg_max)

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.det_head(neck_feats), self.pose_head(neck_feats)

    def forward(self, x):
        return self._forward_head(x)


class ModelE_BiFPN(_DualHeadModel):
    """M-E: BiFPN neck + dual-head."""

    def __init__(self, num_det_classes=19, num_kpts=17, reg_max=16,
                 backbone_depth=0.67, backbone_width=0.75):
        super().__init__(num_det_classes, num_kpts, reg_max)
        self.backbone = CSPDarkNet(depth=backbone_depth, width=backbone_width)
        self.neck = BiFPN(self.backbone.out_channels[1:],
                          depth=backbone_depth, width=backbone_width)
        ch = self.neck.out_channels
        self.det_head = DetectHead(ch[0], num_classes=num_det_classes, reg_max=reg_max)
        self.pose_head = PoseHead(ch[0], num_kpts=num_kpts, reg_max=reg_max)

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.det_head(neck_feats), self.pose_head(neck_feats)

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
