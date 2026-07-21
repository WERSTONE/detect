"""Pure COCO detection model with YOLOv8 backbone and BiFPN neck."""

import torch
import torch.nn as nn

from test_model.model.backbone import CSPDarkNet
from test_model.model.bifpn import _batched_nms, _dfl_decode, _make_grid
from test_model.model.common import Conv
from test_model.model.head import YOLOLikeDetectHead
from test_model.model.loss import YOLODetectionLoss
from test_model.model.neck import BiFPN


class BifpnDetectModel(nn.Module):
    """Detection-only model used to isolate BiFPN detection accuracy."""

    def __init__(self, num_det_classes=80, reg_max=16, strides=(8, 16, 32),
                 backbone_depth=0.67, backbone_width=0.75, input_size=640,
                 neck_use_p2_context=False, neck_downsample='conv',
                 neck_out_channels=None,
                 assigner_topk=10, assigner_alpha=0.5, assigner_beta=6.0,
                 assigner_eps=1.0e-9):
        super().__init__()
        self.num_det_classes = int(num_det_classes)
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
        self.det_head = YOLOLikeDetectHead(
            ch,
            num_classes=self.num_det_classes,
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

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _forward_head(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        det_feats = [adapter(feat) for adapter, feat in zip(self.det_adapter, neck_feats)]
        return self.det_head(det_feats)

    def forward(self, x):
        return self._forward_head(x)

    def compute_loss(self, images, gt_dict_list):
        images = images.to(next(self.parameters()).device)
        return self.det_loss(self._forward_head(images), gt_dict_list)

    @torch.no_grad()
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6, max_det=300):
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device)
        head_outs = self._forward_head(images)
        return self._decode_predictions(
            head_outs['cls'], head_outs['reg'],
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            max_det=max_det,
        )

    def _decode_predictions(self, cls_list, reg_list,
                            score_thresh=0.01, iou_thresh=0.6,
                            max_det=300, max_nms=30000):
        device = cls_list[0].device
        bsz = cls_list[0].shape[0]
        num_cls = cls_list[0].shape[1]
        results = []

        for b in range(bsz):
            all_boxes, all_scores, all_cls = [], [], []
            for lvl, stride in enumerate(self.strides):
                _, _, h, w = cls_list[lvl].shape
                cls_l = cls_list[lvl][b:b + 1].permute(0, 2, 3, 1).reshape(h * w, num_cls)
                scores_l = cls_l.sigmoid()
                grid = _make_grid(w, h, device) * stride
                boxes_l = _dfl_decode(reg_list[lvl][b:b + 1], self.reg_max, stride, grid)[0]

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
                all_cls.append(cls_idx.long())

            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                classes = torch.cat(all_cls)
                keep = _batched_nms(boxes, scores, classes, iou_thresh=iou_thresh, max_det=max_det)
                results.append({
                    'boxes': boxes[keep],
                    'scores': scores[keep],
                    'classes': classes[keep],
                    'kpts': torch.zeros((keep.numel(), 17, 3), device=device),
                })
            else:
                results.append({
                    'boxes': torch.zeros(0, 4, device=device),
                    'scores': torch.zeros(0, device=device),
                    'classes': torch.zeros(0, dtype=torch.long, device=device),
                    'kpts': torch.zeros(0, 17, 3, device=device),
                })
        return results
