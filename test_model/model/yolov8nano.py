"""YOLOv8n detection model replica for COCO80 experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from test_model.model.backbone import CSPDarkNet
from test_model.model.bifpn import _batched_nms, _dfl_decode, _make_grid
from test_model.model.common import C2f, Conv
from test_model.model.head import YOLOLikeDetectHead
from test_model.model.loss import YOLODetectionLoss


class YOLOv8PAN(nn.Module):
    """YOLOv8 PAN-FPN neck for P3/P4/P5 detection features."""

    def __init__(self, backbone_channels, depth=0.33, width=0.25, max_ch=1024):
        super().__init__()

        def ch(x):
            return max(8, int((min(x, max_ch) * width + 7) // 8 * 8))

        def n_blocks(x):
            return max(1, int(round(x * depth)))

        c3, c4, c5 = backbone_channels[-3:]
        n = n_blocks(3)
        p3_out = ch(256)
        p4_out = ch(512)
        p5_out = ch(1024)

        self.p4_fuse = C2f(c5 + c4, p4_out, n=n, shortcut=False)
        self.p3_fuse = C2f(p4_out + c3, p3_out, n=n, shortcut=False)

        self.p3_down = Conv(p3_out, p3_out, 3, 2)
        self.p4_out = C2f(p3_out + p4_out, p4_out, n=n, shortcut=False)

        self.p4_down = Conv(p4_out, p4_out, 3, 2)
        self.p5_out = C2f(p4_out + c5, p5_out, n=n, shortcut=False)

        self.out_channels = [p3_out, p4_out, p5_out]

    def forward(self, feats):
        _, p3, p4, p5 = feats

        p5_up = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4_td = self.p4_fuse(torch.cat((p5_up, p4), dim=1))

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode="nearest")
        p3_out = self.p3_fuse(torch.cat((p4_up, p3), dim=1))

        p3_down = self.p3_down(p3_out)
        p4_out = self.p4_out(torch.cat((p3_down, p4_td), dim=1))

        p4_down = self.p4_down(p4_out)
        p5_out = self.p5_out(torch.cat((p4_down, p5), dim=1))

        return [p3_out, p4_out, p5_out]


class YOLOv8NanoModel(nn.Module):
    """YOLOv8n-style detection model with local loss/eval interfaces."""

    def __init__(self, num_det_classes=80, reg_max=16, strides=(8, 16, 32),
                 input_size=640, assigner_topk=10, assigner_alpha=0.5,
                 assigner_beta=6.0, assigner_eps=1.0e-9):
        super().__init__()
        self.num_det_classes = int(num_det_classes)
        self.reg_max = int(reg_max)
        self.strides = list(strides)
        self.input_size = int(input_size)

        self.backbone = CSPDarkNet(depth=0.33, width=0.25, max_ch=1024)
        self.neck = YOLOv8PAN(self.backbone.out_channels, depth=0.33, width=0.25, max_ch=1024)
        self.det_head = YOLOLikeDetectHead(
            self.neck.out_channels,
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
        return self.det_head(self.neck(self.backbone(x)))

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
            head_outs["cls"], head_outs["reg"],
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
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "classes": classes[keep],
                    "kpts": torch.zeros((keep.numel(), 17, 3), device=device),
                })
            else:
                results.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "classes": torch.zeros(0, dtype=torch.long, device=device),
                    "kpts": torch.zeros(0, 17, 3, device=device),
                })
        return results
