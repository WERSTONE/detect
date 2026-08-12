"""YOLOv8-style single-head pose models."""

import torch
import torch.nn as nn

from test_model.final.model.backbone import CSPDarkNet
from test_model.final.model.bifpn import _batched_nms, _dfl_decode, _make_grid
from test_model.final.model.common import Conv
from test_model.final.model.head import YOLOLikeDetectHead
from test_model.final.model.loss import YOLOPoseLoss
from test_model.final.model.neck import BiFPN


class YOLOLikePoseHead(nn.Module):
    """YOLOv8 pose head: person detect branches plus keypoint branches."""

    def __init__(self, channels, num_kpts=17, reg_max=16, strides=(8, 16, 32),
                 img_size=640):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels] * len(strides)
        self.channels = list(channels)
        self.num_kpts = int(num_kpts)
        self.kpt_dim = self.num_kpts * 3
        self.detect = YOLOLikeDetectHead(
            self.channels,
            num_classes=1,
            reg_max=reg_max,
            strides=strides,
            img_size=img_size,
        )
        c4 = max(self.channels[0] // 4, self.kpt_dim)
        self.kpt_branches = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c4, 3),
                Conv(c4, c4, 3),
                nn.Conv2d(c4, self.kpt_dim, 1),
            )
            for ch in self.channels
        )
        self._init_kpt_bias()

    def _init_kpt_bias(self):
        for branch in self.kpt_branches:
            nn.init.normal_(branch[-1].weight, 0.0, 0.01)
            nn.init.constant_(branch[-1].bias, 0.0)

    def forward(self, features):
        outs = self.detect(features)
        outs['kpt'] = [branch(feat) for branch, feat in zip(self.kpt_branches, features)]
        return outs


class _SinglePoseModel(nn.Module):
    """Shared single-head pose training and COCOeval prediction interface."""

    def __init__(self, num_kpts=17, reg_max=16, strides=(8, 16, 32),
                 input_size=640, assigner_topk=10, assigner_alpha=0.5,
                 assigner_beta=6.0, assigner_eps=1.0e-9):
        super().__init__()
        self.num_kpts = int(num_kpts)
        self.reg_max = int(reg_max)
        self.strides = list(strides)
        self.input_size = int(input_size)
        self.pose_loss = YOLOPoseLoss(
            num_kpts=self.num_kpts,
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
        raise NotImplementedError

    def forward(self, x):
        return self._forward_head(x)

    def compute_loss(self, images, gt_dict_list):
        images = images.to(next(self.parameters()).device)
        return self.pose_loss(self._forward_head(images), gt_dict_list)

    @staticmethod
    def _decode_kpts(kpt_raw, anchor_idx, grid, stride):
        raw = kpt_raw[anchor_idx].view(-1, 17, 3)
        anchor_points = grid.view(-1, 1, 2)[anchor_idx] + 0.5
        xy = (raw[..., :2] * 2.0 + anchor_points - 0.5) * stride
        conf = raw[..., 2:3].sigmoid()
        return torch.cat((xy, conf), dim=-1)

    @torch.no_grad()
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6, max_det=300):
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device)
        head_outs = self._forward_head(images)
        return self._decode_predictions(
            head_outs['cls'], head_outs['reg'], head_outs['kpt'],
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            max_det=max_det,
        )

    def _decode_predictions(self, cls_list, reg_list, kpt_list,
                            score_thresh=0.01, iou_thresh=0.6,
                            max_det=300, max_nms=30000):
        device = cls_list[0].device
        bsz = cls_list[0].shape[0]
        results = []

        for b in range(bsz):
            all_boxes, all_scores, all_kpts = [], [], []
            for lvl, stride in enumerate(self.strides):
                _, _, h, w = cls_list[lvl].shape
                scores_l = cls_list[lvl][b:b + 1].permute(0, 2, 3, 1).reshape(h * w).sigmoid()
                grid_cells = _make_grid(w, h, device)
                boxes_l = _dfl_decode(reg_list[lvl][b:b + 1], self.reg_max, stride, grid_cells * stride)[0]
                kpt_l = kpt_list[lvl][b:b + 1].permute(0, 2, 3, 1).reshape(h * w, self.num_kpts * 3)

                anchor_idx = (scores_l > score_thresh).nonzero(as_tuple=True)[0]
                if anchor_idx.numel() == 0:
                    continue
                selected_scores = scores_l[anchor_idx]
                if max_nms and selected_scores.numel() > max_nms:
                    topk = selected_scores.argsort(descending=True)[:max_nms]
                    anchor_idx = anchor_idx[topk]
                    selected_scores = selected_scores[topk]
                all_boxes.append(boxes_l[anchor_idx])
                all_scores.append(selected_scores)
                all_kpts.append(self._decode_kpts(kpt_l, anchor_idx, grid_cells, stride))

            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                kpts = torch.cat(all_kpts)
                classes = torch.zeros(len(boxes), dtype=torch.long, device=device)
                keep = _batched_nms(boxes, scores, classes, iou_thresh=iou_thresh, max_det=max_det)
                results.append({
                    'boxes': boxes[keep],
                    'scores': scores[keep],
                    'classes': classes[keep],
                    'kpts': kpts[keep],
                })
            else:
                results.append({
                    'boxes': torch.zeros(0, 4, device=device),
                    'scores': torch.zeros(0, device=device),
                    'classes': torch.zeros(0, dtype=torch.long, device=device),
                    'kpts': torch.zeros(0, self.num_kpts, 3, device=device),
                })
        return results


class BifpnPoseModel(_SinglePoseModel):
    """BiFPN neck with a YOLOv8-style single pose head."""

    def __init__(self, num_kpts=17, reg_max=16, input_size=640,
                 backbone_depth=0.67, backbone_width=0.75,
                 neck_use_p2_context=False, neck_downsample='conv',
                 neck_out_channels=None,
                 assigner_topk=10, assigner_alpha=0.5, assigner_beta=6.0,
                 assigner_eps=1.0e-9):
        super().__init__(
            num_kpts=num_kpts,
            reg_max=reg_max,
            input_size=input_size,
            assigner_topk=assigner_topk,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            assigner_eps=assigner_eps,
        )
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
        self.pose_adapter = nn.ModuleList(Conv(c, c, 1) for c in ch)
        self.pose_head = YOLOLikePoseHead(
            ch,
            num_kpts=self.num_kpts,
            reg_max=self.reg_max,
            strides=self.strides,
            img_size=self.input_size,
        )

    def _forward_head(self, x):
        feats = self.neck(self.backbone(x))
        feats = [adapter(feat) for adapter, feat in zip(self.pose_adapter, feats)]
        return self.pose_head(feats)
