"""Single-stage domain-detect + pose-attribute model.

This model keeps domain detection separate, but attaches person attributes to
the pose anchors. A person prediction therefore carries box, keypoints, and
attributes from the same feature-map location through NMS.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from test_model.final.model.backbone import CSPDarkNet
from test_model.final.model.bifpn import _batched_nms, _dfl_decode, _make_grid
from test_model.final.model.common import Conv
from test_model.final.model.head import CLS_BIAS_INIT, YOLOLikeDetectHead
from test_model.final.model.loss import (
    KPT_SIGMAS,
    YOLODetectionLoss,
    _bbox2dist,
    _bbox_ciou,
    _dfl_loss_per_anchor,
)
from test_model.final.model.neck import BiFPN


class LocalAppearanceAttrBranch(nn.Module):
    """Local appearance branch for small person attributes."""

    def __init__(self, in_ch, mid_ch, out_ch=2, context_kernel=5):
        super().__init__()
        context_kernel = int(context_kernel)
        if context_kernel < 3 or context_kernel % 2 == 0:
            context_kernel = 5
        self.net = nn.Sequential(
            Conv(in_ch, mid_ch, 3),
            Conv(mid_ch, mid_ch, context_kernel, g=mid_ch),
            Conv(mid_ch, mid_ch, 1),
            nn.Conv2d(mid_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


class YOLOLikePoseAttrHead(nn.Module):
    """YOLOv8 pose head with per-anchor person attributes."""

    def __init__(
        self,
        channels,
        num_kpts=17,
        num_attrs=4,
        reg_max=16,
        strides=(8, 16, 32),
        img_size=640,
        attr_dropout=0.1,
        appearance_context_kernel=5,
    ):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels] * len(strides)
        self.channels = list(channels)
        self.num_kpts = int(num_kpts)
        self.num_attrs = int(num_attrs)
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

        attr_mid = max(self.channels[0] // 2, 96)
        action_mid = max(self.channels[0] // 2, 96)
        self.attr_dropout = nn.Dropout2d(float(attr_dropout)) if attr_dropout > 0 else nn.Identity()
        self.appearance_branches = nn.ModuleList(
            LocalAppearanceAttrBranch(
                ch,
                attr_mid,
                out_ch=2,
                context_kernel=appearance_context_kernel,
            )
            for ch in self.channels
        )
        self.action_branches = nn.ModuleList(
            nn.Sequential(
                Conv(ch + self.kpt_dim, action_mid, 3),
                Conv(action_mid, action_mid, 3),
                nn.Conv2d(action_mid, 2, 1),
            )
            for ch in self.channels
        )
        self._init_extra_bias()

    def _init_extra_bias(self):
        for branch in self.kpt_branches:
            nn.init.normal_(branch[-1].weight, 0.0, 0.01)
            nn.init.constant_(branch[-1].bias, 0.0)
        for branch in self.appearance_branches:
            nn.init.normal_(branch.net[-1].weight, 0.0, 0.01)
            nn.init.constant_(branch.net[-1].bias, CLS_BIAS_INIT)
        for branch in self.action_branches:
            nn.init.normal_(branch[-1].weight, 0.0, 0.01)
            nn.init.constant_(branch[-1].bias, CLS_BIAS_INIT)

    def forward(self, features):
        outs = self.detect(features)
        kpts = [branch(feat) for branch, feat in zip(self.kpt_branches, features)]
        attrs = []
        for feat, kpt, app_branch, action_branch in zip(
            features, kpts, self.appearance_branches, self.action_branches
        ):
            app = app_branch(self.attr_dropout(feat))
            action_feat = torch.cat([feat, kpt.detach()], dim=1)
            action = action_branch(self.attr_dropout(action_feat))
            # Attribute order is fixed: smoking, falling, waving, helmet_on.
            attrs.append(torch.cat([app[:, 0:1], action[:, 0:1], action[:, 1:2], app[:, 1:2]], dim=1))
        outs["kpt"] = kpts
        outs["attr"] = attrs
        return outs


class YOLOPoseAttrLoss(YOLODetectionLoss):
    """Person box/keypoint/attribute loss sharing one pose assignment."""

    def __init__(
        self,
        num_kpts=17,
        num_attrs=4,
        reg_max=16,
        strides=(8, 16, 32),
        w_box=7.5,
        w_cls=0.5,
        w_dfl=1.5,
        w_pose=12.0,
        w_kobj=1.0,
        w_attr=1.0,
        attr_consistency_weight=0.05,
        assigner_topk=10,
        assigner_alpha=0.5,
        assigner_beta=6.0,
        assigner_eps=1e-9,
    ):
        super().__init__(
            num_classes=1,
            reg_max=reg_max,
            strides=strides,
            w_box=w_box,
            w_cls=w_cls,
            w_dfl=w_dfl,
            assigner_topk=assigner_topk,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            assigner_eps=assigner_eps,
        )
        self.num_kpts = int(num_kpts)
        self.num_attrs = int(num_attrs)
        self.w_pose = float(w_pose)
        self.w_kobj = float(w_kobj)
        self.w_attr = float(w_attr)
        self.attr_consistency_weight = float(attr_consistency_weight)
        self.register_buffer("sigmas", KPT_SIGMAS.clone())

    def _preprocess_targets(self, gt_dict_list, batch_size, device):
        max_gt = max(
            (int((gt.get("classes", torch.zeros(0)) == 0).sum().item()) for gt in gt_dict_list),
            default=0,
        )
        gt_labels = torch.zeros(batch_size, max_gt, 1, device=device)
        gt_bboxes = torch.zeros(batch_size, max_gt, 4, device=device)
        gt_kpts = torch.zeros(batch_size, max_gt, self.num_kpts, 3, device=device)
        gt_attrs = torch.zeros(batch_size, max_gt, self.num_attrs, device=device)
        gt_attr_mask = torch.zeros(batch_size, max_gt, self.num_attrs, device=device)
        gt_box_weight = torch.ones(batch_size, max_gt, device=device)
        gt_kpt_weight = torch.ones(batch_size, max_gt, device=device)
        gt_attr_weight = torch.ones(batch_size, max_gt, device=device)
        mask_gt = torch.zeros(batch_size, max_gt, 1, device=device)

        for b, gt in enumerate(gt_dict_list):
            boxes = gt["boxes"].to(device, non_blocking=True).float()
            classes = gt["classes"].to(device, non_blocking=True).long()
            kpts = gt.get(
                "kpts",
                torch.zeros(len(boxes), self.num_kpts, 3, device=boxes.device),
            ).to(device, non_blocking=True).float()
            attrs = gt.get(
                "attrs",
                torch.zeros(len(boxes), self.num_attrs, device=boxes.device),
            ).to(device, non_blocking=True).float()
            attr_mask = gt.get(
                "attr_mask",
                torch.zeros(len(boxes), self.num_attrs, device=boxes.device),
            ).to(device, non_blocking=True).float()

            person = classes == 0
            boxes = boxes[person]
            if len(kpts) == len(classes):
                kpts = kpts[person]
            else:
                kpts = torch.zeros(len(boxes), self.num_kpts, 3, device=device)
            if len(attrs) == len(classes):
                attrs = attrs[person, : self.num_attrs]
                attr_mask = attr_mask[person, : self.num_attrs]
            else:
                attrs = torch.zeros(len(boxes), self.num_attrs, device=device)
                attr_mask = torch.zeros(len(boxes), self.num_attrs, device=device)

            n = min(len(boxes), max_gt)
            if n:
                gt_bboxes[b, :n] = boxes[:n]
                gt_kpts[b, :n] = kpts[:n]
                gt_attrs[b, :n] = attrs[:n]
                gt_attr_mask[b, :n] = attr_mask[:n]
                gt_box_weight[b, :n] = float(gt.get("_person_box_weight", 1.0))
                gt_kpt_weight[b, :n] = float(gt.get("_kpt_weight", 1.0))
                gt_attr_weight[b, :n] = float(gt.get("_attr_weight", 1.0))
                mask_gt[b, :n] = 1.0
        return (
            gt_labels,
            gt_bboxes,
            gt_kpts,
            gt_attrs,
            gt_attr_mask,
            gt_box_weight,
            gt_kpt_weight,
            gt_attr_weight,
            mask_gt,
        )

    @staticmethod
    def _decode_keypoints(pred_kpts, anchor_points, stride_tensor):
        kpts = pred_kpts.view(pred_kpts.shape[0], pred_kpts.shape[1], -1, 3).clone()
        ax = anchor_points[:, 0].view(1, -1, 1)
        ay = anchor_points[:, 1].view(1, -1, 1)
        stride = stride_tensor.squeeze(-1).view(1, -1, 1)
        kpts[..., 0] = (kpts[..., 0] * 2.0 + ax - 0.5) * stride
        kpts[..., 1] = (kpts[..., 1] * 2.0 + ay - 0.5) * stride
        return kpts

    def _attr_consistency_loss(self, pred_attrs, fg_mask, target_gt_idx, target_attr_mask):
        if self.attr_consistency_weight <= 0 or not fg_mask.any():
            return torch.zeros((), device=pred_attrs.device)
        loss_sum = torch.zeros((), device=pred_attrs.device)
        count = torch.zeros((), device=pred_attrs.device)
        bs = pred_attrs.shape[0]
        for b in range(bs):
            pos = fg_mask[b]
            if not pos.any():
                continue
            for gt_idx in target_gt_idx[b, pos].unique():
                group = pos.clone()
                group[pos] = target_gt_idx[b, pos] == gt_idx
                if int(group.sum().item()) < 2:
                    continue
                valid_mask = target_attr_mask[b, group].amax(dim=0) > 0
                if not valid_mask.any():
                    continue
                logits = pred_attrs[b, group][:, valid_mask]
                mean = logits.mean(dim=0, keepdim=True).detach()
                loss_sum = loss_sum + F.mse_loss(logits, mean.expand_as(logits), reduction="sum")
                count = count + logits.numel()
        return loss_sum / count.clamp(min=1.0)

    def forward(self, head_outs, gt_dict_list, attr_pos_weight=None):
        cls_feats = head_outs["cls"]
        reg_feats = head_outs["reg"]
        kpt_feats = head_outs["kpt"]
        attr_feats = head_outs["attr"]
        device = cls_feats[0].device
        dtype = cls_feats[0].dtype
        batch_size = cls_feats[0].shape[0]

        pred_scores = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in cls_feats],
            dim=1,
        )
        pred_distri = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * self.reg_max) for x in reg_feats],
            dim=1,
        )
        pred_kpts = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_kpts * 3) for x in kpt_feats],
            dim=1,
        )
        pred_attrs = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_attrs) for x in attr_feats],
            dim=1,
        )

        anchor_points, stride_tensor = self._make_anchors(cls_feats, device, dtype)
        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)
        (
            gt_labels,
            gt_bboxes,
            gt_kpts,
            gt_attrs,
            gt_attr_mask,
            gt_box_weight,
            gt_kpt_weight,
            gt_attr_weight,
            mask_gt,
        ) = self._preprocess_targets(gt_dict_list, batch_size, device)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        batch_idx = torch.arange(batch_size, device=device)[:, None].expand_as(target_gt_idx)
        if gt_box_weight.shape[1] == 0:
            anchor_shape = target_gt_idx.shape
            anchor_box_weight = torch.zeros(anchor_shape, device=device)
            anchor_kpt_weight = torch.zeros(anchor_shape, device=device)
            anchor_attr_weight = torch.zeros(anchor_shape, device=device)
        else:
            safe_gt_idx = target_gt_idx.clamp(min=0)
            anchor_box_weight = gt_box_weight[batch_idx, safe_gt_idx]
            anchor_kpt_weight = gt_kpt_weight[batch_idx, safe_gt_idx]
            anchor_attr_weight = gt_attr_weight[batch_idx, safe_gt_idx]
        sample_cls_weight = torch.zeros(batch_size, 1, 1, device=device)
        for b, gt in enumerate(gt_dict_list):
            sample_cls_weight[b] = float(gt.get("_person_box_weight", 1.0))

        weighted_scores = target_scores * anchor_box_weight.unsqueeze(-1)
        target_scores_sum = torch.clamp(weighted_scores.sum(), min=1.0)
        cls_raw = self.bce(pred_scores.float(), target_scores.to(dtype).float())
        loss_cls = (cls_raw * sample_cls_weight).sum() / target_scores_sum

        loss_box = torch.zeros((), device=device)
        loss_dfl = torch.zeros((), device=device)
        loss_kpt = torch.zeros((), device=device)
        loss_kobj = torch.zeros((), device=device)
        loss_attr = torch.zeros((), device=device)
        loss_attr_cons = torch.zeros((), device=device)
        attr_count = torch.zeros((), device=device)
        per_attr_loss = torch.zeros(self.num_attrs, device=device)

        if fg_mask.sum():
            weight = weighted_scores.sum(-1)[fg_mask].unsqueeze(-1)
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

            decoded_kpts = self._decode_keypoints(pred_kpts, anchor_points, stride_tensor)
            target_kpts = gt_kpts[batch_idx, target_gt_idx.clamp(min=0)]
            pred_pos_kpts = decoded_kpts[fg_mask]
            target_pos_kpts = target_kpts[fg_mask]
            target_pos_boxes = target_bboxes[fg_mask]
            kpt_mask = target_pos_kpts[..., 2] > 0
            pos_kpt_weight = anchor_kpt_weight[fg_mask]

            if kpt_mask.any() and (pos_kpt_weight > 0).any():
                area = (
                    (target_pos_boxes[:, 2] - target_pos_boxes[:, 0])
                    * (target_pos_boxes[:, 3] - target_pos_boxes[:, 1])
                ).clamp(min=1.0)
                sigmas = self.sigmas[: self.num_kpts].to(device).view(1, -1)
                d = (pred_pos_kpts[..., :2] - target_pos_kpts[..., :2]).pow(2).sum(-1)
                e = d / (((2 * sigmas).pow(2) * area[:, None] * 2.0) + 1e-9)
                loss_factor = self.num_kpts / kpt_mask.sum(1).clamp(min=1)
                kpt_weighted = (
                    loss_factor[:, None]
                    * (1.0 - torch.exp(-e))
                    * kpt_mask
                    * pos_kpt_weight[:, None]
                )
                loss_kpt = kpt_weighted.sum() / (kpt_mask.float() * pos_kpt_weight[:, None]).sum().clamp(min=1.0)

            if (pos_kpt_weight > 0).any():
                kobj_raw = F.binary_cross_entropy_with_logits(
                    pred_kpts.view(batch_size, -1, self.num_kpts, 3)[fg_mask][..., 2],
                    kpt_mask.float(),
                    reduction="none",
                )
                loss_kobj = (kobj_raw * pos_kpt_weight[:, None]).sum() / (
                    pos_kpt_weight[:, None].expand_as(kobj_raw).sum().clamp(min=1.0)
                )

            target_attrs = gt_attrs[batch_idx, target_gt_idx.clamp(min=0)]
            target_attr_mask = gt_attr_mask[batch_idx, target_gt_idx.clamp(min=0)]
            attr_weight = target_attr_mask * anchor_attr_weight.unsqueeze(-1) * fg_mask.unsqueeze(-1)
            if attr_weight.sum() > 0:
                pos_weight = (
                    torch.ones(self.num_attrs, device=device, dtype=torch.float32)
                    if attr_pos_weight is None
                    else attr_pos_weight.to(device=device, dtype=torch.float32)
                )
                raw_attr = F.binary_cross_entropy_with_logits(
                    pred_attrs.float(),
                    target_attrs.float(),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                loss_attr = (raw_attr * attr_weight).sum() / attr_weight.sum().clamp(min=1.0)
                per_attr_loss = (raw_attr * attr_weight).sum(dim=(0, 1)) / attr_weight.sum(dim=(0, 1)).clamp(min=1.0)
                attr_count = attr_weight.sum().detach()
                loss_attr_cons = self._attr_consistency_loss(
                    pred_attrs.float(),
                    fg_mask,
                    target_gt_idx,
                    target_attr_mask * anchor_attr_weight.unsqueeze(-1),
                )

        box = self.w_box * loss_box
        cls = self.w_cls * loss_cls
        dfl = self.w_dfl * loss_dfl
        kpt = self.w_pose * loss_kpt
        kobj = self.w_kobj * loss_kobj
        attr = self.w_attr * loss_attr
        attr_cons = self.attr_consistency_weight * loss_attr_cons
        det_total = box + cls + dfl
        pose_total = kpt + kobj
        attr_total = attr + attr_cons
        total = (det_total + pose_total + attr_total) * batch_size
        return {
            "total": total,
            "_pa_det_loss": det_total * batch_size,
            "_pa_pose_loss": pose_total * batch_size,
            "_pa_attr_loss": attr_total * batch_size,
            "det_total": det_total.detach(),
            "det_ciou": box.detach(),
            "det_cls": cls.detach(),
            "det_dfl": dfl.detach(),
            "pose_total": pose_total.detach(),
            "pose_kpt": kpt.detach(),
            "pose_kobj": kobj.detach(),
            "attr_total": attr_total.detach(),
            "attr_bce": attr.detach(),
            "attr_smoking": (self.w_attr * per_attr_loss[0]).detach() if self.num_attrs > 0 else torch.zeros((), device=device),
            "attr_falling": (self.w_attr * per_attr_loss[1]).detach() if self.num_attrs > 1 else torch.zeros((), device=device),
            "attr_waving": (self.w_attr * per_attr_loss[2]).detach() if self.num_attrs > 2 else torch.zeros((), device=device),
            "attr_helmet_on": (self.w_attr * per_attr_loss[3]).detach() if self.num_attrs > 3 else torch.zeros((), device=device),
            "attr_consistency": attr_cons.detach(),
            "attr_count": attr_count.float(),
            "num_pos": fg_mask.sum().detach().float(),
            "target_scores_sum": target_scores_sum.detach(),
        }


class DomainPoseAttrBiFPN(nn.Module):
    """Domain detection plus pose-anchor attributes.

    The domain head can also learn a falling-person class from Attr labels while
    the original falling attribute branch remains supervised on pose anchors.
    """

    def __init__(
        self,
        domain_num_classes=8,
        num_attrs=4,
        domain_class_map=None,
        domain_class_names=None,
        attr_names=None,
        num_kpts=17,
        reg_max=16,
        strides=(8, 16, 32),
        input_size=640,
        backbone_depth=0.67,
        backbone_width=0.75,
        neck_use_p2_context=False,
        neck_downsample="conv",
        neck_out_channels=None,
        assigner_topk=10,
        assigner_alpha=0.5,
        assigner_beta=6.0,
        assigner_eps=1.0e-9,
        domain_class_weights=None,
        attr_consistency_weight=0.05,
        attr_dropout=0.1,
        appearance_context_kernel=5,
    ):
        super().__init__()
        self.domain_num_classes = int(domain_num_classes)
        default_domain_names = [
            "puddle", "fire", "smoke", "other",
            "helmet", "head", "cigarette", "falling",
        ]
        self.domain_class_names = list(
            domain_class_names
            or default_domain_names[: self.domain_num_classes]
            or [f"class_{idx}" for idx in range(self.domain_num_classes)]
        )
        if len(self.domain_class_names) != self.domain_num_classes:
            raise ValueError("domain_class_names length must match domain_num_classes")
        if len(set(self.domain_class_names)) != len(self.domain_class_names):
            raise ValueError("domain_class_names must be unique")
        self.num_attrs = int(num_attrs)
        self.attr_names = list(attr_names or ["smoking", "falling", "waving", "helmet_on"])
        if len(self.attr_names) != self.num_attrs:
            raise ValueError("attr_names length must match num_attrs")
        self.domain_class_map = {int(k): int(v) for k, v in (domain_class_map or {}).items()}
        self.falling_attr_index = self.attr_names.index("falling") if "falling" in self.attr_names else None
        self.falling_domain_index = (
            self.domain_class_names.index("falling")
            if "falling" in self.domain_class_names
            else None
        )
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
        self.domain_det_head = YOLOLikeDetectHead(
            ch,
            num_classes=self.domain_num_classes,
            reg_max=self.reg_max,
            strides=self.strides,
            img_size=self.input_size,
        )
        self.pose_head = YOLOLikePoseAttrHead(
            ch,
            num_kpts=self.num_kpts,
            num_attrs=self.num_attrs,
            reg_max=self.reg_max,
            strides=self.strides,
            img_size=self.input_size,
            attr_dropout=attr_dropout,
            appearance_context_kernel=appearance_context_kernel,
        )
        self.domain_det_loss = YOLODetectionLoss(
            num_classes=self.domain_num_classes,
            reg_max=self.reg_max,
            strides=self.strides,
            assigner_topk=assigner_topk,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            assigner_eps=assigner_eps,
            class_weights=domain_class_weights,
        )
        self.pose_attr_loss = YOLOPoseAttrLoss(
            num_kpts=self.num_kpts,
            num_attrs=self.num_attrs,
            reg_max=self.reg_max,
            strides=self.strides,
            attr_consistency_weight=attr_consistency_weight,
            assigner_topk=assigner_topk,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            assigner_eps=assigner_eps,
        )
        self.det_task_weight = 1.0
        self.pose_task_weight = 1.0
        self.attr_task_weight = 1.0
        self.attr_loss_weight = 1.0
        self.attr_person_box_weight = 0.25
        self.train_domain_det = True
        self.train_det = True
        self.train_pose = True
        self.train_attr = True
        self.register_buffer("attr_pos_weight", torch.ones(self.num_attrs), persistent=False)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_task_weights(self, det_weight=1.0, pose_weight=1.0):
        self.det_task_weight = float(det_weight)
        self.pose_task_weight = float(pose_weight)

    def set_attr_task_weight(self, attr_weight=1.0):
        self.attr_task_weight = float(attr_weight)

    def set_attr_pos_weight(self, pos_weight):
        value = torch.as_tensor(pos_weight, dtype=self.attr_pos_weight.dtype)
        if value.numel() == 1:
            value = value.repeat(self.num_attrs)
        if value.numel() != self.num_attrs:
            raise ValueError(f"attr_pos_weight has {value.numel()} values, expected {self.num_attrs}")
        self.attr_pos_weight.copy_(value.reshape(self.num_attrs).to(self.attr_pos_weight.device))

    def _forward_features(self, x):
        return self.neck(self.backbone(x))

    def forward_pose_outputs(self, x, return_features=False):
        """Run only the person/pose branch used by teacher distillation."""
        neck_feats = self._forward_features(x)
        pose_feats = [adapter(feat) for adapter, feat in zip(self.pose_adapter, neck_feats)]
        out = self.pose_head(pose_feats)
        if return_features:
            return {
                "out": out,
                "neck_feats": neck_feats,
                "pose_feats": pose_feats,
            }
        return out

    def forward(self, x):
        neck_feats = self._forward_features(x)
        det_feats = [adapter(feat) for adapter, feat in zip(self.det_adapter, neck_feats)]
        pose_feats = [adapter(feat) for adapter, feat in zip(self.pose_adapter, neck_feats)]
        return {
            "domain_det": self.domain_det_head(det_feats),
            "pose": self.pose_head(pose_feats),
        }

    def _zero_loss(self, device):
        zero = torch.zeros((), device=device)
        result = {
            "total": zero,
            "det_total": zero.detach(),
            "det_ciou": zero.detach(),
            "det_cls": zero.detach(),
            "det_dfl": zero.detach(),
            "pose_total": zero.detach(),
            "pose_kpt": zero.detach(),
            "pose_kobj": zero.detach(),
            "attr_total": zero.detach(),
            "attr_bce": zero.detach(),
            "attr_smoking": zero.detach(),
            "attr_falling": zero.detach(),
            "attr_waving": zero.detach(),
            "attr_helmet_on": zero.detach(),
            "attr_consistency": zero.detach(),
            "attr_count": zero.detach(),
            "num_pos": zero.detach(),
            "target_scores_sum": zero.detach(),
        }
        for class_name in self.domain_class_names:
            result[f"det_cls_{class_name}"] = zero.detach()
            result[f"det_valid_images_{class_name}"] = zero.detach()
            result[f"det_valid_logits_{class_name}"] = zero.detach()
            result[f"det_pos_anchors_{class_name}"] = zero.detach()
            result[f"det_target_scores_{class_name}"] = zero.detach()
        return result

    def _filter_domain_gt(self, gt_dict_list, device):
        mapped = []
        for gt in gt_dict_list:
            boxes = gt["boxes"].to(device, non_blocking=True).float()
            classes = gt["classes"].to(device, non_blocking=True).long()
            class_mask = gt.get("domain_valid_mask")
            if class_mask is None:
                class_mask = torch.ones(self.domain_num_classes, device=device)
            else:
                class_mask = class_mask.to(device, non_blocking=True).float().flatten()
                if len(class_mask) != self.domain_num_classes:
                    raise ValueError(
                        f"domain_valid_mask length={len(class_mask)} does not match "
                        f"domain_num_classes={self.domain_num_classes}"
                    )
            keep_boxes = []
            keep_classes = []
            for idx, cls in enumerate(classes.tolist()):
                if self.domain_class_map and int(cls) not in self.domain_class_map:
                    continue
                dst = self.domain_class_map.get(int(cls), int(cls))
                if 0 <= dst < self.domain_num_classes:
                    keep_boxes.append(boxes[idx])
                    keep_classes.append(dst)
            if (
                gt.get("task") == "attr"
                and self.falling_attr_index is not None
                and self.falling_domain_index is not None
                and class_mask[self.falling_domain_index] > 0
            ):
                attrs = gt.get("attrs")
                attr_mask = gt.get("attr_mask")
                if attrs is not None and attr_mask is not None:
                    attrs = attrs.to(device, non_blocking=True).float()
                    attr_mask = attr_mask.to(device, non_blocking=True).float()
                    attr_idx = self.falling_attr_index
                    n = min(len(boxes), attrs.shape[0], attr_mask.shape[0])
                    if n and attr_idx < attrs.shape[1] and attr_idx < attr_mask.shape[1]:
                        falling_pos = (attr_mask[:n, attr_idx] > 0) & (attrs[:n, attr_idx] >= 0.5)
                        for box in boxes[:n][falling_pos]:
                            keep_boxes.append(box)
                            keep_classes.append(self.falling_domain_index)
            mapped.append({
                "boxes": torch.stack(keep_boxes) if keep_boxes else torch.zeros((0, 4), device=device),
                "classes": (
                    torch.tensor(keep_classes, device=device, dtype=torch.long)
                    if keep_classes
                    else torch.zeros((0,), device=device, dtype=torch.long)
                ),
                "kpts": torch.zeros((len(keep_boxes), 17, 3), device=device),
                "class_valid_mask": class_mask,
            })
        return mapped

    @staticmethod
    def _slice_outs(head_outs, indices):
        return {
            key: [feat[indices] for feat in value]
            for key, value in head_outs.items()
        }

    def _task_indices(self, gt_dict_list, task):
        return [idx for idx, gt in enumerate(gt_dict_list) if gt.get("task") == task]

    def _compute_pose_attr_loss_for(self, pose_outs, parts, device):
        if not parts:
            return self._zero_loss(device)
        indices = torch.as_tensor([i for i, _ in parts], device=device, dtype=torch.long)
        return self.pose_attr_loss(
            self._slice_outs(pose_outs, indices),
            [gt for _, gt in parts],
            attr_pos_weight=self.attr_pos_weight,
        )

    def compute_loss(self, images, gt_dict_list):
        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)
        outs = self.forward(images)

        pose_idx = self._task_indices(gt_dict_list, "pose")
        attr_idx = self._task_indices(gt_dict_list, "attr")

        if self.train_domain_det:
            det_losses = self.domain_det_loss(
                outs["domain_det"],
                self._filter_domain_gt(gt_dict_list, device),
            )
        else:
            det_losses = self._zero_loss(device)

        pose_attr_parts = []
        pose_parts = []
        attr_parts = []
        if self.train_pose:
            for i in pose_idx:
                gt = dict(gt_dict_list[i])
                gt["_person_box_weight"] = 1.0
                gt["_kpt_weight"] = 1.0
                gt["_attr_weight"] = 0.0
                pose_attr_parts.append((i, gt))
                pose_parts.append((i, gt))
        if self.train_attr:
            for i in attr_idx:
                gt = dict(gt_dict_list[i])
                gt["_person_box_weight"] = self.attr_person_box_weight
                gt["_kpt_weight"] = 0.0
                gt["_attr_weight"] = 1.0
                pose_attr_parts.append((i, gt))
                attr_parts.append((i, gt))

        pose_attr_losses = self._compute_pose_attr_loss_for(outs["pose"], pose_attr_parts, device)
        pose_only_losses = self._compute_pose_attr_loss_for(outs["pose"], pose_parts, device)
        attr_only_losses = self._compute_pose_attr_loss_for(outs["pose"], attr_parts, device)

        pose_raw = pose_attr_losses.get("det_total", torch.zeros((), device=device)) + pose_attr_losses.get(
            "pose_total", torch.zeros((), device=device)
        )
        pose_person_obj = pose_only_losses.get("_pa_det_loss", torch.zeros((), device=device)) + pose_only_losses.get(
            "_pa_pose_loss", torch.zeros((), device=device)
        )
        attr_obj = (
            self.pose_task_weight * attr_only_losses.get("_pa_det_loss", torch.zeros((), device=device))
            + self.attr_task_weight
            * attr_only_losses.get("_pa_attr_loss", torch.zeros((), device=device))
            * self.attr_loss_weight
        )
        det_obj = (
            self.det_task_weight * det_losses["total"]
            if self.train_domain_det
            else torch.zeros((), device=device)
        )
        pose_obj = self.pose_task_weight * pose_person_obj
        total = det_obj + pose_obj + attr_obj
        raw_total = det_losses["total"] + pose_attr_losses["total"]

        class_metrics = det_losses.get("_class_metrics", {})
        class_zero = torch.zeros(self.domain_num_classes, device=device)
        result = {
            "total": total,
            # Trainer uses these graph-connected objectives for stage-specific
            # gradient projection. The pose objective is intentionally pure
            # pose data so auxiliary detection/attribute gradients can be
            # projected away from conflicting with pose on shared parameters.
            "_gp_det_loss": det_obj,
            "_gp_attr_loss": attr_obj,
            "_gp_pose_loss": pose_obj,
            "loss_total": raw_total.detach(),
            "det_total": det_losses["det_total"].detach(),
            "det_ciou": det_losses["det_ciou"].detach(),
            "det_cls": det_losses["det_cls"].detach(),
            "det_dfl": det_losses["det_dfl"].detach(),
            "det_num_pos": det_losses.get("num_pos", torch.zeros((), device=device)).detach().float(),
            "pose_total": pose_raw.detach(),
            "pose_det_total": pose_attr_losses.get("det_total", torch.zeros((), device=device)).detach(),
            "pose_det_ciou": pose_attr_losses.get("det_ciou", torch.zeros((), device=device)).detach(),
            "pose_det_cls": pose_attr_losses.get("det_cls", torch.zeros((), device=device)).detach(),
            "pose_det_dfl": pose_attr_losses.get("det_dfl", torch.zeros((), device=device)).detach(),
            "pose_kpt_total": pose_attr_losses.get("pose_total", torch.zeros((), device=device)).detach(),
            "pose_kpt": pose_attr_losses.get("pose_kpt", torch.zeros((), device=device)).detach(),
            "pose_kobj": pose_attr_losses.get("pose_kobj", torch.zeros((), device=device)).detach(),
            "pose_num_pos": pose_attr_losses.get("num_pos", torch.zeros((), device=device)).detach().float(),
            "attr_total": pose_attr_losses.get("attr_total", torch.zeros((), device=device)).detach(),
            "attr_bce": pose_attr_losses.get("attr_bce", torch.zeros((), device=device)).detach(),
            "attr_smoking": pose_attr_losses.get("attr_smoking", torch.zeros((), device=device)).detach(),
            "attr_falling": pose_attr_losses.get("attr_falling", torch.zeros((), device=device)).detach(),
            "attr_waving": pose_attr_losses.get("attr_waving", torch.zeros((), device=device)).detach(),
            "attr_helmet_on": pose_attr_losses.get("attr_helmet_on", torch.zeros((), device=device)).detach(),
            "attr_consistency": pose_attr_losses.get("attr_consistency", torch.zeros((), device=device)).detach(),
            "attr_count": pose_attr_losses.get("attr_count", torch.zeros((), device=device)).detach().float(),
            "num_pos": (
                det_losses.get("num_pos", torch.zeros((), device=device)).detach().float()
                + pose_attr_losses.get("num_pos", torch.zeros((), device=device)).detach().float()
            ),
            "target_scores_sum": (
                det_losses.get("target_scores_sum", torch.zeros((), device=device)).detach()
                + pose_attr_losses.get("target_scores_sum", torch.zeros((), device=device)).detach()
            ),
            "task_w_det": torch.tensor(self.det_task_weight, device=device),
            "task_w_pose": torch.tensor(self.pose_task_weight, device=device),
            "task_w_attr": torch.tensor(self.attr_task_weight if self.train_attr else 0.0, device=device),
        }
        if self.training:
            # Reuse the graph-connected pose outputs for teacher distillation;
            # the trainer will run the frozen teacher only once per batch.
            result["_distill_pose_out"] = outs["pose"]
        for idx, class_name in enumerate(self.domain_class_names):
            result[f"det_cls_{class_name}"] = class_metrics.get("cls", class_zero)[idx].detach()
            result[f"det_valid_images_{class_name}"] = class_metrics.get("valid_images", class_zero)[idx].detach()
            result[f"det_valid_logits_{class_name}"] = class_metrics.get("valid_logits", class_zero)[idx].detach()
            result[f"det_pos_anchors_{class_name}"] = class_metrics.get("pos_anchors", class_zero)[idx].detach()
            result[f"det_target_scores_{class_name}"] = class_metrics.get("target_scores", class_zero)[idx].detach()
        return result

    @staticmethod
    def _decode_kpts(kpt_raw, anchor_idx, grid, stride, num_kpts):
        raw = kpt_raw[anchor_idx].view(-1, num_kpts, 3)
        anchor_points = grid.view(-1, 1, 2)[anchor_idx] + 0.5
        xy = (raw[..., :2] * 2.0 + anchor_points - 0.5) * stride
        conf = raw[..., 2:3].sigmoid()
        return torch.cat((xy, conf), dim=-1)

    def _decode_one_head(
        self,
        cls_list,
        reg_list,
        kpt_list=None,
        attr_list=None,
        cls_offset=0,
        score_thresh=0.01,
        max_nms=30000,
    ):
        device = cls_list[0].device
        bsz = cls_list[0].shape[0]
        num_cls = cls_list[0].shape[1]
        decoded = []
        for b in range(bsz):
            all_boxes, all_scores, all_cls, all_kpts, all_attrs = [], [], [], [], []
            for lvl, stride in enumerate(self.strides):
                _, _, h, w = cls_list[lvl].shape
                cls_l = cls_list[lvl][b : b + 1].permute(0, 2, 3, 1).reshape(h * w, num_cls)
                scores_l = cls_l.sigmoid()
                grid_cells = _make_grid(w, h, device)
                boxes_l = _dfl_decode(reg_list[lvl][b : b + 1], self.reg_max, stride, grid_cells * stride)[0]
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
                    kpt_l = kpt_list[lvl][b : b + 1].permute(0, 2, 3, 1).reshape(h * w, self.num_kpts * 3)
                    all_kpts.append(self._decode_kpts(kpt_l, anchor_idx, grid_cells, stride, self.num_kpts))
                else:
                    all_kpts.append(torch.zeros((anchor_idx.numel(), self.num_kpts, 3), device=device))
                if attr_list is not None:
                    attr_l = attr_list[lvl][b : b + 1].permute(0, 2, 3, 1).reshape(h * w, self.num_attrs)
                    all_attrs.append(attr_l[anchor_idx].sigmoid())
                else:
                    all_attrs.append(torch.zeros((anchor_idx.numel(), self.num_attrs), device=device))

            if all_boxes:
                decoded.append({
                    "boxes": torch.cat(all_boxes),
                    "scores": torch.cat(all_scores),
                    "classes": torch.cat(all_cls),
                    "kpts": torch.cat(all_kpts),
                    "attrs": torch.cat(all_attrs),
                })
            else:
                decoded.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "classes": torch.zeros(0, dtype=torch.long, device=device),
                    "kpts": torch.zeros(0, self.num_kpts, 3, device=device),
                    "attrs": torch.zeros(0, self.num_attrs, device=device),
                })
        return decoded

    @torch.no_grad()
    def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6, max_det=300):
        self.eval()
        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)
        outs = self.forward(images)
        domain_preds = self._decode_one_head(
            outs["domain_det"]["cls"],
            outs["domain_det"]["reg"],
            kpt_list=None,
            attr_list=None,
            cls_offset=1,
            score_thresh=score_thresh,
        )
        pose_preds = self._decode_one_head(
            outs["pose"]["cls"],
            outs["pose"]["reg"],
            kpt_list=outs["pose"]["kpt"],
            attr_list=outs["pose"]["attr"],
            cls_offset=0,
            score_thresh=score_thresh,
        )

        results = []
        for domain, pose in zip(domain_preds, pose_preds):
            boxes = torch.cat([pose["boxes"], domain["boxes"]], dim=0)
            scores = torch.cat([pose["scores"], domain["scores"]], dim=0)
            classes = torch.cat([pose["classes"], domain["classes"]], dim=0)
            kpts = torch.cat([pose["kpts"], domain["kpts"]], dim=0)
            attrs = torch.cat([pose["attrs"], domain["attrs"]], dim=0)
            if boxes.numel() == 0:
                results.append({"boxes": boxes, "scores": scores, "classes": classes, "kpts": kpts, "attrs": attrs})
                continue
            keep = _batched_nms(boxes, scores, classes, iou_thresh=iou_thresh, max_det=max_det)
            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "classes": classes[keep],
                "kpts": kpts[keep],
                "attrs": attrs[keep],
            })
        return results


def create_final_pose_attr_model(name="final_pose_attr", **kwargs):
    if name not in ("final_pose_attr", "bifpn_final_pose_attr"):
        raise ValueError('Supported models: "final_pose_attr", "bifpn_final_pose_attr".')
    model = DomainPoseAttrBiFPN(**kwargs)
    print(f"  Created {name}: {model.num_params / 1e6:.2f}M params")
    return model
