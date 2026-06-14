"""VigilModel v2: 解耦多任务检测器.

v2 相比 v1 的核心改进:
  - 解耦 head (cls/reg/kpt/attr 独立分支, YOLOv8-style)
  - DFL 分布框回归 (替代 exp(ltrb))
  - TaskAlignedAssigner (动态 top-k, 替代 FCOS center)
  - CIoU + DFL loss (替代 WIoU v3)
  - Gather-Distribute Neck (替代 FPN+PAN)
  - 3 尺度 [8,16,32] (去掉冗余 stride 4)
  - 直接 cls score (去掉 centerness)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from Vigil.models.base import VigilModelBase
from Vigil.models.registry import register_model
from Vigil.models.vigil_v2.assigner import TaskAlignedAssigner
from Vigil.models.vigil_v2.backbone import CSPDarkNetV2
from Vigil.models.vigil_v2.head import VigilHeadV2, _dfl_decode, _make_grid, decode_outputs_v2
from Vigil.models.vigil_v2.loss import VigilLossV2
from Vigil.models.vigil_v2.neck import GatherDistributeNeck


class VigilModelV2(VigilModelBase, nn.Module):

    def __init__(self, backbone_w=1.5, neck_ch=320, reg_max=16,
                 w_box=5.0, w_cls=1.0, w_dfl=12.0,
                 w_kpt=10.0, w_helm=10.0, w_smoke=10.0,
                 assigner_topk=13, cls_weights=None):
        super().__init__()
        self.backbone = CSPDarkNetV2(w=backbone_w)
        self.neck = GatherDistributeNeck(
            in_channels=self.backbone.out_channels[1:4],  # p3, p4, p5
            out_ch=neck_ch,
        )
        self.head = VigilHeadV2(neck_ch, num_classes=3, reg_max=reg_max)
        self.strides = [8, 16, 32]
        self.reg_max = reg_max
        self._input_size = (640, 640)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.assigner = TaskAlignedAssigner(topk=assigner_topk)
        self.loss_fn = VigilLossV2(
            w_box=w_box, w_cls=w_cls, w_dfl=w_dfl,
            w_kpt=w_kpt, w_helm=w_helm, w_smoke=w_smoke,
            reg_max=reg_max, cls_weights=cls_weights)

    @property
    def input_size(self):
        return self._input_size

    # ── 训练用 forward ──

    def forward(self, x):
        feats = self.backbone(x)            # p2, p3, p4, p5
        neck_feats = self.neck(feats[1:])   # p3, p4, p5
        return self.head(neck_feats)

    # ── 训练接口: compute_loss ──

    def compute_loss(self, samples):
        """samples: List[VigilSample] — 一个 batch 的样本列表."""
        device = next(self.parameters()).device
        if not isinstance(samples, list):
            samples = [samples]

        imgs = torch.stack([s.image for s in samples]).to(device)

        gt_boxes, gt_classes, attrs, batch_indices = self._build_targets(samples, device)

        head_outs = self.forward(imgs)
        # Loss/assigner 在 fp32 下计算避免溢出
        head_outs = {k: [t.float() for t in v] for k, v in head_outs.items()}
        feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]

        # 解码 (用于 assigner 的 alignment 计算)
        pred_scores, pred_boxes = self._decode_for_assigner(
            head_outs, feat_sizes)

        targets = self.assigner(
            pred_scores, pred_boxes,
            gt_boxes, gt_classes, attrs,
            feat_sizes, self.strides, batch_indices)

        losses = self.loss_fn(head_outs, targets, self.strides, feat_sizes)
        total = (losses["cls"] + losses["ciou"] + losses["dfl"] +
                 losses["kpt"] + losses["helmet"] + losses["smoke"])
        losses["total"] = total
        return losses

    def _build_targets(self, samples, device):
        """构建 GT tensor, 附加 batch 索引.

        samples: List[VigilSample]
        Returns: gt_boxes [M,4], gt_classes [M], attrs dict, batch_indices [M]
        """
        gt_boxes_list, gt_classes_list, batch_idx_list = [], [], []
        kpts_list, helmet_list, smoke_list = [], [], []

        for b, sample in enumerate(samples):
            n_p = len(sample.person_boxes)
            if n_p > 0:
                gt_boxes_list.append(sample.person_boxes)
                gt_classes_list.append(torch.zeros(n_p, dtype=torch.long))
                batch_idx_list.append(torch.full((n_p,), b, dtype=torch.long))
                if sample.person_kpts.numel() > 0:
                    kpts_list.append(sample.person_kpts)
                if sample.person_helmet.numel() > 0:
                    helmet_list.append(sample.person_helmet)
                if sample.person_smoke.numel() > 0:
                    smoke_list.append(sample.person_smoke)

            if sample.detect_boxes.numel() > 0:
                n_d = len(sample.detect_boxes)
                gt_boxes_list.append(sample.detect_boxes)
                gt_classes_list.append(sample.detect_classes)
                batch_idx_list.append(torch.full((n_d,), b, dtype=torch.long))

        if not gt_boxes_list:
            return (torch.empty(0, 4), torch.empty(0, dtype=torch.long),
                    {}, torch.empty(0, dtype=torch.long))

        all_boxes = torch.cat(gt_boxes_list, dim=0).to(device)
        all_classes = torch.cat(gt_classes_list, dim=0).to(device)
        batch_indices = torch.cat(batch_idx_list, dim=0).to(device)

        attrs = {}
        if kpts_list:
            attrs["kpts"] = torch.cat(kpts_list, dim=0).to(device)
        if helmet_list:
            attrs["helmet"] = torch.cat(helmet_list, dim=0).to(device)
        if smoke_list:
            attrs["smoking"] = torch.cat(smoke_list, dim=0).to(device)

        return all_boxes, all_classes, attrs, batch_indices

    @torch.no_grad()
    def _decode_for_assigner(self, head_outs, feat_sizes):
        """解码所有格点的得分和框 (不设阈值), 供 assigner 使用."""
        device = head_outs["cls"][0].device
        pred_scores, pred_boxes = [], []

        B = head_outs["cls"][0].shape[0]
        for lvl, ((H, W), stride) in enumerate(zip(feat_sizes, self.strides)):
            cls_pred = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, H * W, 3)
            scores = cls_pred.sigmoid()
            pred_scores.append(scores)

            grid = _make_grid(W, H, device) * stride
            boxes = _dfl_decode(head_outs["reg"][lvl], self.reg_max, stride, grid)
            pred_boxes.append(boxes)

        return pred_scores, pred_boxes

    # ── 推理接口: detect ──

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        tensor, scale, (pad_l, pad_t) = self._preprocess(frame)
        tensor = tensor.to(next(self.parameters()).device)
        raw = self.forward(tensor)
        det = self._decode(raw)

        for entry in det.values():
            entry["boxes"][:, [0, 2]] = (entry["boxes"][:, [0, 2]] - pad_l) / scale
            entry["boxes"][:, [1, 3]] = (entry["boxes"][:, [1, 3]] - pad_t) / scale
            entry["boxes"][:, [0, 2]] = entry["boxes"][:, [0, 2]].clamp(0, w)
            entry["boxes"][:, [1, 3]] = entry["boxes"][:, [1, 3]].clamp(0, h)
            if "kpts" in entry:
                entry["kpts"][..., 0] = ((entry["kpts"][..., 0] - pad_l) / scale).clamp(0, w)
                entry["kpts"][..., 1] = ((entry["kpts"][..., 1] - pad_t) / scale).clamp(0, h)
        return det

    def _preprocess(self, frame: np.ndarray):
        in_w, in_h = self._input_size
        h, w = frame.shape[:2]
        scale = min(in_w / w, in_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = in_w - new_w, in_h - new_h
        pad_l, pad_t = pad_w // 2, pad_h // 2
        img = cv2.copyMakeBorder(
            img, pad_t, pad_h - pad_t, pad_l, pad_w - pad_l,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0), scale, (pad_l, pad_t)

    def _decode(self, raw_outputs, score_thresh=0.0) -> dict:
        boxes, scores, kpts, helmet, smoking = decode_outputs_v2(
            raw_outputs, self.strides, self.reg_max, score_thresh)
        boxes, scores = boxes[0], scores[0]
        kpts, helmet, smoking = kpts[0], helmet[0], smoking[0]
        result = {}
        for cls_idx, cls_name in [(0, "person"), (1, "fire"), (2, "water")]:
            cls_scores = scores[:, cls_idx]
            mask = cls_scores > score_thresh
            if mask.any():
                entry = {"boxes": boxes[mask], "scores": cls_scores[mask]}
                if cls_name == "person":
                    entry["kpts"]    = kpts[mask]
                    entry["helmet"]  = helmet[mask]
                    entry["smoking"] = smoking[mask]
                result[cls_name] = entry
        return result

    # ── 验证用预测 (640×640 空间, per-class 独立, 不做 argmax) ──

    @torch.no_grad()
    def predict_val(self, sample):
        """Run detection on a preprocessed VigilSample.

        Uses per-class independent thresholding (no argmax), so a single
        grid position can contribute to multiple classes.  Applies per-class
        NMS before returning.

        Returns:
            boxes:   [K, 4] xyxy in 640×640
            scores:  [K]
            classes: [K] 0=person, 1=fire, 2=water
        """
        device = next(self.parameters()).device
        img = sample.image.unsqueeze(0).to(device)
        head_outs = self.forward(img)

        boxes_3d, scores_3d, _, _, _ = decode_outputs_v2(
            head_outs, self.strides, self.reg_max, score_thresh=0.01)

        boxes = boxes_3d[0]    # [N, 4]
        scores = scores_3d[0]  # [N, 3]

        all_boxes, all_scores, all_cls = [], [], []
        for c in range(3):
            cls_scores = scores[:, c]
            keep = cls_scores > 0.01
            if keep.any():
                c_boxes = boxes[keep]
                c_scores = cls_scores[keep]
                nms_k = _nms(c_boxes, c_scores, 0.6)
                all_boxes.append(c_boxes[nms_k])
                all_scores.append(c_scores[nms_k])
                all_cls.append(torch.full((len(nms_k),), c, dtype=torch.long, device=device))

        if all_boxes:
            return (torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_cls))
        return (torch.zeros(0, 4, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def predict_val_full(self, sample):
        """Run validation detection with person keypoints and attributes."""
        device = next(self.parameters()).device
        img = sample.image.unsqueeze(0).to(device)
        head_outs = self.forward(img)

        boxes_3d, scores_3d, kpts_3d, helmet_3d, smoke_3d = decode_outputs_v2(
            head_outs, self.strides, self.reg_max, score_thresh=0.01)

        boxes = boxes_3d[0]
        scores = scores_3d[0]
        kpts = kpts_3d[0]
        helmet = helmet_3d[0]
        smoke = smoke_3d[0]

        all_boxes, all_scores, all_cls = [], [], []
        person_boxes, person_scores, person_kpts = [], [], []
        person_helmet, person_smoke = [], []
        for c in range(3):
            cls_scores = scores[:, c]
            keep = cls_scores > 0.01
            if keep.any():
                c_boxes = boxes[keep]
                c_scores = cls_scores[keep]
                nms_k = _nms(c_boxes, c_scores, 0.6)
                all_boxes.append(c_boxes[nms_k])
                all_scores.append(c_scores[nms_k])
                all_cls.append(torch.full((len(nms_k),), c, dtype=torch.long, device=device))
                if c == 0:
                    person_boxes.append(c_boxes[nms_k])
                    person_scores.append(c_scores[nms_k])
                    person_kpts.append(kpts[keep][nms_k])
                    person_helmet.append(helmet[keep][nms_k])
                    person_smoke.append(smoke[keep][nms_k])

        if all_boxes:
            flat_boxes = torch.cat(all_boxes)
            flat_scores = torch.cat(all_scores)
            flat_cls = torch.cat(all_cls)
        else:
            flat_boxes = torch.zeros(0, 4, device=device)
            flat_scores = torch.zeros(0, device=device)
            flat_cls = torch.zeros(0, dtype=torch.long, device=device)

        if person_boxes:
            p_boxes = torch.cat(person_boxes)
            p_scores = torch.cat(person_scores)
            p_kpts = torch.cat(person_kpts)
            p_helmet = torch.cat(person_helmet)
            p_smoke = torch.cat(person_smoke)
        else:
            p_boxes = torch.zeros(0, 4, device=device)
            p_scores = torch.zeros(0, device=device)
            p_kpts = torch.zeros(0, 17, 3, device=device)
            p_helmet = torch.zeros(0, device=device)
            p_smoke = torch.zeros(0, device=device)

        return {
            "boxes": flat_boxes,
            "scores": flat_scores,
            "classes": flat_cls,
            "person_boxes": p_boxes,
            "person_scores": p_scores,
            "person_kpts": p_kpts,
            "person_helmet": p_helmet,
            "person_smoke": p_smoke,
        }

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_classes(self):
        return 3


def _nms(boxes, scores, iou_thresh):
    """向量化 NMS, boxes [N,4] xyxy."""
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


@register_model("vigil_v2")
def create_model(pretrained=None, **kwargs):
    model = VigilModelV2(**kwargs)

    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state, strict=False)
        except RuntimeError as exc:
            if "size mismatch" in str(exc):
                raise RuntimeError(
                    f"Failed to load {pretrained!r}: checkpoint tensor shapes do not "
                    "match the requested VigilModelV2 architecture. Make sure "
                    "backbone_w/neck_ch are the same as the training run, or retrain "
                    "before using the new model size."
                ) from exc
            raise

    return model

