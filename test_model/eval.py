"""Evaluation script for multi-head verification models.

Computes:
- Detection mAP@0.5 and mAP@0.5:0.95
- Keypoint mAP (OKS-based)
- Per-class metrics
- Outputs comparison-ready JSON

Usage:
    python -m test_model.eval --model dual_head --weights checkpoints/dual_head/dual_head_best.pt --data /data/coco2017
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def compute_iou(box1, box2):
    """IoU between box [4] and boxes [N,4]."""
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1 + area2 - inter + 1e-16)


def _interp_ap(tp, fp, total_gt):
    """COCO-style 101-point interpolated AP."""
    if total_gt <= 0:
        return -1.0
    if len(tp) == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        ap += (np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0) / 101.0
    return float(np.clip(ap, 0.0, 1.0))


def compute_ap_by_class_multi(predictions, ground_truths, iou_thresholds=None, num_classes=80):
    """Compute per-class AP for several IoU thresholds in one pass."""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float32)
    num_thr = len(iou_thresholds)

    aps = {cls_id: np.full(num_thr, -1.0, dtype=np.float32) for cls_id in range(num_classes)}

    gt_by_class = {}
    for cls_id in range(num_classes):
        cls_gt_boxes = []
        total_gt = 0
        for gt in ground_truths:
            classes = np.asarray(gt['classes'], dtype=np.int32)
            boxes = np.asarray(gt['boxes'], dtype=np.float32).reshape(-1, 4)
            mask = classes == cls_id
            boxes_cls = boxes[mask]
            cls_gt_boxes.append(boxes_cls)
            total_gt += len(boxes_cls)
        gt_by_class[cls_id] = (cls_gt_boxes, total_gt)

    for cls_id in range(num_classes):
        all_dets = []
        for img_idx, pred in enumerate(predictions):
            classes = np.asarray(pred['classes'], dtype=np.int32)
            boxes = np.asarray(pred['boxes'], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(pred['scores'], dtype=np.float32)
            mask = classes == cls_id
            if not mask.any():
                continue
            for score, box in zip(scores[mask], boxes[mask]):
                all_dets.append((img_idx, float(score), box))

        all_dets.sort(key=lambda x: x[1], reverse=True)
        gt_boxes_by_img, total_gt = gt_by_class[cls_id]
        if total_gt == 0:
            continue

        tp = np.zeros((num_thr, len(all_dets)), dtype=np.float32)
        fp = np.zeros((num_thr, len(all_dets)), dtype=np.float32)
        gt_matched = [[np.zeros(len(boxes), dtype=bool) for boxes in gt_boxes_by_img]
                      for _ in range(num_thr)]

        for det_idx, (img_idx, _score, det_box) in enumerate(all_dets):
            gt_boxes_cls = gt_boxes_by_img[img_idx]
            if len(gt_boxes_cls) == 0:
                fp[:, det_idx] = 1
                continue

            ious = compute_iou(np.asarray(det_box, dtype=np.float32), gt_boxes_cls)
            order = ious.argsort()[::-1]
            for ti, iou_thresh in enumerate(iou_thresholds):
                best_local = -1
                for li in order:
                    if ious[li] >= iou_thresh and not gt_matched[ti][img_idx][li]:
                        best_local = int(li)
                        break
                if best_local >= 0:
                    tp[ti, det_idx] = 1
                    gt_matched[ti][img_idx][best_local] = True
                else:
                    fp[ti, det_idx] = 1

        aps[cls_id] = np.asarray([
            _interp_ap(tp[ti], fp[ti], total_gt) for ti in range(num_thr)
        ], dtype=np.float32)

    return aps


def compute_ap_by_class(predictions, ground_truths, iou_thresh=0.5, num_classes=80):
    """Compute AP for each class using 101-point interpolation."""
    multi = compute_ap_by_class_multi(
        predictions, ground_truths, iou_thresholds=np.array([iou_thresh], dtype=np.float32),
        num_classes=num_classes)
    return {cls_id: float(values[0]) for cls_id, values in multi.items()}


def compute_pose_oks(pred_kpts, gt_kpts, gt_boxes):
    """Compute OKS between predicted and GT keypoints.

    Args:
        pred_kpts: [K, 17, 3] predicted keypoints
        gt_kpts: [M, 17, 3] GT keypoints
        gt_boxes: [M, 4] GT boxes (for scale)

    Returns:
        oks: [K, M]
    """
    sigmas = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072,
        0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    ])

    pred_kpts = np.asarray(pred_kpts, dtype=np.float32).reshape(-1, 17, 3)
    gt_kpts = np.asarray(gt_kpts, dtype=np.float32).reshape(-1, 17, 3)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    K, M = len(pred_kpts), len(gt_kpts)
    if K == 0 or M == 0:
        return np.zeros((K, M), dtype=np.float32)

    area = np.maximum((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]), 1.0)
    scale = np.sqrt(area)
    k2 = (sigmas.reshape(1, 1, 17) ** 2) * (2 * scale.reshape(1, M, 1)) ** 2
    d2 = ((pred_kpts[:, None, :, :2] - gt_kpts[None, :, :, :2]) ** 2).sum(axis=-1)
    visible = gt_kpts[None, :, :, 2] > 0
    oks_raw = np.exp(-d2 / (2 * k2 + 1e-16)) * visible
    visible_count = visible.sum(axis=-1)
    return np.divide(
        oks_raw.sum(axis=-1),
        np.maximum(visible_count, 1),
        out=np.zeros((K, M), dtype=np.float32),
        where=visible_count > 0,
    ).astype(np.float32)


def compute_pose_ap_multi(predictions, ground_truths, oks_thresholds=None):
    """Compute pose AP over one or more OKS thresholds."""
    if oks_thresholds is None:
        oks_thresholds = np.arange(0.5, 1.0, 0.05)
    oks_thresholds = np.asarray(oks_thresholds, dtype=np.float32)
    num_thr = len(oks_thresholds)

    all_dets = []
    gt_person_by_img = []
    total_gt = 0

    for gt in ground_truths:
        person_mask = gt['classes'] == 0
        if 'kpts' in gt and len(gt['kpts']) == len(gt['classes']):
            visible_mask = (gt['kpts'][:, :, 2] > 0).any(axis=1)
            person_mask = person_mask & visible_mask
        else:
            person_mask = np.zeros_like(person_mask, dtype=bool)
        gt_boxes_person = gt['boxes'][person_mask].astype(np.float32).reshape(-1, 4)
        gt_kpts_person = gt['kpts'][person_mask].astype(np.float32).reshape(-1, 17, 3)
        gt_person_by_img.append((gt_boxes_person, gt_kpts_person))
        total_gt += len(gt_boxes_person)

    for img_idx, pred in enumerate(predictions):
        if 'person_boxes' not in pred:
            continue
        p_boxes = pred['person_boxes']
        p_scores = pred['person_scores']
        p_kpts = pred['person_kpts']
        if len(p_kpts) != len(p_boxes):
            p_kpts = np.zeros((len(p_boxes), 17, 3), dtype=np.float32)

        for i in range(len(p_boxes)):
            all_dets.append((img_idx, float(p_scores[i]),
                            p_boxes[i].tolist(), p_kpts[i]))

    all_dets.sort(key=lambda x: x[1], reverse=True)

    if total_gt == 0:
        return None

    tp = np.zeros((num_thr, len(all_dets)), dtype=np.float32)
    fp = np.zeros((num_thr, len(all_dets)), dtype=np.float32)
    gt_matched = [[np.zeros(len(gt_person_by_img[i][0]), dtype=bool)
                   for i in range(len(gt_person_by_img))] for _ in range(num_thr)]

    for det_idx, (img_idx, score, det_box, det_kpts) in enumerate(all_dets):
        gt_boxes_person, gt_kpts_person = gt_person_by_img[img_idx]
        if len(gt_boxes_person) == 0:
            fp[:, det_idx] = 1
            continue

        oks = compute_pose_oks(
            det_kpts.reshape(1, 17, 3),
            gt_kpts_person.reshape(-1, 17, 3),
            gt_boxes_person.reshape(-1, 4),
        )[0]
        order = oks.argsort()[::-1]
        for ti, oks_thresh in enumerate(oks_thresholds):
            best_local = -1
            for li in order:
                if oks[li] >= oks_thresh and not gt_matched[ti][img_idx][li]:
                    best_local = int(li)
                    break
            if best_local >= 0:
                tp[ti, det_idx] = 1
                gt_matched[ti][img_idx][best_local] = True
            else:
                fp[ti, det_idx] = 1

    return np.asarray([_interp_ap(tp[ti], fp[ti], total_gt) for ti in range(num_thr)], dtype=np.float32)


def compute_pose_ap(predictions, ground_truths):
    """Compute pose AP@0.5 using OKS."""
    aps = compute_pose_ap_multi(
        predictions, ground_truths, oks_thresholds=np.array([0.5], dtype=np.float32))
    return None if aps is None else float(aps[0])


def _to_numpy(value, dtype, shape):
    """Convert torch/numpy/list predictions to a numpy array with a stable shape."""
    if value is None:
        return np.zeros(shape, dtype=dtype)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        return np.zeros(shape, dtype=dtype)
    return arr


def evaluate(model, dataloader, device='cuda', score_thresh=0.01, iou_thresh=0.6, max_det=300):
    """Run inference and collect predictions + ground truths."""
    model.eval()
    model.to(device)

    all_preds = []
    all_gts = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)

            try:
                predictions = model.predict_val(
                    images, score_thresh=score_thresh, iou_thresh=iou_thresh, max_det=max_det)
            except TypeError:
                predictions = model.predict_val(
                    images, score_thresh=score_thresh, iou_thresh=iou_thresh)

            for i in range(len(images)):
                pred = predictions[i]
                boxes_t = pred['boxes']
                scores_t = pred['scores']
                classes_t = pred['classes']
                person_mask_t = classes_t == 0
                person_boxes = pred.get('person_boxes', boxes_t[person_mask_t])
                person_scores = pred.get('person_scores', scores_t[person_mask_t])
                person_kpts = pred.get('person_kpts')
                if person_kpts is None:
                    kpts_t = pred.get('kpts')
                    if kpts_t is not None and len(kpts_t) == len(boxes_t):
                        person_kpts = kpts_t[person_mask_t]
                    else:
                        person_kpts = kpts_t
                person_boxes_np = _to_numpy(person_boxes, np.float32, (0, 4))
                person_scores_np = _to_numpy(person_scores, np.float32, (0,))
                person_kpts_np = _to_numpy(person_kpts, np.float32, (0, 17, 3))
                if len(person_kpts_np) != len(person_boxes_np):
                    person_kpts_np = np.zeros((len(person_boxes_np), 17, 3), dtype=np.float32)
                all_preds.append({
                    'boxes': _to_numpy(boxes_t, np.float32, (0, 4)),
                    'scores': _to_numpy(scores_t, np.float32, (0,)),
                    'classes': _to_numpy(classes_t, np.int32, (0,)),
                    'person_boxes': person_boxes_np,
                    'person_scores': person_scores_np,
                    'person_kpts': person_kpts_np,
                })

                all_gts.append({
                    'boxes': batch['boxes'][i].cpu().numpy().astype(np.float32),
                    'classes': batch['classes'][i].cpu().numpy().astype(np.int32),
                    'kpts': batch['kpts'][i].cpu().numpy().astype(np.float32),
                })

    return all_preds, all_gts


def _scale_boxes_to_letterbox(boxes, scale, pad):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if len(boxes) == 0:
        return boxes
    pad_l, pad_t = pad
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * float(scale) + float(pad_l)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * float(scale) + float(pad_t)
    return boxes


def _scale_kpts_to_letterbox(kpts, scale, pad):
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 17, 3).copy()
    if len(kpts) == 0:
        return kpts
    pad_l, pad_t = pad
    kpts[..., 0] = kpts[..., 0] * float(scale) + float(pad_l)
    kpts[..., 1] = kpts[..., 1] * float(scale) + float(pad_t)
    return kpts


def evaluate_ultralytics(yolo_model, dataloader, device='cuda', score_thresh=0.01,
                         iou_thresh=0.6, max_det=300, provider='ultralytics_detect',
                         input_size=640, num_classes=80):
    """Evaluate an official Ultralytics model on the same labels/metrics."""
    from test_model.dataset import YOLO80_ID_TO_20

    all_preds = []
    all_gts = []

    for batch in tqdm(dataloader, desc=f'Evaluating {provider}'):
        paths = batch.get('img_path', [])
        if not paths:
            raise RuntimeError("Dataloader batch does not include img_path; update dataset.collate_fn first")

        results = yolo_model.predict(
            source=paths,
            imgsz=input_size,
            conf=score_thresh,
            iou=iou_thresh,
            max_det=max_det,
            device=device,
            verbose=False,
        )

        for i, result in enumerate(results):
            scale = batch['scale'][i]
            pad = batch['pad'][i]
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_classes = np.zeros((0,), dtype=np.int32)
            person_boxes = np.zeros((0, 4), dtype=np.float32)
            person_scores = np.zeros((0,), dtype=np.float32)
            person_kpts = np.zeros((0, 17, 3), dtype=np.float32)

            if result.boxes is not None and len(result.boxes) > 0:
                raw_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                raw_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
                raw_classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

                boxes_lb = _scale_boxes_to_letterbox(raw_boxes, scale, pad)

                if provider == 'ultralytics_pose':
                    keep = raw_classes == 0
                    pred_boxes = boxes_lb[keep]
                    pred_scores = raw_scores[keep]
                    pred_classes = np.zeros(len(pred_boxes), dtype=np.int32)
                    person_boxes = pred_boxes
                    person_scores = pred_scores

                    if result.keypoints is not None and len(result.keypoints) > 0:
                        raw_xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)[keep]
                        if getattr(result.keypoints, 'conf', None) is not None:
                            raw_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)[keep]
                        else:
                            raw_conf = np.ones(raw_xy.shape[:2], dtype=np.float32)
                        person_kpts = np.concatenate([raw_xy, raw_conf[..., None]], axis=-1)
                        person_kpts = _scale_kpts_to_letterbox(person_kpts, scale, pad)
                    else:
                        person_kpts = np.zeros((len(person_boxes), 17, 3), dtype=np.float32)
                else:
                    mapped = np.array([YOLO80_ID_TO_20.get(int(c), -1) for c in raw_classes], dtype=np.int32)
                    if num_classes == 80:
                        mapped = raw_classes
                    keep = mapped >= 0
                    pred_boxes = boxes_lb[keep]
                    pred_scores = raw_scores[keep]
                    pred_classes = mapped[keep]
                    person_mask = pred_classes == 0
                    person_boxes = pred_boxes[person_mask]
                    person_scores = pred_scores[person_mask]
                    person_kpts = np.zeros((len(person_boxes), 17, 3), dtype=np.float32)

            all_preds.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'classes': pred_classes,
                'person_boxes': person_boxes,
                'person_scores': person_scores,
                'person_kpts': person_kpts,
            })
            all_gts.append({
                'boxes': batch['boxes'][i].cpu().numpy().astype(np.float32),
                'classes': batch['classes'][i].cpu().numpy().astype(np.int32),
                'kpts': batch['kpts'][i].cpu().numpy().astype(np.float32),
            })

    return all_preds, all_gts


def compute_all_metrics(all_preds, all_gts, num_classes=80):
    """Compute comprehensive metrics.

    Returns dict with:
    - mAP@0.5 (all classes)
    - mAP@0.5:0.95 (all classes)
    - mAP@0.5 (no person)
    - AP@0.5 (person box)
    - AP@0.5 (pose)
    - Per-class AP@0.5
    """
    results = {}
    iou_thresholds = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
    det_aps = compute_ap_by_class_multi(
        all_preds, all_gts, iou_thresholds=iou_thresholds, num_classes=num_classes)
    ap50 = {cls_id: float(values[0]) for cls_id, values in det_aps.items()}

    # Detection AP@0.5
    valid_ap50 = [v for v in ap50.values() if v >= 0]
    results['mAP@0.5'] = float(np.mean(valid_ap50)) if valid_ap50 else 0.0
    results['mAP50'] = results['mAP@0.5']
    results['metrics/mAP50(B)'] = results['mAP@0.5']

    # Detection AP@0.5 (excluding person = class 0)
    ap50_no_person = [v for c, v in ap50.items() if c != 0 and v >= 0]
    results['mAP@0.5_no_person'] = float(np.mean(ap50_no_person)) if ap50_no_person else 0.0
    results['mAP50_no_person'] = results['mAP@0.5_no_person']

    # Person box AP@0.5
    results['AP_person_box@0.5'] = float(ap50.get(0, 0.0)) if ap50.get(0, -1) >= 0 else 0.0

    # Detection AP@0.5:0.95 (average over IoU thresholds)
    class_ap5095 = {}
    for cls_id, values in det_aps.items():
        valid = values[values >= 0]
        class_ap5095[cls_id] = float(np.mean(valid)) if len(valid) else -1.0
    valid_ap5095 = [v for v in class_ap5095.values() if v >= 0]
    results['mAP@0.5:0.95'] = float(np.mean(valid_ap5095)) if valid_ap5095 else 0.0
    results['mAP50-95'] = results['mAP@0.5:0.95']
    results['metrics/mAP50-95(B)'] = results['mAP@0.5:0.95']

    # Detection AP@0.5:0.95 (no person)
    valid_ap5095_np = [v for c, v in class_ap5095.items() if c != 0 and v >= 0]
    results['mAP@0.5:0.95_no_person'] = float(np.mean(valid_ap5095_np)) if valid_ap5095_np else 0.0
    results['mAP50-95_no_person'] = results['mAP@0.5:0.95_no_person']
    results['AP_person_box@0.5:0.95'] = (
        float(class_ap5095.get(0, 0.0)) if class_ap5095.get(0, -1) >= 0 else 0.0
    )

    # Pose AP (OKS-based)
    pose_aps = compute_pose_ap_multi(all_preds, all_gts, oks_thresholds=iou_thresholds)
    if pose_aps is None:
        results['AP_pose@0.5'] = 0.0
        results['AP_pose@0.5:0.95'] = 0.0
    else:
        results['AP_pose@0.5'] = float(pose_aps[0])
        valid_pose = pose_aps[pose_aps >= 0]
        results['AP_pose@0.5:0.95'] = float(np.mean(valid_pose)) if len(valid_pose) else 0.0
    results['mAPpose50'] = results['AP_pose@0.5']
    results['mAPpose50-95'] = results['AP_pose@0.5:0.95']
    results['metrics/mAP50(P)'] = results['AP_pose@0.5']
    results['metrics/mAP50-95(P)'] = results['AP_pose@0.5:0.95']

    # Per-class AP@0.5
    results['per_class_AP@0.5'] = {int(k): float(v) for k, v in ap50.items()}
    results['per_class_AP@0.5:0.95'] = {int(k): float(v) for k, v in class_ap5095.items()}

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None, help='Training config path')
    p.add_argument('--provider', type=str, default='test_model',
                   choices=['test_model', 'ultralytics_detect', 'ultralytics_pose'],
                   help='Evaluation model provider')
    p.add_argument('--weights', type=str, default=None, help='test_model checkpoint path')
    p.add_argument('--ultralytics-weights', type=str, default=None,
                   help='Ultralytics model weights, e.g. yolov8m.pt or yolov8m-pose.pt')
    p.add_argument('--model', type=str, default=None,
                   choices=['dual_head', 'unified_head', 'dual_neck', 'attn_dual', 'bifpn_dual'])
    p.add_argument('--data', type=str, required=True, help='Dataset directory')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--img-dir', type=str, default=None, help='Validation image directory under data root')
    p.add_argument('--label-dir', type=str, default=None, help='Validation label directory under data root')
    p.add_argument('--input-size', type=int, default=None, help='Evaluation input size')
    p.add_argument('--class-id-format', type=str, default=None,
                   choices=['yolo80', 'internal80', 'coco', 'coco80',
                            'coco20', 'internal', 'internal20',
                            'coco_category20', 'coco20_category', 'auto'],
                   help='Label class id format')
    p.add_argument('--max-samples', type=int, default=0,
                   help='Optional sample limit for quick evaluation smoke tests')
    p.add_argument('--score-thresh', type=float, default=None, help='Prediction score threshold')
    p.add_argument('--iou-thresh', type=float, default=None, help='NMS IoU threshold')
    p.add_argument('--max-det', type=int, default=None, help='Max detections per image after NMS')
    p.add_argument('--output', type=str, default=None, help='JSON output path')
    args = p.parse_args()

    from test_model.dataset import create_dataloader, collate_fn

    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}

    d_cfg = cfg.get('data', {})
    e_cfg = cfg.get('eval', {})

    model = None
    yolo_model = None
    if args.provider == 'test_model':
        if not args.model:
            p.error("--model is required when --provider test_model")
        if not args.weights:
            p.error("--weights is required when --provider test_model")
        from test_model.models import create_model

        model_kwargs = {
            'num_kpts': cfg.get('num_kpts', 17),
            'reg_max': cfg.get('reg_max', 16),
        }
        if args.model == 'unified_head':
            model_kwargs['num_classes'] = cfg.get('num_classes', 80)
        else:
            model_kwargs['num_det_classes'] = cfg.get('num_det_classes', 80)
        model = create_model(args.model, **model_kwargs)
        ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint: {args.weights}")
        print(f"Model params: {model.num_params / 1e6:.2f}M")
    else:
        if not args.ultralytics_weights:
            p.error("--ultralytics-weights is required for ultralytics providers")
        from ultralytics import YOLO
        yolo_model = YOLO(args.ultralytics_weights)
        print(f"Loaded Ultralytics weights: {args.ultralytics_weights}")

    # Create dataloader
    data_root = Path(args.data)
    img_dir = args.img_dir
    if img_dir is None:
        img_dir = d_cfg.get(
            'val_img',
            'images/val2017' if (data_root / 'images/val2017').exists() else 'val2017')
    label_dir = args.label_dir
    if label_dir is None:
        label_dir = d_cfg.get('val_label', 'labels/val2017')

    batch = args.batch or e_cfg.get('batch_size', 16)
    workers = args.workers if args.workers is not None else cfg.get('training', {}).get('workers', 4)
    input_size = args.input_size or d_cfg.get('input_size', 640)
    class_id_format = args.class_id_format or d_cfg.get('class_id_format', 'yolo80')
    score_thresh = args.score_thresh if args.score_thresh is not None else e_cfg.get('score_thresh', 0.01)
    iou_thresh = args.iou_thresh if args.iou_thresh is not None else e_cfg.get('iou_thresh', 0.6)
    max_det = args.max_det if args.max_det is not None else e_cfg.get('max_det', 300)
    num_classes = cfg.get('num_classes', 80)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    loader = create_dataloader(
        data_dir=data_root,
        img_dir=img_dir,
        label_dir=label_dir,
        input_size=input_size,
        batch_size=batch,
        use_mosaic=False,
        augment=False,
        shuffle=False,
        num_workers=workers,
        class_id_format=class_id_format,
    )
    if args.max_samples and 0 < args.max_samples < len(loader.dataset):
        loader = DataLoader(
            Subset(loader.dataset, range(args.max_samples)),
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    print(f"Eval samples: {len(loader.dataset)} | input_size={input_size} | "
          f"class_id_format={class_id_format} | provider={args.provider} | max_det={max_det}")

    # Evaluate
    if args.provider == 'test_model':
        all_preds, all_gts = evaluate(
            model, loader, device, score_thresh=score_thresh, iou_thresh=iou_thresh,
            max_det=max_det)
    else:
        all_preds, all_gts = evaluate_ultralytics(
            yolo_model, loader, device, score_thresh=score_thresh, iou_thresh=iou_thresh,
            max_det=max_det, provider=args.provider, input_size=input_size,
            num_classes=num_classes)
    metrics = compute_all_metrics(all_preds, all_gts, num_classes=num_classes)

    # Report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue
        print(f"  {k}: {v:.4f}")
    print()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {args.output}")

    return metrics


if __name__ == '__main__':
    main()
