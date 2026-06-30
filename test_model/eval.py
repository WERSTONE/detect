"""Evaluation script for multi-head verification models.

Computes:
- Detection mAP@0.5 and mAP@0.5:0.95
- Keypoint mAP (OKS-based)
- Per-class metrics
- Outputs comparison-ready JSON

Usage:
    python -m test_model.eval --model dual_head --weights checkpoints/dual_head/dual_head_best.pt --data data/coco20
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


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


def compute_ap_by_class(predictions, ground_truths, iou_thresh=0.5):
    """Compute AP for each class using 101-point interpolation.

    Args:
        predictions: List[dict] with 'boxes'[K,4], 'scores'[K], 'classes'[K]
        ground_truths: List[dict] with 'boxes'[M,4], 'classes'[M]

    Returns:
        dict mapping class_id -> AP value
    """
    num_classes = 20
    aps = {}

    for cls_id in range(num_classes):
        # Collect all predictions for this class
        all_dets = []
        for img_idx, pred in enumerate(predictions):
            mask = pred['classes'] == cls_id
            if not mask.any():
                continue
            boxes = pred['boxes'][mask]
            scores = pred['scores'][mask]
            for i in range(len(boxes)):
                all_dets.append((img_idx, float(scores[i]), boxes[i].tolist()))

        all_dets.sort(key=lambda x: x[1], reverse=True)

        # Count GTs for this class
        gt_counts = []
        gt_matched = []
        for gt in ground_truths:
            mask = gt['classes'] == cls_id
            gt_counts.append(int(mask.sum()))
            gt_matched.append(np.zeros(int(mask.sum()), dtype=bool))
        total_gt = sum(gt_counts)

        if total_gt == 0:
            aps[cls_id] = -1  # undefined
            continue

        tp = np.zeros(len(all_dets))
        fp = np.zeros(len(all_dets))

        for det_idx, (img_idx, score, det_box) in enumerate(all_dets):
            gt_boxes_all = ground_truths[img_idx]['boxes']
            gt_cls_all = ground_truths[img_idx]['classes']
            mask = gt_cls_all == cls_id

            if not mask.any():
                fp[det_idx] = 1
                continue

            gt_boxes_cls = gt_boxes_all[mask]
            local_to_global = np.where(mask)[0]
            ious = compute_iou(np.array(det_box), gt_boxes_cls)

            best_iou, best_local = 0.0, -1
            for li in range(len(gt_boxes_cls)):
                gi = local_to_global[li]
                if not gt_matched[img_idx][gi] and ious[li] > best_iou:
                    best_iou = float(ious[li])
                    best_local = li

            if best_iou >= iou_thresh:
                tp[det_idx] = 1
                gt_matched[img_idx][local_to_global[best_local]] = True
            else:
                fp[det_idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / total_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

        ap = 0.0
        for t in np.linspace(0, 1, 101):
            ap += (np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0) / 101.0
        aps[cls_id] = float(ap)

    return aps


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

    K, M = len(pred_kpts), len(gt_kpts)
    oks = np.zeros((K, M))

    for k in range(K):
        for m in range(M):
            gt_box = gt_boxes[m]
            area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            scale = np.sqrt(max(area, 1.0))

            d2 = np.sum((pred_kpts[k, :, :2] - gt_kpts[m, :, :2]) ** 2, axis=1)
            k2 = (sigmas ** 2) * (2 * scale) ** 2
            visible = gt_kpts[m, :, 2] > 0

            if visible.sum() == 0:
                oks[k, m] = 0.0
            else:
                oks[k, m] = float(np.mean(np.exp(-d2[visible] / (2 * k2[visible]))))

    return oks


def compute_pose_ap(predictions, ground_truths):
    """Compute pose AP@0.5 using OKS.

    Args:
        predictions: List[dict] with person_boxes, person_scores, person_kpts
        ground_truths: List[dict] with boxes, classes, kpts
    """
    oks_thresh = 0.5
    all_dets = []
    gt_person_counts = []

    for img_idx, pred in enumerate(predictions):
        if 'person_boxes' not in pred:
            continue
        p_boxes = pred['person_boxes']
        p_scores = pred['person_scores']
        p_kpts = pred['person_kpts']

        for i in range(len(p_boxes)):
            all_dets.append((img_idx, float(p_scores[i]),
                            p_boxes[i].tolist(), p_kpts[i]))

        gt = ground_truths[img_idx]
        person_mask = gt['classes'] == 0
        gt_person_counts.append(int(person_mask.sum()))

    all_dets.sort(key=lambda x: x[1], reverse=True)
    total_gt = sum(gt_person_counts)

    if total_gt == 0:
        return None

    tp = np.zeros(len(all_dets))
    fp = np.zeros(len(all_dets))
    gt_matched = [np.zeros(c, dtype=bool) for c in gt_person_counts]

    for det_idx, (img_idx, score, det_box, det_kpts) in enumerate(all_dets):
        gt = ground_truths[img_idx]
        person_mask = gt['classes'] == 0
        if not person_mask.any():
            fp[det_idx] = 1
            continue

        gt_boxes_person = gt['boxes'][person_mask]
        gt_kpts_person = gt['kpts'][person_mask]
        local_to_global = np.where(person_mask)[0]

        # Compute OKS
        oks = compute_pose_oks(
            det_kpts.reshape(1, 17, 3),
            gt_kpts_person.reshape(-1, 17, 3),
            gt_boxes_person.reshape(-1, 4),
        )[0]

        # Find best unmatched GT
        best_oks, best_local = 0.0, -1
        for li in range(len(gt_boxes_person)):
            gi = local_to_global[li]
            if not gt_matched[img_idx][gi] and oks[li] > best_oks:
                best_oks = float(oks[li])
                best_local = li

        if best_oks >= oks_thresh:
            tp[det_idx] = 1
            gt_matched[img_idx][local_to_global[best_local]] = True
        else:
            fp[det_idx] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        ap += (np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0) / 101.0
    return float(ap)


def evaluate(model, dataloader, device='cuda'):
    """Run inference and collect predictions + ground truths."""
    model.eval()
    model.to(device)

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)

            # Forward pass
            predictions = model.predict_val(images, score_thresh=0.01, iou_thresh=0.6)

            for i in range(len(images)):
                pred = predictions[i]
                all_preds.append({
                    'boxes': pred['boxes'].cpu().numpy().astype(np.float32),
                    'scores': pred['scores'].cpu().numpy().astype(np.float32),
                    'classes': pred['classes'].cpu().numpy().astype(np.int32),
                    'person_boxes': pred.get('person_boxes', pred['boxes'][pred['classes'] == 0]).cpu().numpy().astype(np.float32),
                    'person_scores': pred.get('person_scores', pred['scores'][pred['classes'] == 0]).cpu().numpy().astype(np.float32),
                    'person_kpts': pred.get('person_kpts',
                        pred.get('kpts', np.zeros((0, 17, 3), dtype=np.float32))).cpu().numpy().astype(np.float32),
                })

                all_gts.append({
                    'boxes': batch['boxes'][i].cpu().numpy().astype(np.float32),
                    'classes': batch['classes'][i].cpu().numpy().astype(np.int32),
                    'kpts': batch['kpts'][i].cpu().numpy().astype(np.float32),
                })

    return all_preds, all_gts


def compute_all_metrics(all_preds, all_gts):
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

    # Detection AP@0.5
    ap50 = compute_ap_by_class(all_preds, all_gts, iou_thresh=0.5)
    valid_ap50 = [v for v in ap50.values() if v >= 0]
    results['mAP@0.5'] = float(np.mean(valid_ap50)) if valid_ap50 else 0.0

    # Detection AP@0.5 (excluding person = class 0)
    ap50_no_person = [v for c, v in ap50.items() if c != 0 and v >= 0]
    results['mAP@0.5_no_person'] = float(np.mean(ap50_no_person)) if ap50_no_person else 0.0

    # Person box AP@0.5
    results['AP_person_box@0.5'] = float(ap50.get(0, 0.0)) if ap50.get(0, -1) >= 0 else 0.0

    # Detection AP@0.5:0.95 (average over IoU thresholds)
    aps_5095 = []
    for iou_t in np.arange(0.5, 1.0, 0.05):
        ap_t = compute_ap_by_class(all_preds, all_gts, iou_thresh=float(iou_t))
        valid = [v for v in ap_t.values() if v >= 0]
        if valid:
            aps_5095.append(np.mean(valid))
    results['mAP@0.5:0.95'] = float(np.mean(aps_5095)) if aps_5095 else 0.0

    # Detection AP@0.5:0.95 (no person)
    aps_5095_np = []
    for iou_t in np.arange(0.5, 1.0, 0.05):
        ap_t = compute_ap_by_class(all_preds, all_gts, iou_thresh=float(iou_t))
        valid_np = [v for c, v in ap_t.items() if c != 0 and v >= 0]
        if valid_np:
            aps_5095_np.append(np.mean(valid_np))
    results['mAP@0.5:0.95_no_person'] = float(np.mean(aps_5095_np)) if aps_5095_np else 0.0

    # Pose AP@0.5 (OKS-based)
    pose_ap = compute_pose_ap(all_preds, all_gts)
    results['AP_pose@0.5'] = pose_ap if pose_ap is not None else 0.0

    # Per-class AP@0.5
    results['per_class_AP@0.5'] = {int(k): float(v) for k, v in ap50.items()}

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True, help='Model checkpoint path')
    p.add_argument('--model', type=str, required=True,
                   choices=['dual_head', 'unified_head', 'dual_neck', 'attn_dual', 'bifpn_dual'])
    p.add_argument('--data', type=str, required=True, help='Dataset directory')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=str, default=None, help='JSON output path')
    args = p.parse_args()

    from test_model.models import create_model
    from test_model.dataset import create_dataloader

    # Load model
    model = create_model(args.model)
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint: {args.weights}")
    print(f"Model params: {model.num_params / 1e6:.2f}M")

    # Create dataloader
    loader = create_dataloader(
        data_dir=Path(args.data),
        img_dir='images/val2017',
        label_dir='labels/val2017',
        batch_size=args.batch,
        use_mosaic=False,
        augment=False,
        shuffle=False,
        num_workers=args.workers,
    )
    print(f"Eval samples: {len(loader.dataset)}")

    # Evaluate
    all_preds, all_gts = evaluate(model, loader, args.device)
    metrics = compute_all_metrics(all_preds, all_gts)

    # Report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if k == 'per_class_AP@0.5':
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
