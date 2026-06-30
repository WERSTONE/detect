"""Evaluate Vigil-v2 on the prepared COCO person-pose pretraining dataset.

Metrics:
  - bbox mAP: person-box AP averaged over IoU thresholds 0.50:0.95.
  - pose mAP: person-keypoint AP averaged over OKS thresholds 0.50:0.95.

Example:
  uv run python scripts/evaluate_coco_pose_map.py --limit 100
  uv run python scripts/evaluate_coco_pose_map.py --model vigil_v2 --weights checkpoints/vigil_v2/pretrain_best.pt
  uv run python scripts/evaluate_coco_pose_map.py --model yolov8_pose --weights checkpoints/yolo_pose/yolov8n-pose.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from Vigil.models.registry import create_model
from Vigil.train.dataset import UnifiedDataset

COCO_OKS_SIGMAS = torch.tensor(
    [
        0.26,
        0.25,
        0.25,
        0.35,
        0.35,
        0.79,
        0.79,
        0.72,
        0.72,
        0.62,
        0.62,
        1.07,
        1.07,
        0.87,
        0.87,
        0.89,
        0.89,
    ],
    dtype=torch.float32,
) / 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a registered model on prepared COCO person-pose data.")
    parser.add_argument("--data-root", default="data/processed/coco_person_pose")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None, help="Defaults to config.model.name")
    parser.add_argument("--weights", default=None, help="Defaults to the registered model's automatic checkpoint lookup")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    parser.add_argument("--conf-thres", type=float, default=0.01, help="Prediction score threshold for evaluation")
    parser.add_argument("--max-det", type=int, default=None, help="Maximum predictions kept per class per image")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0)
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = ((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0))
    area_b = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    return inter / (area_a + area_b - inter + 1e-9)


def oks(pred_kpt: torch.Tensor, gt_kpts: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if gt_kpts.numel() == 0:
        return torch.zeros(0)
    sigmas = COCO_OKS_SIGMAS.to(gt_kpts.device)
    visible = gt_kpts[..., 2] > 0
    dx = pred_kpt[None, :, 0] - gt_kpts[..., 0]
    dy = pred_kpt[None, :, 1] - gt_kpts[..., 1]
    areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1))
    exponent = (dx.square() + dy.square()) / ((2 * sigmas).square()[None, :] * areas[:, None] * 2 + 1e-9)
    score = torch.exp(-exponent) * visible
    denom = visible.sum(dim=1).clamp(min=1)
    return score.sum(dim=1) / denom


def average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0
    points = torch.linspace(0, 1, 101)
    ap = 0.0
    for recall_threshold in points:
        mask = recalls >= recall_threshold
        ap += float(precisions[mask].max()) if mask.any() else 0.0
    return ap / 101


def ap_at_threshold(
    predictions: list[dict],
    targets: dict[int, dict],
    threshold: float,
    metric: str,
    total_gt: int,
) -> float:
    if total_gt == 0:
        return 0.0

    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    matched: dict[int, set[int]] = {}
    tp, fp = [], []

    for pred in predictions:
        image_id = pred["image_id"]
        gt = targets.get(image_id)
        if gt is None or gt["boxes"].numel() == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        scores = box_iou(pred["box"], gt["boxes"]) if metric == "bbox" else oks(pred["kpt"], gt["kpts"], gt["boxes"])
        best_score, best_idx = scores.max(dim=0)
        used = matched.setdefault(image_id, set())

        if float(best_score) >= threshold and int(best_idx) not in used:
            used.add(int(best_idx))
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    tp_t = torch.tensor(tp).cumsum(0)
    fp_t = torch.tensor(fp).cumsum(0)
    recalls = tp_t / total_gt
    precisions = tp_t / (tp_t + fp_t + 1e-9)
    return average_precision(recalls, precisions)


def summarize_map(predictions: list[dict], targets: dict[int, dict], total_gt: int, metric: str) -> dict[str, float]:
    thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]
    aps = {f"AP{int(t * 100)}": ap_at_threshold(predictions, targets, t, metric, total_gt) for t in thresholds}
    return {
        "mAP": sum(aps.values()) / len(aps),
        "mAP50": aps["AP50"],
        "mAP75": aps["AP75"],
        **aps,
    }


def load_model(args: argparse.Namespace, device: torch.device):
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_name = args.model or config.get("model", {}).get("name", "vigil_v2")
    model_kwargs = config.get("model", {}).get("kwargs", {})

    pretrained = True
    if args.weights:
        weights = Path(args.weights)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        pretrained = str(weights)

    model = create_model(model_name, pretrained=pretrained, **model_kwargs)
    model.eval().to(device)
    return model_name, model


def sample_to_rgb_image(sample) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (sample.image.cpu() * std + mean).clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return image


def predict_persons(
    model,
    sample,
    device: torch.device,
    conf_thres: float,
    max_det: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return person boxes, scores and keypoints in the sample's 640x640 coordinate space."""
    if hasattr(model, "predict_val_full"):
        sample.image = sample.image.to(device)
        pred = model.predict_val_full(sample, score_thresh=conf_thres, max_det=max_det)
        return (
            pred["person_boxes"].detach().cpu(),
            pred["person_scores"].detach().cpu(),
            pred["person_kpts"].detach().cpu(),
        )

    # Generic registered-model fallback: use the public detect() contract.
    # The image is reconstructed from the non-augmented dataset sample, so both
    # predictions and GT are in the same 640x640 letterboxed coordinate space.
    rgb = sample_to_rgb_image(sample)
    detections = model.detect(rgb)
    person = detections.get("person", {})
    boxes = person.get("boxes", torch.empty(0, 4)).detach().cpu()
    scores = person.get("scores", torch.empty(0)).detach().cpu()
    kpts = person.get("kpts", torch.empty((len(boxes), 17, 3))).detach().cpu()
    keep = scores > conf_thres
    boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
    if max_det is not None and len(scores) > max_det:
        order = scores.argsort(descending=True)[:max_det]
        boxes, scores, kpts = boxes[order], scores[order], kpts[order]
    if kpts.numel() == 0:
        kpts = torch.empty((len(boxes), 17, 3))
    return boxes, scores, kpts


@torch.no_grad()
def collect_predictions(
    model,
    dataset: UnifiedDataset,
    device: torch.device,
    limit: int | None,
    conf_thres: float,
    max_det: int | None,
):
    targets: dict[int, dict] = {}
    bbox_predictions: list[dict] = []
    pose_predictions: list[dict] = []
    total_gt = 0
    count = min(len(dataset), limit) if limit else len(dataset)

    for image_id in range(count):
        sample = dataset[image_id]
        gt_boxes = sample.person_boxes.cpu()
        gt_kpts = sample.person_kpts.cpu()
        targets[image_id] = {"boxes": gt_boxes, "kpts": gt_kpts}
        total_gt += len(gt_boxes)

        boxes, scores, kpts = predict_persons(model, sample, device, conf_thres, max_det)

        for box, score, kpt in zip(boxes, scores, kpts):
            item = {
                "image_id": image_id,
                "score": float(score),
                "box": box,
                "kpt": kpt,
            }
            bbox_predictions.append(item)
            if kpt.numel() > 0 and bool((kpt[:, 2] > 0).any()):
                pose_predictions.append(item)

        if (image_id + 1) % 100 == 0 or image_id + 1 == count:
            logger.info(f"Evaluated {image_id + 1}/{count} images")

    return targets, bbox_predictions, pose_predictions, total_gt, count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    model_name, model = load_model(args, device)
    dataset = UnifiedDataset(args.data_root, "coco_person_pose", augment=False)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {args.data_root} ({len(dataset)} images)")
    logger.info(f"Prediction filter: conf_thres={args.conf_thres}, max_det={args.max_det}")

    targets, bbox_predictions, pose_predictions, total_gt, count = collect_predictions(
        model, dataset, device, args.limit, args.conf_thres, args.max_det)
    bbox = summarize_map(bbox_predictions, targets, total_gt, "bbox")
    pose = summarize_map(pose_predictions, targets, total_gt, "pose")

    result = {
        "model": model_name,
        "weights": args.weights or "auto",
        "data_root": args.data_root,
        "images": count,
        "conf_thres": args.conf_thres,
        "max_det": args.max_det,
        "person_gt": total_gt,
        "person_predictions": len(bbox_predictions),
        "person_pose_predictions": len(pose_predictions),
        "bbox": bbox,
        "pose": pose,
    }

    print(json.dumps(result, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"Wrote metrics to {out}")


if __name__ == "__main__":
    main()
