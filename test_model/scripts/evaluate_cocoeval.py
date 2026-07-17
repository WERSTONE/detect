#!/usr/bin/env python
"""Evaluate custom test_model checkpoints with official pycocotools COCOeval.

This script exports model predictions to COCO result JSON files in original
image coordinates, then runs COCOeval for bbox and/or keypoints.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from test_model.dataset import (  # noqa: E402
    COCO_CATEGORY_ID_TO_20,
    COCO_CATEGORY_ID_TO_80,
    collate_fn,
    create_dataloader,
)
from test_model.models import create_model  # noqa: E402


COCO80_TO_CATEGORY_ID = {v: k for k, v in COCO_CATEGORY_ID_TO_80.items()}
COCO20_TO_CATEGORY_ID = {v: k for k, v in COCO_CATEGORY_ID_TO_20.items()}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run official COCOeval on a custom test_model checkpoint")
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "test_model/config.yaml"))
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--model", type=str, default=None,
                   choices=["dual_head", "unified_head", "dual_neck", "attn_dual", "bifpn_dual"])
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--img-dir", type=str, default=None)
    p.add_argument("--label-dir", type=str, default=None)
    p.add_argument("--instances-json", type=str, default=None)
    p.add_argument("--keypoints-json", type=str, default=None)
    p.add_argument("--task", type=str, default="both", choices=["both", "bbox", "keypoints"])
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--class-id-format", type=str, default=None,
                   choices=["yolo80", "internal80", "coco", "coco80",
                            "coco20", "internal", "internal20",
                            "coco_category20", "coco20_category", "auto"])
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=None)
    p.add_argument("--iou-thresh", type=float, default=None)
    p.add_argument("--max-det", type=int, default=None,
                   help="Max detections kept after model NMS")
    p.add_argument("--coco-max-det", type=int, default=100,
                   help="COCOeval bbox maxDets summary value")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--prefix", type=str, default=None)
    return p.parse_args()


def load_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(args, cfg, device):
    model_name = args.model or cfg.get("model", "bifpn_dual")
    model_kwargs = {
        "num_kpts": cfg.get("num_kpts", 17),
        "reg_max": cfg.get("reg_max", 16),
    }
    if model_name == "unified_head":
        model_kwargs["num_classes"] = cfg.get("num_classes", cfg.get("num_det_classes", 80))
    else:
        model_kwargs["num_det_classes"] = cfg.get("num_det_classes", cfg.get("num_classes", 80))

    model = create_model(model_name, **model_kwargs)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.weights}")
    print(f"Model: {model_name} | params={model.num_params / 1e6:.2f}M")
    return model


def resolve_annotation_path(data_root, user_path, filename):
    if user_path:
        return Path(user_path)
    candidates = [
        data_root / "annotations" / filename,
        data_root / filename,
        data_root.parent / "annotations" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def class_to_category_id(cls_id, num_classes, keep_classes):
    cls_id = int(cls_id)
    if num_classes == 1 or keep_classes == [0]:
        return 1 if cls_id == 0 else None
    if num_classes == 80:
        return COCO80_TO_CATEGORY_ID.get(cls_id)
    if num_classes == 20:
        return COCO20_TO_CATEGORY_ID.get(cls_id)
    return COCO80_TO_CATEGORY_ID.get(cls_id)


def boxes_from_letterbox(boxes, scale, pad, orig_shape):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if len(boxes) == 0:
        return boxes
    pad_l, pad_t = pad
    orig_h, orig_w = orig_shape
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(pad_l)) / float(scale)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(pad_t)) / float(scale)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    return boxes


def kpts_from_letterbox(kpts, scale, pad, orig_shape):
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 17, 3).copy()
    if len(kpts) == 0:
        return kpts
    pad_l, pad_t = pad
    orig_h, orig_w = orig_shape
    kpts[..., 0] = (kpts[..., 0] - float(pad_l)) / float(scale)
    kpts[..., 1] = (kpts[..., 1] - float(pad_t)) / float(scale)
    kpts[..., 0] = np.clip(kpts[..., 0], 0, orig_w)
    kpts[..., 1] = np.clip(kpts[..., 1], 0, orig_h)
    kpts[..., 2] = np.clip(kpts[..., 2], 0, 1)
    return kpts


def export_predictions(model, loader, device, num_classes, keep_classes,
                       score_thresh, iou_thresh, max_det):
    bbox_results = []
    kpt_results = []
    image_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            preds = model.predict_val(
                images,
                score_thresh=score_thresh,
                iou_thresh=iou_thresh,
                max_det=max_det,
            )

            for i, pred in enumerate(preds):
                image_id = batch["image_id"][i]
                if image_id is None:
                    img_path = batch["img_path"][i]
                    raise ValueError(
                        f"Cannot infer COCO image_id from image path: {img_path}")
                image_id = int(image_id)
                image_ids.append(image_id)

                scale = batch["scale"][i]
                pad = batch["pad"][i]
                orig_shape = batch["orig_shape"][i]

                boxes = boxes_from_letterbox(
                    pred["boxes"].detach().cpu().numpy(), scale, pad, orig_shape)
                scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
                classes = pred["classes"].detach().cpu().numpy().astype(np.int32)
                kpts = kpts_from_letterbox(
                    pred.get("kpts", torch.zeros(0, 17, 3, device=images.device))
                    .detach().cpu().numpy(),
                    scale, pad, orig_shape)
                if len(kpts) != len(boxes):
                    kpts = np.zeros((len(boxes), 17, 3), dtype=np.float32)

                for box, score, cls_id in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w <= 0.0 or h <= 0.0:
                        continue
                    category_id = class_to_category_id(cls_id, num_classes, keep_classes)
                    if category_id is None:
                        continue
                    bbox_results.append({
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    })

                person_mask = classes == 0
                for box, score, kpt in zip(boxes[person_mask], scores[person_mask], kpts[person_mask]):
                    x1, y1, x2, y2 = box.tolist()
                    if (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
                        continue
                    if not np.any(kpt[:, 2] > 0):
                        continue
                    kpt_flat = kpt.reshape(-1).astype(float).tolist()
                    kpt_results.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "keypoints": kpt_flat,
                        "score": float(score),
                    })

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {(batch_idx + 1) * loader.batch_size} images...")

    return bbox_results, kpt_results, sorted(set(image_ids))


def empty_metrics(task_name):
    return {
        f"{task_name}/AP": 0.0,
        f"{task_name}/AP50": 0.0,
        f"{task_name}/AP75": 0.0,
        f"{task_name}/note": "no predictions",
    }


def summarize_cocoeval(coco_eval, iou_type):
    stats = [float(x) for x in coco_eval.stats]
    if iou_type == "bbox":
        keys = [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
        ]
    else:
        keys = [
            "AP", "AP50", "AP75", "AP_medium", "AP_large",
            "AR", "AR50", "AR75", "AR_medium", "AR_large",
        ]
    return {f"{iou_type}/{k}": stats[i] for i, k in enumerate(keys[:len(stats)])}


def run_cocoeval(annotation_json, results_json, iou_type, image_ids, coco_max_det=100):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit(
            "pycocotools is required for official COCOeval. "
            "Install it with: pip install pycocotools"
        ) from exc

    with open(results_json, encoding="utf-8") as f:
        predictions = json.load(f)
    if not predictions:
        return empty_metrics(iou_type)

    coco_gt = COCO(str(annotation_json))
    coco_dt = coco_gt.loadRes(str(results_json))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.imgIds = list(image_ids)
    if iou_type == "bbox":
        coco_eval.params.maxDets = [1, 10, int(coco_max_det)]
    elif iou_type == "keypoints":
        coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return summarize_cocoeval(coco_eval, iou_type)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    d_cfg = cfg.get("data", {})
    e_cfg = cfg.get("eval", {})

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    data_root = Path(args.data or d_cfg.get("root", "data/coco2017"))
    img_dir = args.img_dir or d_cfg.get(
        "val_img",
        "images/val2017" if (data_root / "images/val2017").exists() else "val2017",
    )
    label_dir = args.label_dir or d_cfg.get("val_label", "labels/val2017")
    input_size = args.input_size or d_cfg.get("input_size", 640)
    batch = args.batch or e_cfg.get("batch_size", 16)
    workers = args.workers if args.workers is not None else cfg.get("training", {}).get("workers", 4)
    class_id_format = args.class_id_format or d_cfg.get("class_id_format", "yolo80")
    score_thresh = args.score_thresh if args.score_thresh is not None else e_cfg.get("score_thresh", 0.001)
    iou_thresh = args.iou_thresh if args.iou_thresh is not None else e_cfg.get("iou_thresh", 0.7)
    max_det = args.max_det if args.max_det is not None else e_cfg.get("max_det", 300)

    num_classes = cfg.get("num_classes", cfg.get("num_det_classes", 80))
    keep_classes = d_cfg.get("keep_classes", None)
    if keep_classes is None and cfg.get("num_det_classes", 80) == 1:
        keep_classes = [0]
    if keep_classes is not None:
        keep_classes = [int(c) for c in keep_classes]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.weights).resolve().parent / "cocoeval"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or Path(args.weights).stem

    model = build_model(args, cfg, device)
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
        drop_last=False,
        class_id_format=class_id_format,
        keep_classes=keep_classes,
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

    print(
        f"COCOeval export samples={len(loader.dataset)} input_size={input_size} "
        f"score={score_thresh} nms_iou={iou_thresh} max_det={max_det} "
        f"coco_max_det={args.coco_max_det} keep_classes={keep_classes}"
    )
    bbox_results, kpt_results, image_ids = export_predictions(
        model, loader, device, num_classes, keep_classes,
        score_thresh, iou_thresh, max_det,
    )

    bbox_json = output_dir / f"{prefix}_bbox_predictions.json"
    kpt_json = output_dir / f"{prefix}_keypoint_predictions.json"
    with open(bbox_json, "w", encoding="utf-8") as f:
        json.dump(bbox_results, f)
    with open(kpt_json, "w", encoding="utf-8") as f:
        json.dump(kpt_results, f)
    print(f"Saved bbox predictions: {bbox_json} ({len(bbox_results)} results)")
    print(f"Saved keypoint predictions: {kpt_json} ({len(kpt_results)} results)")

    metrics = {
        "weights": str(args.weights),
        "config": str(args.config),
        "num_images": len(image_ids),
        "num_bbox_predictions": len(bbox_results),
        "num_keypoint_predictions": len(kpt_results),
    }

    if args.task in ("both", "bbox"):
        instances_json = resolve_annotation_path(
            data_root, args.instances_json, "instances_val2017.json")
        if not instances_json.exists():
            print(f"Skip bbox COCOeval, annotation not found: {instances_json}")
        else:
            metrics.update(run_cocoeval(
                instances_json, bbox_json, "bbox", image_ids,
                coco_max_det=args.coco_max_det,
            ))

    if args.task in ("both", "keypoints"):
        keypoints_json = resolve_annotation_path(
            data_root, args.keypoints_json, "person_keypoints_val2017.json")
        if not keypoints_json.exists():
            print(f"Skip keypoint COCOeval, annotation not found: {keypoints_json}")
        else:
            metrics.update(run_cocoeval(
                keypoints_json, kpt_json, "keypoints", image_ids,
                coco_max_det=args.coco_max_det,
            ))

    metrics_json = output_dir / f"{prefix}_cocoeval_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_json}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
