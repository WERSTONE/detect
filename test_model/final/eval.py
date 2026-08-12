"""Final three-task evaluation: domain mAP + attribute metrics + COCO pose.

Runs the final ``DomainAttrBiFPN`` model on the val splits and reports:
  - domain detection mAP (classes 1..4) on Detect valid/
  - person detection AP (class 0) on Attr/Pose val
  - attribute precision/recall/F1 per attribute on Attr val
  - optional official COCO keypoint AP when the server COCO annotations exist

Prediction/GT boxes are compared in letterboxed 640px input space, matching
how the model is trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.final.data import (  # noqa: E402
    AttrDataset,
    DetectDataset,
    PoseDataset,
    _resolve_root,
    make_loader,
)
from test_model.final.model import create_final_model  # noqa: E402

DETECT_DIR_MAP = {
    "train": ("train", "train"),
    "val": ("valid", "valid"),
}
ATTR_SPLITS = ("train", "val")
POSE_DIR_MAP = {
    "train": ("train2017", "labels/train2017"),
    "val": ("val2017", "labels/val2017"),
}
_NO_AUG = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0, flip_lr=0.0
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the final three-head model on val splits")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "test_model/final/yaml/final_three_head.yaml"),
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", default="val", choices=["train", "val", "all"])
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=0,
                   help="Limit images per split (quick sanity runs)")
    p.add_argument("--score-thresh", type=float, default=None)
    p.add_argument("--domain-conf", type=float, default=None)
    p.add_argument("--person-conf", type=float, default=None)
    p.add_argument("--attr-thresh", type=float, default=None)
    p.add_argument("--iou-thresh", type=float, default=None)
    p.add_argument("--match-iou", type=float, default=None)
    p.add_argument("--max-det", type=int, default=None)
    p.add_argument("--no-coco-pose", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def normalize_device(device):
    device = str(device or "cuda")
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    return device


def build_model(cfg, weights, device):
    data_cfg = cfg.get("data", {}) or {}
    domain_cfg = cfg.get("domain_det", {}) or {}
    attr_cfg = cfg.get("attributes", {}) or {}
    neck_cfg = cfg.get("neck", {}) or {}
    assigner_cfg = cfg.get("assigner", {}) or {}
    model = create_final_model(
        name=cfg.get("model", "final_three_head"),
        domain_num_classes=domain_cfg.get("num_classes", 4),
        domain_class_map=domain_cfg.get("class_map", {}),
        num_attrs=attr_cfg.get("num_attrs", 4),
        attr_names=attr_cfg.get("names", None),
        num_kpts=cfg.get("num_kpts", 17),
        reg_max=cfg.get("reg_max", 16),
        input_size=data_cfg.get("input_size", 640),
        neck_use_p2_context=neck_cfg.get("use_p2_context", False),
        neck_downsample=neck_cfg.get("downsample", "conv"),
        neck_out_channels=neck_cfg.get("out_channels", None),
        assigner_topk=assigner_cfg.get("topk", 10),
        assigner_alpha=assigner_cfg.get("alpha", 0.5),
        assigner_beta=assigner_cfg.get("beta", 6.0),
        assigner_eps=assigner_cfg.get("eps", 1.0e-9),
    ).to(device)

    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = None
    if hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        for key in ("ema", "model"):
            obj = ckpt.get(key)
            if hasattr(obj, "state_dict"):
                state = obj.state_dict()
                break
        if state is None:
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if state is None:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")
    target = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    model.load_state_dict(matched, strict=False)
    print(
        f"Loaded: {weights} | compatible={len(matched)}/{len(target)} "
        f"epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else 'n/a'}"
    )
    model.eval()
    return model


def _merge_task_cfg(train_cfg, val_cfg):
    merged = dict(train_cfg)
    merged.update(val_cfg or {})
    return merged


def build_eval_loaders(cfg, split, batch, workers, input_size, max_samples=0):
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = data_cfg.get("train", {}) or {}
    val_cfg = data_cfg.get("val", {}) or {}
    loaders = []

    def _maybe_trim(ds):
        if max_samples and 0 < max_samples < len(ds):
            return Subset(ds, range(max_samples))
        return ds

    det_cfg = _merge_task_cfg(train_cfg.get("detect", {}), val_cfg.get("detect", {}))
    det_root = _resolve_root(det_cfg.get("root", ""))
    det_keys = ("train", "val") if split == "all" else (split,)
    for key in det_keys:
        img_dir, lbl_dir = DETECT_DIR_MAP.get(key, ("valid", "valid"))
        img_dir = det_cfg.get(f"{key}_images", f"{img_dir}/images")
        lbl_dir = det_cfg.get(f"{key}_labels", f"{lbl_dir}/labels")
        if (det_root / img_dir).exists() and (det_root / lbl_dir).exists():
            ds = DetectDataset(
                det_root, img_dir, lbl_dir,
                input_size=input_size, augment=False, **_NO_AUG,
            )
            ds = _maybe_trim(ds)
            loaders.append(("detect", key, make_loader(
                ds, batch, workers, shuffle=False, drop_last=False)))
            print(f"Eval detect[{key}]: {img_dir} / {lbl_dir} | {len(ds)}")

    attr_cfg = _merge_task_cfg(train_cfg.get("attr", {}), val_cfg.get("attr", {}))
    attr_root = _resolve_root(attr_cfg.get("root", ""))
    if attr_root.exists():
        attr_keys = ATTR_SPLITS if split == "all" else (split,)
        for key in attr_keys:
            ds = AttrDataset(
                attr_root, split=key,
                input_size=input_size, augment=False, **_NO_AUG,
            )
            ds = _maybe_trim(ds)
            loaders.append(("attr", key, make_loader(
                ds, batch, workers, shuffle=False, drop_last=False)))
            print(f"Eval attr[{key}]: split={key} | {len(ds)}")

    pose_cfg = _merge_task_cfg(train_cfg.get("pose", {}), val_cfg.get("pose", {}))
    pose_root = _resolve_root(pose_cfg.get("root", ""))
    pose_keys = ("train", "val") if split == "all" else (split,)
    for key in pose_keys:
        img_dir, lbl_dir = POSE_DIR_MAP.get(key, ("val2017", "labels/val2017"))
        img_dir = pose_cfg.get(f"{key}_images", img_dir)
        lbl_dir = pose_cfg.get(f"{key}_labels", lbl_dir)
        if (pose_root / img_dir).exists() and (pose_root / lbl_dir).exists():
            ds = PoseDataset(
                pose_root, img_dir, lbl_dir,
                input_size=input_size,
                source_class_format=pose_cfg.get("class_id_format", "yolo80"),
                augment=False, **_NO_AUG,
            )
            ds = _maybe_trim(ds)
            loaders.append(("pose", key, make_loader(
                ds, batch, workers, shuffle=False, drop_last=False)))
            print(f"Eval pose[{key}]: {img_dir} / {lbl_dir} | {len(ds)}")

    if not loaders:
        raise RuntimeError("No eval data sources available")
    return loaders


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 0, None)
    area_b = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def average_precision(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def eval_ap(preds_by_class, gts_by_image_class, class_ids, iou_thresholds):
    out = {}
    for cls_id in class_ids:
        gt_count = sum(
            len(v) for (img_idx, c), v in gts_by_image_class.items()
            if c == cls_id
        )
        cls_preds = sorted(
            preds_by_class.get(cls_id, []),
            key=lambda x: x["score"], reverse=True,
        )
        ap_values = []
        ap50 = 0.0
        no_preds = len(cls_preds) == 0
        for thr in iou_thresholds:
            matched = {
                key: np.zeros(len(boxes), dtype=bool)
                for key, boxes in gts_by_image_class.items()
                if key[1] == cls_id
            }
            tp = np.zeros(len(cls_preds), dtype=np.float32)
            fp = np.zeros(len(cls_preds), dtype=np.float32)
            for i, pred in enumerate(cls_preds):
                key = (pred["image_idx"], cls_id)
                gt_boxes = gts_by_image_class.get(key, [])
                if len(gt_boxes) == 0:
                    fp[i] = 1.0
                    continue
                ious = box_iou(
                    np.asarray([pred["box"]], dtype=np.float32), gt_boxes)[0]
                best = int(np.argmax(ious))
                if ious[best] >= thr and not matched[key][best]:
                    tp[i] = 1.0
                    matched[key][best] = True
                else:
                    fp[i] = 1.0
            if gt_count == 0:
                ap = None
            elif no_preds:
                ap = 0.0
            else:
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recalls = tp_cum / max(gt_count, 1)
                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
                ap = average_precision(recalls, precisions)
            if abs(thr - 0.5) < 1e-9:
                ap50 = ap
            if ap is not None:
                ap_values.append(ap)
        out[cls_id] = {
            "gt": int(gt_count),
            "pred": int(len(cls_preds)),
            "AP50": ap50,
            "AP50_95": float(np.mean(ap_values)) if ap_values else None,
        }
    valid = [v["AP50_95"] for v in out.values() if v["AP50_95"] is not None]
    valid50 = [v["AP50"] for v in out.values() if v["AP50"] is not None]
    out["mAP50_95"] = float(np.mean(valid)) if valid else None
    out["mAP50"] = float(np.mean(valid50)) if valid50 else None
    return out


def eval_attrs(gt_persons, pred_persons, attr_names, attr_thresh, match_iou):
    counts = {
        name: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "unknown": 0}
        for name in attr_names
    }
    matched_persons = 0
    total_gt_persons = 0
    for gt_boxes, gt_attrs, gt_masks, pred_boxes, pred_attrs, pred_scores in zip(
        gt_persons["boxes"],
        gt_persons["attrs"],
        gt_persons["masks"],
        pred_persons["boxes"],
        pred_persons["attrs"],
        pred_persons["scores"],
    ):
        total_gt_persons += len(gt_boxes)
        if len(gt_boxes) == 0:
            continue
        order = np.argsort(-pred_scores) if len(pred_scores) else []
        used_gt = np.zeros(len(gt_boxes), dtype=bool)
        for pi in order:
            if len(gt_boxes) == 0:
                break
            ious = box_iou(
                np.asarray([pred_boxes[pi]], dtype=np.float32), gt_boxes)[0]
            if len(ious) == 0:
                continue
            gi = int(np.argmax(ious))
            if ious[gi] < match_iou or used_gt[gi]:
                continue
            used_gt[gi] = True
            matched_persons += 1
            for ai, name in enumerate(attr_names):
                if ai >= gt_masks.shape[1] or gt_masks[gi, ai] <= 0.0:
                    counts[name]["unknown"] += 1
                    continue
                gt_pos = gt_attrs[gi, ai] > 0.5
                pred_pos = pred_attrs[pi, ai] >= attr_thresh
                if gt_pos and pred_pos:
                    counts[name]["tp"] += 1
                elif (not gt_pos) and pred_pos:
                    counts[name]["fp"] += 1
                elif gt_pos and (not pred_pos):
                    counts[name]["fn"] += 1
                else:
                    counts[name]["tn"] += 1
    metrics = {}
    micro = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "unknown": 0}
    for name, c in counts.items():
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        known = tp + fp + tn + fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(known, 1)
        metrics[name] = {
            **c,
            "known": int(known),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }
        for key in micro:
            micro[key] += c[key]
    known = micro["tp"] + micro["fp"] + micro["tn"] + micro["fn"]
    p = micro["tp"] / max(micro["tp"] + micro["fp"], 1)
    r = micro["tp"] / max(micro["tp"] + micro["fn"], 1)
    metrics["micro"] = {
        **micro,
        "known": int(known),
        "precision": float(p),
        "recall": float(r),
        "f1": float(2 * p * r / max(p + r, 1e-9)),
        "accuracy": float((micro["tp"] + micro["tn"]) / max(known, 1)),
    }
    return {
        "match_iou": float(match_iou),
        "attr_thresh": float(attr_thresh),
        "gt_persons": int(total_gt_persons),
        "matched_persons": int(matched_persons),
        "person_match_rate": float(matched_persons / max(total_gt_persons, 1)),
        "per_attr": metrics,
    }


def run_coco_pose(model, cfg, device, input_size, workers, thresholds):
    """Optional official COCO keypoint eval using server-side annotations."""
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = data_cfg.get("train", {}) or {}
    val_cfg = data_cfg.get("val", {}) or {}
    pose_cfg = _merge_task_cfg(train_cfg.get("pose", {}), val_cfg.get("pose", {}))
    pose_root = _resolve_root(pose_cfg.get("root", ""))
    img_dir = pose_cfg.get("images", "val2017")
    lbl_dir = pose_cfg.get("labels", "labels/val2017")
    if not ((pose_root / img_dir).exists() and (pose_root / lbl_dir).exists()):
        return {"note": "pose val split not available"}
    instances = pose_root / "annotations" / "instances_val2017.json"
    keypoints = pose_root / "annotations" / "person_keypoints_val2017.json"
    if not (instances.exists() and keypoints.exists()):
        return {"note": "COCO annotations not found under pose root"}
    try:
        from test_model.train.cocoeval import evaluate_model
        from test_model.train.dataset import create_dataloader
    except ImportError as exc:
        return {"note": f"cocoeval/pycocotools unavailable: {exc}"}

    loader = create_dataloader(
        data_dir=pose_root,
        img_dir=img_dir,
        label_dir=lbl_dir,
        input_size=input_size,
        use_mosaic=False,
        augment=False,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
        class_id_format=pose_cfg.get("class_id_format", "yolo80"),
    )
    try:
        return evaluate_model(
            model, loader, device,
            task="keypoints",
            data_root=pose_root,
            num_classes=1,
            keep_classes=[0],
            score_thresh=thresholds["score_thresh"],
            iou_thresh=thresholds["iou_thresh"],
            max_det=thresholds["max_det"],
        )
    except Exception as exc:  # pragma: no cover - server-only path
        return {"note": f"COCO pose eval failed: {exc}"}


def evaluate_checkpoint(cfg, weights, device="cuda", split="val",
                        batch=None, workers=None, input_size=None,
                        max_samples=0,
                        score_thresh=0.01, domain_conf=0.25, person_conf=0.25,
                        attr_thresh=0.5, iou_thresh=0.6, match_iou=0.5,
                        max_det=300, coco_pose=True, output_dir=None,
                        prefix="final"):
    """Run the full final evaluation and return a metrics dict."""
    eval_cfg = cfg.get("eval", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    attr_names = list((cfg.get("attributes", {}) or {}).get(
        "names",
        ["smoking", "falling", "waving", "helmet_on"],
    ))
    input_size = int(input_size or data_cfg.get("input_size", 640))
    batch = int(batch or eval_cfg.get("batch_size", 16))
    workers = int(workers if workers is not None else 8)
    score_thresh = float(score_thresh if score_thresh is not None
                         else eval_cfg.get("score_thresh", 0.01))
    domain_conf = float(domain_conf if domain_conf is not None
                        else eval_cfg.get("domain_conf", 0.25))
    person_conf = float(person_conf if person_conf is not None
                        else eval_cfg.get("person_conf", 0.25))
    attr_thresh = float(attr_thresh if attr_thresh is not None
                        else eval_cfg.get("attr_thresh", 0.5))
    iou_thresh = float(iou_thresh if iou_thresh is not None
                       else eval_cfg.get("iou_thresh", 0.6))
    match_iou = float(match_iou if match_iou is not None
                      else eval_cfg.get("match_iou", 0.5))
    max_det = int(max_det if max_det is not None
                  else eval_cfg.get("max_det", 300))
    output_dir = output_dir or eval_cfg.get("output_dir")

    model = build_model(cfg, weights, device)
    loaders = build_eval_loaders(
        cfg, split, batch, workers, input_size, max_samples=max_samples,
    )

    domain_pred_classes = [1, 2, 3, 4]
    preds_by_class = {cls_id: [] for cls_id in domain_pred_classes + [0]}
    gts_by_image_class = {}
    gt_persons = {"boxes": [], "attrs": [], "masks": []}
    pred_persons = {"boxes": [], "attrs": [], "scores": []}
    image_idx = 0
    total_images = 0

    with torch.no_grad():
        for task, split_name, loader in loaders:
            split_seen = 0
            for batch_idx, batch_data in enumerate(loader, start=1):
                images = batch_data["image"].to(device, non_blocking=True)
                preds = model.predict_val(
                    images,
                    score_thresh=score_thresh,
                    iou_thresh=iou_thresh,
                    max_det=max_det,
                )
                for bi in range(len(preds)):
                    gt_boxes = batch_data["boxes"][bi].numpy()
                    gt_classes = batch_data["classes"][bi].numpy().astype(int)

                    if task == "detect":
                        for cls_id in domain_pred_classes:
                            gts_by_image_class[(image_idx, cls_id)] = (
                                gt_boxes[gt_classes == cls_id].astype(np.float32)
                            )
                    else:
                        gts_by_image_class[(image_idx, 0)] = (
                            gt_boxes[gt_classes == 0].astype(np.float32)
                        )

                    if task == "attr":
                        person_mask = gt_classes == 0
                        gt_persons["boxes"].append(
                            gt_boxes[person_mask].astype(np.float32))
                        gt_persons["attrs"].append(
                            batch_data["attrs"][bi].numpy()[person_mask]
                            .astype(np.float32))
                        gt_persons["masks"].append(
                            batch_data["attr_mask"][bi].numpy()[person_mask]
                            .astype(np.float32))

                    pred = {
                        key: (value.detach().cpu().numpy() if torch.is_tensor(value)
                              else value)
                        for key, value in preds[bi].items()
                    }
                    p_boxes = pred["boxes"].astype(np.float32)
                    p_scores = pred["scores"].astype(np.float32)
                    p_classes = pred["classes"].astype(int)
                    p_attrs = pred.get(
                        "attrs",
                        np.zeros((len(p_boxes), len(attr_names)), dtype=np.float32),
                    )

                    for cls_id in domain_pred_classes + [0]:
                        keep = p_classes == cls_id
                        for box, score in zip(p_boxes[keep], p_scores[keep]):
                            preds_by_class.setdefault(cls_id, []).append({
                                "image_idx": image_idx,
                                "box": box.astype(np.float32),
                                "score": float(score),
                            })

                    if task == "attr":
                        keep_person = (p_classes == 0) & (p_scores >= person_conf)
                        pred_persons["boxes"].append(
                            p_boxes[keep_person].astype(np.float32))
                        pred_persons["attrs"].append(
                            p_attrs[keep_person].astype(np.float32))
                        pred_persons["scores"].append(
                            p_scores[keep_person].astype(np.float32))

                    image_idx += 1
                    split_seen += 1
                if batch_idx == 1 or batch_idx % 20 == 0:
                    print(
                        f"Evaluated {task}[{split_name}]: "
                        f"{split_seen}/{len(loader.dataset)} total={image_idx}",
                        flush=True,
                    )
            total_images += len(loader.dataset)

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    domain_ap = eval_ap(
        preds_by_class, gts_by_image_class, domain_pred_classes, iou_thresholds)
    person_ap = eval_ap(
        preds_by_class, gts_by_image_class, [0], iou_thresholds)
    attr_metrics = eval_attrs(
        gt_persons, pred_persons,
        attr_names=attr_names,
        attr_thresh=attr_thresh,
        match_iou=match_iou,
    )

    domain_names = list((cfg.get("domain_det", {}) or {}).get(
        "names", ["puddle", "fire", "smoke", "other"]))
    per_class = {}
    for idx, cls_id in enumerate(domain_pred_classes):
        name = domain_names[idx] if idx < len(domain_names) else str(cls_id)
        per_class[name] = {
            "class_id": int(cls_id),
            **domain_ap.get(cls_id, {}),
        }

    result = {
        "weights": str(weights),
        "config": str(Path(__file__).resolve().parent / "yaml/final_three_head.yaml"),
        "split": split,
        "num_images": int(image_idx),
        "thresholds": {
            "decode_score": score_thresh,
            "domain_conf": domain_conf,
            "person_conf": person_conf,
            "attr_thresh": attr_thresh,
            "nms_iou": iou_thresh,
            "match_iou": match_iou,
        },
        "domain_det": {
            "mAP50": domain_ap["mAP50"],
            "mAP50_95": domain_ap["mAP50_95"],
            "per_class": per_class,
        },
        "person_det": {
            "AP50": person_ap.get(0, {}).get("AP50"),
            "AP50_95": person_ap.get(0, {}).get("AP50_95"),
            "gt": person_ap.get(0, {}).get("gt"),
            "pred": person_ap.get(0, {}).get("pred"),
        },
        "attributes": attr_metrics,
    }

    if coco_pose:
        thresholds = {
            "score_thresh": score_thresh,
            "iou_thresh": iou_thresh,
            "max_det": max_det,
        }
        result["pose_coco"] = run_coco_pose(
            model, cfg, device, input_size, workers, thresholds)

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report = output / f"{prefix}_{split}_eval_metrics.json"
        with open(report, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics: {report}", flush=True)
    return result


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = normalize_device(args.device)
    evaluate_checkpoint(
        cfg,
        args.weights,
        device=device,
        split=args.split,
        batch=args.batch,
        workers=args.workers,
        input_size=args.input_size,
        max_samples=args.max_samples,
        score_thresh=args.score_thresh,
        domain_conf=args.domain_conf,
        person_conf=args.person_conf,
        attr_thresh=args.attr_thresh,
        iou_thresh=args.iou_thresh,
        match_iou=args.match_iou,
        max_det=args.max_det,
        coco_pose=not args.no_coco_pose,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
