"""Evaluate domain detection and person attributes on the custom val set."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from test_model.model import create_model  # noqa: E402
from test_model.train.dataset import create_dataloader  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate bifpn_dual_domain_attr on custom validation labels")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "test_model/config/bifpn_domain_attr_finetune.yaml"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--img-dir", default=None)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--domain-conf", type=float, default=0.25)
    parser.add_argument("--attr-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_device(device):
    device = str(device or "cuda")
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"
    return device


def model_kwargs_from_config(cfg):
    data_cfg = cfg.get("data", {}) or {}
    neck_cfg = cfg.get("neck", {}) or {}
    assigner_cfg = cfg.get("assigner", {}) or {}
    domain_cfg = cfg.get("domain_det", {}) or {}
    attr_cfg = cfg.get("attributes", {}) or {}
    return {
        "num_kpts": cfg.get("num_kpts", 17),
        "num_det_classes": cfg.get("num_det_classes", cfg.get("num_classes", 80)),
        "reg_max": cfg.get("reg_max", 16),
        "input_size": data_cfg.get("input_size", 640),
        "neck_use_p2_context": neck_cfg.get("use_p2_context", False),
        "neck_downsample": neck_cfg.get("downsample", "conv"),
        "neck_out_channels": neck_cfg.get("out_channels", None),
        "assigner_topk": assigner_cfg.get("topk", 10),
        "assigner_alpha": assigner_cfg.get("alpha", 0.5),
        "assigner_beta": assigner_cfg.get("beta", 6.0),
        "assigner_eps": assigner_cfg.get("eps", 1.0e-9),
        "domain_num_classes": domain_cfg.get("num_classes", 2),
        "domain_class_map": domain_cfg.get("class_map", None),
        "num_attrs": attr_cfg.get(
            "num_attrs",
            len(attr_cfg.get("names", [])) if attr_cfg.get("names") else 4,
        ),
        "attr_names": attr_cfg.get("names", None),
    }


def extract_state_dict(checkpoint, prefer_ema=False):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if prefer_ema and isinstance(checkpoint.get("ema_state"), dict):
        return checkpoint["ema_state"]
    for key in ("model_state_dict", "state_dict", "ema_state"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("No state dict found in checkpoint")


def build_model(cfg, weights, device, prefer_ema=False):
    model_name = cfg.get("model", "bifpn_dual_domain_attr")
    model = create_model(model_name, **model_kwargs_from_config(cfg)).to(device)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint, prefer_ema=prefer_ema)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded: {weights} | tensors={len(state)} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else 'n/a'}",
        flush=True,
    )
    if missing:
        print(f"  missing sample: {missing[:6]}", flush=True)
    if unexpected:
        print(f"  unexpected sample: {unexpected[:6]}", flush=True)
    model.eval()
    return model


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
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def eval_ap(preds_by_class, gts_by_image_class, class_ids, iou_thresholds):
    out = {}
    for cls_id in class_ids:
        gt_count = sum(len(v) for (img_idx, c), v in gts_by_image_class.items() if c == cls_id)
        cls_preds = sorted(preds_by_class.get(cls_id, []), key=lambda x: x["score"], reverse=True)
        ap_values = []
        ap50 = 0.0
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
                ious = box_iou(np.asarray([pred["box"]], dtype=np.float32), gt_boxes)[0]
                best = int(np.argmax(ious))
                if ious[best] >= thr and not matched[key][best]:
                    tp[i] = 1.0
                    matched[key][best] = True
                else:
                    fp[i] = 1.0
            if gt_count == 0:
                ap = None
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
    class_metrics = [v for v in out.values() if isinstance(v, dict)]
    valid = [v["AP50_95"] for v in class_metrics if v["AP50_95"] is not None]
    out["mAP50_95"] = float(np.mean(valid)) if valid else None
    valid50 = [v["AP50"] for v in class_metrics if v["AP50"] is not None]
    out["mAP50"] = float(np.mean(valid50)) if valid50 else None
    return out


def precision_recall_at_conf(preds_by_class, gts_by_image_class, class_ids, conf, match_iou):
    out = {}
    total_tp = total_fp = total_fn = 0
    for cls_id in class_ids:
        gt_count = sum(len(v) for (img_idx, c), v in gts_by_image_class.items() if c == cls_id)
        cls_preds = [
            p for p in sorted(preds_by_class.get(cls_id, []), key=lambda x: x["score"], reverse=True)
            if p["score"] >= conf
        ]
        matched = {
            key: np.zeros(len(boxes), dtype=bool)
            for key, boxes in gts_by_image_class.items()
            if key[1] == cls_id
        }
        tp = fp = 0
        for pred in cls_preds:
            key = (pred["image_idx"], cls_id)
            gt_boxes = gts_by_image_class.get(key, [])
            if len(gt_boxes) == 0:
                fp += 1
                continue
            ious = box_iou(np.asarray([pred["box"]], dtype=np.float32), gt_boxes)[0]
            best = int(np.argmax(ious))
            if ious[best] >= match_iou and not matched[key][best]:
                tp += 1
                matched[key][best] = True
            else:
                fp += 1
        fn = int(gt_count - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(gt_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        out[cls_id] = {
            "gt": int(gt_count),
            "pred_conf": int(len(cls_preds)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    p = total_tp / max(total_tp + total_fp, 1)
    r = total_tp / max(total_tp + total_fn, 1)
    out["micro"] = {
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": float(p),
        "recall": float(r),
        "f1": float(2 * p * r / max(p + r, 1e-9)),
    }
    return out


def eval_attrs(gt_persons, pred_persons, attr_names, attr_thresh, match_iou):
    n_attr = len(attr_names)
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
            ious = box_iou(np.asarray([pred_boxes[pi]], dtype=np.float32), gt_boxes)[0]
            if len(ious) == 0:
                continue
            gi = int(np.argmax(ious))
            if ious[gi] < match_iou or used_gt[gi]:
                continue
            used_gt[gi] = True
            matched_persons += 1
            for ai, name in enumerate(attr_names):
                if ai >= gt_masks.shape[1] or gt_masks[gi, ai] <= 0.5:
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
            "positive_gt": int(tp + fn),
            "negative_gt": int(tn + fp),
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    domain_cfg = cfg.get("domain_det", {}) or {}
    attr_cfg = cfg.get("attributes", {}) or {}

    data_root = args.data or data_cfg.get("root")
    img_dir = args.img_dir or data_cfg.get("val_img", "images/val")
    label_dir = args.label_dir or data_cfg.get("val_label", "labels/val")
    input_size = int(args.input_size or data_cfg.get("input_size", 640))
    batch = int(args.batch or eval_cfg.get("batch_size", 16))
    workers = int(args.workers if args.workers is not None else 8)
    iou_thresh = float(args.iou_thresh or eval_cfg.get("iou_thresh", 0.6))
    max_det = int(args.max_det or eval_cfg.get("max_det", 300))
    device = normalize_device(args.device)

    model = build_model(cfg, args.weights, device, prefer_ema=args.prefer_ema)
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
        class_id_format=data_cfg.get("class_id_format", "yolo80"),
    )

    domain_names = list(domain_cfg.get("names", ["fire", "water"]))
    domain_class_map = {int(k): int(v) for k, v in (domain_cfg.get("class_map", {}) or {}).items()}
    inv_domain_map = {v: k for k, v in domain_class_map.items()}
    domain_pred_classes = [inv_domain_map.get(i, i) for i in range(len(domain_names))]
    attr_names = list(attr_cfg.get("names", getattr(model, "attr_names", [])))

    preds_by_class = {cls_id: [] for cls_id in domain_pred_classes}
    gts_by_image_class = {}
    gt_persons = {"boxes": [], "attrs": [], "masks": []}
    pred_persons = {"boxes": [], "attrs": [], "scores": []}
    image_idx = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader, start=1):
            images = batch_data["image"].to(device, non_blocking=True)
            preds = model.predict_val(
                images,
                score_thresh=args.score_thresh,
                iou_thresh=iou_thresh,
                max_det=max_det,
            )
            bs = len(preds)
            for bi in range(bs):
                gt_boxes = batch_data["boxes"][bi].numpy()
                gt_classes = batch_data["classes"][bi].numpy().astype(int)
                gt_attrs_i = batch_data["attrs"][bi].numpy()
                gt_masks_i = batch_data["attr_mask"][bi].numpy()

                for cls_id in domain_pred_classes:
                    boxes = gt_boxes[gt_classes == cls_id]
                    gts_by_image_class[(image_idx, cls_id)] = boxes.astype(np.float32)

                person_mask = gt_classes == 0
                gt_persons["boxes"].append(gt_boxes[person_mask].astype(np.float32))
                gt_persons["attrs"].append(gt_attrs_i[person_mask].astype(np.float32))
                gt_persons["masks"].append(gt_masks_i[person_mask].astype(np.float32))

                pred = {
                    key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
                    for key, value in preds[bi].items()
                }
                p_boxes = pred["boxes"].astype(np.float32)
                p_scores = pred["scores"].astype(np.float32)
                p_classes = pred["classes"].astype(int)
                p_attrs = pred.get("attrs", np.zeros((len(p_boxes), len(attr_names)), dtype=np.float32))

                for cls_id in domain_pred_classes:
                    keep = p_classes == cls_id
                    for box, score in zip(p_boxes[keep], p_scores[keep]):
                        preds_by_class.setdefault(cls_id, []).append({
                            "image_idx": image_idx,
                            "box": box.astype(np.float32),
                            "score": float(score),
                        })

                keep_person = (p_classes == 0) & (p_scores >= args.person_conf)
                pred_persons["boxes"].append(p_boxes[keep_person].astype(np.float32))
                pred_persons["attrs"].append(p_attrs[keep_person].astype(np.float32))
                pred_persons["scores"].append(p_scores[keep_person].astype(np.float32))

                image_idx += 1
            if batch_idx == 1 or batch_idx % 20 == 0:
                print(f"Evaluated {image_idx}/{len(loader.dataset)} images", flush=True)

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    det_ap = eval_ap(preds_by_class, gts_by_image_class, domain_pred_classes, iou_thresholds)
    det_pr = precision_recall_at_conf(
        preds_by_class,
        gts_by_image_class,
        domain_pred_classes,
        conf=args.domain_conf,
        match_iou=args.match_iou,
    )
    attr_metrics = eval_attrs(
        gt_persons,
        pred_persons,
        attr_names=attr_names,
        attr_thresh=args.attr_thresh,
        match_iou=args.match_iou,
    )

    per_class = {}
    for idx, cls_id in enumerate(domain_pred_classes):
        name = domain_names[idx] if idx < len(domain_names) else str(cls_id)
        per_class[name] = {
            "class_id": int(cls_id),
            **det_ap.get(cls_id, {}),
            **{f"conf_{k}": v for k, v in det_pr.get(cls_id, {}).items() if k not in ("gt",)},
        }

    result = {
        "weights": str(args.weights),
        "config": str(args.config),
        "data_root": str(data_root),
        "val_img": str(img_dir),
        "val_label": str(label_dir),
        "num_images": int(image_idx),
        "thresholds": {
            "decode_score": float(args.score_thresh),
            "domain_conf": float(args.domain_conf),
            "person_conf": float(args.person_conf),
            "attr_thresh": float(args.attr_thresh),
            "nms_iou": float(iou_thresh),
            "match_iou": float(args.match_iou),
        },
        "domain_det": {
            "mAP50": det_ap["mAP50"],
            "mAP50_95": det_ap["mAP50_95"],
            "micro_at_conf": det_pr["micro"],
            "per_class": per_class,
        },
        "attributes": attr_metrics,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
