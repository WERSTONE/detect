"""Search deployment confidence thresholds for expanded-detect pose-attr models.

The script runs model inference once, optionally caches raw post-NMS predictions,
and sweeps confidence thresholds for domain detection, person detection, and
person attributes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.final_pose_attr_expanded_detect.eval import (  # noqa: E402
    box_iou,
    build_eval_loaders,
    build_model,
    normalize_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search confidence thresholds for deployment")
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "test_model/final_pose_attr_expanded_detect/yaml/"
            "final_pose_attr_expanded_detect_vlm_mix_source_mask_smoking_gp.yaml"
        ),
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--decode-conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--person-conf-for-attrs", type=float, default=0.25)
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument(
        "--domain-thresholds",
        default="0.05:0.80:0.05",
        help="Comma list or start:end:step, e.g. 0.05:0.80:0.05",
    )
    parser.add_argument(
        "--person-thresholds",
        default="0.10:0.80:0.05",
        help="Comma list or start:end:step",
    )
    parser.add_argument(
        "--attr-thresholds",
        default="0.30:0.80:0.05",
        help="Comma list or start:end:step",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional .pkl cache for predictions/GT. Reused when --reuse-cache is set.",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/final_pose_attr_expanded_detect_threshold_search",
    )
    return parser.parse_args()


def parse_thresholds(spec):
    text = str(spec).strip()
    if ":" in text:
        start, end, step = [float(x) for x in text.split(":")]
        values = []
        cur = start
        while cur <= end + step * 0.5:
            values.append(round(cur, 6))
            cur += step
        return values
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collect_predictions(cfg, weights, device, args):
    eval_cfg = cfg.get("eval", {}) or {}
    input_size = int(args.input_size or cfg.get("data", {}).get("input_size", 640))
    batch = int(args.batch or eval_cfg.get("batch_size", 16))
    workers = int(args.workers or cfg.get("training", {}).get("workers", 8))
    max_det = int(args.max_det or eval_cfg.get("max_det", 300))

    model = build_model(cfg, weights, device)
    loaders = build_eval_loaders(
        cfg,
        args.split,
        batch,
        workers,
        input_size,
        max_samples=args.max_samples,
    )
    domain_cfg = cfg.get("domain_det", {}) or {}
    domain_class_ids = [int(x) for x in domain_cfg.get(
        "eval_class_ids",
        list(range(1, int(domain_cfg.get("num_classes", 4)) + 1)),
    )]

    cache = {
        "meta": {
            "weights": str(weights),
            "split": args.split,
            "decode_conf": float(args.decode_conf),
            "nms_iou": float(args.nms_iou),
            "match_iou": float(args.match_iou),
            "max_det": int(max_det),
            "input_size": int(input_size),
        },
        "domain_class_ids": domain_class_ids,
        "preds_by_class": {cls_id: [] for cls_id in domain_class_ids + [0]},
        "gts_by_image_class": {},
        "gt_persons": {"boxes": [], "attrs": [], "masks": []},
        "pred_persons": {"boxes": [], "attrs": [], "scores": []},
        "task_counts": {},
        "num_images": 0,
    }

    image_idx = 0
    with torch.no_grad():
        for task, split_name, loader in loaders:
            split_seen = 0
            for batch_idx, batch_data in enumerate(loader, start=1):
                images = batch_data["image"].to(device, non_blocking=True)
                preds = model.predict_val(
                    images,
                    score_thresh=args.decode_conf,
                    iou_thresh=args.nms_iou,
                    max_det=max_det,
                )
                for bi, pred_t in enumerate(preds):
                    gt_boxes = to_numpy(batch_data["boxes"][bi]).astype(np.float32)
                    gt_classes = to_numpy(batch_data["classes"][bi]).astype(int)

                    if task == "detect":
                        domain_mask = to_numpy(batch_data["domain_valid_mask"][bi]) > 0.5
                        for cls_id in domain_class_ids:
                            if domain_mask[cls_id - 1]:
                                cache["gts_by_image_class"][(image_idx, cls_id)] = (
                                    gt_boxes[gt_classes == cls_id].astype(np.float32)
                                )
                    else:
                        cache["gts_by_image_class"][(image_idx, 0)] = (
                            gt_boxes[gt_classes == 0].astype(np.float32)
                        )

                    if task == "attr":
                        person_mask = gt_classes == 0
                        cache["gt_persons"]["boxes"].append(
                            gt_boxes[person_mask].astype(np.float32)
                        )
                        cache["gt_persons"]["attrs"].append(
                            to_numpy(batch_data["attrs"][bi])[person_mask].astype(np.float32)
                        )
                        cache["gt_persons"]["masks"].append(
                            to_numpy(batch_data["attr_mask"][bi])[person_mask].astype(np.float32)
                        )

                    pred = {key: to_numpy(value) for key, value in pred_t.items()}
                    p_boxes = pred["boxes"].astype(np.float32)
                    p_scores = pred["scores"].astype(np.float32)
                    p_classes = pred["classes"].astype(int)
                    p_attrs = pred.get(
                        "attrs",
                        np.zeros((len(p_boxes), len(get_attr_names(cfg))), dtype=np.float32),
                    ).astype(np.float32)

                    if task == "detect":
                        domain_mask = to_numpy(batch_data["domain_valid_mask"][bi]) > 0.5
                        valid_classes = [
                            cls_id for cls_id in domain_class_ids
                            if domain_mask[cls_id - 1]
                        ]
                    else:
                        valid_classes = []

                    for cls_id in valid_classes + [0]:
                        keep = p_classes == cls_id
                        for box, score in zip(p_boxes[keep], p_scores[keep]):
                            cache["preds_by_class"].setdefault(cls_id, []).append({
                                "image_idx": int(image_idx),
                                "box": box.astype(np.float32),
                                "score": float(score),
                            })

                    if task == "attr":
                        keep_person = p_classes == 0
                        cache["pred_persons"]["boxes"].append(
                            p_boxes[keep_person].astype(np.float32)
                        )
                        cache["pred_persons"]["attrs"].append(
                            p_attrs[keep_person].astype(np.float32)
                        )
                        cache["pred_persons"]["scores"].append(
                            p_scores[keep_person].astype(np.float32)
                        )

                    image_idx += 1
                    split_seen += 1

                if batch_idx == 1 or batch_idx % 20 == 0:
                    print(
                        f"Collected {task}[{split_name}]: "
                        f"{split_seen}/{len(loader.dataset)} total={image_idx}",
                        flush=True,
                    )
            cache["task_counts"][f"{task}/{split_name}"] = int(len(loader.dataset))

    cache["num_images"] = int(image_idx)
    return cache


def get_attr_names(cfg):
    attr_cfg = cfg.get("pose_attr", {}) or cfg.get("attributes", {}) or {}
    names = attr_cfg.get("names", None)
    return list(names) if names else ["smoking", "falling", "waving", "helmet_on"]


def get_domain_names(cfg, class_ids):
    domain_cfg = cfg.get("domain_det", {}) or {}
    names = list(domain_cfg.get("names", []))
    out = {}
    for idx, cls_id in enumerate(class_ids):
        out[cls_id] = names[idx] if idx < len(names) else str(cls_id)
    return out


def match_detection_at_threshold(preds_by_class, gts_by_image_class, cls_id, conf, match_iou):
    gt_items = {
        key: boxes
        for key, boxes in gts_by_image_class.items()
        if key[1] == cls_id
    }
    gt_count = int(sum(len(v) for v in gt_items.values()))
    matched = {key: np.zeros(len(boxes), dtype=bool) for key, boxes in gt_items.items()}
    preds = [
        pred for pred in preds_by_class.get(cls_id, [])
        if float(pred["score"]) >= conf
    ]
    preds.sort(key=lambda x: x["score"], reverse=True)

    tp = 0
    fp = 0
    for pred in preds:
        key = (int(pred["image_idx"]), cls_id)
        gt_boxes = gt_items.get(key)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp += 1
            continue
        ious = box_iou(np.asarray([pred["box"]], dtype=np.float32), gt_boxes)[0]
        best = int(np.argmax(ious))
        if ious[best] >= match_iou and not matched[key][best]:
            tp += 1
            matched[key][best] = True
        else:
            fp += 1
    fn = gt_count - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(conf),
        "gt": int(gt_count),
        "pred": int(len(preds)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def choose_operating_points(rows, target_recall, target_precision):
    def score_key(row):
        return (row["f1"], row["precision"], row["recall"], -row["threshold"])

    best_f1 = max(rows, key=score_key) if rows else None
    high_recall_rows = [row for row in rows if row["recall"] >= target_recall]
    high_precision_rows = [row for row in rows if row["precision"] >= target_precision]
    return {
        "best_f1": best_f1,
        "target_recall": (
            max(high_recall_rows, key=lambda r: (r["precision"], r["f1"], r["threshold"]))
            if high_recall_rows else None
        ),
        "target_precision": (
            max(high_precision_rows, key=lambda r: (r["recall"], r["f1"], -r["threshold"]))
            if high_precision_rows else None
        ),
    }


def sweep_detection(cache, class_ids, thresholds, match_iou, target_recall, target_precision):
    out = {}
    for cls_id in class_ids:
        rows = [
            match_detection_at_threshold(
                cache["preds_by_class"],
                cache["gts_by_image_class"],
                cls_id,
                conf,
                match_iou,
            )
            for conf in thresholds
        ]
        out[cls_id] = {
            "sweep": rows,
            "selected": choose_operating_points(rows, target_recall, target_precision),
        }
    return out


def make_attr_matches(gt_persons, pred_persons, person_conf, match_iou):
    matches = []
    total_gt = 0
    for gt_boxes, gt_attrs, gt_masks, pred_boxes, pred_attrs, pred_scores in zip(
        gt_persons["boxes"],
        gt_persons["attrs"],
        gt_persons["masks"],
        pred_persons["boxes"],
        pred_persons["attrs"],
        pred_persons["scores"],
    ):
        total_gt += len(gt_boxes)
        if len(gt_boxes) == 0:
            continue
        keep = pred_scores >= person_conf
        pred_boxes = pred_boxes[keep]
        pred_attrs = pred_attrs[keep]
        pred_scores = pred_scores[keep]
        order = np.argsort(-pred_scores) if len(pred_scores) else []
        used_gt = np.zeros(len(gt_boxes), dtype=bool)
        for pi in order:
            ious = box_iou(np.asarray([pred_boxes[pi]], dtype=np.float32), gt_boxes)[0]
            if len(ious) == 0:
                continue
            gi = int(np.argmax(ious))
            if ious[gi] < match_iou or used_gt[gi]:
                continue
            used_gt[gi] = True
            matches.append((
                gt_attrs[gi].astype(np.float32),
                gt_masks[gi].astype(np.float32),
                pred_attrs[pi].astype(np.float32),
            ))
    return matches, total_gt


def attr_metrics_for_threshold(matches, attr_idx, attr_thresh):
    tp = fp = tn = fn = unknown = 0
    for gt_attrs, gt_masks, pred_attrs in matches:
        if attr_idx >= len(gt_masks) or gt_masks[attr_idx] <= 0.0:
            unknown += 1
            continue
        gt_pos = gt_attrs[attr_idx] > 0.5
        pred_pos = pred_attrs[attr_idx] >= attr_thresh
        if gt_pos and pred_pos:
            tp += 1
        elif (not gt_pos) and pred_pos:
            fp += 1
        elif gt_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1
    known = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(attr_thresh),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "unknown": int(unknown),
        "known": int(known),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(known, 1)),
    }


def sweep_attrs(cache, attr_names, thresholds, person_conf, match_iou,
                target_recall, target_precision):
    matches, total_gt = make_attr_matches(
        cache["gt_persons"],
        cache["pred_persons"],
        person_conf,
        match_iou,
    )
    out = {
        "person_conf": float(person_conf),
        "match_iou": float(match_iou),
        "gt_persons": int(total_gt),
        "matched_persons": int(len(matches)),
        "person_match_rate": float(len(matches) / max(total_gt, 1)),
        "per_attr": {},
    }
    for attr_idx, name in enumerate(attr_names):
        rows = [
            attr_metrics_for_threshold(matches, attr_idx, threshold)
            for threshold in thresholds
        ]
        out["per_attr"][name] = {
            "sweep": rows,
            "selected": choose_operating_points(rows, target_recall, target_precision),
        }
    return out


def summarize_rows(result, domain_names, attr_names):
    lines = []
    lines.append("# Detection thresholds")
    lines.append("class\tbest_f1_thr\tprecision\trecall\tf1\tpred\tgt")
    for cls_id, payload in result["domain_det"]["per_class"].items():
        row = payload["selected"]["best_f1"]
        lines.append(
            f"{domain_names[int(cls_id)]}\t{row['threshold']:.2f}\t"
            f"{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1']:.4f}\t"
            f"{row['pred']}\t{row['gt']}"
        )
    lines.append("")
    lines.append("# Person threshold")
    row = result["person_det"]["selected"]["best_f1"]
    lines.append(
        f"person\t{row['threshold']:.2f}\t{row['precision']:.4f}\t"
        f"{row['recall']:.4f}\t{row['f1']:.4f}\t{row['pred']}\t{row['gt']}"
    )
    lines.append("")
    lines.append("# Attribute thresholds")
    lines.append("attr\tbest_f1_thr\tprecision\trecall\tf1\taccuracy\tknown")
    for name in attr_names:
        row = result["attributes"]["per_attr"][name]["selected"]["best_f1"]
        lines.append(
            f"{name}\t{row['threshold']:.2f}\t{row['precision']:.4f}\t"
            f"{row['recall']:.4f}\t{row['f1']:.4f}\t"
            f"{row['accuracy']:.4f}\t{row['known']}"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = normalize_device(args.device)
    cache_path = Path(args.cache) if args.cache else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_path and args.reuse_cache:
        print(f"Loading cache: {cache_path}")
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
    else:
        cache = collect_predictions(cfg, args.weights, device, args)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved cache: {cache_path}")

    domain_thresholds = parse_thresholds(args.domain_thresholds)
    person_thresholds = parse_thresholds(args.person_thresholds)
    attr_thresholds = parse_thresholds(args.attr_thresholds)
    domain_class_ids = [int(x) for x in cache["domain_class_ids"]]
    domain_names = get_domain_names(cfg, domain_class_ids)
    attr_names = get_attr_names(cfg)

    domain = sweep_detection(
        cache,
        domain_class_ids,
        domain_thresholds,
        args.match_iou,
        args.target_recall,
        args.target_precision,
    )
    person = sweep_detection(
        cache,
        [0],
        person_thresholds,
        args.match_iou,
        args.target_recall,
        args.target_precision,
    )[0]
    attrs = sweep_attrs(
        cache,
        attr_names,
        attr_thresholds,
        args.person_conf_for_attrs,
        args.match_iou,
        args.target_recall,
        args.target_precision,
    )

    result = {
        "weights": str(args.weights),
        "config": str(args.config),
        "split": args.split,
        "num_images": int(cache["num_images"]),
        "decode_conf": float(cache["meta"]["decode_conf"]),
        "nms_iou": float(cache["meta"]["nms_iou"]),
        "match_iou": float(args.match_iou),
        "target_recall": float(args.target_recall),
        "target_precision": float(args.target_precision),
        "domain_det": {
            "thresholds": domain_thresholds,
            "per_class": {
                str(cls_id): {
                    "name": domain_names[cls_id],
                    **domain[cls_id],
                }
                for cls_id in domain_class_ids
            },
        },
        "person_det": {
            "thresholds": person_thresholds,
            **person,
        },
        "attributes": {
            "thresholds": attr_thresholds,
            **attrs,
        },
    }

    json_path = output_dir / "threshold_search.json"
    txt_path = output_dir / "threshold_search_summary.tsv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(
        summarize_rows(result, domain_names, attr_names),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved JSON: {json_path}")
    print(f"Saved summary: {txt_path}")


if __name__ == "__main__":
    main()
