"""Diagnose smoking/helmet attribute quality as a function of person scale."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.final.data import AttrDataset  # noqa: E402
from test_model.final_pose_attr_expanded_detect.eval import (  # noqa: E402
    _NO_AUG,
    box_iou,
    build_model,
    normalize_device,
)


ATTR_NAMES = ("smoking", "falling", "waving", "helmet_on")
TARGET_ATTRS = ("smoking", "helmet_on")
SCALE_BINS = (
    ("tiny", 0.0, 0.02),
    ("small", 0.02, 0.08),
    ("medium", 0.08, 0.25),
    ("large", 0.25, 1.01),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "test_model/final_pose_attr_expanded_detect/yaml/final_pose_attr_expanded_detect_vlm_mix.yaml"),
    )
    parser.add_argument("--attr-root", default=str(PROJECT_ROOT / "data/Attr"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--attr-thresh", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-per-source", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--groups",
        default="smoking,helmet",
        help="Comma-separated Attr top-level groups to evaluate",
    )
    parser.add_argument("--output", default="outputs/attr_scale_analysis.json")
    return parser.parse_args()


def scale_bin(area_ratio: float) -> str:
    for name, low, high in SCALE_BINS:
        if low <= area_ratio < high:
            return name
    return "large"


def dataset_name(path: str) -> str:
    parts = Path(path).parts
    try:
        idx = parts.index("Attr")
        return "/".join(parts[idx + 1 : idx + 3])
    except ValueError:
        return Path(path).parent.parent.name


def update_stats(bucket, truth: int, prob: float, threshold: float):
    pred = int(prob >= threshold)
    bucket["count"] += 1
    bucket["prob_sum"] += prob
    bucket["positive_prob_sum"] += prob if truth else 0.0
    bucket["negative_prob_sum"] += prob if not truth else 0.0
    bucket["positive_count"] += truth
    bucket["negative_count"] += 1 - truth
    bucket["tp"] += int(truth == 1 and pred == 1)
    bucket["fp"] += int(truth == 0 and pred == 1)
    bucket["tn"] += int(truth == 0 and pred == 0)
    bucket["fn"] += int(truth == 1 and pred == 0)


def finish_stats(raw):
    result = {}
    for key, value in raw.items():
        item = dict(value)
        tp, fp, tn, fn = (item[k] for k in ("tp", "fp", "tn", "fn"))
        item["mean_prob"] = item.pop("prob_sum") / max(item["count"], 1)
        item["positive_mean_prob"] = item.pop("positive_prob_sum") / max(item["positive_count"], 1)
        item["negative_mean_prob"] = item.pop("negative_prob_sum") / max(item["negative_count"], 1)
        item["precision"] = tp / max(tp + fp, 1)
        item["recall"] = tp / max(tp + fn, 1)
        item["accuracy"] = (tp + tn) / max(tp + fp + tn + fn, 1)
        result[key] = item
    return result


def blank_stats():
    return defaultdict(lambda: defaultdict(float))


def finish_score_stats(raw):
    result = {}
    for key, value in raw.items():
        result[key] = {
            "count": int(value["count"]),
            "mean_prob": value["prob_sum"] / max(value["count"], 1),
            "rate_ge_0_5": value["above_thresh"] / max(value["count"], 1),
        }
    return result


@torch.inference_mode()
def main():
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    device = normalize_device(args.device)
    model = build_model(cfg, args.weights, device)
    dataset = AttrDataset(
        args.attr_root,
        split=args.split,
        input_size=int(cfg.get("data", {}).get("input_size", 640)),
        augment=False,
        **_NO_AUG,
    )
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    selected = [
        ref for ref in dataset.samples
        if any(group in ref.image.parts for group in groups)
    ]
    if args.max_per_source > 0:
        grouped = defaultdict(list)
        for ref in selected:
            grouped[dataset_name(str(ref.image))].append(ref)
        rng = random.Random(20260816)
        limited = []
        for refs in grouped.values():
            limited.extend(rng.sample(refs, min(args.max_per_source, len(refs))))
        selected = limited
    dataset.samples = selected

    by_attr_scale = {name: blank_stats() for name in TARGET_ATTRS}
    by_attr_source = {name: blank_stats() for name in TARGET_ATTRS}
    output_by_source = {name: blank_stats() for name in TARGET_ATTRS}
    matched = 0
    gt_total = 0
    image_limit = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)

    for start in range(0, image_limit, args.batch):
        samples = [dataset[index] for index in range(start, min(start + args.batch, image_limit))]
        images = torch.stack([sample["image"] for sample in samples]).to(device)
        predictions = model.predict_val(images, score_thresh=0.01, iou_thresh=0.6, max_det=300)
        for sample, pred in zip(samples, predictions):
            keep = (pred["classes"] == 0) & (pred["scores"] >= args.person_conf)
            pred_boxes = pred["boxes"][keep].detach().cpu().numpy()
            pred_attrs = pred["attrs"][keep].detach().cpu().numpy()
            gt_boxes = sample["boxes"].numpy()
            gt_attrs = sample["attrs"].numpy()
            gt_masks = sample["attr_mask"].numpy()
            gt_total += len(gt_boxes)
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            candidates = []
            for pi, gi in zip(*np.where(ious >= args.match_iou)):
                candidates.append((float(ious[pi, gi]), int(pi), int(gi)))
            used_pred, used_gt = set(), set()
            for _, pi, gi in sorted(candidates, reverse=True):
                if pi in used_pred or gi in used_gt:
                    continue
                used_pred.add(pi)
                used_gt.add(gi)
                matched += 1
                box = gt_boxes[gi]
                area_ratio = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / float(images.shape[-2] * images.shape[-1])
                size = scale_bin(area_ratio)
                source = dataset_name(sample["img_path"])
                for attr_name in TARGET_ATTRS:
                    ai = ATTR_NAMES.index(attr_name)
                    prob = float(pred_attrs[pi, ai])
                    output_by_source[attr_name][source]["count"] += 1
                    output_by_source[attr_name][source]["prob_sum"] += prob
                    output_by_source[attr_name][source]["above_thresh"] += int(prob >= args.attr_thresh)
                    if gt_masks[gi, ai] <= 0:
                        continue
                    truth = int(gt_attrs[gi, ai] >= 0.5)
                    update_stats(by_attr_scale[attr_name][size], truth, prob, args.attr_thresh)
                    update_stats(by_attr_source[attr_name][source], truth, prob, args.attr_thresh)

        done = min(start + args.batch, image_limit)
        if done % 200 < args.batch or done == image_limit:
            print(f"Processed {done}/{image_limit}", flush=True)

    report = {
        "weights": str(Path(args.weights).resolve()),
        "split": args.split,
        "images": image_limit,
        "gt_persons": gt_total,
        "matched_persons": matched,
        "person_match_rate": matched / max(gt_total, 1),
        "thresholds": {
            "person_conf": args.person_conf,
            "match_iou": args.match_iou,
            "attr_thresh": args.attr_thresh,
        },
        "scale_bins_area_ratio": {name: [low, high] for name, low, high in SCALE_BINS},
        "by_scale": {name: finish_stats(stats) for name, stats in by_attr_scale.items()},
        "by_source": {name: finish_stats(stats) for name, stats in by_attr_source.items()},
        "all_outputs_by_source": {
            name: finish_score_stats(stats) for name, stats in output_by_source.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
