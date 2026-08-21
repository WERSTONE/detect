"""Measure attribute probability drift when the same labeled person is zoomed."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.final.data import AttrDataset  # noqa: E402
from test_model.final_pose_attr_expanded_detect.eval import _NO_AUG, box_iou, build_model, normalize_device  # noqa: E402


ATTR_NAMES = ("smoking", "falling", "waving", "helmet_on")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "test_model/final_pose_attr_expanded_detect/yaml/final_pose_attr_expanded_detect_vlm_mix.yaml"),
    )
    parser.add_argument("--attr-root", default=str(PROJECT_ROOT / "data/Attr"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-attr", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--attr-thresh", type=float, default=0.5)
    parser.add_argument("--factors", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--output", default="outputs/attr_zoom_analysis.json")
    return parser.parse_args()


def source_name(path: str) -> str:
    parts = Path(path).parts
    idx = parts.index("Attr")
    return "/".join(parts[idx + 1 : idx + 3])


def zoom_about_box(image_t, box, factor):
    image = (image_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    cx = float((box[0] + box[2]) / 2)
    cy = float((box[1] + box[3]) / 2)
    matrix = cv2.getRotationMatrix2D((cx, cy), 0.0, float(factor))
    zoomed = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    corners = np.array(
        [[box[0], box[1], 1.0], [box[2], box[3], 1.0]], dtype=np.float32
    )
    transformed = corners @ matrix.T
    target = np.array(
        [transformed[0, 0], transformed[0, 1], transformed[1, 0], transformed[1, 1]],
        dtype=np.float32,
    )
    if target[0] < 0 or target[1] < 0 or target[2] >= w or target[3] >= h:
        return None
    tensor = torch.from_numpy(zoomed.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor, target


@torch.inference_mode()
def main():
    args = parse_args()
    factors = [float(item) for item in args.factors.split(",")]
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    device = normalize_device(args.device)
    model = build_model(cfg, args.weights, device)
    dataset = AttrDataset(
        args.attr_root,
        split="val",
        input_size=int(cfg.get("data", {}).get("input_size", 640)),
        augment=False,
        **_NO_AUG,
    )
    dataset.samples = [
        ref for ref in dataset.samples
        if "smoking" in ref.image.parts or "helmet" in ref.image.parts
    ]

    candidates = {"smoking": [], "helmet_on": []}
    for index in range(len(dataset)):
        sample = dataset[index]
        source = source_name(sample["img_path"])
        for attr_name in candidates:
            ai = ATTR_NAMES.index(attr_name)
            for gi in range(len(sample["boxes"])):
                if sample["attr_mask"][gi, ai] >= 0.99 and sample["attrs"][gi, ai] < 0.5:
                    candidates[attr_name].append((index, gi, source))

    rng = random.Random(20260816)
    experiments = []
    selected_counts = {}
    for attr_name, items in candidates.items():
        rng.shuffle(items)
        selected = items[: args.samples_per_attr]
        selected_counts[attr_name] = len(selected)
        ai = ATTR_NAMES.index(attr_name)
        for sample_id, (index, gi, source) in enumerate(selected):
            sample = dataset[index]
            box = sample["boxes"][gi].numpy()
            for factor in factors:
                transformed = zoom_about_box(sample["image"], box, factor)
                if transformed is None:
                    continue
                image_t, target_box = transformed
                experiments.append({
                    "attr": attr_name,
                    "attr_index": ai,
                    "sample_id": sample_id,
                    "source": source,
                    "factor": factor,
                    "image": image_t,
                    "target_box": target_box,
                })

    for start in range(0, len(experiments), args.batch):
        chunk = experiments[start : start + args.batch]
        images = torch.stack([item["image"] for item in chunk]).to(device)
        predictions = model.predict_val(images, score_thresh=0.01, iou_thresh=0.6, max_det=300)
        for item, pred in zip(chunk, predictions):
            keep = (pred["classes"] == 0) & (pred["scores"] >= args.person_conf)
            boxes = pred["boxes"][keep].detach().cpu().numpy()
            attrs = pred["attrs"][keep].detach().cpu().numpy()
            item["prob"] = None
            item["match_iou"] = 0.0
            if len(boxes):
                ious = box_iou(boxes, item["target_box"][None])[..., 0]
                pi = int(np.argmax(ious))
                item["match_iou"] = float(ious[pi])
                if ious[pi] >= 0.3:
                    item["prob"] = float(attrs[pi, item["attr_index"]])
            item.pop("image")
            item["target_box"] = item["target_box"].tolist()

    summary = {}
    for attr_name in candidates:
        summary[attr_name] = {}
        for factor in factors:
            values = [
                item["prob"] for item in experiments
                if item["attr"] == attr_name and item["factor"] == factor and item["prob"] is not None
            ]
            summary[attr_name][str(factor)] = {
                "matched": len(values),
                "mean_prob": float(np.mean(values)) if values else None,
                "median_prob": float(np.median(values)) if values else None,
                "false_positive_rate": float(np.mean(np.asarray(values) >= args.attr_thresh)) if values else None,
            }

    report = {
        "selected_negative_samples": selected_counts,
        "factors": factors,
        "person_conf": args.person_conf,
        "attr_thresh": args.attr_thresh,
        "summary": summary,
        "experiments": experiments,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
