"""
数据集标注可视化 — 每个数据集采样若干张，将 GT 标注绘制在图上。
用法: python data/scripts/viz_datasets.py [--samples 5] [--output data/viz]
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 每个数据集的类别名和颜色
DATASET_META = {
    "person": {
        "names": {0: "person"},
        "colors": {0: (0, 255, 0)},  # green
    },
    "helmet": {
        "names": {0: "helmet_on", 1: "helmet_off"},
        "colors": {0: (0, 255, 0), 1: (0, 0, 255)},
    },
    "fire_smoke": {
        "names": {0: "fire"},
        "colors": {0: (0, 0, 255)},
    },
    "smoking": {
        "names": {0: "smoking"},
        "colors": {0: (255, 0, 0)},
    },
    "water_leak": {
        "names": {0: "stagnant_water"},
        "colors": {0: (255, 128, 0)},
    },
}

# COCO 17 关键点骨架连接
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]
KPT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170),
]


def draw_boxes(img, labels_dir, img_name, meta, img_w, img_h):
    """绘制 YOLO 格式标注框。"""
    label_file = labels_dir / (Path(img_name).stem + ".txt")
    if not label_file.exists():
        return

    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)

            name = meta["names"].get(cls_id, f"cls_{cls_id}")
            color = meta["colors"].get(cls_id, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # 标签背景
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, name, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_keypoints(img, ann_file, img_name, img_w, img_h):
    """绘制 COCO 关键点。兼容 annotations.json (含 images 数组) 和 kpt_annotations.json (仅 annotations)。"""
    if not ann_file.exists():
        return

    with open(ann_file) as f:
        data = json.load(f)

    target_stem = Path(img_name).stem
    img_id = None

    # 尝试从 images 数组中查找
    if "images" in data:
        for img_info in data["images"]:
            fn = img_info.get("file_name", "")
            if Path(fn).stem == target_stem:
                img_id = img_info["id"]
                break

    # 回退: 直接用文件名 stem 作为 image_id (COCO 标准命名)
    if img_id is None:
        try:
            img_id = int(target_stem)
        except ValueError:
            return

    anns = [a for a in data["annotations"] if a["image_id"] == img_id]
    for ann in anns:
        kpts = ann.get("keypoints", [])
        if len(kpts) != 51:
            continue

        pts = []
        for i in range(17):
            x = kpts[i * 3] / img_w * img.shape[1] if img_w > 0 else kpts[i * 3]
            y = kpts[i * 3 + 1] / img_h * img.shape[0] if img_h > 0 else kpts[i * 3 + 1]
            v = kpts[i * 3 + 2]
            if v > 0:
                pts.append((int(x), int(y), i))
                cv2.circle(img, (int(x), int(y)), 3, KPT_COLORS[i], -1)

        # 画骨架
        for p1, p2 in SKELETON:
            p1_pts = [p for p in pts if p[2] == p1]
            p2_pts = [p for p in pts if p[2] == p2]
            if p1_pts and p2_pts:
                cv2.line(img, (p1_pts[0][0], p1_pts[0][1]),
                         (p2_pts[0][0], p2_pts[0][1]), (0, 255, 255), 1)


def viz_dataset(dataset_name, n_samples, output_dir):
    """可视化单个数据集。"""
    ds_dir = PROCESSED_DIR / dataset_name
    if not ds_dir.exists():
        print(f"  [skip] {dataset_name}: not found")
        return

    meta = DATASET_META.get(dataset_name, {"names": {}, "colors": {}})

    # 从 train 目录采样
    img_dir = ds_dir / "images" / "train"
    if not img_dir.exists():
        print(f"  [skip] {dataset_name}: no images/train/")
        return

    all_imgs = sorted(img_dir.glob("*"))
    all_imgs = [p for p in all_imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not all_imgs:
        print(f"  [skip] {dataset_name}: no images")
        return

    random.seed(42)
    sampled = random.sample(all_imgs, min(n_samples, len(all_imgs)))

    labels_dir = ds_dir / "labels" / "train"
    kpt_file = ds_dir / "kpt_annotations.json"

    ds_out = output_dir / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)

    for img_path in sampled:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        draw_boxes(img, labels_dir, img_path.name, meta, w, h)

        # person 数据集额外绘制关键点 (如果有关键点标注)
        if dataset_name == "person" and kpt_file.exists():
            draw_keypoints(img, kpt_file, img_path.name, w, h)

        out_path = ds_out / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"    {img_path.name}")

    print(f"  [{dataset_name}] {len(sampled)} samples → {ds_out}")


def main():
    parser = argparse.ArgumentParser(description="Vigil 数据集标注可视化")
    parser.add_argument("--samples", type=int, default=5, help="每个数据集采样数")
    parser.add_argument("--output", type=str, default="data/viz", help="输出目录")
    parser.add_argument("--dataset", type=str, default=None,
                        help="仅可视化指定数据集 (默认全部)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else list(DATASET_META.keys())

    for name in datasets:
        viz_dataset(name, args.samples, output_dir)

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
