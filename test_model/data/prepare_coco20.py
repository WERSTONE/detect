"""Prepare COCO 2017 labels for multi-head verification.

Merges instances_*.json (bbox for all 80 classes) with
person_keypoints_*.json (keypoints for person class) to produce
a YOLO label dataset. By default it writes the legacy 20-class subset;
with --full80 it writes standard COCO 80-class ids.

Usage:
    python test_model/data/prepare_coco20.py --data-dir /data/coco2017 [--download] [--full80]
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

CLASS_MAP = {
    1: 0,    # person (COCO category_id=1)
    2: 1,    # bicycle
    3: 2,    # car
    4: 3,    # motorcycle
    6: 4,    # bus
    8: 5,    # truck
    18: 6,   # dog
    17: 7,   # cat
    19: 8,   # horse
    16: 9,   # bird
    21: 10,  # cow
    20: 11,  # sheep
    62: 12,  # chair
    67: 13,  # dining table
    73: 14,  # laptop
    27: 15,  # backpack
    37: 16,  # sports ball
    44: 17,  # bottle
    47: 18,  # cup
    77: 19,  # cell phone
}

COCO80_CLASS_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}

COCO_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}


def download_file(url, dest, desc=''):
    if Path(dest).exists():
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {desc}...")
    def report(bn, bs, total):
        if total > 0 and bn % 50 == 0:
            pct = min(100, bn * bs * 100 / total)
            print(f"\r    {pct:.0f}%", end='', flush=True)
    urlretrieve(url, dest, reporthook=report)
    print("\r    100% - done")


def download_coco(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in COCO_URLS.items():
        dest = data_dir / Path(url).name
        download_file(url, dest, name)
    for name in COCO_URLS:
        zip_path = data_dir / Path(COCO_URLS[name]).name
        if zip_path.exists():
            print(f"  Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
            print(f"  Done")


def filter_coco_to_yolo(data_dir, class_map=None, label_name='20-class'):
    """Merge instances + person_keypoints → 20-class YOLO labels.

    Strategy:
      1. Load instances_*.json as primary source (bbox for all 80 classes)
      2. Load person_keypoints_*.json and build kpt lookup {image_id: {ann_id: kpts}}
      3. Write labels: bbox from instances, kpts from person_keypoints for person class
      4. Include ALL images with >=1 annotation from our 20 classes
    """
    class_map = class_map or CLASS_MAP
    data_dir = Path(data_dir)
    ann_dir = data_dir / 'annotations'

    for split in ('train2017', 'val2017'):
        inst_file = ann_dir / f'instances_{split}.json'
        kpt_file = ann_dir / f'person_keypoints_{split}.json'

        if not inst_file.exists():
            print(f"  WARNING: {inst_file} not found — skipping {split}")
            continue

        print(f"\n  Processing {split}...")

        # ---- Load instance annotations (bbox, all classes) ----
        with open(inst_file) as f:
            instances = json.load(f)
        print(f"    instances: {len(instances['images'])} images, {len(instances['annotations'])} annotations")

        # ---- Load keypoint annotations (person only) ----
        kpt_lookup = {}  # {image_id: {ann_id: [kpt_x, kpt_y, kpt_v, ...]}}
        if kpt_file.exists():
            with open(kpt_file) as f:
                kpts = json.load(f)
            for ann in kpts['annotations']:
                img_id = ann['image_id']
                if img_id not in kpt_lookup:
                    kpt_lookup[img_id] = {}
                # Use ann['id'] to match with instance annotations later
                kpt_lookup[img_id][ann['id']] = ann.get('keypoints', [0] * 51)
            print(f"    keypoints: {len(kpts['annotations'])} person annotations with kpts")
        else:
            print(f"    keypoints: not found, person will have zero kpts")

        # ---- Build image lookup ----
        images = {img['id']: img for img in instances['images']}

        # ---- Group instance annotations by image, filter to 20 classes ----
        img_anns = {}  # {image_id: [(ann, cls_20), ...]}
        for ann in instances['annotations']:
            cat_id = ann['category_id']
            if cat_id not in class_map:
                continue
            img_id = ann['image_id']
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append((ann, class_map[cat_id]))

        print(f"    {len(img_anns)} images with {label_name} annotations")

        # ---- Write YOLO labels ----
        label_dir = data_dir / 'labels' / split
        label_dir.mkdir(parents=True, exist_ok=True)
        for old_label in label_dir.glob('*.txt'):
            old_label.unlink()

        converted = 0
        skipped_no_image = 0

        for img_id, anns in img_anns.items():
            if img_id not in images:
                skipped_no_image += 1
                continue

            img_info = images[img_id]
            img_w, img_h = img_info['width'], img_info['height']
            label_path = label_dir / f"{img_info['file_name'].rsplit('.', 1)[0]}.txt"

            lines = []
            for ann, mapped_cls in anns:
                bx, by, bw, bh = ann['bbox']
                xc = (bx + bw / 2) / img_w
                yc = (by + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h

                line = f"{mapped_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"

                # Keypoints: merge from person_keypoints if available
                if mapped_cls == 0:
                    ann_id = ann['id']
                    kpt_data = kpt_lookup.get(img_id, {}).get(ann_id, [0] * 51)
                    for j in range(17):
                        kx = kpt_data[j * 3] / img_w if kpt_data[j * 3] > 0 else 0
                        ky = kpt_data[j * 3 + 1] / img_h if kpt_data[j * 3 + 1] > 0 else 0
                        kv = min(float(kpt_data[j * 3 + 2]), 2)
                        line += f" {kx:.6f} {ky:.6f} {kv:.0f}"
                else:
                    line += " " + " ".join(["0 0 0"] * 17)

                lines.append(line)

            with open(label_path, 'w') as f:
                f.write('\n'.join(lines))
            converted += 1

        print(f"    Converted {converted} images → {label_dir}")
        print(f"    Person boxes: {sum(1 for a in instances['annotations'] if a['category_id'] == 1)} (all)")
        print(f"    Person with kpts: {len(kpt_lookup)} images")
        if skipped_no_image:
            print(f"    Skipped {skipped_no_image} annotations (missing image)")


def verify_labels(data_dir, expected_num_classes=None):
    data_dir = Path(data_dir)
    for split in ('train2017', 'val2017'):
        img_dir = data_dir / 'images' / split
        label_dir = data_dir / 'labels' / split
        if not img_dir.exists() or not label_dir.exists():
            continue
        label_files = {p.stem for p in label_dir.glob('*.txt')}
        img_files = {p.stem for p in img_dir.glob('*.jpg')}
        img_files |= {p.stem for p in img_dir.glob('*.png')}
        missing_imgs = label_files - img_files
        missing_labels = img_files - label_files
        bad_class = 0
        bad_format = 0
        total_rows = 0
        for label_path in label_dir.glob('*.txt'):
            with open(label_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    total_rows += 1
                    if len(parts) not in (5, 56):
                        bad_format += 1
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        bad_format += 1
                        continue
                    if expected_num_classes is not None and not (0 <= cls_id < expected_num_classes):
                        bad_class += 1
        print(f"\n  {split}: {len(img_files)} images, {len(label_files)} labels")
        if missing_labels:
            print(f"    Images without labels: {len(missing_labels)}")
        if missing_imgs:
            print(f"    Labels without images: {len(missing_imgs)}")
        if total_rows:
            print(f"    Label rows checked: {total_rows}")
        if bad_format:
            print(f"    Invalid label rows: {bad_format}")
        if bad_class:
            print(f"    Out-of-range class ids: {bad_class}")
        if missing_imgs or bad_format or bad_class:
            raise RuntimeError(f"Label verification failed for {split}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--full80', action='store_true',
                   help='Write standard COCO/YOLO 80-class ids instead of the legacy 20-class subset')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    expected_num_classes = 80 if args.full80 else 20
    if args.verify_only:
        verify_labels(data_dir, expected_num_classes=expected_num_classes)
        return
    if args.download:
        print("Downloading COCO 2017...")
        download_coco(data_dir)
    print("\nConverting COCO labels (merging bbox + keypoints)...")
    if args.full80:
        filter_coco_to_yolo(data_dir, class_map=COCO80_CLASS_MAP, label_name='80-class')
    else:
        filter_coco_to_yolo(data_dir)
    verify_labels(data_dir, expected_num_classes=expected_num_classes)
    print("\nDone!")


if __name__ == '__main__':
    main()
