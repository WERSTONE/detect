#!/usr/bin/env python
"""Audit COCO multi-task labels before training.

Checks the class-id mapping used by COCOMultiTaskDataset and reports the split
that dual-head models will see:
  - all COCO boxes/classes -> detection head
  - person keypoints -> pose head
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from test_model.dataset import COCOMultiTaskDataset, COCO80_CLASSES


def parse_args():
    p = argparse.ArgumentParser(description='Audit multi-task COCO labels')
    p.add_argument('--data', type=str, required=True, help='Dataset root')
    p.add_argument('--img-dir', type=str, default='train2017')
    p.add_argument('--label-dir', type=str, default='labels/train2017')
    p.add_argument('--input-size', type=int, default=640)
    p.add_argument('--class-id-format', type=str, default='yolo80',
                   choices=['yolo80', 'internal80', 'coco', 'coco80',
                            'coco20', 'internal', 'internal20',
                            'coco_category20', 'coco20_category', 'auto'])
    p.add_argument('--max-samples', type=int, default=0,
                   help='0 means audit all samples')
    p.add_argument('--output', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = COCOMultiTaskDataset(
        data_dir=args.data,
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        input_size=args.input_size,
        use_mosaic=False,
        augment=False,
        class_id_format=args.class_id_format,
    )

    raw_lines = 0
    ignored_lines = 0
    raw_class_counts = Counter()
    mapped_class_counts = Counter()
    sanitized_class_counts = Counter()
    images_with_labels = 0
    person_boxes = 0
    person_with_visible_kpts = 0
    det_boxes = 0

    limit = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    for idx in range(limit):
        _, label_path = dataset.samples[idx]
        if label_path and Path(label_path).exists():
            images_with_labels += 1
            with open(label_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    raw_lines += 1
                    raw_cls = int(float(parts[0]))
                    raw_class_counts[raw_cls] += 1
                    mapped = dataset._map_class_id(raw_cls, has_kpts=len(parts) > 5)
                    if mapped is None:
                        ignored_lines += 1
                    else:
                        mapped_class_counts[mapped] += 1

        sample = dataset[idx]
        classes = sample['classes']
        kpts = sample['kpts']
        for i, cls_t in enumerate(classes.tolist()):
            sanitized_class_counts[cls_t] += 1
            if cls_t == 0:
                person_boxes += 1
                if i < len(kpts) and (kpts[i, :, 2] > 0).any().item():
                    person_with_visible_kpts += 1
            det_boxes += 1

    result = {
        'dataset_root': str(Path(args.data).resolve()),
        'img_dir': args.img_dir,
        'label_dir': args.label_dir,
        'class_id_format': args.class_id_format,
        'samples_checked': limit,
        'images_with_labels': images_with_labels,
        'raw_label_lines': raw_lines,
        'ignored_label_lines': ignored_lines,
        'dual_head_split': {
            'det_head_all_boxes': det_boxes,
            'det_head_person_boxes': person_boxes,
            'pose_head_person_boxes_with_visible_keypoints': person_with_visible_kpts,
        },
        'mapped_class_counts': {
            COCO80_CLASSES[k]: int(v) for k, v in sorted(mapped_class_counts.items())
            if 0 <= k < len(COCO80_CLASSES)
        },
        'sanitized_class_counts': {
            COCO80_CLASSES[k]: int(v) for k, v in sorted(sanitized_class_counts.items())
            if 0 <= k < len(COCO80_CLASSES)
        },
        'raw_class_counts': {str(k): int(v) for k, v in sorted(raw_class_counts.items())},
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
