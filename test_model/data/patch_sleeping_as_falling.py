"""Patch prepared domain-attr labels so fall class 2 sleeping means falling.

This avoids rerunning YOLO pose after ``prepare_domain_attr.py`` has already
created person boxes/keypoints. It only rewrites the 4 attr values and 4 attr
masks on person lines for records whose source dataset is ``fall``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from test_model.data.prepare_domain_attr import (
    FALLING_ACTION_CLASSES,
    WAVING_ACTION_CLASSES,
    Person,
    greedy_match,
    parse_box_labels,
    read_image_shape,
    yolo_xywh_to_xyxy,
)


def parse_args():
    p = argparse.ArgumentParser(description="Patch sleeping labels as falling")
    p.add_argument("--data-root", default="/root/detect/data/train_data")
    p.add_argument("--candidate-root", default="/root/detect/data/train_data_candidates/full")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_path(path, data_root):
    path = Path(path)
    if path.exists():
        return path
    alt = Path(data_root) / path
    return alt if alt.exists() else path


def source_label_for_record(record, candidate_root):
    source_image = Path(record.get("source_image", ""))
    candidates = []
    if source_image:
        candidates.append(source_image.parent.parent / "labels" / f"{source_image.stem}.txt")
    if candidate_root:
        candidates.append(Path(candidate_root) / "fall" / "labels" / f"{source_image.stem}.txt")
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_person_lines(label_path, img_w, img_h):
    lines = label_path.read_text(encoding="utf-8").splitlines()
    persons = []
    person_line_indices = []
    attrs = []
    masks = []
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 64:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls != 0:
            continue
        box = yolo_xywh_to_xyxy(parts[1:5], img_w, img_h)
        kpts = np.zeros((17, 3), dtype=np.float32)
        persons.append(Person(box, kpts, score=1.0))
        person_line_indices.append(idx)
        attrs.append([float(x) for x in parts[56:60]])
        masks.append([float(x) for x in parts[60:64]])
    return lines, persons, person_line_indices, attrs, masks


def update_line_attrs(line, attrs, masks):
    parts = line.strip().split()
    parts[56:60] = [f"{float(v):.6f}" for v in attrs]
    parts[60:64] = [f"{float(v):.6f}" for v in masks]
    return " ".join(parts)


def patch_one(label_path, image_path, source_label):
    shape = read_image_shape(image_path)
    if shape is None:
        return None
    img_w, img_h = shape
    lines, persons, person_line_indices, attrs, masks = parse_person_lines(
        label_path, img_w, img_h)
    if not persons:
        return {
            "patched": False,
            "persons": 0,
            "changed": 0,
            "sleeping_matches": 0,
        }

    action_boxes = parse_box_labels(source_label, img_w, img_h)
    for row in attrs:
        row[1] = 0.0
        row[2] = 0.0
    for row in masks:
        row[1] = 1.0
        row[2] = 1.0

    matches = greedy_match(action_boxes, persons, min_score=0.2, metric="ioa")
    sleeping_matches = 0
    for ai, pi in matches.items():
        cls = action_boxes[ai].cls
        if cls in FALLING_ACTION_CLASSES:
            if cls == 2:
                sleeping_matches += 1
            attrs[pi][1] = 1.0
            attrs[pi][2] = 0.0
        elif cls in WAVING_ACTION_CLASSES:
            attrs[pi][1] = 0.0
            attrs[pi][2] = 1.0

    changed = 0
    for local_idx, line_idx in enumerate(person_line_indices):
        new_line = update_line_attrs(lines[line_idx], attrs[local_idx], masks[local_idx])
        if new_line != lines[line_idx]:
            changed += 1
            lines[line_idx] = new_line

    return {
        "patched": changed > 0,
        "persons": len(persons),
        "changed": changed,
        "sleeping_matches": sleeping_matches,
        "falling_positive": sum(1 for row in attrs if row[1] > 0.5),
        "waving_positive": sum(1 for row in attrs if row[2] > 0.5),
        "lines": lines,
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    index_path = data_root / "meta" / "dataset_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing dataset index: {index_path}")
    records = json.loads(index_path.read_text(encoding="utf-8"))

    stats = {
        "data_root": str(data_root),
        "candidate_root": str(args.candidate_root),
        "dry_run": bool(args.dry_run),
        "fall_records": 0,
        "missing_source_label": 0,
        "missing_output": 0,
        "patched_files": 0,
        "changed_persons": 0,
        "sleeping_matches": 0,
        "falling_positive": 0,
        "waving_positive": 0,
    }

    for rec in records:
        if rec.get("dataset") != "fall":
            continue
        stats["fall_records"] += 1
        label_path = resolve_path(rec.get("label", ""), data_root)
        image_path = resolve_path(rec.get("image", ""), data_root)
        if not label_path.exists() or not image_path.exists():
            stats["missing_output"] += 1
            continue
        source_label = source_label_for_record(rec, args.candidate_root)
        if source_label is None:
            stats["missing_source_label"] += 1
            continue

        result = patch_one(label_path, image_path, source_label)
        if not result:
            stats["missing_output"] += 1
            continue
        stats["changed_persons"] += result["changed"]
        stats["sleeping_matches"] += result["sleeping_matches"]
        stats["falling_positive"] += result["falling_positive"]
        stats["waving_positive"] += result["waving_positive"]
        if result["patched"]:
            stats["patched_files"] += 1
            if not args.dry_run:
                label_path.write_text(
                    "\n".join(result["lines"]) + "\n",
                    encoding="utf-8",
                )

    out_path = data_root / "meta" / "patch_sleeping_as_falling.json"
    if not args.dry_run:
        out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
