"""Patch prepared domain-attr labels with direct helmet/head assignment.

This avoids rerunning YOLO pose after ``prepare_domain_attr.py`` has already
created person boxes/keypoints. It rewrites only the helmet_on attr value and
mask on person lines for records whose source dataset is ``hat``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from test_model.data.prepare_domain_attr import (
    Person,
    center_in,
    ioa,
    parse_box_labels,
    read_image_shape,
    yolo_xywh_to_xyxy,
)


def parse_args():
    p = argparse.ArgumentParser(description="Patch helmet_on labels")
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
    if candidate_root and source_image:
        candidates.append(Path(candidate_root) / "hat" / "labels" / f"{source_image.stem}.txt")
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


def assign_helmet(persons, label_boxes, attrs, masks):
    candidates = []
    for bi, box in enumerate(label_boxes):
        if box.cls not in (0, 1):
            continue
        for pi, person in enumerate(persons):
            score = ioa(box.xyxy, person.xyxy)
            if center_in(box.xyxy, person.xyxy):
                score += 0.05
            if score >= 0.2:
                candidates.append((score, bi, pi))
    candidates.sort(reverse=True)

    used_box = set()
    used_person = set()
    positives = 0
    negatives = 0
    for _score, bi, pi in candidates:
        if bi in used_box or pi in used_person:
            continue
        used_box.add(bi)
        used_person.add(pi)
        value = 1.0 if label_boxes[bi].cls == 0 else 0.0
        attrs[pi][3] = value
        masks[pi][3] = 1.0
        positives += int(value > 0.5)
        negatives += int(value <= 0.5)
    return positives, negatives


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
            "helmet_positive": 0,
            "helmet_negative": 0,
        }

    label_boxes = parse_box_labels(source_label, img_w, img_h)
    for row in masks:
        row[3] = 0.0
    helmet_positive, helmet_negative = assign_helmet(persons, label_boxes, attrs, masks)

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
        "helmet_positive": helmet_positive,
        "helmet_negative": helmet_negative,
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
        "hat_records": 0,
        "missing_source_label": 0,
        "missing_output": 0,
        "patched_files": 0,
        "changed_persons": 0,
        "persons": 0,
        "helmet_positive": 0,
        "helmet_negative": 0,
    }

    for rec in records:
        if rec.get("dataset") != "hat":
            continue
        stats["hat_records"] += 1
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
        stats["persons"] += result["persons"]
        stats["changed_persons"] += result["changed"]
        stats["helmet_positive"] += result["helmet_positive"]
        stats["helmet_negative"] += result["helmet_negative"]
        if result["patched"]:
            stats["patched_files"] += 1
            if not args.dry_run:
                label_path.write_text(
                    "\n".join(result["lines"]) + "\n",
                    encoding="utf-8",
                )

    out_path = data_root / "meta" / "patch_helmet_attrs.json"
    if not args.dry_run:
        out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
