"""Create a compact candidate subset for server-side pose relabeling.

This script does not run YOLO pose. It only samples useful images, removes
irrelevant detection labels where possible, and keeps dataset-specific raw
labels needed for later attribute assignment.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Prepare compact domain candidates")
    p.add_argument("--output", default="D:/AI4PumpRoom/data/train_data_candidates")
    p.add_argument("--sample-per-dataset", type=int, default=1500)
    p.add_argument("--fall-sample", type=int, default=None)
    p.add_argument("--hat-sample", type=int, default=None)
    p.add_argument("--water-sample", type=int, default=None)
    p.add_argument("--fire-sample", type=int, default=None)
    p.add_argument("--smoking-sample", type=int, default=None)
    p.add_argument("--coco-pose-sample", type=int, default=None)
    p.add_argument("--debug-sample-per-dataset", type=int, default=80)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    p.add_argument("--preview", type=int, default=36)
    return p.parse_args()


def collect_pairs(img_dir, label_dir):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    pairs = []
    if not img_dir.exists() or not label_dir.exists():
        return pairs
    for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
        label = label_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs


def collect_split_pairs(root, splits=("train", "valid", "val", "test")):
    root = Path(root)
    pairs = []
    for split in splits:
        pairs.extend(collect_pairs(root / split / "images", root / split / "labels"))
        pairs.extend(collect_pairs(root / "images" / split, root / "labels" / split))
    if not pairs:
        pairs.extend(collect_pairs(root / "images", root / "labels"))
    return pairs


def image_shape(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def read_yolo_lines(label_path):
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            rows.append((cls, parts))
    return rows


def has_any_label(_img, label, _shape):
    return bool(read_yolo_lines(label))


def has_class(class_id):
    def _fn(_img, label, _shape):
        return any(cls == class_id for cls, _parts in read_yolo_lines(label))
    return _fn


def has_pose_person(_img, label, _shape):
    for cls, parts in read_yolo_lines(label):
        if cls == 0 and len(parts) >= 56:
            return True
    return False


def clean_label_lines(dataset, label_path):
    rows = read_yolo_lines(label_path)
    keep = []
    if dataset == "fire":
        # fire_smoke raw labels: 0=person, 1=fire. Keep fire only.
        keep = [parts for cls, parts in rows if cls == 1]
    elif dataset == "water":
        # water_leak raw labels: 0=person, 2=water. Keep water only.
        keep = [parts for cls, parts in rows if cls == 2]
    elif dataset == "fall":
        # Keep all action classes; non-fall/non-wave classes are useful negatives.
        keep = [parts for _cls, parts in rows]
    elif dataset == "hat":
        # Keep both hat and head; head is required for helmet_on assignment.
        keep = [parts for _cls, parts in rows]
    elif dataset == "smoking":
        # Old pose labels are not used for final person assignment, but keeping
        # them helps inspect source data if needed.
        keep = [parts for _cls, parts in rows]
    elif dataset == "coco_pose":
        keep = [parts for cls, parts in rows if cls == 0 and len(parts) >= 56]
    return [" ".join(parts) for parts in keep]


def sample_valid(name, pairs, sample_n, rng, valid_fn):
    valid = []
    for img, label in pairs:
        shape = image_shape(img)
        if shape is None:
            continue
        if valid_fn(img, label, shape):
            valid.append((img, label, shape))
    rng.shuffle(valid)
    if sample_n and len(valid) > sample_n:
        valid = valid[:sample_n]
    print(f"{name}: valid={len(valid)}")
    return valid


def copy_dataset(output, dataset, items, debug_items):
    records = []
    for subset_name, subset_items in (("full", items), ("debug", debug_items)):
        img_out = output / subset_name / dataset / "images"
        label_out = output / subset_name / dataset / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for idx, (img, label, shape) in enumerate(subset_items):
            suffix = img.suffix.lower()
            stem = f"{dataset}_{idx:05d}_{img.stem}"
            out_img = img_out / f"{stem}{suffix}"
            out_label = label_out / f"{stem}.txt"
            shutil.copy2(img, out_img)
            lines = clean_label_lines(dataset, label)
            out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            records.append({
                "subset": subset_name,
                "dataset": dataset,
                "source_image": str(img),
                "source_label": str(label),
                "image": str(out_img),
                "label": str(out_label),
                "width": shape[0],
                "height": shape[1],
                "label_count": len(lines),
            })
    return records


def yolo_to_xyxy(parts, w, h):
    cx, cy, bw, bh = map(float, parts[1:5])
    cx *= w
    cy *= h
    bw *= w
    bh *= h
    return [int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)]


def draw_preview(output, records, limit, rng):
    debug_records = [r for r in records if r["subset"] == "debug"]
    rng.shuffle(debug_records)
    chosen = debug_records[:limit]
    if not chosen:
        return
    cells = []
    for rec in chosen:
        img = cv2.imread(rec["image"])
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in Path(rec["label"]).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x1, y1, x2, y2 = yolo_to_xyxy(parts, w, h)
            color = (0, 255, 0) if cls == 0 else ((0, 128, 255) if cls == 1 else (255, 0, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(cls), (x1, max(18, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        img = cv2.resize(img, (480, 480))
        cv2.putText(img, rec["dataset"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cells.append(img)
    if not cells:
        return
    cols = 3
    rows = int(np.ceil(len(cells) / cols))
    blank = np.zeros((480, 480, 3), dtype=np.uint8)
    while len(cells) < rows * cols:
        cells.append(blank.copy())
    grid = np.concatenate([
        np.concatenate(cells[i * cols:(i + 1) * cols], axis=1)
        for i in range(rows)
    ], axis=0)
    cv2.imwrite(str(output / "meta" / "candidate_preview.jpg"), grid)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_root = Path("D:/AI4PumpRoom/data")
    output = Path(args.output)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to rebuild")
        shutil.rmtree(output)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    dataset_cfg = {
        "fall": {
            "pairs": collect_split_pairs(data_root / "data-fall"),
            "valid": has_any_label,
        },
        "hat": {
            "pairs": collect_split_pairs(data_root / "hat"),
            "valid": has_any_label,
        },
        "water": {
            "pairs": collect_pairs(data_root / "processed" / "water_leak" / "images",
                                   data_root / "processed" / "water_leak" / "labels"),
            "valid": has_class(2),
        },
        "fire": {
            "pairs": collect_pairs(data_root / "processed" / "fire_smoke" / "images",
                                   data_root / "processed" / "fire_smoke" / "labels"),
            "valid": has_class(1),
        },
        "smoking": {
            "pairs": collect_pairs(data_root / "processed" / "smoking" / "images",
                                   data_root / "processed" / "smoking" / "labels"),
            "valid": lambda _img, _label, _shape: True,
        },
        "coco_pose": {
            "pairs": collect_pairs(data_root / "processed" / "coco_person_pose" / "images",
                                   data_root / "processed" / "coco_person_pose" / "labels"),
            "valid": has_pose_person,
        },
    }

    all_records = []
    stats = {}
    sample_overrides = {
        "fall": args.fall_sample,
        "hat": args.hat_sample,
        "water": args.water_sample,
        "fire": args.fire_sample,
        "smoking": args.smoking_sample,
        "coco_pose": args.coco_pose_sample,
    }
    for dataset, cfg in dataset_cfg.items():
        sample_n = (
            args.sample_per_dataset
            if sample_overrides.get(dataset) is None
            else sample_overrides[dataset]
        )
        items = sample_valid(dataset, cfg["pairs"], sample_n, rng, cfg["valid"])
        debug_items = items[:min(args.debug_sample_per_dataset, len(items))]
        print(f"Copying {dataset}: full={len(items)} debug={len(debug_items)}")
        records = copy_dataset(output, dataset, items, debug_items)
        all_records.extend(records)
        stats[dataset] = {
            "source_pairs": len(cfg["pairs"]),
            "full": len(items),
            "debug": len(debug_items),
        }

    (output / "meta" / "candidate_index.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "meta" / "candidate_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "Candidate subset for server-side domain_attr preparation.\n"
        "full/: sampled upload set. debug/: small local validation subset.\n"
        "Datasets: fall, hat, water, fire, smoking, coco_pose.\n",
        encoding="utf-8",
    )
    draw_preview(output, all_records, args.preview, rng)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved candidates to {output}")


if __name__ == "__main__":
    main()
