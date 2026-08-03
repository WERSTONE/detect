"""Prepare target-domain detection + pose-attribute fine-tuning data.

Output label format:
  domain detection:
    cls cx cy w h
    where cls: 1=fire, 2=water

  person pose with attributes:
    0 cx cy w h kx ky kv ... 17 times ... a0 a1 a2 a3 m0 m1 m2 m3
    attrs/masks order: smoking, falling, waving, helmet_on
"""

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ATTR_NAMES = ["smoking", "falling", "waving", "helmet_on"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_NAMES = ("fall", "hat", "water", "fire", "smoking", "coco_pose")
FALLING_ACTION_CLASSES = {0, 2}  # falling, sleeping
WAVING_ACTION_CLASSES = {5}      # waving hands


@dataclass
class YoloBox:
    cls: int
    xyxy: np.ndarray


@dataclass
class Person:
    xyxy: np.ndarray
    kpts: np.ndarray
    score: float


def parse_args():
    p = argparse.ArgumentParser(description="Prepare domain attr training data")
    p.add_argument(
        "--candidate-root",
        default=None,
        help=(
            "Optional compact candidate root created by prepare_domain_candidates. "
            "Expected layout: fall/images+labels, hat/images+labels, ..."
        ),
    )
    p.add_argument("--output", default="D:/AI4PumpRoom/data/train_data")
    p.add_argument("--sample-per-dataset", type=int, default=1500)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pose-model", default="yolov8x-pose.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--pose-batch", type=int, default=16)
    p.add_argument("--pose-conf", type=float, default=0.25)
    p.add_argument("--kpt-conf", type=float, default=0.25)
    p.add_argument("--pose-imgsz", type=int, default=640)
    p.add_argument("--pose-half", action="store_true")
    p.add_argument("--pose-max-det", type=int, default=100)
    p.add_argument("--force", action="store_true")
    p.add_argument("--preview", type=int, default=36)
    p.add_argument("--weak-neg-fall-wave", action="store_true", default=True)
    p.add_argument("--no-weak-neg-fall-wave", dest="weak_neg_fall_wave", action="store_false")
    return p.parse_args()


def yolo_xywh_to_xyxy(vals, w, h):
    cx, cy, bw, bh = [float(v) for v in vals]
    bw *= w
    bh *= h
    cx *= w
    cy *= h
    return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32)


def xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    return [cx / w, cy / h, bw / w, bh / h]


def ioa(inner, outer):
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area = max(float(inner[2] - inner[0]) * float(inner[3] - inner[1]), 1.0)
    return inter / area


def iou(a, b):
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(float(a[2] - a[0]) * float(a[3] - a[1]), 1.0)
    area_b = max(float(b[2] - b[0]) * float(b[3] - b[1]), 1.0)
    return inter / max(area_a + area_b - inter, 1.0)


def center_in(box, container):
    cx = (float(box[0]) + float(box[2])) / 2
    cy = (float(box[1]) + float(box[3])) / 2
    return float(container[0]) <= cx <= float(container[2]) and float(container[1]) <= cy <= float(container[3])


def expand_box(box, scale, w, h):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    return np.array([
        max(0.0, cx - bw / 2),
        max(0.0, cy - bh / 2),
        min(float(w - 1), cx + bw / 2),
        min(float(h - 1), cy + bh / 2),
    ], dtype=np.float32)


def read_image_shape(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def collect_pairs(img_dir, label_dir):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    pairs = []
    for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
        label = label_dir / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
    return pairs


def candidate_has_standard_layout(root):
    root = Path(root)
    return any((root / name / "images").exists() and
               (root / name / "labels").exists()
               for name in DATASET_NAMES)


def normalize_backslash_candidate_root(candidate_root):
    """Repair candidate zips extracted with literal Windows backslashes.

    Some Linux unzip builds warn about backslash path separators but still
    create files like ``full\\fall\\images\\xxx.jpg``.  The training-data
    processor expects normal directories, so this copies those files into the
    requested candidate root using POSIX-style separators.
    """
    candidate_root = Path(candidate_root)
    if candidate_root.exists() and candidate_has_standard_layout(candidate_root):
        return candidate_root

    search_root = candidate_root if candidate_root.exists() else candidate_root.parent
    if not search_root.exists():
        raise FileNotFoundError(
            f"candidate root not found: {candidate_root}; "
            f"also cannot scan parent: {search_root}"
        )

    prefix = candidate_root.name
    repaired = 0
    for src in search_root.rglob("*"):
        if not src.is_file():
            continue
        rel = str(src.relative_to(search_root)).replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        if len(parts) < 4 or parts[0] != prefix:
            continue
        if parts[1] not in DATASET_NAMES or parts[2] not in ("images", "labels"):
            continue
        dst = candidate_root.joinpath(*parts[1:])
        if src.resolve() == dst.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
        repaired += 1

    if repaired:
        print(
            f"Normalized {repaired} backslash-path candidate files into "
            f"{candidate_root}",
            flush=True,
        )

    if not candidate_has_standard_layout(candidate_root):
        raise FileNotFoundError(
            f"No candidate dataset layout found under {candidate_root}. "
            "Expected fall/images+labels, hat/images+labels, ..."
        )
    return candidate_root


def collect_split_pairs(root, splits=("train", "valid", "val", "test")):
    root = Path(root)
    pairs = []
    for split in splits:
        img_dir = root / split / "images"
        label_dir = root / split / "labels"
        if img_dir.exists() and label_dir.exists():
            pairs.extend(collect_pairs(img_dir, label_dir))
    if not pairs and (root / "images").exists() and (root / "labels").exists():
        pairs.extend(collect_pairs(root / "images", root / "labels"))
    return pairs


def parse_box_labels(label_path, img_w, img_h):
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            boxes.append(YoloBox(cls=cls, xyxy=yolo_xywh_to_xyxy(parts[1:5], img_w, img_h)))
    return boxes


def parse_pose_label(label_path, img_w, img_h):
    persons = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 56:
                continue
            cls = int(float(parts[0]))
            if cls != 0:
                continue
            box = yolo_xywh_to_xyxy(parts[1:5], img_w, img_h)
            kpts = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                kx = float(parts[5 + i * 3]) * img_w
                ky = float(parts[5 + i * 3 + 1]) * img_h
                kv = float(parts[5 + i * 3 + 2])
                kpts[i] = [kx, ky, kv]
            persons.append(Person(box, kpts, score=1.0))
    return persons


def sample_pairs(name, pairs, sample_n, rng, valid_fn):
    valid = []
    for img, label in pairs:
        shape = read_image_shape(img)
        if shape is None:
            continue
        if valid_fn(img, label, shape):
            valid.append((img, label, shape))
    rng.shuffle(valid)
    if sample_n and len(valid) > sample_n:
        valid = valid[:sample_n]
    print(f"{name}: selected {len(valid)} samples")
    return valid


def load_candidate_pairs(candidate_root):
    candidate_root = normalize_backslash_candidate_root(candidate_root)

    selected = {}
    for name in DATASET_NAMES:
        pairs = collect_pairs(candidate_root / name / "images",
                              candidate_root / name / "labels")
        items = []
        for img, label in pairs:
            shape = read_image_shape(img)
            if shape is not None:
                items.append((img, label, shape))
        selected[name] = items
        print(f"{name}: loaded {len(items)} candidate samples")
    return selected


def load_pose_model(model_name):
    from ultralytics import YOLO
    return YOLO(model_name)


def predict_persons(model, image_paths, device, batch, conf, kpt_conf, imgsz, half, max_det):
    out = {}
    paths = list(image_paths)
    for idx, path_in in enumerate(paths):
        if idx == 0 or (idx + 1) % 50 == 0 or (idx + 1) == len(paths):
            print(f"  pose {idx + 1}/{len(paths)}", flush=True)
        results = model.predict(
            source=str(path_in),
            device=device,
            batch=1,
            conf=conf,
            imgsz=imgsz,
            half=half,
            max_det=max_det,
            verbose=False,
        )
        for result in results:
            path = Path(result.path)
            persons = []
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                kpts_xy = result.keypoints.xy.cpu().numpy()
                kpts_conf = result.keypoints.conf.cpu().numpy()
                for box, score, xy, kc in zip(boxes, scores, kpts_xy, kpts_conf):
                    kpts = np.zeros((17, 3), dtype=np.float32)
                    kpts[:, :2] = xy[:, :2]
                    kpts[:, 2] = np.where(kc >= kpt_conf, kc, 0.0)
                    persons.append(Person(box.astype(np.float32), kpts, float(score)))
            out[path.resolve()] = persons
    return out


def greedy_match(source_boxes, persons, min_score=0.2, metric="ioa"):
    candidates = []
    for si, src in enumerate(source_boxes):
        for pi, person in enumerate(persons):
            score = ioa(src.xyxy, person.xyxy) if metric == "ioa" else iou(src.xyxy, person.xyxy)
            if center_in(src.xyxy, person.xyxy):
                score += 0.05
            if score >= min_score:
                candidates.append((score, si, pi))
    candidates.sort(reverse=True)
    used_src = set()
    used_person = set()
    matches = {}
    for score, si, pi in candidates:
        if si in used_src or pi in used_person:
            continue
        used_src.add(si)
        used_person.add(pi)
        matches[si] = pi
    return matches


def assign_fall_attrs(persons, action_boxes, attrs, masks):
    for row in attrs:
        row[1] = 0.0
        row[2] = 0.0
    for row in masks:
        row[1] = 1.0
        row[2] = 1.0
    matches = greedy_match(action_boxes, persons, min_score=0.2, metric="ioa")
    for ai, pi in matches.items():
        cls = action_boxes[ai].cls
        if cls in FALLING_ACTION_CLASSES:
            attrs[pi][1] = 1.0
            attrs[pi][2] = 0.0
        elif cls in WAVING_ACTION_CLASSES:
            attrs[pi][1] = 0.0
            attrs[pi][2] = 1.0


def assign_helmet_attrs(persons, label_boxes, attrs, masks, img_w, img_h):
    # The safety-helmet source dataset uses mutually exclusive head labels:
    #   0=hat/helmet head, 1=bare head.
    # Match each small head label to at most one YOLO-pose person and assign
    # the per-person helmet_on attribute directly.
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
    for _score, bi, pi in candidates:
        if bi in used_box or pi in used_person:
            continue
        used_box.add(bi)
        used_person.add(pi)
        attrs[pi][3] = 1.0 if label_boxes[bi].cls == 0 else 0.0
        masks[pi][3] = 1.0


def weak_negative_attrs(dataset_name, attrs, masks, weak_neg_fall_wave):
    for attr, mask in zip(attrs, masks):
        if dataset_name == "smoking":
            attr[0] = 1.0
        else:
            attr[0] = 0.0
        mask[0] = 1.0
        if weak_neg_fall_wave and dataset_name != "fall":
            attr[1] = 0.0
            attr[2] = 0.0
            mask[1] = 1.0
            mask[2] = 1.0


def write_label(path, persons, domain_boxes, img_w, img_h):
    lines = []
    for person, attrs, masks in persons:
        xywh = xyxy_to_yolo(person.xyxy, img_w, img_h)
        vals = [0, *xywh]
        for x, y, v in person.kpts:
            vals.extend([
                float(np.clip(x / img_w, 0.0, 1.0)),
                float(np.clip(y / img_h, 0.0, 1.0)),
                float(max(v, 0.0)),
            ])
        vals.extend([float(v) for v in attrs])
        vals.extend([float(v) for v in masks])
        lines.append(" ".join([str(vals[0])] + [f"{v:.6f}" for v in vals[1:]]))

    for cls, box in domain_boxes:
        xywh = xyxy_to_yolo(box, img_w, img_h)
        lines.append(" ".join([str(cls)] + [f"{v:.6f}" for v in xywh]))

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_name(index, val_ratio, rng):
    return "val" if rng.random() < val_ratio else "train"


def draw_preview(output_root, records, limit, rng):
    if limit <= 0 or not records:
        return
    chosen = records[:]
    rng.shuffle(chosen)
    chosen = chosen[:limit]
    cells = []
    cell_w, cell_h = 480, 480
    for rec in chosen:
        img = cv2.imread(str(rec["image"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        label = Path(rec["label"])
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                box = yolo_xywh_to_xyxy(parts[1:5], w, h)
                color = (0, 255, 0) if cls == 0 else ((0, 128, 255) if cls == 1 else (255, 0, 0))
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = "person" if cls == 0 else ("fire" if cls == 1 else "water")
                if cls == 0 and len(parts) >= 64:
                    attrs = [float(v) for v in parts[56:60]]
                    masks = [float(v) for v in parts[60:64]]
                    active = [f"{n}={int(a)}" for n, a, m in zip(ATTR_NAMES, attrs, masks) if m > 0]
                    if active:
                        text += " " + ",".join(active)
                cv2.putText(img, text, (x1, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        img = cv2.resize(img, (cell_w, cell_h))
        cv2.putText(img, rec["dataset"], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cells.append(img)
    if not cells:
        return
    cols = 3
    rows = int(np.ceil(len(cells) / cols))
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    while len(cells) < rows * cols:
        cells.append(blank.copy())
    grid = np.concatenate([
        np.concatenate(cells[r * cols:(r + 1) * cols], axis=1)
        for r in range(rows)
    ], axis=0)
    cv2.imwrite(str(output_root / "meta" / "preview.jpg"), grid)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    repo_data = Path("D:/AI4PumpRoom/data")
    output = Path(args.output)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to rebuild")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    if args.candidate_root:
        selected = load_candidate_pairs(args.candidate_root)
    else:
        datasets = {
            "fall": {
                "pairs": (
                    collect_pairs(repo_data / "data-fall" / "images" / "train", repo_data / "data-fall" / "labels" / "train") +
                    collect_pairs(repo_data / "data-fall" / "images" / "val", repo_data / "data-fall" / "labels" / "val") +
                    collect_pairs(repo_data / "data-fall" / "images" / "test", repo_data / "data-fall" / "labels" / "test")
                ),
                "valid": lambda _img, label, shape: len(parse_box_labels(label, *shape)) > 0,
            },
            "hat": {
                "pairs": collect_split_pairs(repo_data / "hat"),
                "valid": lambda _img, label, shape: len(parse_box_labels(label, *shape)) > 0,
            },
            "water": {
                "pairs": collect_pairs(repo_data / "processed" / "water_leak" / "images",
                                       repo_data / "processed" / "water_leak" / "labels"),
                "valid": lambda _img, label, shape: any(b.cls == 2 for b in parse_box_labels(label, *shape)),
            },
            "fire": {
                "pairs": collect_pairs(repo_data / "processed" / "fire_smoke" / "images",
                                       repo_data / "processed" / "fire_smoke" / "labels"),
                "valid": lambda _img, label, shape: any(b.cls == 1 for b in parse_box_labels(label, *shape)),
            },
            "smoking": {
                "pairs": collect_pairs(repo_data / "processed" / "smoking" / "images",
                                       repo_data / "processed" / "smoking" / "labels"),
                "valid": lambda _img, _label, _shape: True,
            },
            "coco_pose": {
                "pairs": collect_pairs(repo_data / "processed" / "coco_person_pose" / "images",
                                       repo_data / "processed" / "coco_person_pose" / "labels"),
                "valid": lambda _img, label, shape: len(parse_pose_label(label, *shape)) > 0,
            },
        }

        selected = {}
        for name, cfg in datasets.items():
            selected[name] = sample_pairs(
                name, cfg["pairs"], args.sample_per_dataset, rng, cfg["valid"])

    pose_model = load_pose_model(args.pose_model)
    all_pose_images = []
    for name, items in selected.items():
        if name == "coco_pose":
            continue
        all_pose_images.extend([img for img, _label, _shape in items])
    print(f"Running YOLO pose on {len(all_pose_images)} images with {args.pose_model}")
    pose_by_path = predict_persons(
        pose_model,
        all_pose_images,
        device=args.device,
        batch=args.pose_batch,
        conf=args.pose_conf,
        kpt_conf=args.kpt_conf,
        imgsz=args.pose_imgsz,
        half=args.pose_half,
        max_det=args.pose_max_det,
    )

    stats = {
        "attrs": ATTR_NAMES,
        "classes": {"0": "person", "1": "fire", "2": "water"},
        "datasets": {},
        "written": 0,
        "skipped_no_person": 0,
    }
    records = []

    for dataset_name, items in selected.items():
        ds_stats = {"selected": len(items), "written": 0, "persons": 0, "fire": 0, "water": 0}
        print(f"Writing {dataset_name}: {len(items)} selected", flush=True)
        for idx, (img_path, label_path, shape) in enumerate(items):
            if idx == 0 or (idx + 1) % 100 == 0 or (idx + 1) == len(items):
                print(
                    f"  {dataset_name} {idx + 1}/{len(items)} "
                    f"written={ds_stats['written']}",
                    flush=True,
                )
            img_w, img_h = shape
            label_boxes = parse_box_labels(label_path, img_w, img_h)
            domain_boxes = []
            persons = []
            if dataset_name == "coco_pose":
                raw_persons = parse_pose_label(label_path, img_w, img_h)
            else:
                raw_persons = pose_by_path.get(img_path.resolve(), [])

            if dataset_name == "fire":
                domain_boxes = [(1, b.xyxy) for b in label_boxes if b.cls == 1]
            elif dataset_name == "water":
                domain_boxes = [(2, b.xyxy) for b in label_boxes if b.cls == 2]

            if dataset_name in ("fall", "hat", "smoking") and not raw_persons:
                stats["skipped_no_person"] += 1
                continue

            attrs = [np.zeros(4, dtype=np.float32) for _ in raw_persons]
            masks = [np.zeros(4, dtype=np.float32) for _ in raw_persons]
            weak_negative_attrs(dataset_name, attrs, masks, args.weak_neg_fall_wave)

            if dataset_name == "fall":
                assign_fall_attrs(raw_persons, label_boxes, attrs, masks)
            elif dataset_name == "hat":
                assign_helmet_attrs(raw_persons, label_boxes, attrs, masks, img_w, img_h)

            persons = list(zip(raw_persons, attrs, masks))
            if not persons and not domain_boxes:
                continue

            split = split_name(idx, args.val_ratio, rng)
            stem = f"{dataset_name}_{idx:05d}_{img_path.stem}"
            out_img = output / "images" / split / f"{stem}{img_path.suffix.lower()}"
            out_label = output / "labels" / split / f"{stem}.txt"
            shutil.copy2(img_path, out_img)
            write_label(out_label, persons, domain_boxes, img_w, img_h)

            ds_stats["written"] += 1
            ds_stats["persons"] += len(persons)
            ds_stats["fire"] += sum(1 for c, _b in domain_boxes if c == 1)
            ds_stats["water"] += sum(1 for c, _b in domain_boxes if c == 2)
            stats["written"] += 1
            records.append({
                "dataset": dataset_name,
                "source_image": str(img_path),
                "image": str(out_img),
                "label": str(out_label),
                "split": split,
                "persons": len(persons),
                "domain_boxes": len(domain_boxes),
            })
        stats["datasets"][dataset_name] = ds_stats

    (output / "meta" / "dataset_index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "meta" / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data.yaml").write_text(
        "path: \"{}\"\ntrain: images/train\nval: images/val\nnames:\n"
        "  0: person\n  1: fire\n  2: water\n".format(str(output).replace("\\", "/")),
        encoding="utf-8",
    )
    draw_preview(output, records, args.preview, rng)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
