"""Merge the smoking-people YOLO dataset into Detect_head_helmet_cigarette.

The source dataset has one class (class 0: cigarette). The target auxiliary
dataset uses three classes: helmet=0, head=1, cigarette=2.

Source split policy:
  train -> train/valid with a deterministic 90/10 split
  test  -> test

The operation is additive and content-deduplicated, so rerunning it is safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TARGET_CIGARETTE_CLASS = 2


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(image_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def parse_label(path: Path) -> tuple[list[str], Counter]:
    output: list[str] = []
    stats = Counter()
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{path}:{line_no}: expected YOLO label with at least 5 fields")
        source_class = int(float(parts[0]))
        if source_class != 0:
            raise ValueError(f"{path}:{line_no}: unexpected source class {source_class}; expected 0")
        values = [float(value) for value in parts[1:5]]
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{path}:{line_no}: normalized box values out of range: {values}")
        cx, cy, width, height = values
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_no}: non-positive box size: {values}")
        output.append(
            f"{TARGET_CIGARETTE_CLASS} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
        )
        stats["boxes"] += 1
        stats["area_gt_0.20"] += int(width * height > 0.20)
        stats["area_gt_0.50"] += int(width * height > 0.50)
        stats["width_gt_0.65"] += int(width > 0.65)
        stats["height_gt_0.65"] += int(height > 0.65)
    if not output:
        raise ValueError(f"{path}: empty label file")
    return output, stats


def deterministic_split(source_split: str, image: Path, valid_ratio: float) -> str:
    if source_split == "test":
        return "test"
    digest = hashlib.sha1(image.name.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "valid" if bucket < valid_ratio else "train"


def existing_image_hashes(target_root: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "valid", "test"):
        image_dir = target_root / split / "images"
        if not image_dir.exists():
            continue
        for image in image_dir.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                hashes.add(file_sha1(image))
    return hashes


def merge_dataset(source_root: Path, target_root: Path, valid_ratio: float) -> dict:
    target_root.mkdir(parents=True, exist_ok=True)
    known_hashes = existing_image_hashes(target_root)
    stats = Counter()
    split_stats: dict[str, Counter] = {
        split: Counter() for split in ("train", "valid", "test")
    }

    for source_split in ("train", "test"):
        image_dir = source_root / source_split / "images"
        label_dir = source_root / source_split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source split: {image_dir} / {label_dir}")

        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                stats["missing_images"] += 1
                continue

            image_hash = file_sha1(image_path)
            if image_hash in known_hashes:
                stats["duplicate_images"] += 1
                continue

            label_lines, label_stats = parse_label(label_path)
            target_split = deterministic_split(source_split, image_path, valid_ratio)
            base = f"smoking_people_{image_hash[:12]}_{image_path.stem}"
            image_out = target_root / target_split / "images" / f"{base}{image_path.suffix.lower()}"
            label_out = target_root / target_split / "labels" / f"{base}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_path, image_out)
            label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            known_hashes.add(image_hash)

            stats["images_added"] += 1
            stats["boxes_added"] += label_stats["boxes"]
            split_stats[target_split]["images_added"] += 1
            split_stats[target_split]["boxes_added"] += label_stats["boxes"]
            for key in ("area_gt_0.20", "area_gt_0.50", "width_gt_0.65", "height_gt_0.65"):
                stats[key] += label_stats[key]
                split_stats[target_split][key] += label_stats[key]

    return {
        "source_root": str(source_root.resolve()),
        "target_root": str(target_root.resolve()),
        "source_class_mapping": {"0": "cigarette"},
        "target_class_mapping": {"0": "helmet", "1": "head", "2": "cigarette"},
        "split_policy": {
            "source_train_valid_ratio": valid_ratio,
            "source_train": ["train", "valid"],
            "source_test": ["test"],
        },
        "stats": dict(stats),
        "by_split": {split: dict(values) for split, values in split_stats.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/smoking people.v1i.yolov8"),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/Detect_head_helmet_cigarette"),
    )
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")

    summary = merge_dataset(args.source.resolve(), args.target.resolve(), args.valid_ratio)
    summary_path = args.summary or (args.target / "merge_smoking_people_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
