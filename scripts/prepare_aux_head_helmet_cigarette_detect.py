from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CLASS_NAMES = ["helmet", "head", "cigarette"]


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "sample"


@dataclass
class Record:
    source: str
    split: str
    image: Path
    boxes: list[list[float]]


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def read_hardhat_record(label: Path, image: Path, split: str, source: str) -> Record | None:
    boxes = []
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        source_cls = int(float(parts[0]))
        if source_cls == 1:
            target_cls = 0  # helmet
        elif source_cls == 0:
            target_cls = 1  # bare head
        else:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        if w > 0 and h > 0:
            boxes.append([float(target_cls), cx, cy, w, h])
    if not boxes:
        return None
    return Record(source, split, image, boxes)


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_hardhat(roots: list[Path], dedupe_images: bool = True) -> tuple[list[Record], list[Record]]:
    train, test = [], []
    seen_images: set[str] = set()
    for root_idx, root in enumerate(roots):
        source = f"hardhat{root_idx + 1}"
        for source_split, bucket in (("train", train), ("test", test)):
            label_dir = root / source_split / "labels"
            image_dir = root / source_split / "images"
            if not label_dir.exists() or not image_dir.exists():
                continue
            for label in sorted(label_dir.glob("*.txt")):
                image = find_image(image_dir, label.stem)
                if image is None:
                    continue
                if dedupe_images:
                    digest = file_sha1(image)
                    if digest in seen_images:
                        continue
                    seen_images.add(digest)
                record = read_hardhat_record(label, image, source_split, source)
                if record is not None:
                    bucket.append(record)
    return train, test


def split_records(records: list[Record], rng: random.Random, valid_ratio: float) -> tuple[list[Record], list[Record]]:
    records = list(records)
    rng.shuffle(records)
    n_valid = int(round(len(records) * valid_ratio))
    valid = records[:n_valid]
    test = records[n_valid:]
    for r in valid:
        r.split = "valid"
    for r in test:
        r.split = "test"
    return valid, test


def class_counts(records: list[Record]) -> Counter:
    counts = Counter()
    for record in records:
        for box in record.boxes:
            counts[int(box[0])] += 1
    return counts


def select_hardhat(
    records: list[Record],
    target_helmet_boxes: int,
    target_head_boxes: int,
    rng: random.Random,
) -> list[Record]:
    # Keep all target labels in selected images. Dropping helmet boxes from a
    # head image, or vice versa, would turn known positives into false background.
    pool = list(records)
    rng.shuffle(pool)
    selected: list[Record] = []
    selected_ids: set[int] = set()
    counts = Counter()

    def add(record: Record) -> None:
        selected_ids.add(id(record))
        selected.append(record)
        for box in record.boxes:
            counts[int(box[0])] += 1

    head_pool = [r for r in pool if any(int(b[0]) == 1 for b in r.boxes)]
    head_total = class_counts(head_pool)[1]
    head_target = min(target_head_boxes, head_total)
    for record in head_pool:
        if counts[1] >= head_target:
            break
        add(record)

    helmet_pool = [r for r in pool if id(r) not in selected_ids and any(int(b[0]) == 0 for b in r.boxes)]
    helmet_total = class_counts(records)[0]
    helmet_target = min(target_helmet_boxes, helmet_total)
    for record in helmet_pool:
        if counts[0] >= helmet_target:
            break
        add(record)

    return selected


def read_cigarette_boxes(label: Path, max_area: float, max_wh: float) -> list[list[float]]:
    boxes = []
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        source_cls = int(float(parts[0]))
        if source_cls != 0:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        if w <= 0 or h <= 0 or w > max_wh or h > max_wh or w * h > max_area:
            continue
        boxes.append([2.0, cx, cy, w, h])
    return boxes


def collect_cigarette(raw_root: Path, max_area: float, max_wh: float) -> dict[str, dict[str, list[Record]]]:
    out: dict[str, dict[str, list[Record]]] = {
        "cigarette": defaultdict(list),
        "smoke": defaultdict(list),
    }
    for source in ("cigarette", "smoke"):
        for split in ("train", "val", "test"):
            label_dir = raw_root / source / "labels" / split
            image_dir = raw_root / source / "images" / split
            for label in sorted(label_dir.glob("*.txt")):
                image = find_image(image_dir, label.stem)
                if image is None:
                    continue
                boxes = read_cigarette_boxes(label, max_area=max_area, max_wh=max_wh)
                if boxes:
                    norm_split = "valid" if split == "val" else split
                    out[source][norm_split].append(Record(source, norm_split, image, boxes))
    return out


def select_cigarette(
    records_by_source: dict[str, list[Record]],
    target_boxes: int,
    rng: random.Random,
) -> list[Record]:
    # Prefer the dataset that explicitly contains cigarette/face/smoking classes,
    # then fill the remaining count from the larger smoke dataset.
    selected = []
    total = 0
    for source in ("cigarette", "smoke"):
        pool = list(records_by_source.get(source, []))
        rng.shuffle(pool)
        for record in pool:
            if total >= target_boxes:
                break
            selected.append(record)
            total += len(record.boxes)
    return selected


def copy_records(records: list[Record], output: Path) -> dict:
    stats = {
        "images": Counter(),
        "boxes": Counter(),
        "boxes_by_class": defaultdict(Counter),
        "images_by_source": defaultdict(Counter),
        "boxes_by_source": defaultdict(Counter),
    }
    seen_names = Counter()
    for idx, record in enumerate(records):
        split = "valid" if record.split == "val" else record.split
        stem = safe_component(record.image.stem)
        digest = file_sha1(record.image)[:10]
        base = f"{safe_component(record.source)}_{idx:06d}_{digest}_{stem}"
        seen_names[base] += 1
        if seen_names[base] > 1:
            base = f"{base}_{seen_names[base]}"
        image_name = f"{base}{record.image.suffix.lower()}"
        label_name = f"{base}.txt"
        image_out = output / split / "images" / image_name
        label_out = output / split / "labels" / label_name
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image, image_out)
        label_out.write_text(
            "\n".join(
                f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for cls, cx, cy, w, h in record.boxes
            )
            + "\n",
            encoding="utf-8",
        )
        stats["images"][split] += 1
        stats["images_by_source"][split][record.source] += 1
        for box in record.boxes:
            cls = int(box[0])
            stats["boxes"][split] += 1
            stats["boxes_by_class"][split][CLASS_NAMES[cls]] += 1
            stats["boxes_by_source"][split][record.source] += 1
    return stats


def write_data_yaml(output: Path):
    content = "\n".join(
        [
            f"path: {output.resolve().as_posix()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {len(CLASS_NAMES)}",
            f"names: {CLASS_NAMES}",
            "",
        ]
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def jsonable(obj):
    if isinstance(obj, (Counter, defaultdict)):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description="Build a standalone head/helmet/cigarette detection dataset.")
    parser.add_argument(
        "--hardhat-roots",
        type=Path,
        nargs="+",
        default=[Path("data/processed/hardhat.v1i.yolov8"), Path("data/hardhat")],
    )
    parser.add_argument("--cigarette-root", type=Path, default=Path("data/processed/raw_cigarette_annotations/downloaded"))
    parser.add_argument("--output", type=Path, default=Path("data/Detect_head_helmet_cigarette"))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--hardhat-valid-ratio", type=float, default=0.68)
    parser.add_argument("--target-train-boxes", type=int, default=7000)
    parser.add_argument("--target-valid-boxes", type=int, default=1700)
    parser.add_argument("--target-test-boxes", type=int, default=900)
    parser.add_argument("--cigarette-max-area", type=float, default=0.20)
    parser.add_argument("--cigarette-max-wh", type=float, default=0.65)
    parser.add_argument("--keep-duplicate-hardhat-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.cigarette_root.exists():
        raise FileNotFoundError(f"Missing cigarette dataset root: {args.cigarette_root}")

    rng = random.Random(args.seed)
    hardhat_train, hardhat_test = collect_hardhat(
        args.hardhat_roots,
        dedupe_images=not args.keep_duplicate_hardhat_images,
    )
    hardhat_valid, hardhat_test = split_records(hardhat_test, rng, args.hardhat_valid_ratio)
    target_boxes_by_split = {
        "train": args.target_train_boxes,
        "valid": args.target_valid_boxes,
        "test": args.target_test_boxes,
    }
    hardhat_by_split = {
        split: select_hardhat(
            records,
            target_helmet_boxes=target_boxes_by_split[split],
            target_head_boxes=target_boxes_by_split[split],
            rng=rng,
        )
        for split, records in (
            ("train", hardhat_train),
            ("valid", hardhat_valid),
            ("test", hardhat_test),
        )
    }

    cigarette_sources = collect_cigarette(
        args.cigarette_root,
        max_area=args.cigarette_max_area,
        max_wh=args.cigarette_max_wh,
    )

    selected = []
    split_targets = {}
    for split, hardhat_records in hardhat_by_split.items():
        target_cigarette = target_boxes_by_split[split]
        split_targets[split] = int(target_cigarette)
        selected.extend(hardhat_records)
        selected.extend(
            select_cigarette(
                {
                    "cigarette": cigarette_sources["cigarette"].get(split, []),
                    "smoke": cigarette_sources["smoke"].get(split, []),
                },
                target_boxes=target_cigarette,
                rng=rng,
            )
        )

    stats = copy_records(selected, args.output)
    write_data_yaml(args.output)
    summary = {
        "class_names": CLASS_NAMES,
        "mapping": {
            "hardhat class 1 helmet": "helmet",
            "hardhat class 0 head": "head",
            "cigarette/smoke class 0": "cigarette",
            "ignored": ["hardhat person", "cigarette face", "cigarette smoking"],
        },
        "filters": {
            "cigarette_max_area": args.cigarette_max_area,
            "cigarette_max_wh": args.cigarette_max_wh,
            "dedupe_hardhat_images": not args.keep_duplicate_hardhat_images,
        },
        "hardhat_roots": [str(p) for p in args.hardhat_roots],
        "target_boxes_per_class_by_split": target_boxes_by_split,
        "target_cigarette_boxes_by_split": split_targets,
        **jsonable(stats),
    }
    (args.output / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
