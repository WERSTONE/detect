"""Build lightweight source-aware class-supervision manifests.

The manifests describe which unified detection classes are semantically
defined for each image. They are intentionally separate from the uploaded
images and YOLO label files so the same preprocessing can run on the server.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ALL_CLASSES = [1, 2, 3, 4, 5, 6, 7]
ORIGINAL_VALID = [1, 2, 3, 4, 7]
ATTR_SMOKING_POS_VALID = [1, 2, 3, 4]
ATTR_FALLING_VALID = [1, 2, 3, 4, 7, 8]
ATTR_SMOKING_FALLING_VALID = [1, 2, 3, 4, 8]


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(image_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _original_smoking_positive(label_path: Path) -> bool:
    """Use only the original 13-column Attr label, never the VLM replacement."""
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            smoking = float(parts[5])
            smoking_mask = float(parts[9])
        except ValueError:
            continue
        if smoking >= 0.5 and smoking_mask > 0.0:
            return True
    return False


def _original_falling_supervised(label_path: Path) -> bool:
    """Use only the original 13-column Attr label, never the VLM replacement."""
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            falling_mask = float(parts[10])
        except ValueError:
            continue
        if falling_mask > 0.0:
            return True
    return False


def build_attr_manifest(attr_root: Path, output_name: str, overwrite: bool) -> dict:
    records: list[dict] = []
    by_split: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    for group_dir in sorted(p for p in attr_root.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            for split in ("train", "val", "test"):
                image_dir = dataset_dir / "images" / split
                label_dir = dataset_dir / "labels" / split
                if not image_dir.exists() or not label_dir.exists():
                    continue
                for label_path in sorted(label_dir.glob("*.txt")):
                    image_path = _find_image(image_dir, label_path.stem)
                    if image_path is None:
                        continue
                    smoking_positive = _original_smoking_positive(label_path)
                    falling_supervised = _original_falling_supervised(label_path)
                    valid_class_ids = set(ORIGINAL_VALID)
                    if smoking_positive:
                        valid_class_ids.discard(7)
                    if falling_supervised:
                        valid_class_ids.add(8)
                    if smoking_positive and falling_supervised:
                        policy = "attr_smoking_falling_positive"
                    elif smoking_positive:
                        policy = "attr_smoking_positive"
                    elif falling_supervised:
                        policy = "attr_falling_positive"
                    else:
                        policy = "attr_default"
                    records.append({
                        "image": _rel(image_path, attr_root),
                        "source_group": group_dir.name,
                        "source_dataset": dataset_dir.name,
                        "split": split,
                        "original_smoking_positive": smoking_positive,
                        "original_falling_supervised": falling_supervised,
                        "valid_class_ids": sorted(valid_class_ids),
                    })
                    by_split[split] += 1
                    by_policy[policy] += 1

    output = attr_root / output_name
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    _write_jsonl(output, records)
    return {
        "path": str(output),
        "records": len(records),
        "by_split": dict(by_split),
        "by_policy": dict(by_policy),
    }


def _aux_policy(image_path: Path) -> tuple[str, list[int]]:
    name = image_path.name.lower()
    if name.startswith("hardhat1_"):
        return "head_helmet", ALL_CLASSES
    if (
        name.startswith("cigarette_")
        or name.startswith("smoke_")
        or name.startswith("smoking_people_")
    ):
        return "cigarette", ORIGINAL_VALID
    raise ValueError(
        f"Unknown auxiliary image prefix: {image_path.name}. "
        "Expected hardhat1_, cigarette_, smoke_, or smoking_people_."
    )


def build_aux_manifest(aux_root: Path, output_name: str, overwrite: bool) -> dict:
    records: list[dict] = []
    by_split: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    for split in ("train", "valid", "test"):
        image_dir = aux_root / split / "images"
        label_dir = aux_root / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = _find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            source_group, valid_ids = _aux_policy(image_path)
            records.append({
                "image": _rel(image_path, aux_root),
                "source_group": source_group,
                "split": split,
                "valid_class_ids": valid_ids,
            })
            by_split[split] += 1
            by_group[source_group] += 1

    output = aux_root / output_name
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    _write_jsonl(output, records)
    return {
        "path": str(output),
        "records": len(records),
        "by_split": dict(by_split),
        "by_group": dict(by_group),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attr-root", required=True, type=Path)
    parser.add_argument("--aux-detect-root", required=True, type=Path)
    parser.add_argument("--manifest-name", default="domain_supervision_manifest_falling.jsonl")
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attr_root = args.attr_root.resolve()
    aux_root = args.aux_detect_root.resolve()
    if not attr_root.is_dir():
        raise FileNotFoundError(attr_root)
    if not aux_root.is_dir():
        raise FileNotFoundError(aux_root)
    summary = {
        "policy": {
            "unified_class_ids": {
                "1": "puddle", "2": "fire", "3": "smoke", "4": "other",
                "5": "helmet", "6": "head", "7": "cigarette", "8": "falling",
            },
            "original_detect": ORIGINAL_VALID,
            "aux_head_helmet": ALL_CLASSES,
            "aux_cigarette": ORIGINAL_VALID,
            "attr_original_smoking_positive": ATTR_SMOKING_POS_VALID,
            "attr_falling_positive": ATTR_FALLING_VALID,
            "attr_smoking_falling_positive": ATTR_SMOKING_FALLING_VALID,
            "attr_other": ORIGINAL_VALID,
        },
        "attr": build_attr_manifest(attr_root, args.manifest_name, args.overwrite),
        "aux_detect": build_aux_manifest(aux_root, args.manifest_name, args.overwrite),
    }
    summary_path = args.summary_json or (aux_root / "domain_supervision_manifest_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
