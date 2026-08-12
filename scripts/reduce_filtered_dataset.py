"""Create a reduced copy of a filtered domain-attribute image dataset."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample selected class folders into a reduced dataset copy.")
    parser.add_argument("--input", required=True, help="Filtered dataset root containing class folders")
    parser.add_argument("--output", required=True, help="Reduced dataset output root")
    parser.add_argument("--reduce", nargs="+", default=["hat", "rest"], help="Class folders to downsample")
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Ratio to keep for folders listed in --reduce")
    parser.add_argument("--classes", nargs="+", default=["hat", "smoking", "falling", "rest"])
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing empty output")
    return parser.parse_args()


def list_images(path):
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def discrete_sample(paths, keep_ratio):
    if keep_ratio >= 1.0:
        return list(paths)
    keep = max(1, int(math.ceil(len(paths) * keep_ratio))) if paths else 0
    if keep >= len(paths):
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, keep, dtype=np.int64)
    return [paths[int(i)] for i in indices]


def prepare_output(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    prepare_output(output_root, args.overwrite)

    reduce_set = set(args.reduce)
    summary = {
        "input": str(input_root),
        "output": str(output_root),
        "keep_ratio": args.keep_ratio,
        "reduced_classes": sorted(reduce_set),
        "classes": {},
    }
    rows = []

    for class_name in args.classes:
        src_dir = input_root / class_name
        dst_dir = output_root / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        paths = list_images(src_dir)
        selected = discrete_sample(paths, args.keep_ratio) if class_name in reduce_set else paths
        for src in selected:
            shutil.copy2(src, dst_dir / src.name)
        summary["classes"][class_name] = {
            "source_count": len(paths),
            "kept_count": len(selected),
            "source_bytes": int(sum(p.stat().st_size for p in paths)),
            "kept_bytes": int(sum(p.stat().st_size for p in selected)),
        }
        for src in selected:
            rows.append({"class": class_name, "source": str(src), "output": str(dst_dir / src.name)})
        print(f"{class_name}: kept {len(selected)}/{len(paths)}", flush=True)

    with open(output_root / "reduction_manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "source", "output"])
        writer.writeheader()
        writer.writerows(rows)
    with open(output_root / "reduction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
