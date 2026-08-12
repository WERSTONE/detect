"""Filter full COCO labels to person-only pose supervision data.

Reads the 56-column YOLO COCO80 labels produced by ``prepare_coco80`` (or, when
the label directory is missing, the official ``person_keypoints_*.json``),
keeps only images that contain at least one qualifying person, and writes a
label set that contains only person rows.

Original COCO image filenames are preserved so image lookup and the official
COCO pose eval (which keys results by numeric image id) keep working.

Output layout (matches ``test_model.final.PoseDataset``):
    <coco-root>/<labels-out>/<split>/*.txt     only person rows, class 0
    <coco-root>/<split>/<original>.jpg         images stay in place
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COCO_CATEGORY_ID_TO_80 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}


def parse_args():
    p = argparse.ArgumentParser(
        description="Keep only person-containing COCO samples as pose supervision")
    p.add_argument("--coco-root", required=True,
                   help="COCO root, e.g. /root/autodl-tmp/coco2017")
    p.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    p.add_argument("--labels-in", default="labels",
                   help="Source YOLO label dir under coco-root (56-col COCO80)")
    p.add_argument("--labels-out", default="labels_person",
                   help="Filtered label dir written under coco-root")
    p.add_argument("--class-id-format", default="yolo80", choices=["yolo80", "coco80"],
                   help="Person class id in the source labels (yolo80: 0, coco80: 1)")
    p.add_argument("--min-persons", type=int, default=1,
                   help="Keep an image when it has at least this many kept persons")
    p.add_argument("--min-visible-kpts", type=int, default=1,
                   help="Drop a person row with fewer visible keypoints (0 = any)")
    p.add_argument("--max-images", type=int, default=0,
                   help="Cap kept images per split to this many (0 = keep all). "
                        "Samples are picked as an evenly spaced spread to stay representative.")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy kept images into a fresh root instead of leaving them in place")
    p.add_argument("--output-root", default=None,
                   help="When --copy-images, copy images to <output-root>/<split> and "
                        "labels to <output-root>/<labels-out>/<split>")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing filtered label directory")
    return p.parse_args()


def person_class_id(class_id_format: str) -> int:
    return 0 if class_id_format == "yolo80" else 1


def count_visible_kpts(parts, n_kpts=17) -> int:
    visible = 0
    for j in range(n_kpts):
        idx = 5 + j * 3 + 2
        if idx < len(parts):
            try:
                if float(parts[idx]) > 0:
                    visible += 1
            except ValueError:
                pass
    return visible


def parse_txt_label(path: Path, person_cls: int, min_visible_kpts: int):
    """Return filtered person rows for one label file, or None to drop the image."""
    kept_rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls != person_cls:
            continue
        if min_visible_kpts > 0 and count_visible_kpts(parts) < min_visible_kpts:
            continue
        parts[0] = "0"
        kept_rows.append(" ".join(parts))
    if not kept_rows:
        return None
    return kept_rows


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_from_json(coco_root: Path, split: str, src_label_dir: Path) -> bool:
    """Fallback: generate person-only labels from person_keypoints_*.json.

    Returns True when json source was used and labels were written.
    """
    ann_path = coco_root / "annotations" / f"person_keypoints_{split}.json"
    if not ann_path.exists():
        return False
    src_label_dir.mkdir(parents=True, exist_ok=True)
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in ann["images"]}
    anns_by_image: dict[int, list] = {}
    for a in ann["annotations"]:
        if a.get("iscrowd", 0):
            continue
        if a.get("category_id") != 1:
            continue
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        if w <= 1 or h <= 1:
            continue
        anns_by_image.setdefault(a["image_id"], []).append(a)

    written = 0
    for image_id, img in sorted(images.items()):
        anns = anns_by_image.get(image_id)
        if not anns:
            continue
        img_w, img_h = float(img["width"]), float(img["height"])
        rows = []
        for a in anns:
            kpts = a.get("keypoints", [0] * 51)
            visible = sum(1 for i in range(17) if kpts[i * 3 + 2] > 0)
            if visible < 1:
                continue
            x, y, w, h = a["bbox"]
            vals = [
                "0",
                f"{(x + w / 2) / img_w:.6f}",
                f"{(y + h / 2) / img_h:.6f}",
                f"{w / img_w:.6f}",
                f"{h / img_h:.6f}",
            ]
            for i in range(17):
                kx, ky, kv = kpts[i * 3:i * 3 + 3]
                vals.extend([
                    f"{kx / img_w:.6f}" if kx > 0 else "0.000000",
                    f"{ky / img_h:.6f}" if ky > 0 else "0.000000",
                    "1" if kv > 0 else "0",
                ])
            rows.append(" ".join(vals))
        if not rows:
            continue
        stem = Path(img["file_name"]).stem
        (src_label_dir / f"{stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
        written += 1
    print(f"[{split}] generated {written} person labels from json -> {src_label_dir}")
    return True


def main():
    args = parse_args()
    coco_root = Path(args.coco_root)
    person_cls = person_class_id(args.class_id_format)

    summary = {"coco_root": str(coco_root), "splits": {}}

    for split in args.splits:
        image_dir = coco_root / split
        src_label_dir = coco_root / args.labels_in / split
        out_label_dir = coco_root / args.labels_out / split

        if not src_label_dir.exists() or not any(src_label_dir.glob("*.txt")):
            print(f"[{split}] no labels in {src_label_dir}; trying person_keypoints json")
            convert_from_json(coco_root, split, src_label_dir)

        if not src_label_dir.exists() or not any(src_label_dir.glob("*.txt")):
            print(f"[{split}] SKIP: no source labels found")
            continue

        if out_label_dir.exists() and any(out_label_dir.iterdir()) and not args.force:
            raise FileExistsError(
                f"{out_label_dir} is not empty; pass --force to overwrite")
        out_label_dir.mkdir(parents=True, exist_ok=True)

        out_image_dir = None
        if args.copy_images:
            out_root = Path(args.output_root or coco_root)
            out_image_dir = out_root / split
            out_label_dir = out_root / args.labels_out / split
            out_label_dir.mkdir(parents=True, exist_ok=True)
            out_image_dir.mkdir(parents=True, exist_ok=True)

        kept = dropped_no_person = dropped_kpts = missing_image = 0
        total_person_rows = 0
        candidates = []
        for label_path in sorted(src_label_dir.glob("*.txt")):
            rows = parse_txt_label(label_path, person_cls, args.min_visible_kpts)
            if rows is None:
                dropped_no_person += 1
                continue
            image = find_image(image_dir, label_path.stem)
            if image is None:
                missing_image += 1
                continue
            candidates.append((label_path.name, image, rows))

        if args.max_images and len(candidates) > args.max_images:
            indices = np.linspace(0, len(candidates) - 1, args.max_images, dtype=np.int64)
            candidates = [candidates[int(i)] for i in indices]

        for name, image, rows in candidates:
            (out_label_dir / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
            total_person_rows += len(rows)
            if args.copy_images:
                shutil.copy2(image, out_image_dir / image.name)
            kept += 1

        summary["splits"][split] = {
            "source_labels": sum(1 for _ in src_label_dir.glob("*.txt")),
            "kept_images": kept,
            "dropped_no_person": dropped_no_person,
            "dropped_kpts": dropped_kpts,
            "missing_image": missing_image,
            "person_rows": total_person_rows,
            "labels_out": str(out_label_dir),
        }
        print(f"[{split}] kept={kept} person_rows={total_person_rows} "
              f"no_person={dropped_no_person} missing_image={missing_image}")

    out_json = coco_root / args.labels_out / "filter_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print("Point test_model/final/yaml/final_three_head.yaml pose labels to:")
    for split in args.splits:
        print(f"  {split}: labels: {args.labels_out}/{split}")


if __name__ == "__main__":
    main()
