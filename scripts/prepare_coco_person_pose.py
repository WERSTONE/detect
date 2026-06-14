"""Prepare a COCO person-pose subset in Vigil label format.

This script downloads COCO keypoint annotations, selects a manageable subset of
images with visible person keypoints, downloads only those images, and writes:

    data/processed/coco_person_pose/images/*.jpg
    data/processed/coco_person_pose/labels/*.txt

Label rows follow the Vigil person format:
    0 cx cy w h [17 * (x y vis)] helmet_attr smoke_attr

Coordinates are normalized to the original image size. helmet_attr and
smoke_attr are set to -1 because COCO does not annotate these attributes.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/{split}/{image_id:012d}.jpg"


def download_file(url: str, dst: Path, retries: int = 3, chunk_size: int = 1024 * 1024):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=chunk_size)
            tmp.replace(dst)
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(1.5 * attempt)


def ensure_annotations(data_dir: Path, split: str) -> Path:
    ann_dir = data_dir / "coco" / "annotations"
    ann_path = ann_dir / f"person_keypoints_{split}.json"
    if ann_path.exists():
        return ann_path

    zip_path = ann_dir / "annotations_trainval2017.zip"
    print(f"Downloading COCO annotations -> {zip_path}")
    download_file(ANNOTATIONS_URL, zip_path)

    wanted = f"annotations/person_keypoints_{split}.json"
    print(f"Extracting {wanted}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(wanted, ann_dir)
    extracted = ann_dir / wanted
    extracted.replace(ann_path)
    extracted.parent.rmdir()
    return ann_path


def coco_box_to_norm_xywh(box, img_w, img_h):
    x, y, w, h = box
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def coco_kpts_to_norm(kpts, img_w, img_h):
    out = []
    for i in range(17):
        x, y, v = kpts[i * 3:i * 3 + 3]
        if v <= 0:
            out.extend([0.0, 0.0, 0.0])
        else:
            out.extend([x / img_w, y / img_h, 1.0 if v > 0 else 0.0])
    return out


def build_subset(ann, max_images, min_visible_kpts, max_persons_per_image, seed):
    images = {img["id"]: img for img in ann["images"]}
    anns_by_image = defaultdict(list)

    for a in ann["annotations"]:
        if a.get("iscrowd", 0):
            continue
        if a.get("category_id") != 1:
            continue
        if a.get("num_keypoints", 0) < min_visible_kpts:
            continue
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        if w <= 2 or h <= 2:
            continue
        anns_by_image[a["image_id"]].append(a)

    candidates = []
    for image_id, anns in anns_by_image.items():
        img = images.get(image_id)
        if not img:
            continue
        anns = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)
        candidates.append((image_id, img, anns[:max_persons_per_image]))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_images]


def write_label(label_path: Path, img, anns):
    img_w, img_h = img["width"], img["height"]
    rows = []
    for a in anns:
        cx, cy, bw, bh = coco_box_to_norm_xywh(a["bbox"], img_w, img_h)
        vals = [0, cx, cy, bw, bh]
        vals.extend(coco_kpts_to_norm(a["keypoints"], img_w, img_h))
        vals.extend([-1, -1])
        rows.append(" ".join(
            str(v) if isinstance(v, int) else f"{v:.6f}" for v in vals))
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare_one(item, args, img_out: Path, lbl_out: Path):
    image_id, img, anns = item
    stem = f"coco_{args.split}_{image_id:012d}"
    dst_img = img_out / f"{stem}.jpg"
    dst_lbl = lbl_out / f"{stem}.txt"
    if not dst_img.exists():
        url = IMAGE_URL.format(split=args.split, image_id=image_id)
        download_file(url, dst_img, retries=args.retries)
    write_label(dst_lbl, img, anns)
    return image_id


def prepare(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ann_path = ensure_annotations(data_dir, args.split)
    print(f"Loading {ann_path}")
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    subset = build_subset(
        ann,
        max_images=args.max_images,
        min_visible_kpts=args.min_visible_kpts,
        max_persons_per_image=args.max_persons_per_image,
        seed=args.seed,
    )

    print(f"Selected {len(subset)} {args.split} images")
    written = 0
    skipped = 0
    if args.workers <= 1:
        for idx, item in enumerate(subset, 1):
            try:
                prepare_one(item, args, img_out, lbl_out)
                written += 1
            except Exception as exc:
                skipped += 1
                print(f"[skip] {item[0]}: {exc}")
            if idx % args.log_interval == 0:
                print(f"  {idx}/{len(subset)} processed, written={written}, skipped={skipped}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(prepare_one, item, args, img_out, lbl_out) for item in subset]
            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    fut.result()
                    written += 1
                except Exception as exc:
                    skipped += 1
                    print(f"[skip] {exc}")
                if idx % args.log_interval == 0:
                    print(f"  {idx}/{len(subset)} processed, written={written}, skipped={skipped}")

    meta = {
        "source": "COCO 2017 person_keypoints",
        "split": args.split,
        "max_images": args.max_images,
        "written": written,
        "min_visible_kpts": args.min_visible_kpts,
        "max_persons_per_image": args.max_persons_per_image,
    }
    (out_dir / "data.yaml").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done: {written} images -> {out_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="data/processed/coco_person_pose")
    p.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    p.add_argument("--max-images", type=int, default=8000)
    p.add_argument("--min-visible-kpts", type=int, default=8)
    p.add_argument("--max-persons-per-image", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-interval", type=int, default=200)
    return p.parse_args(argv)


if __name__ == "__main__":
    prepare(parse_args(sys.argv[1:]))

