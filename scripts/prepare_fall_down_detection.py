"""Prepare the Fall Down Detection dataset from an Ultralytics NDJSON export.

The source NDJSON contains one dataset metadata row and one row per image.
Each image row includes a signed image URL, split name, and normalized boxes.

This script downloads only the referenced images and writes a YOLO-style
dataset under the chosen output directory.

Modes:
  - detect: class 0 = fallen-person
  - attr4:  class 0 = person, 17 zero keypoints, attrs=[smoking, falling,
             waving, helmet_on], masks=[0, 1, 0, 0]

The attr4 mode is meant for the multi-task person-attribute pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ATTR_NAMES = ["smoking", "falling", "waving", "helmet_on"]
ATTR_VALUES = [0.0, 1.0, 0.0, 0.0]
ATTR_MASKS = [0.0, 1.0, 0.0, 0.0]
ZERO_KPTS = [0.0] * 51
SPLIT_MAP = {"valid": "val"}


def _fmt_value(value):
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def _normalize_split(split: str) -> str:
    return SPLIT_MAP.get(split, split)


def _download_file(url: str, dst: Path, retries: int = 3, chunk_size: int = 1024 * 1024):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=chunk_size)
            tmp.replace(dst)
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(1.5 * attempt)


def _load_ndjson(path: Path):
    meta = None
    images = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            typ = obj.get("type")
            if typ == "dataset":
                meta = obj
            elif typ == "image":
                images.append(obj)
            else:
                raise ValueError(f"unexpected row type at line {line_no}: {typ}")
    if meta is None:
        raise ValueError(f"no dataset metadata found in {path}")
    return meta, images


def _build_label_lines(item: dict, label_mode: str) -> str:
    boxes = item.get("annotations", {}).get("boxes", []) or []
    rows = []
    for box in boxes:
        if len(box) < 5:
            continue
        _, cx, cy, bw, bh = box[:5]
        base = [0, cx, cy, bw, bh]
        if label_mode == "attr4":
            base.extend(ZERO_KPTS)
            base.extend(ATTR_VALUES)
            base.extend(ATTR_MASKS)
        rows.append(" ".join(_fmt_value(v) for v in base))
    return ("\n".join(rows) + "\n") if rows else ""


def _process_one(item: dict, out_dir: Path, label_mode: str, retries: int):
    split = _normalize_split(item.get("split", "train"))
    rel = Path(item["file"])
    if rel.is_absolute() or ".." in rel.parts:
        rel = Path(rel.name)
    img_dst = out_dir / "images" / split / rel
    lbl_dst = out_dir / "labels" / split / rel.with_suffix(".txt")

    _download_file(item["url"], img_dst, retries=retries)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.write_text(_build_label_lines(item, label_mode), encoding="utf-8")
    return split


def _write_data_yaml(out_dir: Path, source: Path, meta: dict, label_mode: str):
    names = ["person"] if label_mode == "attr4" else ["fallen-person"]
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 1",
        "names:",
    ]
    for name in names:
        lines.append(f"  - {name}")
    lines.append(f"label_mode: {label_mode}")
    lines.append(f"source_ndjson: {source.resolve().as_posix()}")
    if label_mode == "attr4":
        lines.append("attributes:")
        lines.append("  names:")
        for name in ATTR_NAMES:
            lines.append(f"    - {name}")
    lines.append(f"source_dataset_url: {meta.get('url', '')}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Path to fall-down-detection.ndjson")
    p.add_argument("--output", default="data/processed/fall_down_detection_attr",
                   help="Output dataset root")
    p.add_argument("--label-mode", default="attr4", choices=["attr4", "detect"],
                   help="Write attr-ready person labels or raw detection labels")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on the number of images to process")
    p.add_argument("--splits", default="train,val,test",
                   help="Comma-separated split filter, e.g. train,val")
    p.add_argument("--log-interval", type=int, default=250)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, images = _load_ndjson(source)
    allowed_splits = {_normalize_split(s.strip()) for s in args.splits.split(",") if s.strip()}
    if allowed_splits:
        images = [
            item for item in images
            if _normalize_split(item.get("split", "train")) in allowed_splits
        ]
    if args.limit and args.limit > 0:
        images = images[:args.limit]

    source_copy = out_dir / source.name
    if source.resolve() != source_copy.resolve():
        shutil.copy2(source, source_copy)

    split_counts = Counter(_normalize_split(item.get("split", "train")) for item in images)
    print(f"Found {len(images)} images")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")
    print(f"Output: {out_dir}")

    written = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as ex:
        futures = [
            ex.submit(_process_one, item, out_dir, args.label_mode, args.retries)
            for item in images
        ]
        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
                written += 1
            except Exception as exc:
                failed += 1
                print(f"[skip] {exc}")
            if args.log_interval > 0 and idx % args.log_interval == 0:
                print(f"  processed={idx}/{len(images)} written={written} failed={failed}")

    _write_data_yaml(out_dir, source, meta, args.label_mode)

    print(f"Done: written={written} failed={failed}")
    print(f"labels at: {out_dir / 'labels'}")


if __name__ == "__main__":
    main(sys.argv[1:])
