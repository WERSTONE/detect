"""Prepare COCO2017 labels for COCO80 detection + person keypoints.

The output label format is:
    cls x_center y_center width height [x y v] * 17

Class ids are standard YOLO COCO80 ids. Non-person classes receive zero
keypoints so the dataset has a consistent 56-column label format.
"""

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

COCO80_CATEGORY_TO_CLASS = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url}")
    urlretrieve(url, dest)


def download_coco(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for url in COCO_URLS.values():
        zip_path = data_dir / Path(url).name
        download_file(url, zip_path)
        print(f"Extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)


def _load_keypoints(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for ann in data["annotations"]:
        lookup.setdefault(ann["image_id"], {})[ann["id"]] = ann.get("keypoints", [0] * 51)
    return lookup


def convert_split(data_dir, split):
    data_dir = Path(data_dir)
    ann_dir = data_dir / "annotations"
    inst_file = ann_dir / f"instances_{split}.json"
    kpt_file = ann_dir / f"person_keypoints_{split}.json"
    if not inst_file.exists():
        raise FileNotFoundError(inst_file)

    with open(inst_file, encoding="utf-8") as f:
        instances = json.load(f)
    keypoints = _load_keypoints(kpt_file)
    images = {img["id"]: img for img in instances["images"]}

    grouped = {}
    for ann in instances["annotations"]:
        cls = COCO80_CATEGORY_TO_CLASS.get(ann["category_id"])
        if cls is None or ann.get("iscrowd", 0):
            continue
        if ann["bbox"][2] <= 1 or ann["bbox"][3] <= 1:
            continue
        grouped.setdefault(ann["image_id"], []).append((ann, cls))

    label_dir = data_dir / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    for old_label in label_dir.glob("*.txt"):
        old_label.unlink()

    converted = 0
    for image_id, img in sorted(images.items()):
        anns = grouped.get(image_id, [])
        img_w, img_h = float(img["width"]), float(img["height"])
        lines = []
        for ann, cls in anns:
            x, y, w, h = ann["bbox"]
            values = [
                str(cls),
                f"{(x + w / 2) / img_w:.6f}",
                f"{(y + h / 2) / img_h:.6f}",
                f"{w / img_w:.6f}",
                f"{h / img_h:.6f}",
            ]
            kpt_data = keypoints.get(image_id, {}).get(ann["id"], [0] * 51)
            if cls != 0:
                kpt_data = [0] * 51
            for idx in range(17):
                kx = kpt_data[idx * 3]
                ky = kpt_data[idx * 3 + 1]
                kv = min(float(kpt_data[idx * 3 + 2]), 2.0)
                values.extend([
                    f"{kx / img_w:.6f}" if kx > 0 else "0.000000",
                    f"{ky / img_h:.6f}" if ky > 0 else "0.000000",
                    f"{kv:.0f}",
                ])
            lines.append(" ".join(values))

        label_path = label_dir / f"{Path(img['file_name']).stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        converted += 1

    print(f"{split}: wrote {converted} label files to {label_dir}")


def verify_labels(data_dir):
    data_dir = Path(data_dir)
    for split in ("train2017", "val2017"):
        label_dir = data_dir / "labels" / split
        bad_rows = 0
        rows = 0
        for label_path in label_dir.glob("*.txt"):
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    rows += 1
                    if len(parts) != 56:
                        bad_rows += 1
                        continue
                    cls = int(float(parts[0]))
                    if cls < 0 or cls >= 80:
                        bad_rows += 1
        print(f"{split}: rows={rows} bad_rows={bad_rows}")


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO80 labels with person keypoints")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_coco(args.data_dir)
    for split in ("train2017", "val2017"):
        convert_split(args.data_dir, split)
    if args.verify:
        verify_labels(args.data_dir)


if __name__ == "__main__":
    main()

