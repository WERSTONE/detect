"""
筛选并整理开源预训练数据，统一转为 YOLO 格式，控制总量。

用法:
  python data/scripts/prepare_data.py              # 全部处理
  python data/scripts/prepare_data.py --dry-run     # 仅统计，不生成文件
  python data/scripts/prepare_data.py --max 1000    # 每数据集最多 1000 张

输出: data/processed/
  person/       COCO person bbox (YOLO)
  helmet/       Hardhat → YOLO (0=on, 1=off, 2=none)
  fire_smoke/   D-Fire YOLO (0=fire, 1=smoke)
  smoking/      CigDet YOLO (0=smoking)
  water_leak/   Water YOLO (0=stain, 1=drip)
  keypoints/    COCO person keypoints (JSON)
"""

import os
import sys
import json
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SEED = 42


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clean_out_dir(task: str):
    """清空并重建任务输出目录。"""
    for sub in ["images", "labels"]:
        d = OUT_DIR / task / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)


def write_data_yaml(task: str, names: dict, extra: dict = None):
    """生成 YOLO 风格的 data.yaml。"""
    yaml = {
        "path": str((OUT_DIR / task).resolve()),
        "train": "images",
        "val": "images",
        "names": names,
    }
    if extra:
        yaml.update(extra)
    with open(OUT_DIR / task / "data.yaml", "w") as f:
        for k, v in yaml.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")


# ══════════════════════════════════════════════
# COCO → person bbox (YOLO)
# ══════════════════════════════════════════════

def process_coco_person(max_images: int = 2000, dry: bool = False):
    """
    从 COCO val2017 提取 person 类 bbox，转为 YOLO 格式。
    """
    task = "person"
    print(f"\n{'='*50}\n[{task}] Processing...")

    coco_root = RAW_DIR / "coco2017"
    ann_path = coco_root / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017"

    if not ann_path.exists():
        print(f"  SKIP: {ann_path} not found")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == 1:  # person
            anns_by_img[ann["image_id"]].append(ann)

    # 优先选 person 多 + 有 keypoints 的图片
    scored = []
    for img_id, anns in anns_by_img.items():
        img = img_map[img_id]
        kp_anns = [a for a in anns if a.get("num_keypoints", 0) > 0]
        score = len(anns) * 2 + len(kp_anns) * 3
        scored.append((score, img, anns))
    scored.sort(key=lambda x: -x[0])

    selected = scored[:max_images]
    random.Random(SEED).shuffle(selected)  # 打乱避免顺序偏差

    if not dry:
        clean_out_dir(task)
        for _, img_info, anns in selected:
            src = img_dir / img_info["file_name"]
            dst_img = OUT_DIR / task / "images" / img_info["file_name"]
            if src.exists():
                shutil.copy2(src, dst_img)

            h, w = img_info["height"], img_info["width"]
            lbl_path = OUT_DIR / task / "labels" / f"{Path(img_info['file_name']).stem}.txt"
            with open(lbl_path, "w") as f:
                for a in anns:
                    x, y, bw, bh = a["bbox"]
                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        write_data_yaml(task, {0: "person"})

    print(f"  {'[DRY] ' if dry else ''}Saved {len(selected)} images → {OUT_DIR / task}/")


# ══════════════════════════════════════════════
# COCO → person keypoints (JSON)
# ══════════════════════════════════════════════

def process_coco_keypoints(max_images: int = 1500, dry: bool = False):
    """
    从 COCO val2017 提取有 keypoints 的 person 图片，保留 COCO JSON 格式。
    """
    task = "keypoints"
    print(f"\n{'='*50}\n[{task}] Processing...")

    coco_root = RAW_DIR / "coco2017"
    ann_path = coco_root / "annotations" / "person_keypoints_val2017.json"
    img_dir = coco_root / "val2017"

    if not ann_path.exists():
        print(f"  SKIP: {ann_path} not found")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["num_keypoints"] > 0:
            anns_by_img[ann["image_id"]].append(ann)

    scored = []
    for img_id, anns in anns_by_img.items():
        total_kp = sum(a["num_keypoints"] for a in anns)
        scored.append((total_kp, img_map[img_id], anns))
    scored.sort(key=lambda x: -x[0])

    selected = scored[:max_images]
    random.Random(SEED).shuffle(selected)

    if not dry:
        clean_out_dir(task)
        new_images = []
        new_anns = []
        img_id_map = {}

        for new_id, (_, img_info, anns) in enumerate(selected, start=1):
            old_id = img_info["id"]
            img_id_map[old_id] = new_id

            src = img_dir / img_info["file_name"]
            dst_img = OUT_DIR / task / "images" / img_info["file_name"]
            if src.exists():
                shutil.copy2(src, dst_img)

            new_img = dict(img_info)
            new_img["id"] = new_id
            new_images.append(new_img)

            for a in anns:
                new_a = dict(a)
                new_a["id"] = len(new_anns) + 1
                new_a["image_id"] = new_id
                new_anns.append(new_a)

        subset = {
            "images": new_images,
            "annotations": new_anns,
            "categories": coco["categories"],
        }
        json_path = OUT_DIR / task / "annotations.json"
        with open(json_path, "w") as f:
            json.dump(subset, f)

        write_data_yaml(task, {1: "person"},
                        {"format": "coco", "annotation_file": "annotations.json"})

    print(f"  {'[DRY] ' if dry else ''}Saved {len(selected)} images → {OUT_DIR / task}/")


# ══════════════════════════════════════════════
# Hardhat (VOC XML) → YOLO
# ══════════════════════════════════════════════

HELMET_CLASS_MAP = {"helmet": 0, "head": 1, "person": 2}
# 0=helmet_on, 1=helmet_off (head visible no helmet), 2=helmet_none (person far/no head)


def process_hardhat(max_images: int = 2000, dry: bool = False):
    """
    将 Hardhat Pascal VOC XML 转为 YOLO 格式。
    """
    task = "helmet"
    print(f"\n{'='*50}\n[{task}] Processing...")

    hh_root = RAW_DIR / "Hardhat"
    img_dir = hh_root / "images"
    ann_dir = hh_root / "annotations"

    if not ann_dir.exists():
        print(f"  SKIP: {ann_dir} not found")
        return

    samples = []
    for xml_file in sorted(ann_dir.glob("*.xml")):
        tree = ET.parse(xml_file)
        root_el = tree.getroot()
        img_name = root_el.find("filename").text
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        size_el = root_el.find("size")
        w = int(size_el.find("width").text) if size_el is not None else 416
        h = int(size_el.find("height").text) if size_el is not None else 416

        objects = []
        has_helmet = False
        for obj in root_el.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in HELMET_CLASS_MAP:
                continue
            if cls_name == "helmet":
                has_helmet = True
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)
            objects.append((HELMET_CLASS_MAP[cls_name], x1, y1, x2, y2))

        # 偏好有 helmet 标注的图片
        score = 10 if has_helmet else 1
        samples.append((score, img_path, w, h, objects))

    samples.sort(key=lambda x: -x[0])
    selected = samples[:max_images]
    random.Random(SEED).shuffle(selected)

    if not dry:
        clean_out_dir(task)
        for i, (_, img_path, w, h, objects) in enumerate(selected):
            new_name = f"helmet_{i:05d}.jpg"
            shutil.copy2(img_path, OUT_DIR / task / "images" / new_name)

            # 将 helmet/head 级别 bbox 扩展为近似 person bbox
            # 逻辑与 _parse_voc 一致: head bbox → person bbox
            person_objects = []
            for cls_id, x1, y1, x2, y2 in objects:
                bw, bh = x2 - x1, y2 - y1
                if cls_id in (0, 1):  # helmet or head → 扩展为 person
                    px1 = max(0, x1 - bw * 0.5)
                    py1 = max(0, y1 - bh * 0.3)
                    px2 = min(w, x2 + bw * 0.5)
                    py2 = min(h, y2 + bh * 2.5)
                    person_objects.append((cls_id, px1, py1, px2, py2))
                else:  # person → 不扩展
                    person_objects.append((cls_id, x1, y1, x2, y2))

            lbl_path = OUT_DIR / task / "labels" / f"helmet_{i:05d}.txt"
            with open(lbl_path, "w") as f:
                for cls_id, px1, py1, px2, py2 in person_objects:
                    cx = ((px1 + px2) / 2) / w
                    cy = ((py1 + py2) / 2) / h
                    nw = (px2 - px1) / w
                    nh = (py2 - py1) / h
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        write_data_yaml(task, {0: "helmet_on", 1: "helmet_off", 2: "helmet_none"})

    print(f"  {'[DRY] ' if dry else ''}Saved {len(selected)} images → {OUT_DIR / task}/")


# ══════════════════════════════════════════════
# Fire / Smoke (YOLO)
# ══════════════════════════════════════════════

def process_fire(max_images: int = 2000, dry: bool = False):
    """
    筛选 fire 数据集中有 label 的图片，统一为 YOLO 格式。

    fire 数据集结构特殊: images 和 labels 命名不一致 (部分 NoFileSmoke* 无标签)，
    只保留 stem 匹配的 image-label 对。
    """
    task = "fire_smoke"
    print(f"\n{'='*50}\n[{task}] Processing...")

    fire_root = RAW_DIR / "fire"
    if not fire_root.exists():
        print(f"  SKIP: {fire_root} not found")
        return

    pairs = []
    for split in ["train", "val", "test"]:
        img_dir = fire_root / split
        lbl_dir = fire_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if img_path.stem in lbl_stems:
                pairs.append((img_path, lbl_dir / f"{img_path.stem}.txt"))

    if not pairs:
        print("  SKIP: no matching image-label pairs found")
        return

    # 优先选择含 fire (class 0) 和 smoke (class 1) 都有的样本
    scored = []
    for img_path, lbl_path in pairs:
        has_fire = has_smoke = False
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 0:
                    has_fire = True
                elif cls_id == 1:
                    has_smoke = True
        score = (has_fire + has_smoke) * 5 + 1
        scored.append((score, img_path, lbl_path))
    scored.sort(key=lambda x: -x[0])

    selected = scored[:max_images]
    random.Random(SEED).shuffle(selected)

    if not dry:
        clean_out_dir(task)
        for i, (_, img_path, lbl_path) in enumerate(selected):
            ext = img_path.suffix
            new_name = f"fire_{i:05d}{ext}"
            shutil.copy2(img_path, OUT_DIR / task / "images" / new_name)

            dst_lbl = OUT_DIR / task / "labels" / f"fire_{i:05d}.txt"
            shutil.copy2(lbl_path, dst_lbl)  # class 保持原样: 0=fire, 1=smoke

        write_data_yaml(task, {0: "fire", 1: "smoke"})

    print(f"  {'[DRY] ' if dry else ''}Saved {len(selected)} images → {OUT_DIR / task}/")
    print(f"  Total matching pairs in raw: {len(pairs)}")


# ══════════════════════════════════════════════
# CigDet (Smoking, YOLO)
# ══════════════════════════════════════════════

def process_smoking(max_images: int = 9999, dry: bool = False):
    """
    CigDet 数据集很小 (557 images)，全部保留。
    """
    task = "smoking"
    print(f"\n{'='*50}\n[{task}] Processing...")

    cig_root = RAW_DIR / "CigDet_dataset"
    if not cig_root.exists():
        print(f"  SKIP: {cig_root} not found")
        return

    # 合并 train + test
    pairs = []
    for split in ["train", "test"]:
        d = cig_root / split
        if not d.exists():
            continue
        for txt_path in sorted(d.glob("*.txt")):
            img_path = d / f"{txt_path.stem}.jpg"
            if img_path.exists():
                pairs.append((img_path, txt_path))

    selected = pairs[:max_images]  # CigDet 很小，不用打分
    random.Random(SEED).shuffle(selected)

    if not dry:
        clean_out_dir(task)
        for i, (img_path, txt_path) in enumerate(selected):
            ext = img_path.suffix
            new_name = f"smoke_{i:05d}{ext}"
            shutil.copy2(img_path, OUT_DIR / task / "images" / new_name)

            dst_lbl = OUT_DIR / task / "labels" / f"smoke_{i:05d}.txt"
            shutil.copy2(txt_path, dst_lbl)

        write_data_yaml(task, {0: "smoking"})

    print(f"  {'[DRY] ' if dry else ''}Saved {len(selected)} images → {OUT_DIR / task}/")


# ══════════════════════════════════════════════
# Water Leak (YOLO)
# ══════════════════════════════════════════════

WATER_CLASS_REMAP = {0: 0, 1: 1}
# 数据原类: 0=water(stagnant)→stain, 1=wet_surface→drip
# 但我们保持原 id，在训练时用 class_map 重映射


def process_water(max_images: int = 2000, dry: bool = False):
    """
    整理 Water/Wet Surface 数据集。
    跳过 Raw data (无标注) 和 Rotated (增强重复)。
    """
    task = "water_leak"
    print(f"\n{'='*50}\n[{task}] Processing...")

    # 选择第一个副本（第二个是重复的）
    water_root = (RAW_DIR / "Dataset of Stagnant Water and Wet Surface with Annotations"
                  / "Stagnant water and Wet surface Dataset")

    if not water_root.exists():
        print(f"  SKIP: {water_root} not found")
        return

    # 只取 Outdoor (数据量大) + Indoor (数量少但都要)
    pairs = []
    for sub in ["Outdoor", "Indoor"]:
        d = water_root / sub
        if not d.exists():
            continue
        for txt_path in sorted(d.glob("*.txt")):
            # 找对应图片 (.jpeg 或 .jpg)
            img_path = d / f"{txt_path.stem}.jpeg"
            if not img_path.exists():
                img_path = d / f"{txt_path.stem}.jpg"
            if img_path.exists():
                pairs.append((img_path, txt_path))

    # 优先选择标签多的图片
    scored = []
    for img_path, txt_path in pairs:
        with open(txt_path) as f:
            n_lines = sum(1 for l in f if l.strip())
        scored.append((n_lines, img_path, txt_path))
    scored.sort(key=lambda x: -x[0])

    selected = scored[:max_images]
    random.Random(SEED).shuffle(selected)

    if not dry:
        clean_out_dir(task)
        for i, (_, img_path, txt_path) in enumerate(selected):
            ext = img_path.suffix
            new_name = f"leak_{i:05d}{ext}"
            shutil.copy2(img_path, OUT_DIR / task / "images" / new_name)

            # 保持原始 class ID (0=stagnant_water, 1=wet_surface)
            dst_lbl = OUT_DIR / task / "labels" / f"leak_{i:05d}.txt"
            shutil.copy2(txt_path, dst_lbl)

        write_data_yaml(task, {0: "stagnant_water", 1: "wet_surface"})

    print(f"  {'[DRY] ' if dry else ''}Saved {len(selected)} images → {OUT_DIR / task}/")
    print(f"  Total labeled pairs in raw: {len(pairs)}")


# ══════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════

def cleanup_raw(dry: bool = False):
    """删除重复/冗余的原始数据。"""
    print(f"\n{'='*50}\n[Cleanup]")

    dup = (RAW_DIR / "Dataset of Stagnant Water and Wet Surface with Annotations"
           / "Stagnant Water and Wet Surface Dataset 1")
    if dup.exists():
        print(f"  Removing duplicate Water dataset: {dup}")
        if not dry:
            shutil.rmtree(dup)

    # 删除 fire 数据集中没有对应 label 的图片（NoFileSmoke*）
    fire_root = RAW_DIR / "fire"
    if fire_root.exists():
        removed = 0
        for split in ["train", "val", "test"]:
            img_dir = fire_root / split
            lbl_dir = fire_root / "labels" / split
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if img_path.stem not in lbl_stems:
                    if not dry:
                        img_path.unlink()
                    removed += 1
        print(f"  {'[DRY] ' if dry else ''}Removed {removed} unlabeled images from fire dataset")


# ══════════════════════════════════════════════
# 汇总统计
# ══════════════════════════════════════════════

def print_summary():
    print(f"\n{'='*50}\n[Summary]")
    total = 0
    for task_dir in sorted(OUT_DIR.glob("*")):
        if not task_dir.is_dir():
            continue
        n_imgs = len(list((task_dir / "images").glob("*"))) if (task_dir / "images").exists() else 0
        size_mb = sum(f.stat().st_size for f in task_dir.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"  {task_dir.name:20s}  {n_imgs:5d} images  {size_mb:6.1f} MB")
        total += n_imgs
    print(f"  {'TOTAL':20s}  {total:5d} images")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Vigil")
    parser.add_argument("--max", type=int, default=2000,
                        help="Max images per big dataset (default: 2000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print stats, don't write files")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Skip raw data cleanup")
    args = parser.parse_args()

    random.seed(SEED)
    dry = args.dry_run
    mx = getattr(args, "max")

    print(f"Vigil Data Preparation")
    print(f"  max_per_dataset: {mx}")
    print(f"  dry_run: {dry}")
    print(f"  output: {OUT_DIR.resolve()}")

    ensure_dir(OUT_DIR)

    # 按依赖关系处理
    process_coco_person(max_images=mx, dry=dry)
    process_coco_keypoints(max_images=max(1, mx * 3 // 4), dry=dry)
    process_hardhat(max_images=mx, dry=dry)
    process_fire(max_images=mx, dry=dry)
    process_smoking(max_images=9999, dry=dry)  # 保留全部
    process_water(max_images=mx, dry=dry)

    if not args.skip_cleanup:
        cleanup_raw(dry=dry)

    print_summary()
    print("\nDone. Update config/train.yaml stages.pretrain.datasets with:")
    print("""
  person:     {path: "data/processed/person",      weight: 1.0}
  helmet:     {path: "data/processed/helmet",      weight: 1.0}
  fire_smoke: {path: "data/processed/fire_smoke",  weight: 1.0}
  smoking:    {path: "data/processed/smoking",     weight: 1.0}
  water_leak: {path: "data/processed/water_leak",  weight: 1.0}
  keypoints:  {path: "data/processed/keypoints",   weight: 1.0}
""")


if __name__ == "__main__":
    main()
