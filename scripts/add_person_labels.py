"""
用 YOLOv8x 为 helmet/fire_smoke/smoking 数据集添加 person 标签，
统一 person=class 0，筛选低质量样本，缩减到 2000 张。

处理顺序:
  1. 读取所有图片 + 现有标签
  2. YOLOv8x 推理 person 检测
  3. 按新 class mapping 生成标签 (person=0，旧类别顺移)
  4. helmet/smoking: 过滤 #persons > #object_labels 的样本
  5. 缩减到 2000 张 (优先保留标签丰富的样本)
  6. 替换旧目录 + 更新 data.yaml

用法: python scripts/add_person_labels.py
"""

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path("data/processed")

# ── 数据集配置 ──
# old→new: None = 丢弃该类
DATASET_SPECS = {
    "helmet": {
        "class_remap": {0: 1, 1: 2, 3: None},
        "new_names": {0: "person", 1: "helmet_on", 2: "helmet_off"},
        "filter": True,
        "max_images": 2000,
    },
    "smoking": {
        "class_remap": {0: 1},
        "new_names": {0: "person", 1: "cigarette"},
        "filter": True,
        "max_images": 557,
    },
    "fire_smoke": {
        "class_remap": {0: 1, 1: None},   # 只保留 fire, 丢弃 smoke
        "new_names": {0: "person", 1: "fire"},
        "filter": False,
        "max_images": 2000,
    },
}


def collect_pairs(dataset_dir):
    """收集 (img_path, lbl_path_or_None) 列表，支持嵌套子目录。"""
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    pairs = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # 尝试匹配 labels 下的对应路径
        rel = img_path.relative_to(img_dir)
        lbl_path = lbl_dir / rel.with_suffix(".txt")
        if not lbl_path.exists():
            lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            lbl_path = None
        pairs.append((img_path, lbl_path))
    return pairs


def read_old_labels(lbl_path):
    """返回 [(class_id, cx, cy, w, h), ...]，lbl_path 为 None 时返回空列表。"""
    labels = []
    if lbl_path is None or not lbl_path.exists():
        return labels
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w_n, h_n = map(float, parts[1:5])
            labels.append((cls_id, cx, cy, w_n, h_n))
    return labels


def remap_labels(old_labels, class_remap):
    """按 class_remap 转换，丢弃 map 值为 None 的类别。"""
    new = []
    for cls_id, cx, cy, wn, hn in old_labels:
        new_cls = class_remap.get(cls_id)
        if new_cls is not None:
            new.append((new_cls, cx, cy, wn, hn))
    return new


def detect_persons(model, img_path, device):
    """YOLOv8x 检测，返回归一化 person bboxes [(cx, cy, w, h), ...]"""
    results = model(str(img_path), device=device, verbose=False)
    persons = []
    h, w = None, None
    for r in results:
        h, w = r.orig_shape
        if r.boxes is None:
            continue
        for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls_id.item()) != 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            persons.append((
                (x1 + x2) / 2 / w,
                (y1 + y2) / 2 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
            ))
    return persons


def process_one(name, spec, model, device):
    dataset_dir = ROOT / name
    pairs = collect_pairs(dataset_dir)
    print(f"  images: {len(pairs)}")

    class_remap = spec["class_remap"]
    do_filter = spec["filter"]
    max_n = spec["max_images"]

    # ── YOLO 检测 + 标签生成 ──
    records = []
    for img_path, old_lbl in tqdm(pairs, desc=f"  {name} detect"):
        yolo_persons = detect_persons(model, img_path, device)
        old_labels = read_old_labels(old_lbl)
        obj_labels = remap_labels(old_labels, class_remap)  # 仅 object 标签
        n_objects = len(obj_labels)
        n_persons = len(yolo_persons)

        # 过滤: 人物数 > 物体标签数 → 存在无标注人物
        if do_filter and n_persons != n_objects:
            continue

        # 组装最终标签: person(0) 在前，object 在后
        final_labels = []
        for pcx, pcy, pw, ph in yolo_persons:
            final_labels.append((0, pcx, pcy, pw, ph))
        final_labels.extend(obj_labels)

        # 打分: object 标签权重更高，person 次之
        score = n_objects * 3 + n_persons
        records.append({
            "src": img_path,
            "labels": final_labels,
            "score": score,
        })

    n_before = len(pairs)
    n_after_filter = len(records)
    print(f"  filter: {n_before} → {n_after_filter} ({n_before - n_after_filter} removed)")

    # ── 缩减到 max_n ──
    records.sort(key=lambda x: -x["score"])
    if len(records) > max_n:
        records = records[:max_n]
        print(f"  reduce: capped at {max_n}")

    # ── 写入临时目录 ──
    tmp_img = dataset_dir / "_images_tmp"
    tmp_lbl = dataset_dir / "_labels_tmp"
    for d in (tmp_img, tmp_lbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    for i, rec in enumerate(tqdm(records, desc=f"  {name} write")):
        ext = rec["src"].suffix
        new_stem = f"{name}_{i:05d}"
        shutil.copy2(rec["src"], tmp_img / f"{new_stem}{ext}")
        with open(tmp_lbl / f"{new_stem}.txt", "w") as f:
            for cls_id, cx, cy, wn, hn in rec["labels"]:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")

    # ── 替换旧目录 ──
    old_img = dataset_dir / "images"
    old_lbl = dataset_dir / "labels"
    old_cache = dataset_dir / "letterbox_cache"
    for d in (old_img, old_lbl, old_cache):
        if d.exists():
            shutil.rmtree(d)
    tmp_img.rename(old_img)
    tmp_lbl.rename(old_lbl)

    # ── 更新 data.yaml ──
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {json.dumps(str(dataset_dir.resolve()))}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        names = spec["new_names"]
        f.write(f"names: {json.dumps({str(k): v for k, v in names.items()})}\n")

    return {"n_before": n_before, "n_kept": len(records)}


def main():
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "" else "cpu"
    print(f"Vigil: Add person labels (YOLOv8x, {device})")
    print(f"  targets: {list(DATASET_SPECS.keys())}")

    model = YOLO("yolov8x.pt")

    for name, spec in DATASET_SPECS.items():
        print(f"\n{'='*50}\n[{name}]")
        try:
            s = process_one(name, spec, model, device)
            print(f"  done: {s['n_kept']} images kept (from {s['n_before']})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone. Backups at: data/processed/{helmet,smoking,fire_smoke}_backup/")


if __name__ == "__main__":
    main()

