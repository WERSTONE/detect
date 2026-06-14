"""
用 YOLO11x-pose 更新 helmet/fire_smoke/smoking/water_leak 的人框+关键点。

处理逻辑:
  1. 备份各数据集
  2. YOLO11n-pose 推理 → person bbox + 17关键点
  3. 保留原标签中所有非 person 行 (class != 0) 不变
  4. 用 YOLO 结果覆盖 person (class 0) 行，格式: cls cx cy w h + 51 kpt值
  5. 更新 data.yaml

用法: python scripts/update_person_kpts.py
"""

import json
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

# 需要处理的数据集
DATASETS = ["smoking"]


def collect_pairs(dataset_dir):
    """收集 (img_path, lbl_path) 列表。"""
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    pairs = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rel = img_path.relative_to(img_dir)
        lbl_path = lbl_dir / rel.with_suffix(".txt")
        if not lbl_path.exists():
            lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            lbl_path = None
        pairs.append((img_path, lbl_path))
    return pairs


def read_non_person_labels(lbl_path):
    """读取标签文件，返回所有非 person (class != 0) 的行。"""
    lines = []
    if lbl_path is None or not lbl_path.exists():
        return lines
    with open(lbl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id != 0:  # 只保留非 person 标签
                lines.append(line)
    return lines


def process_dataset(name, model, device):
    dataset_dir = ROOT / name
    if not dataset_dir.exists():
        print(f"  SKIP: {dataset_dir} not found")
        return

    # ── 备份 ──
    backup_dir = ROOT / f"{name}_backup_v2"
    if not backup_dir.exists():
        shutil.copytree(dataset_dir, backup_dir)
        print(f"  backup → {backup_dir}")

    pairs = collect_pairs(dataset_dir)
    print(f"  images: {len(pairs)}")

    # ── YOLO 推理 ──
    records = []  # [{src, new_name, person_lines, other_lines}]
    for img_path, old_lbl in tqdm(pairs, desc=f"  {name} infer"):
        try:
            results = model(str(img_path), device=device, verbose=False, conf=0.7)
        except Exception as e:
            print(f"  skip {img_path.name}: {e}")
            records.append({"src": img_path, "person_lines": [], "other_lines": read_non_person_labels(old_lbl)})
            continue

        person_lines = []
        for r in results:
            h_img, w_img = r.orig_shape
            if r.keypoints is None or r.keypoints.data is None:
                continue

            boxes = r.boxes
            kpts_data = r.keypoints.data  # [N, 17, 3]

            if boxes is None or len(boxes) == 0:
                continue

            for box, kpt in zip(boxes.xyxy, kpts_data):
                cls_id = int(box.cls.item()) if hasattr(box, 'cls') else 0
                if cls_id != 0:
                    continue  # 只取 person

                x1, y1, x2, y2 = box[:4].tolist()
                # 转 YOLO 归一化坐标
                cx = (x1 + x2) / 2 / w_img
                cy = (y1 + y2) / 2 / h_img
                bw = (x2 - x1) / w_img
                bh = (y2 - y1) / h_img

                # 关键点归一化
                kpt_parts = []
                for kp in kpt:
                    kx, ky, kv = kp[0].item(), kp[1].item(), kp[2].item()
                    kpt_parts.append(f"{kx / w_img:.6f} {ky / h_img:.6f} {kv:.6f}")

                # 格式: cls cx cy w h + 51 kpt values
                line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_parts)
                person_lines.append(line)

        # 保留原有非 person 标签
        other_lines = read_non_person_labels(old_lbl)

        records.append({
            "src": img_path,
            "person_lines": person_lines,
            "other_lines": other_lines,
        })

    # ── 统计 ──
    n_persons = sum(len(r["person_lines"]) for r in records)
    n_others = sum(len(r["other_lines"]) for r in records)
    print(f"  persons: {n_persons}, other labels: {n_others}")

    # ── 写入临时目录 ──
    tmp_img = dataset_dir / "_tmp_img"
    tmp_lbl = dataset_dir / "_tmp_lbl"
    for d in (tmp_img, tmp_lbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    for i, rec in enumerate(tqdm(records, desc=f"  {name} write")):
        ext = rec["src"].suffix
        stem = f"{name}_{i:05d}"
        shutil.copy2(rec["src"], tmp_img / f"{stem}{ext}")

        all_lines = rec["person_lines"] + rec["other_lines"]
        with open(tmp_lbl / f"{stem}.txt", "w") as f:
            for line in all_lines:
                f.write(line + "\n")

    # ── 替换旧目录 ──
    old_img = dataset_dir / "images"
    old_lbl = dataset_dir / "labels"
    for d in (old_img, old_lbl):
        if d.exists():
            shutil.rmtree(d)
    tmp_img.rename(old_img)
    tmp_lbl.rename(old_lbl)

    # 清理缓存
    cache = dataset_dir / "letterbox_cache"
    if cache.exists():
        shutil.rmtree(cache)

    # ── 更新 data.yaml (保持 names 不变，仅更新路径和格式) ──
    yaml_path = dataset_dir / "data.yaml"
    old_yaml = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    old_yaml[k.strip()] = v.strip()

    names = old_yaml.get("names", "{}")
    with open(yaml_path, "w") as f:
        f.write(f"path: {json.dumps(str(dataset_dir.resolve()))}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"names: {names}\n")

    print(f"  done: {len(records)} images")


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Update person boxes + keypoints (YOLO11n-pose, {device})")
    print(f"  targets: {DATASETS}")

    model = YOLO("yolo11x-pose.pt")

    for name in DATASETS:
        print(f"\n{'='*50}\n[{name}]")
        try:
            process_dataset(name, model, device)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone. Backups at: data/processed/*_backup_v2/")


if __name__ == "__main__":
    main()

