"""
数据集标注清洗 — 根据人工审查结果过滤低质量类别。

操作:
  fire_smoke:  只保留 class 0 (fire), 移除 class 1 (smoke)
  water_leak:  只保留 class 0 (stagnant_water), 移除 class 1 (wet_surface)

用法: python data/scripts/clean_labels.py [--dry-run]
"""
import os
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 清洗规则: {dataset_name: keep_class_ids}
CLEAN_RULES = {
    "fire_smoke": {0},      # 只保留 fire
    "water_leak": {0},      # 只保留 stagnant_water
}


def clean_dataset(dataset_name, keep_ids, dry_run=False):
    ds_dir = PROCESSED_DIR / dataset_name
    if not ds_dir.exists():
        print(f"  [skip] {dataset_name}: not found")
        return

    for split in ["train", "val"]:
        lbl_dir = ds_dir / "labels" / split
        img_dir = ds_dir / "images" / split
        if not lbl_dir.exists():
            continue

        removed_imgs = 0
        kept = 0

        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            with open(lbl_path) as f:
                lines = f.readlines()

            filtered = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in keep_ids:
                    filtered.append(line)

            if not filtered:
                # 该图所有标注都被移除 → 删除图片和标签
                stem = lbl_path.stem
                img_candidates = list(img_dir.glob(f"{stem}.*"))
                if not dry_run:
                    lbl_path.unlink()
                    for ip in img_candidates:
                        ip.unlink()
                removed_imgs += 1
            else:
                if not dry_run and len(filtered) < len(lines):
                    with open(lbl_path, "w") as f:
                        f.writelines(filtered)
                kept += 1

        print(f"  [{dataset_name}/{split}] kept={kept} removed={removed_imgs}")

    # 更新 data.yaml
    if not dry_run:
        yaml_path = ds_dir / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                content = f.read()
            # 更新 names 字段
            if dataset_name == "fire_smoke":
                content = content.replace(
                    'names: {"0": "fire", "1": "smoke"}',
                    'names: {"0": "fire"}')
            elif dataset_name == "water_leak":
                content = content.replace(
                    'names: {"0": "stagnant_water", "1": "wet_surface"}',
                    'names: {"0": "stagnant_water"}')
            with open(yaml_path, "w") as f:
                f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Vigil 数据集标注清洗")
    parser.add_argument("--dry-run", action="store_true", help="仅预览")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] 仅预览\n")

    for name, keep_ids in CLEAN_RULES.items():
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Cleaning {name} (keep classes: {keep_ids})...")
        clean_dataset(name, keep_ids, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
