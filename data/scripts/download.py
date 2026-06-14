"""
开源数据集下载脚本。
每个数据集有独立的下载函数，请手动运行所需的部分。
用法: python data/scripts/download.py --dataset coco
      python data/scripts/download.py --dataset all  (全部下载)
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ── COCO 2017 (person + keypoints) ──

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

def download_coco(target_dir=None):
    """下载 COCO 2017 图片 + 关键点标注 (~20GB)。建议只下 val2017(1GB) 先测试。"""
    target_dir = target_dir or os.path.join(DATA_DIR, "coco2017")
    _ensure_dir(target_dir)
    print(f"[COCO] Downloading to {target_dir}")

    for name, url in COCO_URLS.items():
        dst = os.path.join(target_dir, os.path.basename(url))
        if os.path.exists(dst):
            print(f"  {name}: already downloaded")
            continue
        print(f"  {name}: downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"  {name}: download failed ({e})")
            continue

        print(f"  {name}: extracting...")
        with zipfile.ZipFile(dst, "r") as zf:
            zf.extractall(target_dir)
        os.remove(dst)

    print("[COCO] Done. Expected structure:")
    print("  data/raw/coco2017/train2017/")
    print("  data/raw/coco2017/val2017/")
    print("  data/raw/coco2017/annotations/")


# ── SHWD (安全帽) ──

def download_shwd(target_dir=None):
    """
    SHWD 原始链接已失效，可从以下镜像获取:
      - Kaggle: https://www.kaggle.com/datasets/junwide/safetyhelmetwearing
      - GitHub: https://github.com/yuqingbin/Safety-Helmet-Wearing-Dataset
    下载后解压到 data/raw/shwd/，确保结构为:
      data/raw/shwd/
        ├── Annotations/  (Pascal VOC XML)
        └── JPEGImages/   (图片)
    """
    target_dir = target_dir or os.path.join(DATA_DIR, "shwd")
    _ensure_dir(target_dir)
    print(f"[SHWD] Please manually download from:")
    print(f"  Kaggle: https://www.kaggle.com/datasets/junwide/safetyhelmetwearing")
    print(f"  GitHub: https://github.com/yuqingbin/Safety-Helmet-Wearing-Dataset")
    print(f"  Extract to: {target_dir}")
    print(f"  Expected: {target_dir}/Annotations/ + {target_dir}/JPEGImages/")


# ── Smoking123 (吸烟行为) ──

def download_smoking123(target_dir=None):
    target_dir = target_dir or os.path.join(DATA_DIR, "smoking123")
    _ensure_dir(target_dir)
    print(f"[Smoking123] Manual download required:")
    print(f"  GitHub: https://github.com/qunshansj/Smoking-behavior-detection")
    print(f"  Arrange as: {target_dir}/images/ + {target_dir}/labels/")


# ── D-Fire (烟火检测) ──

D_FIRE_URL = "https://drive.google.com/uc?export=download&id=1EwFG7JbTWKQcU6n4Chf72_gFkqCYaH7s"

def download_dfire(target_dir=None):
    target_dir = target_dir or os.path.join(DATA_DIR, "dfire")
    _ensure_dir(target_dir)
    print(f"[D-Fire] Download from:")
    print(f"  GitHub: https://github.com/gaiasd/DFireDataset")
    print(f"  Kaggle: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo")
    print(f"  Arrange as: {target_dir}/images/ + {target_dir}/labels/")


# ── Water Leak (漏水) ──

def download_leak(target_dir=None):
    target_dir = target_dir or os.path.join(DATA_DIR, "leak")
    _ensure_dir(target_dir)
    print(f"[Water Leak] Choose one of:")
    print(f"  Roboflow (free): https://universe.roboflow.com/kai-bsnlf/water-leakage-bnrzx")
    print(f"  Building Stain: https://blog.csdn.net/fl1623863129/article/details/147952571")
    print(f"  Download from Roboflow as YOLO format, extract to {target_dir}/")
    print(f"  Arrange as: {target_dir}/images/ + {target_dir}/labels/")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Download open-source datasets for Vigil")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "coco", "shwd", "smoking123", "dfire", "leak"])
    parser.add_argument("--dir", default=None, help="Target directory")
    args = parser.parse_args()

    datasets = {
        "coco": download_coco,
        "shwd": download_shwd,
        "smoking123": download_smoking123,
        "dfire": download_dfire,
        "leak": download_leak,
    }

    if args.dataset == "all":
        for name, fn in datasets.items():
            print(f"\n{'='*60}")
            fn(args.dir)
    else:
        datasets[args.dataset](args.dir)


if __name__ == "__main__":
    main()
