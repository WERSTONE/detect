"""Datasets and single-loader sampling for final three-task training.

The final training flow keeps detect, attr, and COCO-pose samples as separate
sources so each task can use task-appropriate augmentation, then concatenates
them into one dataset sampled by a single random loader. Batches do not
guarantee per-task presence; task balance is expected from dataset
construction (the COCO person subset is capped to match detect/attr sizes).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_root(root: str | Path) -> Path:
    """Resolve a config data root, treating relative paths as repo-relative."""
    path = Path(root)
    raw = str(root)
    if path.is_absolute() or raw.startswith(("/", "\\")):
        return path
    return PROJECT_ROOT / path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ATTR_NAMES = ("smoking", "falling", "waving", "helmet_on")
KPT_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


@dataclass(frozen=True)
class SampleRef:
    image: Path
    label: Path


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def yolo_xywh_to_xyxy(parts: list[str], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = map(float, parts[:4])
    bw *= width
    bh *= height
    cx *= width
    cy *= height
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def xyxy_to_yolo(box: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    return [(x1 + bw / 2) / width, (y1 + bh / 2) / height, bw / width, bh / height]


class FinalBaseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_dir: str,
        label_dir: str,
        input_size: int = 640,
        augment: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        translate: float = 0.05,
        scale: float = 0.1,
        flip_lr: float = 0.5,
        task: str = "",
    ):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.label_dir = self.root / label_dir
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.translate = float(translate)
        self.scale = float(scale)
        self.flip_lr = float(flip_lr)
        self.task = task
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No samples found: {self.image_dir} / {self.label_dir}")

    def _collect_samples(self) -> list[SampleRef]:
        samples: list[SampleRef] = []
        for label in sorted(self.label_dir.glob("*.txt")):
            image = find_image(self.image_dir, label.stem)
            if image is not None:
                samples.append(SampleRef(image=image, label=label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _letterbox(self, image, boxes, kpts):
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h
        pad_l, pad_t = pad_w // 2, pad_h // 2
        image = cv2.copyMakeBorder(
            image,
            pad_t,
            pad_h - pad_t,
            pad_l,
            pad_w - pad_l,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        boxes = boxes.copy()
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_l
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_t
        kpts = kpts.copy()
        if len(kpts):
            kpts[..., 0] = kpts[..., 0] * scale + pad_l
            kpts[..., 1] = kpts[..., 1] * scale + pad_t
        return image, boxes, kpts, scale, (pad_l, pad_t)

    def _augment_basic(self, image, boxes, kpts, allow_attr_sensitive=True):
        h, w = image.shape[:2]
        image, boxes, kpts, scale, pad = self._letterbox(image, boxes, kpts)

        if self.augment:
            image = self._hsv(image)
            if self.flip_lr > 0 and random.random() < self.flip_lr:
                image = image[:, ::-1].copy()
                if len(boxes):
                    old_x1 = boxes[:, 0].copy()
                    old_x2 = boxes[:, 2].copy()
                    boxes[:, 0] = self.input_size - old_x2
                    boxes[:, 2] = self.input_size - old_x1
                if len(kpts):
                    kpts[..., 0] = self.input_size - kpts[..., 0]
                    kpts = kpts[:, KPT_FLIP_MAP]

            # Attr samples use only safe, full-image translation/scale. No crop,
            # CutMix, or object paste is applied here.
            if allow_attr_sensitive and (self.translate > 0 or self.scale > 0):
                image, boxes, kpts = self._safe_affine(image, boxes, kpts)

        return image, boxes, kpts, scale, pad

    def _safe_affine(self, image, boxes, kpts):
        s = self.input_size
        scale_factor = random.uniform(1.0 - self.scale, 1.0 + self.scale)
        tx = random.uniform(-self.translate, self.translate) * s
        ty = random.uniform(-self.translate, self.translate) * s
        matrix = cv2.getRotationMatrix2D((s / 2, s / 2), 0.0, scale_factor)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        image = cv2.warpAffine(image, matrix, (s, s), borderValue=(114, 114, 114))
        if len(boxes):
            corners = np.ones((len(boxes) * 4, 3), dtype=np.float32)
            corners[:, :2] = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
            warped = corners @ matrix.T
            warped = warped.reshape(len(boxes), 8)
            xs = warped[:, [0, 2, 4, 6]]
            ys = warped[:, [1, 3, 5, 7]]
            boxes[:, 0] = np.clip(xs.min(axis=1), 0, s - 1)
            boxes[:, 1] = np.clip(ys.min(axis=1), 0, s - 1)
            boxes[:, 2] = np.clip(xs.max(axis=1), 0, s - 1)
            boxes[:, 3] = np.clip(ys.max(axis=1), 0, s - 1)
        if len(kpts):
            pts = np.ones((len(kpts) * 17, 3), dtype=np.float32)
            pts[:, :2] = kpts[..., :2].reshape(-1, 2)
            warped = (pts @ matrix.T).reshape(len(kpts), 17, 2)
            kpts[..., :2] = warped
            outside = (
                (kpts[..., 0] < 0)
                | (kpts[..., 0] > s - 1)
                | (kpts[..., 1] < 0)
                | (kpts[..., 1] > s - 1)
            )
            kpts[..., 2] = np.where(outside, 0.0, kpts[..., 2])
            kpts[..., 0] = np.clip(kpts[..., 0], 0, s - 1)
            kpts[..., 1] = np.clip(kpts[..., 1], 0, s - 1)
        return image, boxes, kpts

    def _hsv(self, image):
        if max(abs(self.hsv_h), abs(self.hsv_s), abs(self.hsv_v)) <= 0:
            return image
        gains = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype
        x = np.arange(0, 256, dtype=gains.dtype)
        lut_hue = ((x * gains[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * gains[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * gains[2], 0, 255).astype(dtype)
        return cv2.cvtColor(
            cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))),
            cv2.COLOR_HSV2RGB,
        )

    def _to_sample(self, image, boxes, classes, kpts, attrs, attr_mask, ref, scale, pad):
        keep = []
        for idx, box in enumerate(boxes):
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                keep.append(idx)
        if keep:
            boxes = boxes[keep]
            classes = classes[keep]
            kpts = kpts[keep]
            attrs = attrs[keep]
            attr_mask = attr_mask[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)
            kpts = np.zeros((0, 17, 3), dtype=np.float32)
            attrs = np.zeros((0, 4), dtype=np.float32)
            attr_mask = np.zeros((0, 4), dtype=np.float32)

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {
            "image": image_t,
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "classes": torch.from_numpy(classes.astype(np.int64)),
            "kpts": torch.from_numpy(kpts.astype(np.float32)),
            "attrs": torch.from_numpy(attrs.astype(np.float32)),
            "attr_mask": torch.from_numpy(attr_mask.astype(np.float32)),
            "task": self.task,
            "scale": scale,
            "pad": pad,
            "img_path": str(ref.image),
            "orig_shape": None,
            "image_id": None,
        }


class DetectDataset(FinalBaseDataset):
    def __init__(self, *args, class_offset=1, **kwargs):
        super().__init__(*args, task="detect", **kwargs)
        self.class_offset = int(class_offset)
        self.sample_class_sets = [self._read_class_set(ref.label) for ref in self.samples]

    def _read_class_set(self, label_path: Path) -> set[int]:
        classes = set()
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                classes.add(int(float(parts[0])))
        return classes

    def sampler_weights(self, class_balance: bool = True) -> list[float]:
        if not class_balance:
            return [1.0] * len(self.samples)
        class_counts: dict[int, int] = {}
        for classes in self.sample_class_sets:
            for cls in classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        weights = []
        for classes in self.sample_class_sets:
            if not classes:
                weights.append(1.0)
            else:
                weights.append(sum(1.0 / max(class_counts[c], 1) for c in classes) / len(classes))
        mean = sum(weights) / max(len(weights), 1)
        return [w / max(mean, 1e-12) for w in weights]

    def __getitem__(self, idx):
        ref = self.samples[idx]
        image = self._load_image(ref.image)
        h, w = image.shape[:2]
        boxes, classes = [], []
        for line in ref.label.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            classes.append(int(float(parts[0])) + self.class_offset)
            boxes.append(yolo_xywh_to_xyxy(parts[1:5], w, h))
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        kpts = np.zeros((len(boxes), 17, 3), dtype=np.float32)
        attrs = np.zeros((len(boxes), 4), dtype=np.float32)
        attr_mask = np.zeros((len(boxes), 4), dtype=np.float32)
        image, boxes, kpts, scale, pad = self._augment_basic(image, boxes, kpts, allow_attr_sensitive=True)
        return self._to_sample(image, boxes, classes, kpts, attrs, attr_mask, ref, scale, pad)


class AttrDataset(FinalBaseDataset):
    def __init__(self, root: str | Path, split: str, **kwargs):
        self.split = split
        root = Path(root)
        refs: list[SampleRef] = []
        for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for ds_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                label_dir = ds_dir / "labels" / split
                image_dir = ds_dir / "images" / split
                if not label_dir.exists() or not image_dir.exists():
                    continue
                for label in sorted(label_dir.glob("*.txt")):
                    image = find_image(image_dir, label.stem)
                    if image is not None:
                        refs.append(SampleRef(image=image, label=label))
        self._precollected_refs = refs
        super().__init__(root=root, image_dir=".", label_dir=".", task="attr", **kwargs)

    def _collect_samples(self) -> list[SampleRef]:
        return list(getattr(self, "_precollected_refs", []))

    def sampler_weights(self, target_attrs: tuple[str, ...] = ATTR_NAMES) -> list[float]:
        attr_to_idx = {name: idx for idx, name in enumerate(ATTR_NAMES)}
        counts = {name: {"pos": 0, "neg": 0} for name in target_attrs}
        sample_keys: list[list[tuple[str, str]]] = []
        for ref in self.samples:
            keys: list[tuple[str, str]] = []
            for line in ref.label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 13:
                    continue
                for name in target_attrs:
                    idx = attr_to_idx[name]
                    value = float(parts[5 + idx])
                    mask = float(parts[9 + idx])
                    if mask <= 0:
                        continue
                    key = "pos" if value >= 0.5 else "neg"
                    counts[name][key] += 1
                    keys.append((name, key))
            sample_keys.append(keys)
        weights = []
        for keys in sample_keys:
            if not keys:
                weights.append(1.0)
                continue
            weights.append(sum(1.0 / max(counts[name][key], 1) for name, key in keys) / len(keys))
        mean = sum(weights) / max(len(weights), 1)
        return [w / max(mean, 1e-12) for w in weights]

    def __getitem__(self, idx):
        ref = self.samples[idx]
        image = self._load_image(ref.image)
        h, w = image.shape[:2]
        boxes, classes, attrs, attr_mask = [], [], [], []
        for line in ref.label.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 13:
                continue
            boxes.append(yolo_xywh_to_xyxy(parts[1:5], w, h))
            classes.append(0)
            attrs.append([float(x) for x in parts[5:9]])
            attr_mask.append([float(x) for x in parts[9:13]])
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        kpts = np.zeros((len(boxes), 17, 3), dtype=np.float32)
        attrs = np.asarray(attrs, dtype=np.float32).reshape(-1, 4)
        attr_mask = np.asarray(attr_mask, dtype=np.float32).reshape(-1, 4)
        image, boxes, kpts, scale, pad = self._augment_basic(image, boxes, kpts, allow_attr_sensitive=True)
        return self._to_sample(image, boxes, classes, kpts, attrs, attr_mask, ref, scale, pad)


class PoseDataset(FinalBaseDataset):
    def __init__(self, *args, source_class_format="yolo80", **kwargs):
        super().__init__(*args, task="pose", **kwargs)
        self.source_class_format = str(source_class_format)

    def _map_person_class(self, cls: int) -> int | None:
        if self.source_class_format in ("coco", "coco80"):
            return 0 if cls == 1 else None
        return 0 if cls == 0 else None

    def __getitem__(self, idx):
        ref = self.samples[idx]
        image = self._load_image(ref.image)
        h, w = image.shape[:2]
        boxes, classes, kpts = [], [], []
        for line in ref.label.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 56:
                continue
            mapped = self._map_person_class(int(float(parts[0])))
            if mapped is None:
                continue
            boxes.append(yolo_xywh_to_xyxy(parts[1:5], w, h))
            classes.append(mapped)
            kp = np.zeros((17, 3), dtype=np.float32)
            for j in range(17):
                kp[j] = [
                    float(parts[5 + j * 3]) * w,
                    float(parts[5 + j * 3 + 1]) * h,
                    float(parts[5 + j * 3 + 2]),
                ]
            kpts.append(kp)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 17, 3)
        attrs = np.zeros((len(boxes), 4), dtype=np.float32)
        attr_mask = np.zeros((len(boxes), 4), dtype=np.float32)
        image, boxes, kpts, scale, pad = self._augment_basic(image, boxes, kpts, allow_attr_sensitive=True)
        return self._to_sample(image, boxes, classes, kpts, attrs, attr_mask, ref, scale, pad)


def final_collate(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "classes": [item["classes"] for item in batch],
        "kpts": [item["kpts"] for item in batch],
        "attrs": [item["attrs"] for item in batch],
        "attr_mask": [item["attr_mask"] for item in batch],
        "task": [item["task"] for item in batch],
        "scale": [item.get("scale", 1.0) for item in batch],
        "pad": [item.get("pad", (0, 0)) for item in batch],
        "img_path": [item.get("img_path", "") for item in batch],
        "orig_shape": [item.get("orig_shape", None) for item in batch],
        "image_id": [item.get("image_id", None) for item in batch],
    }


def make_loader(dataset, batch_size, workers, shuffle=True, weights=None, drop_last=True):
    sampler = None
    if weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(len(dataset), int(batch_size)),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        collate_fn=final_collate,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


class FinalTaskBatchLoader:
    """Yield one logical batch made from detect, attr, and pose mini-batches."""

    def __init__(self, loaders: dict[str, DataLoader]):
        self.loaders = loaders
        self.dataset = self
        self.samples = []
        self._iters = None

    def __len__(self):
        return max(len(loader) for loader in self.loaders.values())

    def _ensure_iters(self):
        if self._iters is None:
            self._iters = {name: iter(loader) for name, loader in self.loaders.items()}

    def _next(self, name):
        try:
            return next(self._iters[name])
        except StopIteration:
            self._iters[name] = iter(self.loaders[name])
            return next(self._iters[name])

    def __iter__(self):
        self._ensure_iters()
        for _ in range(len(self)):
            parts = [self._next(name) for name in ("detect", "attr", "pose") if name in self.loaders]
            yield merge_batches(parts)

    def set_epoch(self, _epoch: int):
        return None

    def set_close_mosaic(self, _close: bool = False):
        return None


def merge_batches(parts: list[dict]) -> dict:
    images = torch.cat([part["image"] for part in parts], dim=0)
    merged = {"image": images}
    for key in ("boxes", "classes", "kpts", "attrs", "attr_mask", "task", "scale", "pad", "img_path", "orig_shape", "image_id"):
        values = []
        for part in parts:
            values.extend(part[key])
        merged[key] = values
    return merged


def split_batch_size(total: int, ratios: dict[str, float]) -> dict[str, int]:
    total = int(total)
    names = ["detect", "attr", "pose"]
    raw = {name: float(ratios.get(name, 0.0)) for name in names}
    if sum(raw.values()) <= 0:
        raw = {"detect": 1.0, "attr": 1.0, "pose": 1.0}
    denom = sum(raw.values())
    exact = {name: total * raw[name] / denom for name in names}
    out = {name: max(1, int(math.floor(exact[name]))) for name in names if raw[name] > 0}
    while sum(out.values()) > total:
        name = max(out, key=lambda n: out[n])
        out[name] -= 1
    while sum(out.values()) < total:
        name = max(out, key=lambda n: exact[n] - out.get(n, 0))
        out[name] = out.get(name, 0) + 1
    return out


def build_final_train_loader(cfg: dict):
    """Build a single train loader that samples across all three tasks.

    The three task datasets are concatenated and sampled with a plain random
    shuffle, so a batch does not guarantee per-task presence. Task balance is
    expected to come from dataset construction (the pose filter script caps the
    COCO person subset to roughly match detect/attr sizes).
    """
    data_cfg = cfg["data"]
    train_cfg = data_cfg["train"]
    workers = int(cfg["training"].get("workers", 8))
    input_size = int(data_cfg.get("input_size", 640))
    batch_size = int(cfg["training"]["batch_size"])

    detect_aug = cfg.get("augmentation", {}).get("detect", {})
    attr_aug = cfg.get("augmentation", {}).get("attr", {})
    pose_aug = cfg.get("augmentation", {}).get("pose", {})

    detect_ds = DetectDataset(
        _resolve_root(train_cfg["detect"]["root"]),
        train_cfg["detect"].get("images", "train/images"),
        train_cfg["detect"].get("labels", "train/labels"),
        input_size=input_size,
        **detect_aug,
    )
    attr_ds = AttrDataset(
        _resolve_root(train_cfg["attr"]["root"]),
        split=train_cfg["attr"].get("split", "train"),
        input_size=input_size,
        **attr_aug,
    )
    pose_ds = PoseDataset(
        _resolve_root(train_cfg["pose"]["root"]),
        train_cfg["pose"].get("images", "train2017"),
        train_cfg["pose"].get("labels", "labels/train2017"),
        input_size=input_size,
        source_class_format=train_cfg["pose"].get("class_id_format", "yolo80"),
        **pose_aug,
    )

    combined = ConcatDataset([detect_ds, attr_ds, pose_ds])
    loader = make_loader(
        combined, batch_size, workers,
        shuffle=True, weights=None, drop_last=True,
    )
    print(
        "Final train samples: "
        f"combined={len(combined)} detect={len(detect_ds)} "
        f"attr={len(attr_ds)} pose={len(pose_ds)}"
    )
    print(
        "Final train batch: "
        f"size={batch_size} shuffle=True (single random loader, no per-task "
        "composition guarantee)"
    )
    return loader


_NO_AUG = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0, flip_lr=0.0)


def _merge_task_cfg(train_cfg: dict, val_cfg: dict) -> dict:
    merged = dict(train_cfg)
    merged.update(val_cfg or {})
    return merged


def build_final_val_loader(cfg: dict) -> FinalTaskBatchLoader:
    """Build a task-balanced validation loader from data.val sections.

    Only tasks whose validation directories actually exist are included, so the
    local Detect valid/ + Attr val/ runs stand alone and the server COCO val2017
    is picked up automatically when present. No augmentation or sampling is
    applied (full coverage, deterministic order).
    """
    data_cfg = cfg["data"]
    train_cfg = data_cfg["train"]
    val_cfg = data_cfg.get("val", {}) or {}
    batch_cfg = cfg.get("batch_sampling", {})
    input_size = int(data_cfg.get("input_size", 640))
    train_workers = int(cfg["training"].get("workers", 8))
    val_workers = max(1, train_workers // 4)
    total_bs = int(cfg["training"]["batch_size"])

    tasks: dict[str, Dataset] = {}
    ratio_names: list[str] = []

    det_cfg = _merge_task_cfg(train_cfg.get("detect", {}), val_cfg.get("detect", {}))
    det_root = _resolve_root(det_cfg.get("root", ""))
    det_img = det_cfg.get("images", "valid/images")
    det_lbl = det_cfg.get("labels", "valid/labels")
    if (det_root / det_img).exists() and (det_root / det_lbl).exists():
        tasks["detect"] = DetectDataset(
            det_root, det_img, det_lbl,
            input_size=input_size, augment=False, **_NO_AUG,
        )
        ratio_names.append("detect")

    attr_cfg = _merge_task_cfg(train_cfg.get("attr", {}), val_cfg.get("attr", {}))
    if _resolve_root(attr_cfg.get("root", "")).exists():
        tasks["attr"] = AttrDataset(
            _resolve_root(attr_cfg.get("root", "")),
            split=attr_cfg.get("split", "val"),
            input_size=input_size, augment=False, **_NO_AUG,
        )
        ratio_names.append("attr")

    pose_cfg = _merge_task_cfg(train_cfg.get("pose", {}), val_cfg.get("pose", {}))
    pose_root = _resolve_root(pose_cfg.get("root", ""))
    pose_img = pose_cfg.get("images", "val2017")
    pose_lbl = pose_cfg.get("labels", "labels/val2017")
    if (pose_root / pose_img).exists() and (pose_root / pose_lbl).exists():
        tasks["pose"] = PoseDataset(
            pose_root, pose_img, pose_lbl,
            input_size=input_size,
            source_class_format=pose_cfg.get("class_id_format", "yolo80"),
            augment=False, **_NO_AUG,
        )
        ratio_names.append("pose")

    if not tasks:
        raise RuntimeError(
            "No validation data sources available. Add data.val in the config."
        )

    ratios = {
        name: batch_cfg.get("ratios", {}).get(name, 0.0)
        for name in ratio_names
    }
    task_bs = split_batch_size(total_bs, ratios)
    loaders = {
        name: make_loader(
            tasks[name], task_bs[name], val_workers,
            shuffle=False, drop_last=False,
        )
        for name in ratio_names
    }
    print(
        "Final val batch sizes: "
        + ", ".join(f"{name}={bs}" for name, bs in task_bs.items())
    )
    print(
        "Final val samples: "
        + ", ".join(f"{name}={len(tasks[name])}" for name in ratio_names)
    )
    return FinalTaskBatchLoader(loaders)
