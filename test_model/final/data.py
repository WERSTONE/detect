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
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

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
ATTR_MAIN_BY_GROUP = {
    "smoking": "smoking",
    "falling": "falling",
    "waving": "waving",
    "helmet": "helmet_on",
}
KPT_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


@dataclass(frozen=True)
class SampleRef:
    image: Path
    label: Path
    group: str = ""
    dataset: str = ""
    vlm_label: Path | None = None


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _normal_split_name(split: str) -> str:
    split = str(split or "").strip().lower()
    if split in {"valid", "validation"}:
        return "val"
    return split


def _split_policy_mode(policy: dict | None) -> str:
    return str((policy or {}).get("mode", "keep")).strip().lower()


def _stable_seed(seed: int, salt: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _partition_test_refs(
    refs: list[SampleRef],
    train_ratio: float,
    seed: int,
    salt: str,
) -> tuple[list[SampleRef], list[SampleRef]]:
    refs = list(refs)
    rng = random.Random(_stable_seed(seed, salt))
    rng.shuffle(refs)
    n_train = int(round(len(refs) * float(train_ratio)))
    n_train = max(0, min(n_train, len(refs)))
    return refs[:n_train], refs[n_train:]


def _partition_indices(
    n_items: int,
    train_ratio: float,
    seed: int,
    salt: str,
) -> tuple[list[int], list[int]]:
    indices = list(range(int(n_items)))
    rng = random.Random(_stable_seed(seed, salt))
    rng.shuffle(indices)
    n_train = int(round(len(indices) * float(train_ratio)))
    n_train = max(0, min(n_train, len(indices)))
    return indices[:n_train], indices[n_train:]


def _apply_test_split_policy(
    base_refs: list[SampleRef],
    test_refs: list[SampleRef],
    requested_split: str,
    policy: dict | None,
    salt: str,
) -> list[SampleRef]:
    mode = _split_policy_mode(policy)
    requested = _normal_split_name(requested_split)
    if not test_refs or requested not in {"train", "val"}:
        return base_refs
    if mode in {"keep", "keep_with_eval_test", ""}:
        return base_refs
    if mode == "test_to_train":
        return base_refs + test_refs if requested == "train" else base_refs
    if mode == "test_to_val":
        return base_refs + test_refs if requested == "val" else base_refs
    if mode == "fold_test":
        ratio = float(policy.get("test_train_ratio", policy.get("train_ratio", 0.8)))
        ratio = max(0.0, min(1.0, ratio))
        seed = int(policy.get("seed", 0))
        train_refs, val_refs = _partition_test_refs(test_refs, ratio, seed, salt)
        return base_refs + (train_refs if requested == "train" else val_refs)
    raise ValueError(
        "data.split_policy.mode must be one of: keep, keep_with_eval_test, "
        "fold_test, test_to_train, test_to_val"
    )


def _apply_test_dataset_policy(
    base_dataset: Dataset,
    test_dataset: Dataset | None,
    requested_split: str,
    policy: dict | None,
    salt: str,
) -> Dataset:
    mode = _split_policy_mode(policy)
    requested = _normal_split_name(requested_split)
    if test_dataset is None or len(test_dataset) == 0 or requested not in {"train", "val"}:
        return base_dataset
    if mode in {"keep", "keep_with_eval_test", ""}:
        return base_dataset
    if mode == "test_to_train":
        return ConcatDataset([base_dataset, test_dataset]) if requested == "train" else base_dataset
    if mode == "test_to_val":
        return ConcatDataset([base_dataset, test_dataset]) if requested == "val" else base_dataset
    if mode == "fold_test":
        ratio = float(policy.get("test_train_ratio", policy.get("train_ratio", 0.8)))
        ratio = max(0.0, min(1.0, ratio))
        seed = int(policy.get("seed", 0))
        train_indices, val_indices = _partition_indices(len(test_dataset), ratio, seed, salt)
        indices = train_indices if requested == "train" else val_indices
        if not indices:
            return base_dataset
        return ConcatDataset([base_dataset, Subset(test_dataset, indices)])
    raise ValueError(
        "data.split_policy.mode must be one of: keep, keep_with_eval_test, "
        "fold_test, test_to_train, test_to_val"
    )


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
        degrees: float = 0.0,
        translate: float = 0.05,
        scale: float = 0.1,
        flip_lr: float = 0.5,
        require_full_boxes: bool = False,
        affine_retries: int = 8,
        zoom_out_prob: float = 0.0,
        zoom_out_min_scale: float = 1.0,
        zoom_out_max_scale: float = 1.0,
        mosaic_prob: float = 0.0,
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
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.flip_lr = float(flip_lr)
        self.require_full_boxes = bool(require_full_boxes)
        self.affine_retries = max(int(affine_retries), 1)
        self.zoom_out_prob = max(0.0, min(1.0, float(zoom_out_prob)))
        self.zoom_out_min_scale = max(0.05, min(1.0, float(zoom_out_min_scale)))
        self.zoom_out_max_scale = max(self.zoom_out_min_scale, min(1.0, float(zoom_out_max_scale)))
        self.mosaic_prob = max(0.0, min(1.0, float(mosaic_prob)))
        self._mosaic_enabled = True
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

    def set_close_mosaic(self, close: bool = False):
        self._mosaic_enabled = not bool(close)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _letterbox(self, image, boxes, kpts, size=None):
        size = int(size or self.input_size)
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = size - new_w
        pad_h = size - new_h
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

    def _augment_basic(self, image, boxes, kpts, allow_attr_sensitive=True, already_letterboxed=False):
        if already_letterboxed:
            scale, pad = 1.0, (0, 0)
        else:
            image, boxes, kpts, scale, pad = self._letterbox(image, boxes, kpts)
        image, boxes, kpts = self._augment_post_letterbox(
            image, boxes, kpts, allow_attr_sensitive=allow_attr_sensitive
        )
        return image, boxes, kpts, scale, pad

    def _augment_post_letterbox(self, image, boxes, kpts, allow_attr_sensitive=True):
        if not self.augment:
            return image, boxes, kpts

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

        if self.zoom_out_prob > 0 and random.random() < self.zoom_out_prob:
            image, boxes, kpts = self._zoom_out(image, boxes, kpts)

        if allow_attr_sensitive and (
            self.degrees > 0 or self.translate > 0 or self.scale > 0
        ):
            image, boxes, kpts = self._safe_affine(image, boxes, kpts)
        return image, boxes, kpts

    def _safe_affine(self, image, boxes, kpts):
        s = self.input_size
        original = (image, boxes.copy(), kpts.copy())
        for _ in range(self.affine_retries):
            scale_factor = random.uniform(max(0.05, 1.0 - self.scale), 1.0 + self.scale)
            angle = random.uniform(-self.degrees, self.degrees)
            tx = random.uniform(-self.translate, self.translate) * s
            ty = random.uniform(-self.translate, self.translate) * s
            matrix = cv2.getRotationMatrix2D((s / 2, s / 2), angle, scale_factor)
            matrix[0, 2] += tx
            matrix[1, 2] += ty

            warped_boxes = boxes.copy()
            if len(boxes):
                corners = np.ones((len(boxes) * 4, 3), dtype=np.float32)
                corners[:, :2] = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
                warped = (corners @ matrix.T).reshape(len(boxes), 8)
                xs = warped[:, [0, 2, 4, 6]]
                ys = warped[:, [1, 3, 5, 7]]
                raw = np.stack([xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=1)
                if self.require_full_boxes and (
                    np.any(raw[:, 0] < 0) or np.any(raw[:, 1] < 0)
                    or np.any(raw[:, 2] > s - 1) or np.any(raw[:, 3] > s - 1)
                ):
                    continue
                warped_boxes[:, 0] = np.clip(raw[:, 0], 0, s - 1)
                warped_boxes[:, 1] = np.clip(raw[:, 1], 0, s - 1)
                warped_boxes[:, 2] = np.clip(raw[:, 2], 0, s - 1)
                warped_boxes[:, 3] = np.clip(raw[:, 3], 0, s - 1)

            warped_kpts = kpts.copy()
            if len(kpts):
                pts = np.ones((len(kpts) * 17, 3), dtype=np.float32)
                pts[:, :2] = kpts[..., :2].reshape(-1, 2)
                warped_pts = (pts @ matrix.T).reshape(len(kpts), 17, 2)
                warped_kpts[..., :2] = warped_pts
                outside = (
                    (warped_kpts[..., 0] < 0) | (warped_kpts[..., 0] > s - 1)
                    | (warped_kpts[..., 1] < 0) | (warped_kpts[..., 1] > s - 1)
                )
                warped_kpts[..., 2] = np.where(outside, 0.0, warped_kpts[..., 2])
                warped_kpts[..., 0] = np.clip(warped_kpts[..., 0], 0, s - 1)
                warped_kpts[..., 1] = np.clip(warped_kpts[..., 1], 0, s - 1)

            warped_image = cv2.warpAffine(
                image, matrix, (s, s), borderValue=(114, 114, 114)
            )
            return warped_image, warped_boxes, warped_kpts
        return original

    def _zoom_out(self, image, boxes, kpts):
        s = self.input_size
        factor = random.uniform(self.zoom_out_min_scale, self.zoom_out_max_scale)
        resized = cv2.resize(
            image, (max(1, round(s * factor)), max(1, round(s * factor))),
            interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR,
        )
        h, w = resized.shape[:2]
        max_x, max_y = s - w, s - h
        left = random.randint(0, max(max_x, 0))
        top = random.randint(0, max(max_y, 0))
        canvas = np.full_like(image, 114)
        canvas[top:top + h, left:left + w] = resized
        boxes = boxes.copy()
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * factor + left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * factor + top
        kpts = kpts.copy()
        if len(kpts):
            kpts[..., 0] = kpts[..., 0] * factor + left
            kpts[..., 1] = kpts[..., 1] * factor + top
        return canvas, boxes, kpts

    def _hsv(self, image):
        if max(abs(self.hsv_h), abs(self.hsv_s), abs(self.hsv_v)) <= 0:
            return image
        gains = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        x = np.arange(0, 256, dtype=gains.dtype)
        lut_hue = np.asarray((x * gains[0]) % 180, dtype=np.uint8)
        lut_sat = np.asarray(np.clip(x * gains[1], 0, 255), dtype=np.uint8)
        lut_val = np.asarray(np.clip(x * gains[2], 0, 255), dtype=np.uint8)
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

    def _load_detect_arrays(self, idx):
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
        return ref, image, boxes, classes, kpts, attrs, attr_mask

    def _mosaic_sample_indices(self, idx) -> list[int]:
        return [idx] + random.choices(range(len(self.samples)), k=3)

    def _load_mosaic(self, idx):
        size = self.input_size
        tile = size // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        all_boxes, all_classes = [], []
        offsets = ((0, 0), (tile, 0), (0, tile), (tile, tile))
        indices = self._mosaic_sample_indices(idx)
        ref = self.samples[idx]
        for sample_idx, (off_x, off_y) in zip(indices, offsets):
            _, image, boxes, classes, kpts, _, _ = self._load_detect_arrays(sample_idx)
            image, boxes, _kpts, _scale, _pad = self._letterbox(image, boxes, kpts, size=tile)
            canvas[off_y:off_y + tile, off_x:off_x + tile] = image
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] += off_x
                boxes[:, [1, 3]] += off_y
                all_boxes.append(boxes)
                all_classes.append(classes)
        boxes = (
            np.concatenate(all_boxes, axis=0).astype(np.float32)
            if all_boxes else np.zeros((0, 4), dtype=np.float32)
        )
        classes = (
            np.concatenate(all_classes, axis=0).astype(np.int64)
            if all_classes else np.zeros((0,), dtype=np.int64)
        )
        kpts = np.zeros((len(boxes), 17, 3), dtype=np.float32)
        attrs = np.zeros((len(boxes), 4), dtype=np.float32)
        attr_mask = np.zeros((len(boxes), 4), dtype=np.float32)
        return ref, canvas, boxes, classes, kpts, attrs, attr_mask

    def __getitem__(self, idx):
        use_mosaic = (
            self.augment
            and self._mosaic_enabled
            and self.mosaic_prob > 0
            and len(self.samples) >= 4
            and random.random() < self.mosaic_prob
        )
        if use_mosaic:
            ref, image, boxes, classes, kpts, attrs, attr_mask = self._load_mosaic(idx)
            image, boxes, kpts, scale, pad = self._augment_basic(
                image, boxes, kpts, allow_attr_sensitive=True, already_letterboxed=True
            )
        else:
            ref, image, boxes, classes, kpts, attrs, attr_mask = self._load_detect_arrays(idx)
            image, boxes, kpts, scale, pad = self._augment_basic(
                image, boxes, kpts, allow_attr_sensitive=True
            )
        return self._to_sample(image, boxes, classes, kpts, attrs, attr_mask, ref, scale, pad)


class AttrDataset(FinalBaseDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        vlm_label_dirname: str | None = None,
        mix_vlm_non_main_attrs: bool = False,
        vlm_missing_policy: str = "error",
        split_policy: dict | None = None,
        **kwargs,
    ):
        self.split = split
        root = Path(root)
        self.vlm_label_dirname = str(vlm_label_dirname or "").strip()
        self.mix_vlm_non_main_attrs = bool(mix_vlm_non_main_attrs)
        self.vlm_missing_policy = str(vlm_missing_policy or "error").strip().lower()
        self.split_policy = dict(split_policy or {})
        if self.vlm_missing_policy not in {"error", "zero_mask", "original"}:
            raise ValueError(
                "vlm_missing_policy must be one of: error, zero_mask, original"
            )
        refs: list[SampleRef] = []
        vlm_missing = 0
        vlm_expected = 0

        def collect_dataset_split(ds_dir: Path, group_name: str, split_name: str) -> list[SampleRef]:
            out: list[SampleRef] = []
            label_dir = ds_dir / "labels" / split_name
            image_dir = ds_dir / "images" / split_name
            if not label_dir.exists() or not image_dir.exists():
                return out
            for label in sorted(label_dir.glob("*.txt")):
                image = find_image(image_dir, label.stem)
                if image is None:
                    continue
                vlm_label = None
                if self.vlm_label_dirname:
                    vlm_label = ds_dir / self.vlm_label_dirname / split_name / label.name
                out.append(
                    SampleRef(
                        image=image,
                        label=label,
                        group=group_name,
                        dataset=ds_dir.name,
                        vlm_label=vlm_label,
                    )
                )
            return out

        for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for ds_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                base_refs = collect_dataset_split(ds_dir, group_dir.name, split)
                test_refs = collect_dataset_split(ds_dir, group_dir.name, "test")
                refs.extend(
                    _apply_test_split_policy(
                        base_refs,
                        test_refs,
                        split,
                        self.split_policy,
                        salt=f"attr/{group_dir.name}/{ds_dir.name}",
                    )
                )
        if self.vlm_label_dirname:
            for ref in refs:
                if ref.vlm_label is not None:
                    vlm_expected += 1
                    if not ref.vlm_label.exists():
                        vlm_missing += 1
        if self.mix_vlm_non_main_attrs and not self.vlm_label_dirname:
            raise ValueError("mix_vlm_non_main_attrs requires vlm_label_dirname")
        if self.mix_vlm_non_main_attrs and self.vlm_missing_policy == "error" and vlm_missing:
            raise FileNotFoundError(
                f"Missing {vlm_missing}/{vlm_expected} VLM label files for Attr split={split} "
                f"under dirname={self.vlm_label_dirname}"
            )
        self._vlm_missing = vlm_missing
        self._vlm_expected = vlm_expected
        self._precollected_refs = refs
        super().__init__(root=root, image_dir=".", label_dir=".", task="attr", **kwargs)
        if self.mix_vlm_non_main_attrs:
            print(
                "AttrDataset VLM mix: "
                f"split={split} vlm_dir={self.vlm_label_dirname} "
                f"missing={self._vlm_missing}/{self._vlm_expected} "
                f"policy={self.vlm_missing_policy} split_policy={_split_policy_mode(self.split_policy)}"
            )

    def _collect_samples(self) -> list[SampleRef]:
        return list(getattr(self, "_precollected_refs", []))

    def _main_attr_idx(self, ref: SampleRef) -> int | None:
        name = ATTR_MAIN_BY_GROUP.get(ref.group)
        if name is None:
            return None
        return ATTR_NAMES.index(name)

    def _read_attr_rows(self, label_path: Path) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) == 13:
                rows.append(parts)
        return rows

    def _handle_missing_vlm_rows(
        self,
        ref: SampleRef,
        original_rows: list[list[str]],
        reason: str,
    ) -> list[list[str]]:
        if self.vlm_missing_policy == "error":
            raise FileNotFoundError(
                f"{reason}: original={ref.label} vlm={ref.vlm_label}"
            )
        if self.vlm_missing_policy == "original":
            return original_rows
        main_idx = self._main_attr_idx(ref)
        merged_rows: list[list[str]] = []
        for row in original_rows:
            merged = list(row)
            for attr_idx in range(len(ATTR_NAMES)):
                if attr_idx != main_idx:
                    merged[5 + attr_idx] = "0"
                    merged[9 + attr_idx] = "0"
            merged_rows.append(merged)
        return merged_rows

    def _mixed_attr_rows(self, ref: SampleRef) -> list[list[str]]:
        original_rows = self._read_attr_rows(ref.label)
        if not self.mix_vlm_non_main_attrs:
            return original_rows
        main_idx = self._main_attr_idx(ref)
        if main_idx is None:
            return original_rows
        if ref.vlm_label is None or not ref.vlm_label.exists():
            return self._handle_missing_vlm_rows(ref, original_rows, "Missing VLM label")
        vlm_rows = self._read_attr_rows(ref.vlm_label)
        if len(vlm_rows) != len(original_rows):
            return self._handle_missing_vlm_rows(
                ref,
                original_rows,
                f"VLM row count mismatch original={len(original_rows)} vlm={len(vlm_rows)}",
            )
        mixed_rows: list[list[str]] = []
        for original, vlm in zip(original_rows, vlm_rows):
            merged = list(original)
            for attr_idx in range(len(ATTR_NAMES)):
                if attr_idx == main_idx:
                    continue
                merged[5 + attr_idx] = vlm[5 + attr_idx]
                merged[9 + attr_idx] = vlm[9 + attr_idx]
            mixed_rows.append(merged)
        return mixed_rows

    def sampler_weights(self, target_attrs: tuple[str, ...] = ATTR_NAMES) -> list[float]:
        attr_to_idx = {name: idx for idx, name in enumerate(ATTR_NAMES)}
        counts = {name: {"pos": 0, "neg": 0} for name in target_attrs}
        sample_keys: list[list[tuple[str, str]]] = []
        for ref in self.samples:
            keys: list[tuple[str, str]] = []
            for parts in self._mixed_attr_rows(ref):
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
        for parts in self._mixed_attr_rows(ref):
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


class FinalSequentialValLoader:
    """Yield each task validation loader once without resampling or balancing."""

    def __init__(self, loaders: dict[str, DataLoader]):
        self.loaders = loaders
        self.dataset = self
        self.samples = []

    def __len__(self):
        return sum(len(loader) for loader in self.loaders.values())

    def __iter__(self):
        for name in ("detect", "attr", "pose"):
            loader = self.loaders.get(name)
            if loader is None:
                continue
            yield from loader

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
    split_policy = data_cfg.get("split_policy", {}) or {}
    workers = int(cfg["training"].get("workers", 8))
    input_size = int(data_cfg.get("input_size", 640))
    batch_size = int(cfg["training"]["batch_size"])

    detect_aug = cfg.get("augmentation", {}).get("detect", {})
    attr_aug = cfg.get("augmentation", {}).get("attr", {})
    pose_aug = cfg.get("augmentation", {}).get("pose", {})

    det_root = _resolve_root(train_cfg["detect"]["root"])
    detect_ds = DetectDataset(
        det_root,
        train_cfg["detect"].get("images", "train/images"),
        train_cfg["detect"].get("labels", "train/labels"),
        input_size=input_size,
        **detect_aug,
    )
    detect_test_ds = _make_optional_dataset(
        DetectDataset,
        det_root,
        train_cfg["detect"].get("test_images", "test/images"),
        train_cfg["detect"].get("test_labels", "test/labels"),
        input_size=input_size,
        **detect_aug,
    )
    detect_ds = _apply_test_dataset_policy(
        detect_ds,
        detect_test_ds,
        "train",
        split_policy,
        salt="detect",
    )
    attr_ds = AttrDataset(
        _resolve_root(train_cfg["attr"]["root"]),
        split=train_cfg["attr"].get("split", "train"),
        input_size=input_size,
        **_attr_dataset_options(train_cfg["attr"], split_policy),
        **attr_aug,
    )
    pose_root = _resolve_root(train_cfg["pose"]["root"])
    pose_ds = PoseDataset(
        pose_root,
        train_cfg["pose"].get("images", "train2017"),
        train_cfg["pose"].get("labels", "labels/train2017"),
        input_size=input_size,
        source_class_format=train_cfg["pose"].get("class_id_format", "yolo80"),
        **pose_aug,
    )
    pose_test_ds = _make_optional_dataset(
        PoseDataset,
        pose_root,
        train_cfg["pose"].get("test_images", "test2017"),
        train_cfg["pose"].get("test_labels", "labels_person/test2017"),
        input_size=input_size,
        source_class_format=train_cfg["pose"].get("class_id_format", "yolo80"),
        **pose_aug,
    )
    pose_ds = _apply_test_dataset_policy(
        pose_ds,
        pose_test_ds,
        "train",
        split_policy,
        salt="pose",
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


_NO_AUG = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    flip_lr=0.0,
    require_full_boxes=False,
    affine_retries=1,
    zoom_out_prob=0.0,
    zoom_out_min_scale=1.0,
    zoom_out_max_scale=1.0,
    mosaic_prob=0.0,
)


def _merge_task_cfg(train_cfg: dict, val_cfg: dict) -> dict:
    merged = dict(train_cfg)
    merged.update(val_cfg or {})
    return merged


def _attr_dataset_options(attr_cfg: dict, split_policy: dict | None = None) -> dict:
    keys = ("vlm_label_dirname", "mix_vlm_non_main_attrs", "vlm_missing_policy", "split_policy")
    options = {key: attr_cfg[key] for key in keys if key in attr_cfg}
    if split_policy is not None and "split_policy" not in options:
        options["split_policy"] = split_policy
    return options


def _make_optional_dataset(dataset_cls, root: Path, image_dir: str, label_dir: str, **kwargs):
    if not (root / image_dir).exists() or not (root / label_dir).exists():
        return None
    try:
        return dataset_cls(root, image_dir, label_dir, **kwargs)
    except RuntimeError:
        return None
    return {key: attr_cfg[key] for key in keys if key in attr_cfg}


def build_final_val_loader(cfg: dict) -> FinalSequentialValLoader:
    """Build validation loaders from data.val sections.

    Only tasks whose validation directories actually exist are included, so the
    local Detect valid/ + Attr val/ runs stand alone and the server COCO val2017
    is picked up automatically when present. No augmentation, sampling, or task
    balancing is applied, so each validation sample is evaluated exactly once.
    """
    data_cfg = cfg["data"]
    train_cfg = data_cfg["train"]
    val_cfg = data_cfg.get("val", {}) or {}
    split_policy = data_cfg.get("split_policy", {}) or {}
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
        det_base = DetectDataset(
            det_root, det_img, det_lbl,
            input_size=input_size, augment=False, **_NO_AUG,
        )
        det_test = _make_optional_dataset(
            DetectDataset,
            det_root,
            det_cfg.get("test_images", "test/images"),
            det_cfg.get("test_labels", "test/labels"),
            input_size=input_size,
            augment=False,
            **_NO_AUG,
        )
        tasks["detect"] = _apply_test_dataset_policy(
            det_base,
            det_test,
            "val",
            split_policy,
            salt="detect",
        )
        ratio_names.append("detect")

    attr_cfg = _merge_task_cfg(train_cfg.get("attr", {}), val_cfg.get("attr", {}))
    if _resolve_root(attr_cfg.get("root", "")).exists():
        tasks["attr"] = AttrDataset(
            _resolve_root(attr_cfg.get("root", "")),
            split=attr_cfg.get("split", "val"),
            input_size=input_size, augment=False, **_NO_AUG,
            **_attr_dataset_options(attr_cfg, split_policy),
        )
        ratio_names.append("attr")

    pose_cfg = _merge_task_cfg(train_cfg.get("pose", {}), val_cfg.get("pose", {}))
    pose_root = _resolve_root(pose_cfg.get("root", ""))
    pose_img = pose_cfg.get("images", "val2017")
    pose_lbl = pose_cfg.get("labels", "labels/val2017")
    if (pose_root / pose_img).exists() and (pose_root / pose_lbl).exists():
        pose_base = PoseDataset(
            pose_root, pose_img, pose_lbl,
            input_size=input_size,
            source_class_format=pose_cfg.get("class_id_format", "yolo80"),
            augment=False, **_NO_AUG,
        )
        pose_test = _make_optional_dataset(
            PoseDataset,
            pose_root,
            pose_cfg.get("test_images", "test2017"),
            pose_cfg.get("test_labels", "labels_person/test2017"),
            input_size=input_size,
            source_class_format=pose_cfg.get("class_id_format", "yolo80"),
            augment=False,
            **_NO_AUG,
        )
        tasks["pose"] = _apply_test_dataset_policy(
            pose_base,
            pose_test,
            "val",
            split_policy,
            salt="pose",
        )
        ratio_names.append("pose")

    if not tasks:
        raise RuntimeError(
            "No validation data sources available. Add data.val in the config."
        )

    loaders = {
        name: make_loader(
            tasks[name], total_bs, val_workers,
            shuffle=False, drop_last=False,
        )
        for name in ratio_names
    }
    print(
        "Final val batch size: "
        + ", ".join(f"{name}={total_bs}" for name in ratio_names)
    )
    print(
        "Final val samples: "
        + ", ".join(f"{name}={len(tasks[name])}" for name in ratio_names)
    )
    print(
        "Final val steps: "
        + ", ".join(f"{name}={len(loaders[name])}" for name in ratio_names)
        + f" total={sum(len(loaders[name]) for name in ratio_names)}"
    )
    return FinalSequentialValLoader(loaders)
