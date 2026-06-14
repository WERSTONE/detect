"""统一数据集: 0=person, 1=fire, 2=water. 内置数据增强."""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TRAIN_SIZE = 640

# COCO 17 点水平翻转交换对
KPT_FLIP_PAIRS = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                   (11, 12), (13, 14), (15, 16)]


@dataclass
class VigilSample:
    image: torch.Tensor              # [3, 640, 640]
    person_boxes: torch.Tensor       # [N, 4] xyxy
    person_kpts: torch.Tensor        # [N, 17, 3]
    person_helmet: torch.Tensor      # [N] 0=on, 1=off
    person_smoke: torch.Tensor       # [N] 0=no, 1=yes
    detect_boxes: torch.Tensor       # [M, 4]
    detect_classes: torch.Tensor     # [M] 1=fire, 2=water


# ── 增强 ──

def _random_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-1, 1) * hgain * 180) % 360
    hsv[..., 1] *= 1 + np.random.uniform(-1, 1) * sgain
    hsv[..., 2] *= 1 + np.random.uniform(-1, 1) * vgain
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _letterbox(img, size=640, fill=114):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(h * r), int(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = size - nh, size - nw
    pt, pl = dh // 2, dw // 2
    img = cv2.copyMakeBorder(
        img, pt, dh - pt, pl, dw - pl, cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return img, r, (pl, pt)


def _tx_boxes(boxes, scale, pad_l, pad_t):
    if boxes is None or len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]] * scale + pad_l
    b[:, [1, 3]] = b[:, [1, 3]] * scale + pad_t
    return b


def _tx_kpts(kpts, scale, pad_l, pad_t):
    if kpts is None or len(kpts) == 0:
        return kpts
    k = kpts.copy()
    k[:, :, 0] = k[:, :, 0] * scale + pad_l
    k[:, :, 1] = k[:, :, 1] * scale + pad_t
    return k


def _flip_boxes(boxes, w):
    if boxes is None or len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = w - b[:, [2, 0]]
    return b


def _flip_kpts(kpts, w):
    if kpts is None or len(kpts) == 0:
        return kpts
    k = kpts.copy()
    k[:, :, 0] = w - k[:, :, 0]
    for l_idx, r_idx in KPT_FLIP_PAIRS:
        if l_idx != r_idx and l_idx < k.shape[1] and r_idx < k.shape[1]:
            k[:, l_idx], k[:, r_idx] = k[:, r_idx].copy(), k[:, l_idx].copy()
    return k


def _normalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1))


# ── Mosaic 增强 ──

def _mosaic4(samples, size=640):
    """将 4 张图拼成 1 张, 合并标签."""
    s = size // 2
    cx = int(np.random.uniform(s // 2, s + s // 2))
    cy = int(np.random.uniform(s // 2, s + s // 2))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    all_pb, all_pk, all_ph, all_ps = [], [], [], []
    all_db, all_dc = [], []

    for i, sample in enumerate(samples):
        img_h, img_w = sample["img"].shape[:2]
        r = min(s / img_w, s / img_h)
        nh, nw = int(img_h * r), int(img_w * r)
        img = cv2.resize(sample["img"], (nw, nh))

        # ideal position of resized image on canvas (may be negative)
        if i == 0:      px, py = cx - nw, cy - nh
        elif i == 1:    px, py = cx,      cy - nh
        elif i == 2:    px, py = cx - nw, cy
        else:           px, py = cx,      cy

        # canvas crop region (visible portion, clamped to [0, size])
        x1 = max(0, px);       y1 = max(0, py)
        x2 = min(size, px + nw); y2 = min(size, py + nh)
        # source crop region (corresponding portion of resized image)
        sx1 = x1 - px;         sy1 = y1 - py
        sx2 = x2 - px;         sy2 = y2 - py

        cw, ch = x2 - x1, y2 - y1
        if cw > 0 and ch > 0:
            canvas[y1:y2, x1:x2] = img[sy1:sy2, sx1:sx2]

        def _mx_boxes(b, sc, ox, oy):
            if b is None or len(b) == 0: return b, np.array([], dtype=bool)
            b = b.copy()
            b[:, [0, 2]] = b[:, [0, 2]] * sc + ox
            b[:, [1, 3]] = b[:, [1, 3]] * sc + oy
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, size)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, size)
            keep = (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2)
            return b[keep], keep

        def _mx_kpts(k, sc, ox, oy):
            if k is None or len(k) == 0: return k
            k = k.copy()
            k[:, :, 0] = k[:, :, 0] * sc + ox
            k[:, :, 1] = k[:, :, 1] * sc + oy
            return k

        # label offset = ideal image position (may be negative)
        pb, pb_keep = _mx_boxes(sample.get("person_boxes"), r, px, py)
        db, db_keep = _mx_boxes(sample.get("detect_boxes"), r, px, py)
        pk = _mx_kpts(sample.get("person_kpts"), r, px, py)

        if pb is not None and len(pb) > 0:
            all_pb.append(pb)
            all_pk.append(pk[pb_keep] if len(pb_keep) == len(pk) else pk)
            all_ph.append(sample.get("person_helmet", np.array([]))[pb_keep] if len(pb_keep) == len(sample.get("person_helmet", np.array([]))) else sample.get("person_helmet", np.array([])))
            all_ps.append(sample.get("person_smoke", np.array([]))[pb_keep] if len(pb_keep) == len(sample.get("person_smoke", np.array([]))) else sample.get("person_smoke", np.array([])))
        if db is not None and len(db) > 0:
            all_db.append(db)
            dc = sample.get("detect_classes", np.array([]))
            all_dc.append(dc[db_keep] if len(db_keep) == len(dc) else dc)

    return canvas, all_pb, all_pk, all_ph, all_ps, all_db, all_dc


# ── 解析 ──

def _parse_label(lbl_path, img_w, img_h):
    p_boxes, p_kpts, p_helm, p_smoke = [], [], [], []
    d_boxes, d_cls = [], []

    if not lbl_path.exists():
        return (
            torch.empty(0, 4), torch.empty(0, 17, 3), torch.empty(0),
            torch.empty(0), torch.empty(0, 4), torch.empty(0, dtype=torch.long),
        )

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            vals = list(map(float, parts[1:]))
            cx, cy, w_n, h_n = vals[0], vals[1], vals[2], vals[3]
            x1 = (cx - w_n / 2) * img_w
            y1 = (cy - h_n / 2) * img_h
            x2 = (cx + w_n / 2) * img_w
            y2 = (cy + h_n / 2) * img_h

            if cls_id == 0 and len(vals) >= 57:
                p_boxes.append([x1, y1, x2, y2])
                kpt = np.array(vals[4:55], dtype=np.float32).reshape(17, 3)
                kpt[:, 0] *= img_w
                kpt[:, 1] *= img_h
                p_kpts.append(kpt)
                p_helm.append(int(vals[55]))
                p_smoke.append(int(vals[56]))
            elif cls_id in (1, 2):
                d_boxes.append([x1, y1, x2, y2])
                d_cls.append(cls_id)

    return (
        torch.tensor(p_boxes, dtype=torch.float32) if p_boxes else torch.empty(0, 4),
        torch.tensor(np.stack(p_kpts), dtype=torch.float32) if p_kpts else torch.empty(0, 17, 3),
        torch.tensor(p_helm, dtype=torch.float32) if p_helm else torch.empty(0),
        torch.tensor(p_smoke, dtype=torch.float32) if p_smoke else torch.empty(0),
        torch.tensor(d_boxes, dtype=torch.float32) if d_boxes else torch.empty(0, 4),
        torch.tensor(d_cls, dtype=torch.long) if d_cls else torch.empty(0, dtype=torch.long),
    )


# ── 数据集 ──

class UnifiedDataset(Dataset):

    def __init__(self, root, dataset_name, augment=True, mosaic_pool=None):
        self.root = root
        self.name = dataset_name
        self.mosaic_pool = mosaic_pool  # 跨数据集混合的全局样本池
        # augment: bool or dict (config)
        if isinstance(augment, dict):
            self.aug_cfg = augment
            self.augment = True
        else:
            self.aug_cfg = {}
            self.augment = augment
        img_dir = Path(root) / "images"
        lbl_dir = Path(root) / "labels"
        self.samples = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            self.samples.append((str(img_path), lbl_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Mosaic — 跨数据集混合: 首张图取自当前数据集, 其余 3 张从全局池随机选取
        mosaic_prob = self.aug_cfg.get("mosaic", {}).get("prob", 0.5) if self.aug_cfg.get("mosaic", {}).get("enabled", True) else 0.0
        if self.augment and np.random.random() < mosaic_prob:
            pool = self.mosaic_pool if self.mosaic_pool else self.samples
            pool_size = len(pool)
            indices = [idx] + [np.random.randint(0, pool_size) for _ in range(3)]
            samples = []
            for j, sample_idx in enumerate(indices):
                if j == 0:
                    img_path_j, lbl_path_j = self.samples[sample_idx]
                else:
                    img_path_j, lbl_path_j = pool[sample_idx]
                img_j = cv2.imread(img_path_j)
                if img_j is None:
                    continue
                hj, wj = img_j.shape[:2]
                pj_b, pj_k, pj_h, pj_s, dj_b, dj_c = _parse_label(lbl_path_j, wj, hj)
                samples.append({
                    "img": img_j,
                    "person_boxes": pj_b.numpy() if pj_b.numel() > 0 else np.empty((0, 4)),
                    "person_kpts": pj_k.numpy() if pj_k.numel() > 0 else np.empty((0, 17, 3)),
                    "person_helmet": pj_h.numpy() if pj_h.numel() > 0 else np.empty(0),
                    "person_smoke": pj_s.numpy() if pj_s.numel() > 0 else np.empty(0),
                    "detect_boxes": dj_b.numpy() if dj_b.numel() > 0 else np.empty((0, 4)),
                    "detect_classes": dj_c.numpy() if dj_c.numel() > 0 else np.empty(0, dtype=np.int64),
                })

            if len(samples) == 4:
                img, all_pb, all_pk, all_ph, all_ps, all_db, all_dc = _mosaic4(samples, TRAIN_SIZE)

                # 合并标签
                p_boxes_np = np.concatenate(all_pb) if all_pb else np.empty((0, 4))
                p_kpts_np = np.concatenate(all_pk) if all_pk else np.empty((0, 17, 3))
                p_helm_np = np.concatenate(all_ph) if all_ph else np.empty(0)
                p_smoke_np = np.concatenate(all_ps) if all_ps else np.empty(0)
                d_boxes_np = np.concatenate(all_db) if all_db else np.empty((0, 4))
                d_cls_np = np.concatenate(all_dc) if all_dc else np.empty(0, dtype=np.int64)

                # HSV + flip after mosaic
                hsv_cfg = self.aug_cfg.get("hsv", {})
                if hsv_cfg.get("enabled", True):
                    img = _random_hsv(img, hsv_cfg.get("hgain", 0.015), hsv_cfg.get("sgain", 0.7), hsv_cfg.get("vgain", 0.4))
                flip_prob = self.aug_cfg.get("flip", {}).get("prob", 0.5) if self.aug_cfg.get("flip", {}).get("enabled", True) else 0.0
                if np.random.random() < flip_prob:
                    img = np.ascontiguousarray(img[:, ::-1])
                    nw = img.shape[1]
                    p_boxes_np = _flip_boxes(p_boxes_np, nw)
                    p_kpts_np = _flip_kpts(p_kpts_np, nw)
                    d_boxes_np = _flip_boxes(d_boxes_np, nw)

                img_t = _normalize(img)
                return VigilSample(
                    image=img_t,
                    person_boxes=torch.from_numpy(p_boxes_np).float(),
                    person_kpts=torch.from_numpy(p_kpts_np).float(),
                    person_helmet=torch.from_numpy(p_helm_np).float(),
                    person_smoke=torch.from_numpy(p_smoke_np).float(),
                    detect_boxes=torch.from_numpy(d_boxes_np).float(),
                    detect_classes=torch.from_numpy(d_cls_np).long(),
                )

        # ── 普通数据加载 ──
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        h, w = img.shape[:2]
        p_boxes, p_kpts, p_helm, p_smoke, d_boxes, d_cls = _parse_label(lbl_path, w, h)

        img, scale, (pl, pt) = _letterbox(img, TRAIN_SIZE)

        p_boxes_np = p_boxes.numpy() if p_boxes.numel() > 0 else np.empty((0, 4))
        p_kpts_np = p_kpts.numpy() if p_kpts.numel() > 0 else np.empty((0, 17, 3))
        d_boxes_np = d_boxes.numpy() if d_boxes.numel() > 0 else np.empty((0, 4))

        p_boxes_np = _tx_boxes(p_boxes_np, scale, pl, pt)
        p_kpts_np = _tx_kpts(p_kpts_np, scale, pl, pt)
        d_boxes_np = _tx_boxes(d_boxes_np, scale, pl, pt)

        if self.augment:
            hsv_cfg = self.aug_cfg.get("hsv", {})
            if hsv_cfg.get("enabled", True):
                img = _random_hsv(img, hsv_cfg.get("hgain", 0.015), hsv_cfg.get("sgain", 0.7), hsv_cfg.get("vgain", 0.4))
            flip_prob = self.aug_cfg.get("flip", {}).get("prob", 0.5) if self.aug_cfg.get("flip", {}).get("enabled", True) else 0.0
            if np.random.random() < flip_prob:
                img = np.ascontiguousarray(img[:, ::-1])
                nw = img.shape[1]
                p_boxes_np = _flip_boxes(p_boxes_np, nw)
                p_kpts_np = _flip_kpts(p_kpts_np, nw)
                d_boxes_np = _flip_boxes(d_boxes_np, nw)

        img_t = _normalize(img)

        return VigilSample(
            image=img_t,
            person_boxes=torch.from_numpy(p_boxes_np).float() if len(p_boxes_np) > 0 else torch.empty(0, 4),
            person_kpts=torch.from_numpy(p_kpts_np).float() if len(p_kpts_np) > 0 else torch.empty(0, 17, 3),
            person_helmet=p_helm,
            person_smoke=p_smoke,
            detect_boxes=torch.from_numpy(d_boxes_np).float() if len(d_boxes_np) > 0 else torch.empty(0, 4),
            detect_classes=d_cls,
        )


def collate_fn(batch):
    return batch


def make_dataloaders(dataset_specs, batch_size=1, augment=True, val_ratio=0.0,
                     num_workers=0, pin_memory=False, persistent_workers=False,
                     verbose=True):
    train_loaders = {}
    val_loaders = {}

    # —— 预建所有 train UnifiedDataset, 收集样本构建全局 Mosaic 池 ——
    train_datasets = {}
    for name, spec in dataset_specs.items():
        path = spec["path"]
        if not os.path.exists(path):
            if verbose: print(f"  [skip] {name}: {path} not found")
            continue
        train_datasets[name] = UnifiedDataset(path, name, augment=augment)

    # 跨数据集 Mosaic 全局样本池
    mosaic_pool = []
    for ds in train_datasets.values():
        mosaic_pool.extend(ds.samples)

    for name, ds in train_datasets.items():
        ds.mosaic_pool = mosaic_pool

        if val_ratio > 0:
            val_ds = UnifiedDataset(ds.root, name, augment=False)
            n_val = max(1, int(len(ds) * val_ratio))
            n_train = len(ds) - n_val
            indices = list(range(len(ds)))
            import random
            random.shuffle(indices)
            train_sub = torch.utils.data.Subset(ds, indices[:n_train])
            val_sub = torch.utils.data.Subset(val_ds, indices[n_train:])

            train_loaders[name] = DataLoader(
                train_sub, batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers > 0,
                drop_last=True)
            val_loaders[name] = DataLoader(
                val_sub, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers > 0)
            if verbose: print(f"  [{name}] {n_train} train / {n_val} val")
        else:
            train_loaders[name] = DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers > 0,
                drop_last=True)
            if verbose: print(f"  [{name}] {len(ds)} samples")
    if val_ratio > 0:
        return train_loaders, val_loaders
    return train_loaders

