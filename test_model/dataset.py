"""COCO dataset with YOLO-format labels, augmentations (mosaic, HSV, flip).

Supports:
- YOLO-format label loading with keypoints
- Mosaic (4-image composition)
- HSV, scale, translate, flip augmentations
- Letterbox resize to 640x640
- Multi-GPU distributed sampling
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# COCO 20-class mapping (orig_id -> 0..19 label)
COCO20_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'dog', 'cat', 'horse', 'bird', 'cow', 'sheep',
    'chair', 'dining table', 'laptop',
    'backpack', 'sports ball', 'bottle', 'cup', 'cell phone',
]

COCO_ORIG_ID_TO_20 = {
    0: 0,    # person
    2: 1,    # bicycle
    3: 2,    # car
    4: 3,    # motorcycle
    6: 4,    # bus
    8: 5,    # truck
    17: 6,   # dog
    15: 7,   # cat
    14: 8,   # horse
    16: 9,   # bird
    21: 10,  # cow
    19: 11,  # sheep
    62: 12,  # chair
    67: 13,  # dining table
    73: 14,  # laptop
    27: 15,  # backpack
    37: 16,  # sports ball
    44: 17,  # bottle
    47: 18,  # cup
    77: 19,  # cell phone
}

# COCO keypoint skeleton (for flip mapping)
KPT_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class COCOMultiTaskDataset(Dataset):
    """COCO dataset for multi-task (detection + pose).

    Loads YOLO-format labels.
    Returns dict with:
        'image': [3, 640, 640] normalized tensor
        'boxes': [M, 4] xyxy in 640x640 space
        'classes': [M] 0..19 (0=person)
        'kpts': [M, 17, 3] keypoints (only valid for person class)
    """

    def __init__(self, data_dir, img_dir, label_dir=None,
                 input_size=640, use_mosaic=True, augment=True):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / img_dir
        self.label_dir = self.data_dir / label_dir if label_dir else None
        self.input_size = input_size
        self.use_mosaic = use_mosaic and augment
        self.augment = augment

        # Collect image-label pairs
        self.samples = []
        if self.label_dir and self.label_dir.exists():
            for lb in self.label_dir.glob('*.txt'):
                img_name = lb.stem + '.jpg'
                img_path = self.img_dir / img_name
                if not img_path.exists():
                    img_path = self.img_dir / (lb.stem + '.png')
                if img_path.exists():
                    self.samples.append((str(img_path), str(lb)))
        else:
            # Only images (no labels) — for prediction
            for ext in ('*.jpg', '*.png', '*.jpeg'):
                for p in self.img_dir.glob(ext):
                    self.samples.append((str(p), None))

        if not self.samples:
            raise RuntimeError(f"No samples found in {data_dir} / {img_dir}")

        # Normalization stats (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_mosaic and random.random() < 0.5:
            return self._load_mosaic(idx)
        return self._load_single(idx)

    def _load_single(self, idx):
        img_path, label_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, classes, kpts = [], [], []
        if label_path and Path(label_path).exists():
            boxes, classes, kpts = self._parse_yolo_label(label_path, img.shape[1], img.shape[0])

        # Augment
        if self.augment:
            img, boxes, kpts = self._augment(img, boxes, kpts)

        # Letterbox resize
        img, boxes, kpts, (pad_l, pad_t), scale = self._letterbox(img, boxes, kpts)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros(0, dtype=torch.long)
        kpts_t = np.array(kpts, dtype=np.float32) if kpts else np.zeros((0, 17, 3), dtype=np.float32)
        kpts = torch.from_numpy(kpts_t)

        return {
            'image': img,
            'boxes': boxes,
            'classes': classes,
            'kpts': kpts,
            'scale': scale,
            'pad': (pad_l, pad_t),
        }

    def _load_mosaic(self, idx):
        """Mosaic augmentation: compose 4 images into one."""
        input_size = self.input_size
        s = input_size

        # Mosaic center
        xc = int(random.uniform(s * 0.25, s * 0.75))
        yc = int(random.uniform(s * 0.25, s * 0.75))

        # Select 3 other random images
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        random.shuffle(indices)

        mosaic_img = np.zeros((s, s, 3), dtype=np.uint8)
        mosaic_boxes = []
        mosaic_classes = []
        mosaic_kpts = []

        placements = [
            (0, 0, yc, xc),           # top-left
            (0, xc, yc, s),           # top-right
            (yc, 0, s, xc),           # bottom-left
            (yc, xc, s, s),           # bottom-right
        ]

        for i, idx_i in enumerate(indices):
            img_path, label_path = self.samples[idx_i]
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes_i, classes_i, kpts_i = [], [], []
            if label_path and Path(label_path).exists():
                boxes_i, classes_i, kpts_i = self._parse_yolo_label(
                    label_path, img.shape[1], img.shape[0])

            h0, w0 = img.shape[:2]
            y1, x1, y2, x2 = placements[i]

            # Scale image to fit placement zone
            scale = min((y2 - y1) / h0, (x2 - x1) / w0) * random.uniform(0.8, 1.2)
            new_w, new_h = int(w0 * scale), int(h0 * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # Random offset within zone
            ox = x1 + random.randint(0, max(0, (x2 - x1) - new_w))
            oy = y1 + random.randint(0, max(0, (y2 - y1) - new_h))

            # Place image
            h_place, w_place = img_resized.shape[:2]
            if oy + h_place > s:
                h_place = s - oy
            if ox + w_place > s:
                w_place = s - ox
            mosaic_img[oy:oy + h_place, ox:ox + w_place] = img_resized[:h_place, :w_place]

            # Transform boxes
            for b, boxes_l in enumerate(boxes_i):
                new_box = [
                    boxes_l[0] * scale * w0 + ox,
                    boxes_l[1] * scale * h0 + oy,
                    boxes_l[2] * scale * w0 + ox,
                    boxes_l[3] * scale * h0 + oy,
                ]
                new_box[0] = max(ox, min(ox + w_place, new_box[0]))
                new_box[1] = max(oy, min(oy + h_place, new_box[1]))
                new_box[2] = max(ox, min(ox + w_place, new_box[2]))
                new_box[3] = max(oy, min(oy + h_place, new_box[3]))

                if new_box[2] > new_box[0] and new_box[3] > new_box[1]:
                    mosaic_boxes.append(new_box)
                    mosaic_classes.append(classes_i[b])

                    if kpts_i and b < len(kpts_i):
                        k = kpts_i[b].copy()
                        k[..., 0] = k[..., 0] * scale * w0 + ox
                        k[..., 1] = k[..., 1] * scale * h0 + oy
                        mosaic_kpts.append(k)

        # HSV augment on mosaic
        if self.augment:
            mosaic_img = self._hsv_augment(mosaic_img)

        # Normalize
        mosaic_img = mosaic_img.astype(np.float32) / 255.0
        mosaic_img = (mosaic_img - self.mean) / self.std
        mosaic_img = torch.from_numpy(mosaic_img).permute(2, 0, 1)

        boxes_t = torch.tensor(mosaic_boxes, dtype=torch.float32) if mosaic_boxes else torch.zeros(0, 4)
        classes_t = torch.tensor(mosaic_classes, dtype=torch.long) if mosaic_classes else torch.zeros(0, dtype=torch.long)
        kpts_t = torch.tensor(mosaic_kpts, dtype=torch.float32) if mosaic_kpts else torch.zeros(0, 17, 3)

        return {
            'image': mosaic_img,
            'boxes': boxes_t,
            'classes': classes_t,
            'kpts': kpts_t,
            'scale': 1.0,
            'pad': (0, 0),
        }

    def _parse_yolo_label(self, label_path, img_w, img_h):
        """Parse YOLO-format label file.

        Format per line:
            cls x y w h [px1 py1 pv1 ... px17 py17 pv17]

        Returns:
            boxes: [[x1, y1, x2, y2], ...] in pixel coordinates
            classes: [cls, ...]
            kpts: [[17, 3], ...] in pixel coordinates
        """
        boxes, classes, kpts = [], [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                if cls not in COCO_ORIG_ID_TO_20:
                    continue
                cls_20 = COCO_ORIG_ID_TO_20[cls]

                xc, yc, w, h = map(float, parts[1:5])
                # Convert normalized xywh -> pixel xyxy
                w_px, h_px = w * img_w, h * img_h
                x1 = (xc * img_w) - w_px / 2
                y1 = (yc * img_h) - h_px / 2
                x2 = x1 + w_px
                y2 = y1 + h_px

                boxes.append([x1, y1, x2, y2])
                classes.append(cls_20)

                # Keypoints
                kpt = np.zeros((17, 3), dtype=np.float32)
                if len(parts) > 5 and cls == 0:
                    kpt_data = parts[5:]
                    for j in range(min(17, len(kpt_data) // 3)):
                        px = float(kpt_data[j * 3]) * img_w
                        py = float(kpt_data[j * 3 + 1]) * img_h
                        pv = float(kpt_data[j * 3 + 2])
                        kpt[j] = [px, py, pv]
                kpts.append(kpt)

        return boxes, classes, kpts

    def _augment(self, img, boxes, kpts):
        """Apply HSV + flip augmentations."""
        img = self._hsv_augment(img)

        # Horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            w = img.shape[1]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                boxes[i] = [w - x2, y1, w - x1, y2]
            for i, k in enumerate(kpts):
                if k.any():
                    k[:, 0] = w - k[:, 0]
                    kpts[i] = k[KPT_FLIP_MAP]

        return img, boxes, kpts

    @staticmethod
    def _hsv_augment(img, hgain=0.015, sgain=0.7, vgain=0.4):
        """HSV color augmentation."""
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    def _letterbox(self, img, boxes, kpts):
        """Resize + pad to input_size x input_size."""
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h
        pad_l, pad_t = pad_w // 2, pad_h // 2

        img = cv2.copyMakeBorder(img, pad_t, pad_h - pad_t, pad_l, pad_w - pad_l,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        new_boxes = []
        for box in boxes:
            new_box = [
                box[0] * scale + pad_l,
                box[1] * scale + pad_t,
                box[2] * scale + pad_l,
                box[3] * scale + pad_t,
            ]
            new_boxes.append(new_box)

        new_kpts = []
        for k in kpts:
            nk = k.copy()
            nk[:, 0] = k[:, 0] * scale + pad_l
            nk[:, 1] = k[:, 1] * scale + pad_t
            new_kpts.append(nk)

        return img, new_boxes, new_kpts, (pad_l, pad_t), scale


def collate_fn(batch):
    """Collate batch of dicts."""
    images = torch.stack([x['image'] for x in batch])
    return {
        'image': images,
        'boxes': [x['boxes'] for x in batch],
        'classes': [x['classes'] for x in batch],
        'kpts': [x['kpts'] for x in batch],
    }


def create_dataloader(data_dir, img_dir, label_dir=None,
                      input_size=640, batch_size=16,
                      use_mosaic=True, augment=True,
                      shuffle=True, num_workers=4,
                      distributed=False, rank=0, world_size=1):
    """Create DataLoader for COCO dataset."""
    dataset = COCOMultiTaskDataset(
        data_dir=data_dir,
        img_dir=img_dir,
        label_dir=label_dir,
        input_size=input_size,
        use_mosaic=use_mosaic,
        augment=augment,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=shuffle)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader
