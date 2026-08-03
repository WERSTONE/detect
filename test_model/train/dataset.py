"""COCO dataset with YOLO-format labels and YOLO-style augmentations.

Supports:
- YOLO-format label loading with keypoints
- Mosaic (4-image composition)
- HSV, affine/perspective-style geometry, flip, MixUp, CutMix, Copy-Paste
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


# COCO80 class mapping.
COCO80_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


COCO_CATEGORY_ID_TO_80 = {
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

# COCO keypoint skeleton (for flip mapping)
KPT_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class COCOMultiTaskDataset(Dataset):
    """COCO dataset for multi-task (detection + pose).

    Loads YOLO-format labels.
    Returns dict with:
        'image': [3, 640, 640] normalized tensor
        'boxes': [M, 4] xyxy in 640x640 space
        'classes': [M] 0..79 by default (0=person)
        'kpts': [M, 17, 3] keypoints (only valid for person class)
    """

    def __init__(self, data_dir, img_dir, label_dir=None,
                 input_size=640, use_mosaic=True, augment=True,
                 class_id_format='yolo80',
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                 degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                 perspective=0.0, flip_lr=0.5, flip_ud=0.0, bgr=0.0,
                 mosaic_prob=0.5, mixup_prob=0.0, cutmix_prob=0.0,
                 copy_paste_prob=0.0, copy_paste_ioa=0.3,
                 copy_paste_max_objects=8, keep_classes=None,
                 person_only=False, require_keypoints=False):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / img_dir
        self.label_dir = self.data_dir / label_dir if label_dir else None
        self.input_size = input_size
        self.use_mosaic = use_mosaic and augment
        self.mosaic_prob = mosaic_prob
        self.augment = augment
        self.class_id_format = class_id_format
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.bgr = bgr
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_ioa = copy_paste_ioa
        self.copy_paste_max_objects = int(copy_paste_max_objects)
        self.keep_classes = (
            None if keep_classes is None else {int(c) for c in keep_classes}
        )
        self.person_only = bool(person_only)
        self.require_keypoints = bool(require_keypoints)
        self._base_use_mosaic = self.use_mosaic
        self._base_mixup_prob = self.mixup_prob
        self._base_cutmix_prob = self.cutmix_prob
        self._base_copy_paste_prob = self.copy_paste_prob

        # Collect image-label pairs
        self.samples = []
        n_labels = 0
        n_no_person = 0
        n_no_keypoints = 0
        n_no_image = 0
        if self.label_dir and self.label_dir.exists():
            for lb in self.label_dir.glob('*.txt'):
                n_labels += 1
                if self.person_only or self.require_keypoints:
                    has_person, has_keypoints = self._label_has_person_keypoints(lb)
                    if self.person_only and not has_person:
                        n_no_person += 1
                        continue
                    if self.require_keypoints and not has_keypoints:
                        n_no_keypoints += 1
                        continue

                img_name = lb.stem + '.jpg'
                img_path = self.img_dir / img_name
                if not img_path.exists():
                    img_path = self.img_dir / (lb.stem + '.png')
                if img_path.exists():
                    self.samples.append((str(img_path), str(lb)))
                else:
                    n_no_image += 1

            if self.person_only or self.require_keypoints:
                print(f"  Dataset({img_dir}): {n_labels} labels, "
                      f"{n_no_person} filtered (no person), "
                      f"{n_no_keypoints} filtered (no visible keypoints), "
                      f"{n_no_image} no image, {len(self.samples)} kept", flush=True)
        else:
            # Only images (no labels) for prediction
            for ext in ('*.jpg', '*.png', '*.jpeg'):
                for p in self.img_dir.glob(ext):
                    self.samples.append((str(p), None))

        if not self.samples:
            raise RuntimeError(f"No samples found in {data_dir} / {img_dir}")

        # YOLOv8 pretrained backbones are trained on images scaled to [0, 1],
        # without ImageNet mean/std normalization.

    def __len__(self):
        return len(self.samples)

    def _label_has_person_keypoints(self, label_path):
        has_person = False
        has_keypoints = False
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        raw_cls = int(float(parts[0]))
                    except ValueError:
                        continue
                    mapped_cls = self._map_class_id(raw_cls, len(parts) > 5)
                    if mapped_cls != 0:
                        continue
                    has_person = True
                    kpt_data = parts[5:]
                    for vis_idx in range(2, len(kpt_data), 3):
                        try:
                            if float(kpt_data[vis_idx]) > 0:
                                has_keypoints = True
                                break
                        except ValueError:
                            continue
                    if has_keypoints:
                        break
        except OSError:
            pass
        return has_person, has_keypoints

    def set_close_mosaic(self, close=False):
        """Disable strong composite augmentations during final fine-tuning."""
        if close:
            self.use_mosaic = False
            self.mixup_prob = 0.0
            self.cutmix_prob = 0.0
            self.copy_paste_prob = 0.0
        else:
            self.use_mosaic = self._base_use_mosaic
            self.mixup_prob = self._base_mixup_prob
            self.cutmix_prob = self._base_cutmix_prob
            self.copy_paste_prob = self._base_copy_paste_prob

    def __getitem__(self, idx):
        sample = self._load_composed(idx)
        if self.augment:
            sample = self._apply_composite_augmentations(sample)
        return sample

    def _load_composed(self, idx):
        if self.use_mosaic and random.random() < self.mosaic_prob:
            return self._load_mosaic(idx)
        return self._load_single(idx)

    def _load_single(self, idx):
        img, boxes, classes, kpts, attrs, attr_masks, img_path, orig_shape = self._load_raw(idx)
        orig_h, orig_w = orig_shape

        if self.augment:
            img, boxes, kpts, scale = self._resize_keep_ratio_no_pad(img, boxes, kpts)
            img, boxes, classes, kpts, attrs, attr_masks = self._augment(
                img, boxes, classes, kpts,
                attrs, attr_masks,
                output_size=(self.input_size, self.input_size),
                area_thr=0.10,
            )
            boxes, classes, kpts, attrs, attr_masks = self._sanitize_targets(
                boxes, classes, kpts, attrs, attr_masks)
            pad_l, pad_t = 0, 0
        else:
            img, boxes, kpts, (pad_l, pad_t), scale = self._letterbox(img, boxes, kpts)
            boxes, classes, kpts, attrs, attr_masks = self._sanitize_targets(
                boxes, classes, kpts, attrs, attr_masks)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros(0, dtype=torch.long)
        kpts_t = np.array(kpts, dtype=np.float32) if kpts else np.zeros((0, 17, 3), dtype=np.float32)
        kpts = torch.from_numpy(kpts_t)
        attrs_t = np.array(attrs, dtype=np.float32) if attrs else np.zeros((0, 4), dtype=np.float32)
        attr_masks_t = (
            np.array(attr_masks, dtype=np.float32)
            if attr_masks else np.zeros((0, 4), dtype=np.float32)
        )

        return {
            'image': img,
            'boxes': boxes,
            'classes': classes,
            'kpts': kpts,
            'attrs': torch.from_numpy(attrs_t),
            'attr_mask': torch.from_numpy(attr_masks_t),
            'scale': scale,
            'pad': (pad_l, pad_t),
            'img_path': img_path,
            'orig_shape': (orig_h, orig_w),
            'image_id': self._image_id_from_path(img_path),
        }

    def _load_raw(self, idx):
        img_path, label_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, classes, kpts, attrs, attr_masks = [], [], [], [], []
        if label_path and Path(label_path).exists():
            boxes, classes, kpts, attrs, attr_masks = self._parse_yolo_label(
                label_path, img.shape[1], img.shape[0])
        boxes, classes, kpts, attrs, attr_masks = self._filter_targets_by_class(
            boxes, classes, kpts, attrs, attr_masks)
        return img, boxes, classes, kpts, attrs, attr_masks, img_path, (orig_h, orig_w)

    def _resize_keep_ratio_no_pad(self, img, boxes, kpts):
        """YOLO train-time resize: scale max side to imgsz without padding."""
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        if scale != 1.0:
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            boxes = [[v * scale for v in box] for box in boxes]
            new_kpts = []
            for k in kpts:
                nk = k.copy()
                nk[:, 0] *= scale
                nk[:, 1] *= scale
                new_kpts.append(nk)
            kpts = new_kpts
        return img, boxes, kpts, scale

    def _load_mosaic(self, idx):
        """YOLOv8-style 4-image Mosaic followed by RandomPerspective."""
        s = self.input_size
        yc, xc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_classes = []
        mosaic_kpts = []
        mosaic_attrs = []
        mosaic_attr_masks = []

        for i, idx_i in enumerate(indices):
            img, boxes_i, classes_i, kpts_i, attrs_i, attr_masks_i, _, _ = self._load_raw(idx_i)
            img, boxes_i, kpts_i, _ = self._resize_keep_ratio_no_pad(img, boxes_i, kpts_i)
            h, w = img.shape[:2]

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            if x2a <= x1a or y2a <= y1a:
                continue
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b

            for b, box in enumerate(boxes_i):
                mosaic_boxes.append([box[0] + padw, box[1] + padh, box[2] + padw, box[3] + padh])
                mosaic_classes.append(classes_i[b])
                mosaic_attrs.append(attrs_i[b] if b < len(attrs_i) else np.zeros(4, dtype=np.float32))
                mosaic_attr_masks.append(
                    attr_masks_i[b] if b < len(attr_masks_i) else np.zeros(4, dtype=np.float32))
                if kpts_i and b < len(kpts_i):
                    k = kpts_i[b].copy()
                    k[..., 0] += padw
                    k[..., 1] += padh
                    mosaic_kpts.append(k)
                else:
                    mosaic_kpts.append(np.zeros((17, 3), dtype=np.float32))

        mosaic_img, mosaic_boxes, mosaic_classes, mosaic_kpts, mosaic_attrs, mosaic_attr_masks = self._augment(
            mosaic_img, mosaic_boxes, mosaic_classes, mosaic_kpts,
            mosaic_attrs, mosaic_attr_masks,
            output_size=(s, s),
            area_thr=0.10,
        )

        mosaic_boxes, mosaic_classes, mosaic_kpts, mosaic_attrs, mosaic_attr_masks = self._sanitize_targets(
            mosaic_boxes, mosaic_classes, mosaic_kpts, mosaic_attrs, mosaic_attr_masks)

        # Normalize to YOLO-style [0, 1] tensors.
        mosaic_img = mosaic_img.astype(np.float32) / 255.0
        mosaic_img = torch.from_numpy(mosaic_img).permute(2, 0, 1)

        boxes_t = torch.tensor(mosaic_boxes, dtype=torch.float32) if mosaic_boxes else torch.zeros(0, 4)
        classes_t = torch.tensor(mosaic_classes, dtype=torch.long) if mosaic_classes else torch.zeros(0, dtype=torch.long)
        kpts_t = (torch.from_numpy(np.asarray(mosaic_kpts, dtype=np.float32))
                  if mosaic_kpts else torch.zeros(0, 17, 3))
        attrs_t = (torch.from_numpy(np.asarray(mosaic_attrs, dtype=np.float32))
                   if mosaic_attrs else torch.zeros(0, 4))
        attr_masks_t = (torch.from_numpy(np.asarray(mosaic_attr_masks, dtype=np.float32))
                        if mosaic_attr_masks else torch.zeros(0, 4))

        return {
            'image': mosaic_img,
            'boxes': boxes_t,
            'classes': classes_t,
            'kpts': kpts_t,
            'attrs': attrs_t,
            'attr_mask': attr_masks_t,
            'scale': 1.0,
            'pad': (0, 0),
            'img_path': '',
            'orig_shape': (self.input_size, self.input_size),
            'image_id': None,
        }

    @staticmethod
    def _image_id_from_path(img_path):
        stem = Path(img_path).stem
        return int(stem) if stem.isdigit() else None

    def _parse_yolo_label(self, label_path, img_w, img_h):
        """Parse YOLO-format label file.

        Format per line:
            cls x y w h [px1 py1 pv1 ... px17 py17 pv17]

        Returns:
            boxes: [[x1, y1, x2, y2], ...] in pixel coordinates
            classes: [cls, ...]
            kpts: [[17, 3], ...] in pixel coordinates
        """
        boxes, classes, kpts, attrs, attr_masks = [], [], [], [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                has_kpts = len(parts) > 5
                mapped_cls = self._map_class_id(cls, has_kpts)
                if mapped_cls is None:
                    continue

                xc, yc, w, h = map(float, parts[1:5])
                # Convert normalized xywh -> pixel xyxy
                w_px, h_px = w * img_w, h * img_h
                x1 = (xc * img_w) - w_px / 2
                y1 = (yc * img_h) - h_px / 2
                x2 = x1 + w_px
                y2 = y1 + h_px

                boxes.append([x1, y1, x2, y2])
                classes.append(mapped_cls)

                # Keypoints
                kpt = np.zeros((17, 3), dtype=np.float32)
                if has_kpts and mapped_cls == 0:
                    kpt_data = parts[5:56]
                    for j in range(min(17, len(kpt_data) // 3)):
                        px = float(kpt_data[j * 3]) * img_w
                        py = float(kpt_data[j * 3 + 1]) * img_h
                        pv = float(kpt_data[j * 3 + 2])
                        kpt[j] = [px, py, pv]
                kpts.append(kpt)

                attr = np.zeros(4, dtype=np.float32)
                attr_mask = np.zeros(4, dtype=np.float32)
                if mapped_cls == 0 and len(parts) >= 64:
                    attr[:] = [float(x) for x in parts[56:60]]
                    attr_mask[:] = [float(x) for x in parts[60:64]]
                attrs.append(attr)
                attr_masks.append(attr_mask)

        return boxes, classes, kpts, attrs, attr_masks

    def _filter_targets_by_class(self, boxes, classes, kpts, attrs=None, attr_masks=None):
        if self.keep_classes is None or not classes:
            return boxes, classes, kpts, attrs or [], attr_masks or []
        keep = [i for i, cls in enumerate(classes) if int(cls) in self.keep_classes]
        if len(keep) == len(classes):
            return boxes, classes, kpts, attrs or [], attr_masks or []
        return (
            [boxes[i] for i in keep],
            [classes[i] for i in keep],
            [kpts[i] for i in keep] if kpts else [],
            [attrs[i] for i in keep] if attrs else [],
            [attr_masks[i] for i in keep] if attr_masks else [],
        )

    def _map_class_id(self, cls, has_kpts=False):
        """Map source class id to the model's internal class id.

        label format:
          - yolo80/internal80: standard YOLO COCO ids, person=0, car=2, ...
          - coco/coco80: COCO category ids, person=1, car=3, ...
          - auto: prefer yolo80, except keypoint person annotations may be 0 or 1.
        """
        fmt = str(self.class_id_format).lower()
        if fmt in ('yolo80', 'internal80'):
            return cls if 0 <= cls < len(COCO80_CLASSES) else None
        if fmt in ('coco', 'coco80'):
            return COCO_CATEGORY_ID_TO_80.get(cls)
        if fmt == 'auto':
            if has_kpts and cls in (0, 1):
                return 0
            if 0 <= cls < len(COCO80_CLASSES):
                return cls
            return COCO_CATEGORY_ID_TO_80.get(cls)
        return cls if 0 <= cls < len(COCO80_CLASSES) else None

    def _sanitize_targets(self, boxes, classes, kpts, attrs=None, attr_masks=None):
        """Clip targets to the training image and drop invalid annotations."""
        if not boxes:
            return [], [], [], [], []

        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes_np = np.asarray(classes, dtype=np.int64).reshape(-1)
        if kpts:
            kpts_np = np.asarray(kpts, dtype=np.float32).reshape(-1, 17, 3)
        else:
            kpts_np = np.zeros((len(boxes_np), 17, 3), dtype=np.float32)
        attrs_np = (
            np.asarray(attrs, dtype=np.float32).reshape(-1, 4)
            if attrs else np.zeros((len(boxes_np), 4), dtype=np.float32)
        )
        attr_masks_np = (
            np.asarray(attr_masks, dtype=np.float32).reshape(-1, 4)
            if attr_masks else np.zeros((len(boxes_np), 4), dtype=np.float32)
        )

        n = min(len(boxes_np), len(classes_np), len(kpts_np), len(attrs_np), len(attr_masks_np))
        boxes_np = boxes_np[:n]
        classes_np = classes_np[:n]
        kpts_np = kpts_np[:n]
        attrs_np = attrs_np[:n]
        attr_masks_np = attr_masks_np[:n]

        finite_boxes = np.isfinite(boxes_np).all(axis=1)
        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, self.input_size - 1)
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, self.input_size - 1)
        valid_boxes = (
            finite_boxes &
            (boxes_np[:, 2] - boxes_np[:, 0] > 2) &
            (boxes_np[:, 3] - boxes_np[:, 1] > 2)
        )

        boxes_np = boxes_np[valid_boxes]
        classes_np = classes_np[valid_boxes]
        kpts_np = kpts_np[valid_boxes]
        attrs_np = attrs_np[valid_boxes]
        attr_masks_np = attr_masks_np[valid_boxes]
        if len(boxes_np) == 0:
            return [], [], [], [], []

        xy = kpts_np[..., :2]
        vis = kpts_np[..., 2]
        finite_xy = np.isfinite(xy).all(axis=-1)
        finite_vis = np.isfinite(vis)
        outside = (
            (xy[..., 0] < 0) | (xy[..., 0] > self.input_size - 1) |
            (xy[..., 1] < 0) | (xy[..., 1] > self.input_size - 1)
        )
        keep_vis = (vis > 0) & finite_xy & finite_vis & ~outside
        kpts_np[..., 0] = np.clip(np.nan_to_num(xy[..., 0], nan=0.0), 0, self.input_size - 1)
        kpts_np[..., 1] = np.clip(np.nan_to_num(xy[..., 1], nan=0.0), 0, self.input_size - 1)
        kpts_np[..., 2] = np.where(keep_vis, vis, 0.0)
        kpts_np[classes_np != 0] = 0.0
        attrs_np[classes_np != 0] = 0.0
        attr_masks_np[classes_np != 0] = 0.0

        return (
            boxes_np.tolist(),
            classes_np.tolist(),
            [k for k in kpts_np],
            [a for a in attrs_np],
            [m for m in attr_masks_np],
        )

    def _augment(self, img, boxes, classes, kpts, attrs=None, attr_masks=None,
                 output_size=None, area_thr=0.10):
        """Apply YOLO-style geometry, HSV, channel, and flip augmentations."""
        img, boxes, classes, kpts, attrs, attr_masks = self._random_affine(
            img, boxes, classes, kpts,
            attrs, attr_masks,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            output_size=output_size,
            area_thr=area_thr,
        )
        img = self._hsv_augment(img, self.hsv_h, self.hsv_s, self.hsv_v)

        if self.bgr > 0 and random.random() < self.bgr:
            img = img[..., ::-1].copy()

        if self.flip_ud > 0 and random.random() < self.flip_ud:
            img = img[::-1, :, :].copy()
            h = img.shape[0]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                boxes[i] = [x1, h - y2, x2, h - y1]
            for i, k in enumerate(kpts):
                if k.any():
                    k[:, 1] = h - k[:, 1]
                    kpts[i] = k

        # Horizontal flip
        if random.random() < self.flip_lr:
            img = img[:, ::-1].copy()
            w = img.shape[1]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                boxes[i] = [w - x2, y1, w - x1, y2]
            for i, k in enumerate(kpts):
                if k.any():
                    k[:, 0] = w - k[:, 0]
                    kpts[i] = k[KPT_FLIP_MAP]

        return img, boxes, classes, kpts, attrs or [], attr_masks or []

    def _random_affine(self, img, boxes, classes, kpts, attrs=None, attr_masks=None,
                       degrees=0.0, translate=0.1,
                       scale=0.5, shear=0.0, perspective=0.0,
                       output_size=None, area_thr=0.10):
        """YOLO-style random affine transform with synchronized targets."""
        h, w = img.shape[:2]
        out_w, out_h = output_size if output_size is not None else (w, h)
        if max(abs(degrees), abs(translate), abs(scale), abs(shear), abs(perspective)) <= 0:
            if output_size is not None and (out_w != w or out_h != h):
                img = cv2.warpAffine(
                    img, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
                    dsize=(out_w, out_h), borderValue=(114, 114, 114))
            return img, boxes, classes, kpts, attrs or [], attr_masks or []

        c = np.eye(3, dtype=np.float32)
        c[0, 2] = -w / 2
        c[1, 2] = -h / 2

        p = np.eye(3, dtype=np.float32)
        p[2, 0] = random.uniform(-perspective, perspective)
        p[2, 1] = random.uniform(-perspective, perspective)

        r = np.eye(3, dtype=np.float32)
        angle = random.uniform(-degrees, degrees)
        scale_factor = random.uniform(1 - scale, 1 + scale)
        r[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale_factor)

        s = np.eye(3, dtype=np.float32)
        s[0, 1] = math.tan(math.radians(random.uniform(-shear, shear)))
        s[1, 0] = math.tan(math.radians(random.uniform(-shear, shear)))

        t = np.eye(3, dtype=np.float32)
        t[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_w
        t[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_h

        m = t @ s @ r @ p @ c
        if perspective:
            img = cv2.warpPerspective(
                img, m, dsize=(out_w, out_h), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(
                img, m[:2], dsize=(out_w, out_h), borderValue=(114, 114, 114))

        if not boxes:
            return img, boxes, classes, kpts, attrs or [], attr_masks or []

        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes_np = np.asarray(classes, dtype=np.int64).reshape(-1)
        attrs_np = (
            np.asarray(attrs, dtype=np.float32).reshape(-1, 4)
            if attrs else np.zeros((len(boxes_np), 4), dtype=np.float32)
        )
        attr_masks_np = (
            np.asarray(attr_masks, dtype=np.float32).reshape(-1, 4)
            if attr_masks else np.zeros((len(boxes_np), 4), dtype=np.float32)
        )
        n = len(boxes_np)
        corners = np.ones((n * 4, 3), dtype=np.float32)
        corners[:, :2] = boxes_np[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        warped = corners @ m.T
        warped_xy = (warped[:, :2] / warped[:, 2:3].clip(1e-6)
                     if perspective else warped[:, :2])
        warped_xy = warped_xy.reshape(n, 8)
        x = warped_xy[:, [0, 2, 4, 6]]
        y = warped_xy[:, [1, 3, 5, 7]]
        new_boxes = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1)), dtype=boxes_np.dtype
        ).reshape(4, n).T
        new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, out_w)
        new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, out_h)

        new_kpts_np = np.zeros((n, 17, 3), dtype=np.float32)
        if kpts:
            kpts_np = np.asarray(kpts, dtype=np.float32).reshape(-1, 17, 3)
            pts = np.ones((kpts_np.shape[0] * 17, 3), dtype=np.float32)
            pts[:, :2] = kpts_np[..., :2].reshape(-1, 2)
            warped_pts = pts @ m.T
            warped_pts_xy = (warped_pts[:, :2] / warped_pts[:, 2:3].clip(1e-6)
                             if perspective else warped_pts[:, :2])
            warped_pts_xy = warped_pts_xy.reshape(kpts_np.shape[0], 17, 2)
            kpts_np[..., :2] = warped_pts_xy
            outside = (
                (kpts_np[..., 0] < 0) | (kpts_np[..., 0] > out_w) |
                (kpts_np[..., 1] < 0) | (kpts_np[..., 1] > out_h)
            )
            kpts_np[..., 2] = np.where(outside, 0.0, kpts_np[..., 2])
            kpts_np[..., 0] = np.clip(kpts_np[..., 0], 0, out_w)
            kpts_np[..., 1] = np.clip(kpts_np[..., 1], 0, out_h)
            new_kpts_np[:len(kpts_np)] = kpts_np[:n]

        keep = self._box_candidates(boxes_np.T * scale_factor, new_boxes.T, area_thr=area_thr)
        new_boxes = new_boxes[keep]
        classes_np = classes_np[:n][keep]
        new_kpts_np = new_kpts_np[:n][keep]
        attrs_np = attrs_np[:n][keep]
        attr_masks_np = attr_masks_np[:n][keep]

        return (
            img,
            new_boxes.tolist(),
            classes_np.tolist(),
            [k for k in new_kpts_np],
            [a for a in attrs_np],
            [m for m in attr_masks_np],
        )

    def _apply_composite_augmentations(self, sample):
        """Apply sample-mixing augmentations after resize to input_size."""
        if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
            sample = self._copy_paste(sample, random.randrange(len(self)))
        if self.cutmix_prob > 0 and random.random() < self.cutmix_prob:
            sample = self._cutmix(sample, random.randrange(len(self)))
        if self.mixup_prob > 0 and random.random() < self.mixup_prob:
            sample = self._mixup(sample, random.randrange(len(self)))
        return sample

    def _mixup(self, sample, src_idx):
        src = self._load_composed(src_idx)
        ratio = float(np.random.beta(32.0, 32.0))
        sample['image'] = sample['image'] * ratio + src['image'] * (1.0 - ratio)
        sample['boxes'] = self._cat_targets(sample['boxes'], src['boxes'], 4, torch.float32)
        sample['classes'] = self._cat_targets(sample['classes'], src['classes'], 0, torch.long)
        sample['kpts'] = self._cat_targets(sample['kpts'], src['kpts'], (17, 3), torch.float32)
        sample['attrs'] = self._cat_targets(sample['attrs'], src['attrs'], (4,), torch.float32)
        sample['attr_mask'] = self._cat_targets(sample['attr_mask'], src['attr_mask'], (4,), torch.float32)
        return sample

    def _copy_paste(self, sample, src_idx):
        # Ultralytics CopyPaste depends on instance segments/masks. With bbox-only
        # YOLO labels, official detection training effectively skips it; rectangular
        # patch copy-paste is a different augmentation and hurts parity.
        return sample

    def _cutmix(self, sample, src_idx):
        src = self._load_composed(src_idx)
        if len(src['boxes']) == 0:
            return sample
        j = random.randrange(len(src['boxes']))
        box = src['boxes'][j].round().long()
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        if x2 - x1 <= 4 or y2 - y1 <= 4:
            return sample
        pad = random.uniform(0.0, 0.35)
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pw, ph = min(self.input_size, int(bw * (1 + pad))), min(self.input_size, int(bh * (1 + pad)))
        sx1 = max(0, int(cx - pw / 2))
        sy1 = max(0, int(cy - ph / 2))
        sx2 = min(self.input_size, sx1 + pw)
        sy2 = min(self.input_size, sy1 + ph)
        pw, ph = sx2 - sx1, sy2 - sy1
        if pw <= 4 or ph <= 4:
            return sample
        tx1 = random.randint(0, self.input_size - pw)
        ty1 = random.randint(0, self.input_size - ph)
        patch_box = torch.tensor([tx1, ty1, tx1 + pw, ty1 + ph], dtype=torch.float32)
        if self._max_ioa(patch_box, sample['boxes']) > self.copy_paste_ioa:
            return sample

        sample['image'][:, ty1:ty1 + ph, tx1:tx1 + pw] = src['image'][:, sy1:sy2, sx1:sx2]
        dx, dy = tx1 - sx1, ty1 - sy1
        keep_boxes, keep_classes, keep_kpts, keep_attrs, keep_attr_masks = [], [], [], [], []
        for bi, src_box in enumerate(src['boxes']):
            inter = self._intersect_box(src_box, torch.tensor([sx1, sy1, sx2, sy2], dtype=torch.float32))
            if inter is None:
                continue
            old_area = max(float((src_box[2] - src_box[0]) * (src_box[3] - src_box[1])), 1.0)
            new_area = float((inter[2] - inter[0]) * (inter[3] - inter[1]))
            if new_area / old_area < 0.1:
                continue
            inter[[0, 2]] += dx
            inter[[1, 3]] += dy
            keep_boxes.append(inter)
            keep_classes.append(src['classes'][bi])
            kp = src['kpts'][bi].clone()
            kp[:, 0] += dx
            kp[:, 1] += dy
            outside = (
                (kp[:, 0] < tx1) | (kp[:, 0] > tx1 + pw) |
                (kp[:, 1] < ty1) | (kp[:, 1] > ty1 + ph)
            )
            kp[outside, 2] = 0
            keep_kpts.append(kp)
            keep_attrs.append(src['attrs'][bi].clone())
            keep_attr_masks.append(src['attr_mask'][bi].clone())
        if keep_boxes:
            sample['boxes'] = self._cat_targets(
                sample['boxes'], torch.stack(keep_boxes), 4, torch.float32)
            sample['classes'] = self._cat_targets(
                sample['classes'], torch.stack(keep_classes), 0, torch.long)
            sample['kpts'] = self._cat_targets(
                sample['kpts'], torch.stack(keep_kpts), (17, 3), torch.float32)
            sample['attrs'] = self._cat_targets(
                sample['attrs'], torch.stack(keep_attrs), (4,), torch.float32)
            sample['attr_mask'] = self._cat_targets(
                sample['attr_mask'], torch.stack(keep_attr_masks), (4,), torch.float32)
        return sample

    @staticmethod
    def _cat_targets(a, b, empty_shape, dtype):
        if not torch.is_tensor(b):
            b = torch.tensor(b, dtype=dtype)
        if torch.is_tensor(a):
            b = b.to(device=a.device, dtype=dtype)
        else:
            b = b.to(dtype=dtype)
        if a.numel() == 0:
            return b.clone()
        if b.numel() == 0:
            return a
        return torch.cat([a, b], dim=0)

    @staticmethod
    def _intersect_box(box, crop):
        x1 = max(float(box[0]), float(crop[0]))
        y1 = max(float(box[1]), float(crop[1]))
        x2 = min(float(box[2]), float(crop[2]))
        y2 = min(float(box[3]), float(crop[3]))
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            return None
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    @staticmethod
    def _max_ioa(box, boxes):
        if boxes is None or len(boxes) == 0:
            return 0.0
        box = box.to(dtype=torch.float32)
        boxes = boxes.to(dtype=torch.float32)
        ix1 = torch.maximum(box[0], boxes[:, 0])
        iy1 = torch.maximum(box[1], boxes[:, 1])
        ix2 = torch.minimum(box[2], boxes[:, 2])
        iy2 = torch.minimum(box[3], boxes[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area = ((box[2] - box[0]) * (box[3] - box[1])).clamp(min=1)
        return float((inter / area).max().item())

    @staticmethod
    def _box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.10, eps=1e-16):
        """Ultralytics RandomPerspective box filter."""
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (
            (w2 > wh_thr) &
            (h2 > wh_thr) &
            (w2 * h2 / (w1 * h1 + eps) > area_thr) &
            (ar < ar_thr)
        )

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
        'attrs': [x.get('attrs', torch.zeros(0, 4)) for x in batch],
        'attr_mask': [x.get('attr_mask', torch.zeros(0, 4)) for x in batch],
        'scale': [x.get('scale', 1.0) for x in batch],
        'pad': [x.get('pad', (0, 0)) for x in batch],
        'img_path': [x.get('img_path', '') for x in batch],
        'orig_shape': [x.get('orig_shape', None) for x in batch],
        'image_id': [x.get('image_id', None) for x in batch],
    }


def create_dataloader(data_dir, img_dir, label_dir=None,
                      input_size=640, batch_size=16,
                      use_mosaic=True, augment=True,
                      shuffle=True, num_workers=4,
                      distributed=False, rank=0, world_size=1,
                      drop_last=True, class_id_format='yolo80',
                      hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                      degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                      perspective=0.0, flip_lr=0.5, flip_ud=0.0, bgr=0.0,
                      mosaic_prob=0.5, mixup_prob=0.0, cutmix_prob=0.0,
                      copy_paste_prob=0.0, copy_paste_ioa=0.3,
                      copy_paste_max_objects=8, keep_classes=None,
                      person_only=False, require_keypoints=False,
                      persistent_workers=False):
    """Create DataLoader for COCO dataset."""
    dataset = COCOMultiTaskDataset(
        data_dir=data_dir,
        img_dir=img_dir,
        label_dir=label_dir,
        input_size=input_size,
        use_mosaic=use_mosaic,
        augment=augment,
        class_id_format=class_id_format,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flip_lr=flip_lr,
        flip_ud=flip_ud,
        bgr=bgr,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        cutmix_prob=cutmix_prob,
        copy_paste_prob=copy_paste_prob,
        copy_paste_ioa=copy_paste_ioa,
        copy_paste_max_objects=copy_paste_max_objects,
        keep_classes=keep_classes,
        person_only=person_only,
        require_keypoints=require_keypoints,
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
        drop_last=drop_last,
        persistent_workers=(num_workers > 0 and persistent_workers),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader



