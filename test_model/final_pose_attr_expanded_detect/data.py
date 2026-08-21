"""Data loaders for the pose-anchor attribute redesign."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from test_model.final.data import (
    DetectDataset as BaseDetectDataset,
    FinalSequentialValLoader,
    AttrDataset as BaseAttrDataset,
    PoseDataset as BasePoseDataset,
    _NO_AUG,
    _apply_test_dataset_policy,
    _attr_dataset_options,
    _make_optional_dataset,
    _merge_task_cfg,
    _resolve_root,
)


def _class_mask(class_ids, num_classes=7):
    mask = torch.zeros(int(num_classes), dtype=torch.float32)
    for class_id in class_ids or ():
        index = int(class_id) - 1
        if 0 <= index < len(mask):
            mask[index] = 1.0
    return mask


def _resolve_manifest(root: Path, value):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _load_manifest(root: Path, value):
    path = _resolve_manifest(root, value)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"Supervision manifest not found: {path}. "
            "Run scripts/build_expanded_detect_supervision_manifest.py first."
        )
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[str(record["image"]).replace("\\", "/")] = record
    return records


def _relative_image(root: Path, image_path) -> str:
    return str(Path(image_path).resolve().relative_to(root.resolve())).replace("\\", "/")


class DetectDataset(BaseDetectDataset):
    def __init__(self, *args, domain_valid_class_ids=None,
                 supervision_manifest=None, domain_num_classes=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_num_classes = int(domain_num_classes)
        self._static_domain_mask = _class_mask(domain_valid_class_ids or [1, 2, 3, 4, 7], self.domain_num_classes)
        self._supervision_manifest = _load_manifest(self.root, supervision_manifest)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        record = self._supervision_manifest.get(_relative_image(self.root, self.samples[idx].image))
        if record is None:
            if self._supervision_manifest:
                raise KeyError(
                    f"No supervision-manifest record for {self.samples[idx].image}. "
                    "Regenerate the manifest for this dataset root."
                )
            item["domain_valid_mask"] = self._static_domain_mask.clone()
        else:
            item["domain_valid_mask"] = _class_mask(
                record.get("valid_class_ids", []), self.domain_num_classes
            )
        return item


class AttrDataset(BaseAttrDataset):
    def __init__(self, *args, domain_valid_class_ids=None,
                 supervision_manifest=None, domain_num_classes=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_num_classes = int(domain_num_classes)
        self._static_domain_mask = _class_mask(domain_valid_class_ids or [1, 2, 3, 4, 7], self.domain_num_classes)
        self._supervision_manifest = _load_manifest(self.root, supervision_manifest)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        record = self._supervision_manifest.get(_relative_image(self.root, self.samples[idx].image))
        if record is None:
            if self._supervision_manifest:
                raise KeyError(
                    f"No supervision-manifest record for {self.samples[idx].image}. "
                    "Regenerate the manifest for this dataset root."
                )
            item["domain_valid_mask"] = self._static_domain_mask.clone()
        else:
            item["domain_valid_mask"] = _class_mask(
                record.get("valid_class_ids", []), self.domain_num_classes
            )
        return item


class PoseDataset(BasePoseDataset):
    def __init__(self, *args, domain_valid_class_ids=None,
                 domain_num_classes=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_valid_class_mask = _class_mask(
            domain_valid_class_ids or [1, 2, 3, 4, 7], domain_num_classes
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["domain_valid_mask"] = self.domain_valid_class_mask.clone()
        return item


def final_collate(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "classes": [item["classes"] for item in batch],
        "kpts": [item["kpts"] for item in batch],
        "attrs": [item["attrs"] for item in batch],
        "attr_mask": [item["attr_mask"] for item in batch],
        "domain_valid_mask": [item["domain_valid_mask"] for item in batch],
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


def _detect_source_configs(detect_cfg: dict) -> list[dict]:
    sources = detect_cfg.get("sources")
    if sources:
        return [dict(source) for source in sources]
    return [dict(detect_cfg)]


def _build_detect_dataset(source_cfg: dict, input_size: int, augment_cfg: dict, split_policy: dict, split: str):
    det_root = _resolve_root(source_cfg["root"])
    dataset = DetectDataset(
        det_root,
        source_cfg.get("images", "train/images" if split == "train" else "valid/images"),
        source_cfg.get("labels", "train/labels" if split == "train" else "valid/labels"),
        input_size=input_size,
        class_offset=source_cfg.get("class_offset", 1),
        domain_valid_class_ids=source_cfg.get("valid_class_ids", [1, 2, 3, 4, 7]),
        supervision_manifest=source_cfg.get("supervision_manifest"),
        domain_num_classes=7,
        **augment_cfg,
    )
    test_ds = _make_optional_dataset(
        DetectDataset,
        det_root,
        source_cfg.get("test_images", "test/images"),
        source_cfg.get("test_labels", "test/labels"),
        input_size=input_size,
        class_offset=source_cfg.get("class_offset", 1),
        domain_valid_class_ids=source_cfg.get("valid_class_ids", [1, 2, 3, 4, 7]),
        supervision_manifest=source_cfg.get("supervision_manifest"),
        domain_num_classes=7,
        **augment_cfg,
    )
    return _apply_test_dataset_policy(
        dataset,
        test_ds,
        split,
        split_policy,
        salt=source_cfg.get("split_salt", f"detect:{det_root}"),
    )


def build_final_train_loader(cfg: dict):
    """Build the same single random ConcatDataset train loader used by final."""
    data_cfg = cfg["data"]
    train_cfg = data_cfg["train"]
    split_policy = data_cfg.get("split_policy", {}) or {}
    workers = int(cfg["training"].get("workers", 8))
    input_size = int(data_cfg.get("input_size", 640))
    batch_size = int(cfg["training"]["batch_size"])

    detect_aug = cfg.get("augmentation", {}).get("detect", {})
    attr_aug = cfg.get("augmentation", {}).get("attr", {})
    pose_aug = cfg.get("augmentation", {}).get("pose", {})

    detect_sources = _detect_source_configs(train_cfg["detect"])
    detect_parts = [
        _build_detect_dataset(source, input_size, detect_aug, split_policy, "train")
        for source in detect_sources
    ]
    detect_ds = detect_parts[0] if len(detect_parts) == 1 else ConcatDataset(detect_parts)
    attr_ds = AttrDataset(
        _resolve_root(train_cfg["attr"]["root"]),
        split=train_cfg["attr"].get("split", "train"),
        input_size=input_size,
        **_attr_dataset_options(train_cfg["attr"], split_policy),
        domain_valid_class_ids=train_cfg["attr"].get("domain_valid_class_ids", [1, 2, 3, 4, 7]),
        supervision_manifest=train_cfg["attr"].get("domain_supervision_manifest"),
        domain_num_classes=7,
        **attr_aug,
    )
    pose_root = _resolve_root(train_cfg["pose"]["root"])
    pose_ds = PoseDataset(
        pose_root,
        train_cfg["pose"].get("images", "train2017"),
        train_cfg["pose"].get("labels", "labels/train2017"),
        input_size=input_size,
        source_class_format=train_cfg["pose"].get("class_id_format", "yolo80"),
        domain_valid_class_ids=train_cfg["pose"].get("domain_valid_class_ids", [1, 2, 3, 4, 7]),
        domain_num_classes=7,
        **pose_aug,
    )
    pose_test_ds = _make_optional_dataset(
        PoseDataset,
        pose_root,
        train_cfg["pose"].get("test_images", "test2017"),
        train_cfg["pose"].get("test_labels", "labels_person/test2017"),
        input_size=input_size,
        source_class_format=train_cfg["pose"].get("class_id_format", "yolo80"),
        domain_valid_class_ids=train_cfg["pose"].get("domain_valid_class_ids", [1, 2, 3, 4, 7]),
        domain_num_classes=7,
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
        combined,
        batch_size,
        workers,
        shuffle=True,
        weights=None,
        drop_last=True,
    )
    print(
        "PoseAttr train samples: "
        f"combined={len(combined)} detect={len(detect_ds)} "
        f"attr={len(attr_ds)} pose={len(pose_ds)}"
    )
    if len(detect_parts) > 1:
        print(
            "PoseAttr detect sources: "
            + ", ".join(f"{idx + 1}={len(ds)}" for idx, ds in enumerate(detect_parts))
        )
    print(
        "PoseAttr train batch: "
        f"size={batch_size} shuffle=True (single random loader, no per-task composition guarantee)"
    )
    return loader


def build_final_val_loader(cfg: dict) -> FinalSequentialValLoader:
    """Validation evaluates every available val sample exactly once."""
    data_cfg = cfg["data"]
    train_cfg = data_cfg["train"]
    val_cfg = data_cfg.get("val", {}) or {}
    split_policy = data_cfg.get("split_policy", {}) or {}
    input_size = int(data_cfg.get("input_size", 640))
    train_workers = int(cfg["training"].get("workers", 8))
    val_workers = max(1, train_workers // 4)
    total_bs = int(cfg["training"]["batch_size"])

    tasks: dict[str, Dataset] = {}
    order: list[str] = []

    det_cfg = _merge_task_cfg(train_cfg.get("detect", {}), val_cfg.get("detect", {}))
    det_parts = []
    for source in _detect_source_configs(det_cfg):
        det_root = _resolve_root(source.get("root", ""))
        det_img = source.get("images", "valid/images")
        det_lbl = source.get("labels", "valid/labels")
        if (det_root / det_img).exists() and (det_root / det_lbl).exists():
            det_parts.append(_build_detect_dataset(
                {
                    **source,
                    "images": det_img,
                    "labels": det_lbl,
                },
                input_size,
                {"augment": False, **_NO_AUG},
                split_policy,
                "val",
            ))
    if det_parts:
        tasks["detect"] = det_parts[0] if len(det_parts) == 1 else ConcatDataset(det_parts)
        order.append("detect")

    attr_cfg = _merge_task_cfg(train_cfg.get("attr", {}), val_cfg.get("attr", {}))
    attr_root = _resolve_root(attr_cfg.get("root", ""))
    if attr_root.exists():
        tasks["attr"] = AttrDataset(
            attr_root,
            split=attr_cfg.get("split", "val"),
            input_size=input_size,
            augment=False,
            domain_valid_class_ids=attr_cfg.get("domain_valid_class_ids", [1, 2, 3, 4, 7]),
            supervision_manifest=attr_cfg.get("domain_supervision_manifest"),
            domain_num_classes=7,
            **_NO_AUG,
            **_attr_dataset_options(attr_cfg, split_policy),
        )
        order.append("attr")

    pose_cfg = _merge_task_cfg(train_cfg.get("pose", {}), val_cfg.get("pose", {}))
    pose_root = _resolve_root(pose_cfg.get("root", ""))
    pose_img = pose_cfg.get("images", "val2017")
    pose_lbl = pose_cfg.get("labels", "labels/val2017")
    if (pose_root / pose_img).exists() and (pose_root / pose_lbl).exists():
        pose_base = PoseDataset(
            pose_root,
            pose_img,
            pose_lbl,
            input_size=input_size,
            source_class_format=pose_cfg.get("class_id_format", "yolo80"),
            domain_valid_class_ids=pose_cfg.get("domain_valid_class_ids", [1, 2, 3, 4, 7]),
            domain_num_classes=7,
            augment=False,
            **_NO_AUG,
        )
        pose_test = _make_optional_dataset(
            PoseDataset,
            pose_root,
            pose_cfg.get("test_images", "test2017"),
            pose_cfg.get("test_labels", "labels_person/test2017"),
            input_size=input_size,
            source_class_format=pose_cfg.get("class_id_format", "yolo80"),
            domain_valid_class_ids=pose_cfg.get("domain_valid_class_ids", [1, 2, 3, 4, 7]),
            domain_num_classes=7,
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
        order.append("pose")

    if not tasks:
        raise RuntimeError("No validation data sources available. Add data.val in the config.")

    loaders = {
        name: make_loader(
            tasks[name],
            total_bs,
            val_workers,
            shuffle=False,
            drop_last=False,
        )
        for name in order
    }
    print(
        "PoseAttr val samples: "
        + ", ".join(f"{name}={len(tasks[name])}" for name in order)
    )
    print(
        "PoseAttr val steps: "
        + ", ".join(f"{name}={len(loaders[name])}" for name in order)
        + f" total={sum(len(loaders[name]) for name in order)}"
    )
    return FinalSequentialValLoader(loaders)
