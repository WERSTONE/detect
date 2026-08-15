"""Data loaders for the pose-anchor attribute redesign."""

from __future__ import annotations

from torch.utils.data import ConcatDataset, Dataset

from test_model.final.data import (
    AttrDataset,
    DetectDataset,
    FinalSequentialValLoader,
    PoseDataset,
    _NO_AUG,
    _merge_task_cfg,
    _resolve_root,
    make_loader,
)


def build_final_train_loader(cfg: dict):
    """Build the same single random ConcatDataset train loader used by final."""
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
    input_size = int(data_cfg.get("input_size", 640))
    train_workers = int(cfg["training"].get("workers", 8))
    val_workers = max(1, train_workers // 4)
    total_bs = int(cfg["training"]["batch_size"])

    tasks: dict[str, Dataset] = {}
    order: list[str] = []

    det_cfg = _merge_task_cfg(train_cfg.get("detect", {}), val_cfg.get("detect", {}))
    det_root = _resolve_root(det_cfg.get("root", ""))
    det_img = det_cfg.get("images", "valid/images")
    det_lbl = det_cfg.get("labels", "valid/labels")
    if (det_root / det_img).exists() and (det_root / det_lbl).exists():
        tasks["detect"] = DetectDataset(
            det_root,
            det_img,
            det_lbl,
            input_size=input_size,
            augment=False,
            **_NO_AUG,
        )
        order.append("detect")

    attr_cfg = _merge_task_cfg(train_cfg.get("attr", {}), val_cfg.get("attr", {}))
    attr_root = _resolve_root(attr_cfg.get("root", ""))
    if attr_root.exists():
        tasks["attr"] = AttrDataset(
            attr_root,
            split=attr_cfg.get("split", "val"),
            input_size=input_size,
            augment=False,
            **_NO_AUG,
        )
        order.append("attr")

    pose_cfg = _merge_task_cfg(train_cfg.get("pose", {}), val_cfg.get("pose", {}))
    pose_root = _resolve_root(pose_cfg.get("root", ""))
    pose_img = pose_cfg.get("images", "val2017")
    pose_lbl = pose_cfg.get("labels", "labels/val2017")
    if (pose_root / pose_img).exists() and (pose_root / pose_lbl).exists():
        tasks["pose"] = PoseDataset(
            pose_root,
            pose_img,
            pose_lbl,
            input_size=input_size,
            source_class_format=pose_cfg.get("class_id_format", "yolo80"),
            augment=False,
            **_NO_AUG,
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
