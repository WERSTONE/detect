"""Ultralytics YOLO adapters for direct Vigil inference.

These wrappers are inference-only. They convert Ultralytics Results objects into
the VigilModelBase.detect() dictionary so they can be used with create_engine().
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch

from Vigil.models.base import VigilModelBase


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


_ULTRA_CONFIG_DIR = _project_root() / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRA_CONFIG_DIR))
_ULTRA_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _empty_person(device=None):
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "scores": torch.empty((0,), dtype=torch.float32, device=device),
        "kpts": torch.empty((0, 17, 3), dtype=torch.float32, device=device),
        "helmet": torch.empty((0,), dtype=torch.float32, device=device),
        "smoking": torch.empty((0,), dtype=torch.float32, device=device),
    }


def _empty_det(device=None):
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "scores": torch.empty((0,), dtype=torch.float32, device=device),
    }


class UltralyticsVigilAdapter(VigilModelBase):
    """Inference adapter around ultralytics.YOLO.

    Args:
        weights: Weight path/model name, for example "yolov8n.pt" or
            "yolov8n-pose.pt". A YAML name such as "yolov8n.yaml" is useful for
            smoke tests without downloading weights.
        task: "detect" or "pose".
        input_size: Inference image size passed to Ultralytics.
        conf: Confidence threshold passed to Ultralytics. The Vigil engine still
            applies its own thresholds afterward.
        iou: NMS IoU threshold passed to Ultralytics.
        class_map: Optional mapping from Ultralytics class names to Vigil names.
    """

    DEFAULT_CLASS_MAP = {
        "person": "person",
        "fire": "fire",
        "flame": "fire",
        "smoke": "fire",
        "water": "water",
        "water_leak": "water",
        "water leak": "water",
        "leak": "water",
        "flood": "water",
    }

    def __init__(
        self,
        weights: str,
        task: str = "detect",
        input_size: int | Sequence[int] = 640,
        conf: float = 0.001,
        iou: float = 0.7,
        class_map: dict[str, str] | None = None,
        verbose: bool = False,
        **unused_kwargs,
    ):
        from ultralytics import YOLO

        try:
            self.yolo = YOLO(weights, task=task)
        except KeyError as exc:
            if str(exc).strip("'\"") == "model":
                raise ValueError(
                    "Ultralytics YOLO adapters require Ultralytics-format weights, "
                    f"but got an incompatible checkpoint: {weights!r}. "
                    "For --model yolov8_pose use weights such as 'yolov8n-pose.pt' "
                    "or a local Ultralytics pose checkpoint. For Vigil checkpoints "
                    "such as checkpoints/vigil_v2/pretrain_best.pt, use --model vigil_v2."
                ) from exc
            raise
        self.task = task
        self.imgsz = input_size
        self.conf = conf
        self.iou = iou
        self.verbose = verbose
        self.device = "cpu"
        self.class_map = dict(self.DEFAULT_CLASS_MAP)
        if class_map:
            self.class_map.update({str(k): str(v) for k, v in class_map.items()})

    def eval(self):
        if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "eval"):
            self.yolo.model.eval()
        return self

    def to(self, device):
        self.device = str(device)
        if hasattr(self.yolo, "to"):
            self.yolo.to(device)
        return self

    @property
    def input_size(self):
        if isinstance(self.imgsz, Iterable) and not isinstance(self.imgsz, (str, bytes)):
            vals = list(self.imgsz)
            if len(vals) >= 2:
                return int(vals[1]), int(vals[0])
        return int(self.imgsz), int(self.imgsz)

    @property
    def num_params(self):
        if hasattr(self.yolo, "model"):
            return sum(p.numel() for p in self.yolo.model.parameters())
        return 0

    def compute_loss(self, samples):
        raise NotImplementedError("Ultralytics YOLO adapters are inference-only.")

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> dict:
        results = self.yolo.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
        )
        result = results[0]
        device = getattr(result.boxes.xyxy, "device", None) if result.boxes is not None else None
        out = {
            "person": _empty_person(device),
            "fire": _empty_det(device),
            "water": _empty_det(device),
        }
        if result.boxes is None or result.boxes.xyxy.numel() == 0:
            return out

        boxes = result.boxes.xyxy.float()
        scores = result.boxes.conf.float()
        cls_ids = result.boxes.cls.long()
        names = result.names or getattr(self.yolo.model, "names", {})

        pose_kpts = None
        if getattr(result, "keypoints", None) is not None and result.keypoints.xy is not None:
            xy = result.keypoints.xy.float()
            if getattr(result.keypoints, "conf", None) is not None:
                vis = result.keypoints.conf.float().unsqueeze(-1)
            else:
                vis = torch.ones((*xy.shape[:2], 1), dtype=xy.dtype, device=xy.device)
            pose_kpts = torch.cat([xy, vis], dim=-1)

        buckets = {"person": [], "fire": [], "water": []}
        for i, cls_id in enumerate(cls_ids.tolist()):
            raw_name = str(names.get(cls_id, cls_id)).lower().replace("-", "_").strip()
            vigil_name = self.class_map.get(raw_name)
            if vigil_name in buckets:
                buckets[vigil_name].append(i)

        for vigil_name, idxs in buckets.items():
            if not idxs:
                continue
            idx = torch.tensor(idxs, dtype=torch.long, device=boxes.device)
            entry = {"boxes": boxes[idx], "scores": scores[idx]}
            if vigil_name == "person":
                if pose_kpts is not None and pose_kpts.shape[0] >= boxes.shape[0]:
                    kpts = pose_kpts[idx]
                else:
                    kpts = torch.zeros((len(idxs), 17, 3), dtype=boxes.dtype, device=boxes.device)
                entry.update({
                    "kpts": kpts,
                    "helmet": torch.full((len(idxs),), float("nan"), dtype=boxes.dtype, device=boxes.device),
                    "smoking": torch.full((len(idxs),), float("nan"), dtype=boxes.dtype, device=boxes.device),
                })
            out[vigil_name] = entry
        return out

