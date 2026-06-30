"""Vigil-v3: YOLO-pose based multi-task model.

v3 uses an s/m YOLO-pose checkpoint as the person box/keypoint foundation and
adds separate heads for pump-room tasks:
  - anomaly detection: fire / water
  - person attributes: helmet / smoking
  - action states: fall / wave

The initial implementation keeps inference compatible with the current Vigil
contract by delegating person-pose detection to the YOLO-pose adapter. The new
heads are part of the architecture and state dict, ready for the next training
step where YOLO features, assigners and losses will be wired together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from Vigil.models.base import VigilModelBase
from Vigil.models.registry import register_model
from Vigil.models.ultralytics_adapter import UltralyticsVigilAdapter
from Vigil.models.vigil_v3.heads import ActionStateHead, AnomalyHead, AttributeHead, PersonFeaturePool

YOLO_POSE_WEIGHTS = {
    "n": Path("checkpoints/yolo_pose/yolov8n-pose.pt"),
    "s": Path("checkpoints/yolo_pose/yolov8s-pose.pt"),
    "m": Path("checkpoints/yolo_pose/yolov8m-pose.pt"),
}


def _resolve_pose_weights(variant: str, pose_weights: str | None = None) -> str:
    if pose_weights:
        path = Path(pose_weights)
        if not path.exists():
            raise FileNotFoundError(f"YOLO-pose weights not found: {path}")
        return str(path)

    key = variant.lower().replace("yolov8", "").replace("-pose", "").strip("_-") or "s"
    candidates = []
    if key in YOLO_POSE_WEIGHTS:
        candidates.append(YOLO_POSE_WEIGHTS[key])
    candidates.extend([YOLO_POSE_WEIGHTS["s"], YOLO_POSE_WEIGHTS["m"], YOLO_POSE_WEIGHTS["n"]])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No YOLO-pose weights found for Vigil-v3. Searched: {searched}")


def _looks_like_yolo_pose_checkpoint(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return "yolo" in name and "pose" in name


class VigilModelV3(VigilModelBase, nn.Module):
    """YOLO-pose based v3 architecture.

    Args:
        variant: "s" or "m" are recommended for v3. "n" remains useful for
            smoke tests.
        pose_weights: Ultralytics pose checkpoint used by the person/pose branch.
        input_size: Inference size passed to Ultralytics.
        yolo_conf: YOLO confidence threshold before Vigil post-filtering.
        yolo_iou: YOLO NMS IoU threshold.
        feature_channels: P3/P4/P5 channel plan for the additional heads. This
            is a stable v3 state-dict interface until YOLO feature extraction is
            wired into training.
    """

    def __init__(
        self,
        variant: str = "s",
        pose_weights: str | None = None,
        input_size: int = 640,
        yolo_conf: float = 0.001,
        yolo_iou: float = 0.7,
        feature_channels: list[int] | None = None,
        head_channels: int = 256,
        attr_hidden: int = 256,
        action_hidden: int = 256,
        reg_max: int = 16,
        **unused_kwargs,
    ):
        super().__init__()
        self.variant = variant
        self.pose_weights = _resolve_pose_weights(variant, pose_weights)
        self._input_size = (input_size, input_size)
        self.reg_max = reg_max

        self.pose_branch = UltralyticsVigilAdapter(
            weights=self.pose_weights,
            task="pose",
            input_size=input_size,
            conf=yolo_conf,
            iou=yolo_iou,
        )

        # Channel defaults roughly follow YOLO s/m feature pyramid scales. The
        # heads are intentionally decoupled from Ultralytics internals for now.
        feature_channels = feature_channels or [128, 256, 512]
        self.person_pool = PersonFeaturePool(feature_channels, out_channels=head_channels)
        self.anomaly_head = AnomalyHead(feature_channels, hidden=head_channels, num_classes=2, reg_max=reg_max)
        self.attribute_head = AttributeHead(head_channels, hidden=attr_hidden)
        self.action_head = ActionStateHead(head_channels, hidden=action_hidden)

    def eval(self):
        super().eval()
        self.pose_branch.eval()
        return self

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = None
        if args:
            device = args[0]
        if "device" in kwargs:
            device = kwargs["device"]
        if device is not None:
            self.pose_branch.to(device)
        return module

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @property
    def num_params(self) -> int:
        own_params = sum(param.numel() for param in super().parameters())
        return own_params + int(self.pose_branch.num_params)

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> dict:
        outputs = self.pose_branch.detect(frame)
        person = outputs.setdefault("person", self._empty_person())
        count = len(person["boxes"])
        device = person["boxes"].device if isinstance(person["boxes"], torch.Tensor) else None
        dtype = person["boxes"].dtype if isinstance(person["boxes"], torch.Tensor) else torch.float32

        # v3 action states are per-person single-frame logits. Until the action
        # head is trained, expose neutral zeros and let temporal postprocess rely
        # on geometry/keypoints.
        person.setdefault("fall", torch.zeros(count, dtype=dtype, device=device))
        person.setdefault("wave", torch.zeros(count, dtype=dtype, device=device))
        outputs.setdefault("fire", self._empty_det(device=device, dtype=dtype))
        outputs.setdefault("water", self._empty_det(device=device, dtype=dtype))
        return outputs

    def compute_loss(self, samples) -> dict[str, torch.Tensor | float]:
        raise NotImplementedError(
            "Vigil-v3 architecture is registered for inference and head development, "
            "but full multi-task training still needs YOLO feature extraction, "
            "assigners and losses to be wired in."
        )

    def forward_extra_heads(self, features: list[torch.Tensor]) -> dict[str, dict[str, torch.Tensor | list[torch.Tensor]]]:
        person_features = self.person_pool(features)
        return {
            "anomaly": self.anomaly_head(features),
            "attribute": self.attribute_head(person_features),
            "action": self.action_head(person_features),
        }

    @staticmethod
    def _empty_person(device=None, dtype=torch.float32):
        return {
            "boxes": torch.empty((0, 4), dtype=dtype, device=device),
            "scores": torch.empty((0,), dtype=dtype, device=device),
            "kpts": torch.empty((0, 17, 3), dtype=dtype, device=device),
            "helmet": torch.empty((0,), dtype=dtype, device=device),
            "smoking": torch.empty((0,), dtype=dtype, device=device),
            "fall": torch.empty((0,), dtype=dtype, device=device),
            "wave": torch.empty((0,), dtype=dtype, device=device),
        }

    @staticmethod
    def _empty_det(device=None, dtype=torch.float32):
        return {
            "boxes": torch.empty((0, 4), dtype=dtype, device=device),
            "scores": torch.empty((0,), dtype=dtype, device=device),
        }


def _create_v3(pretrained=None, default_variant: str = "s", **kwargs):
    variant = kwargs.pop("variant", default_variant)
    if pretrained and _looks_like_yolo_pose_checkpoint(pretrained):
        kwargs.setdefault("pose_weights", str(pretrained))
        pretrained = None
    model = VigilModelV3(variant=variant, **kwargs)

    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    return model


@register_model("vigil_v3")
@register_model("vigil_v3_s")
def create_model(pretrained=None, **kwargs):
    return _create_v3(pretrained=pretrained, default_variant="s", **kwargs)


@register_model("vigil_v3_m")
def create_model_m(pretrained=None, **kwargs):
    return _create_v3(pretrained=pretrained, default_variant="m", **kwargs)
