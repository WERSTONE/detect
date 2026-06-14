"""Register Ultralytics YOLOv8-pose for direct inference."""

from pathlib import Path

from Vigil.models.registry import register_model
from Vigil.models.ultralytics_adapter import UltralyticsVigilAdapter

DEFAULT_WEIGHTS = Path("checkpoints/yolov8_pose/yolov8n-pose.pt")


def _resolve_weights(pretrained=None, explicit_weights=None):
    weights = pretrained or explicit_weights
    if weights:
        return str(weights)
    if DEFAULT_WEIGHTS.exists():
        return str(DEFAULT_WEIGHTS)
    raise FileNotFoundError(
        "YOLOv8-pose weights were not found. Place weights at "
        f"{DEFAULT_WEIGHTS.as_posix()} or pass --weights/weights explicitly. "
        "The adapter does not download default weights automatically."
    )


@register_model("yolov8_pose")
@register_model("yolov8-pose")
def create_model(pretrained=None, **kwargs):
    weights = _resolve_weights(pretrained, kwargs.pop("weights", None))
    return UltralyticsVigilAdapter(weights=weights, task="pose", **kwargs)
