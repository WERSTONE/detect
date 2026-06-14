"""Register Ultralytics YOLOv8 detection for direct inference."""

from pathlib import Path

from Vigil.models.registry import register_model
from Vigil.models.ultralytics_adapter import UltralyticsVigilAdapter

DEFAULT_WEIGHTS = Path("checkpoints/yolov8/yolov8n.pt")


def _resolve_weights(pretrained=None, explicit_weights=None):
    weights = pretrained or explicit_weights
    if weights:
        return str(weights)
    if DEFAULT_WEIGHTS.exists():
        return str(DEFAULT_WEIGHTS)
    raise FileNotFoundError(
        "YOLOv8 weights were not found. Place weights at "
        f"{DEFAULT_WEIGHTS.as_posix()} or pass --weights/weights explicitly. "
        "The adapter does not download default weights automatically."
    )


@register_model("yolov8")
def create_model(pretrained=None, **kwargs):
    weights = _resolve_weights(pretrained, kwargs.pop("weights", None))
    return UltralyticsVigilAdapter(weights=weights, task="detect", **kwargs)
